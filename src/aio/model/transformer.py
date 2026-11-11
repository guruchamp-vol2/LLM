"""Decoder-only transformer in plain JAX.

Llama-style architecture:
    token embedding
    -> N x [ RMSNorm -> attention (RoPE, grouped-query) -> residual
             RMSNorm -> SwiGLU MLP, or Mixture of Experts -> residual ]
    -> RMSNorm -> output projection (tied to the embedding by default)

Parameters are a plain dict of arrays. Each layer's weights are stacked along a leading
"layer" axis, so the layers run in a lax.scan: fast to compile, and easy to shard later.

The forward pass can also run with a key/value cache, so text can be generated one token
at a time without recomputing the whole prefix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

F32 = jnp.float32


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32_000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4          # < n_heads = grouped-query attention; 1 = multi-query
    d_ff: int = 0                # 0 = about 8/3 * d_model, rounded up to a multiple of 64
    max_seq_len: int = 2048
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    # Mixture of Experts: 0 = dense MLP in every layer
    n_experts: int = 0
    experts_per_token: int = 2
    moe_aux_weight: float = 0.01  # weight of the load-balancing loss
    dtype: str = "float32"        # compute dtype; "bfloat16" on TPU. Parameters stay float32.
    remat: bool = False           # recompute each layer in the backward pass: ~1/3 more compute, far less memory
    attention: str = "xla"        # "xla" (plain einsum) or "splash" (TPU flash-attention kernel, training only)
    loss_chunk: int = 0           # >0: compute the output layer + loss this many positions at a time

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.n_experts and not 1 <= self.experts_per_token <= self.n_experts:
            raise ValueError("experts_per_token must be between 1 and n_experts")
        if self.attention not in ("xla", "splash"):
            raise ValueError(f"unknown attention {self.attention!r}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ff_dim(self) -> int:
        if self.d_ff:
            return self.d_ff
        return 64 * math.ceil(8 * self.d_model / 3 / 64)


# ---------------------------------------------------------------------- parameters
def init_params(cfg: ModelConfig, key: jax.Array) -> dict:
    """Normal(0, 0.02) weights; layer output projections scaled down by sqrt(2 * n_layers)
    so the residual stream does not grow with depth. Norm weights start at 1."""
    L, D, H, KV, Dh, F, V = (cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
                              cfg.head_dim, cfg.ff_dim, cfg.vocab_size)
    std, out_std = 0.02, 0.02 / math.sqrt(2 * L)
    k = iter(jax.random.split(key, 16))

    def normal(shape, s):
        return jax.random.normal(next(k), shape, F32) * s

    layers = {
        "attn_norm": jnp.ones((L, D), F32),
        "wq": normal((L, D, H * Dh), std),
        "wk": normal((L, D, KV * Dh), std),
        "wv": normal((L, D, KV * Dh), std),
        "wo": normal((L, H * Dh, D), out_std),
        "mlp_norm": jnp.ones((L, D), F32),
    }
    if cfg.n_experts:
        E = cfg.n_experts
        layers |= {
            "router": normal((L, D, E), std),
            "w_gate": normal((L, E, D, F), std),
            "w_up": normal((L, E, D, F), std),
            "w_down": normal((L, E, F, D), out_std),
        }
    else:
        layers |= {
            "w_gate": normal((L, D, F), std),
            "w_up": normal((L, D, F), std),
            "w_down": normal((L, F, D), out_std),
        }
    params = {"embed": normal((V, D), std), "layers": layers, "final_norm": jnp.ones((D,), F32)}
    if not cfg.tie_embeddings:
        params["lm_head"] = normal((D, V), std)
    return params


def count_params(params: dict) -> int:
    return sum(int(x.size) for x in jax.tree.leaves(params))


# ---------------------------------------------------------------------- building blocks
def rms_norm(x: jax.Array, w: jax.Array, eps: float) -> jax.Array:
    x32 = x.astype(F32)
    x32 = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (x32 * w.astype(F32)).astype(x.dtype)


def rope_tables(positions: jax.Array, head_dim: int, theta: float) -> tuple[jax.Array, jax.Array]:
    """cos and sin of position * frequency, shape (T, head_dim / 2)."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=F32) / head_dim))
    angles = positions.astype(F32)[:, None] * inv_freq[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotate pairs (x1[i], x2[i]) of each head by a position-dependent angle. x: (B, T, heads, Dh)."""
    x1, x2 = jnp.split(x.astype(F32), 2, axis=-1)
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1).astype(x.dtype)


def attention(q, k, v, q_pos, k_pos):
    """Causal grouped-query attention.

    q: (B, T, H, Dh); k, v: (B, S, KV, Dh); q_pos: (T,); k_pos: (S,).
    Each group of H / KV query heads shares one key/value head.
    """
    B, T, H, Dh = q.shape
    KV = k.shape[2]
    q = q.reshape(B, T, KV, H // KV, Dh)
    scores = jnp.einsum("btgrd,bsgd->bgrts", q, k).astype(F32) / math.sqrt(Dh)
    causal = k_pos[None, :] <= q_pos[:, None]                       # (T, S)
    scores = jnp.where(causal, scores, jnp.finfo(F32).min)
    weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    out = jnp.einsum("bgrts,bsgd->btgrd", weights, v)
    return out.reshape(B, T, H * Dh)


def splash_attention(q, k, v):
    """Causal grouped-query attention with the Pallas "splash" kernel (flash attention for
    TPU): the T x T score matrix is never stored, so memory grows with T instead of T^2.
    Same shapes as attention(); all positions start at 0 (no cache). Off TPU the kernel runs
    in interpret mode, which is slow but exact, so tests can check it on a laptop.

    The kernel works on one device's data, so on several devices it runs inside shard_map,
    split along the batch ("data" mesh axis, set by the trainer with jax.set_mesh).
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sm

    B, T, H, Dh = q.shape
    KV = k.shape[2]
    G = H // KV
    blk = min(512, T)
    if T % 128 or T % blk:
        raise ValueError(f"splash attention needs seq_len to be a multiple of 128 (got {T})")
    kernel = sk.make_splash_mqa_single_device(
        sm.MultiHeadMask([sm.CausalMask((T, T)) for _ in range(G)]),
        block_sizes=sk.BlockSizes(block_q=blk, block_kv=blk, block_kv_compute=blk, block_q_dkv=blk,
                                  block_kv_dkv=blk, block_kv_dkv_compute=blk, block_q_dq=blk, block_kv_dq=blk),
        interpret=jax.default_backend() != "tpu")

    def local(q, k, v):
        b = q.shape[0]
        qg = (q * (1.0 / math.sqrt(Dh))).astype(q.dtype).transpose(0, 2, 1, 3).reshape(b, KV, G, T, Dh)
        kg, vg = k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)      # (b, KV, T, Dh)
        out = jax.vmap(jax.vmap(kernel))(qg, kg, vg)                      # (b, KV, G, T, Dh)
        return out.reshape(b, H, T, Dh).transpose(0, 2, 1, 3).reshape(b, T, H * Dh)

    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty or "data" not in mesh.axis_names or mesh.shape["data"] == 1:
        return local(q, k, v)
    spec = jax.sharding.PartitionSpec("data")
    return jax.shard_map(local, mesh=mesh, in_specs=(spec, spec, spec), out_specs=spec, check_vma=False)(q, k, v)


def swiglu(x, w_gate, w_up, w_down):
    return (jax.nn.silu(x @ w_gate) * (x @ w_up)) @ w_down


def moe(x, lp, cfg: ModelConfig):
    """Top-k Mixture of Experts with a Switch-style load-balancing loss.

    Every expert runs on every token and the outputs are mixed by the router's gates. That
    is simple and exact, but costs as much as a dense model n_experts times wider; fine for
    small experiments. Routing tokens only to their chosen experts comes with sharding later.
    """
    E, K = cfg.n_experts, cfg.experts_per_token
    probs = jax.nn.softmax((x @ lp["router"]).astype(F32), axis=-1)       # (B, T, E)
    top_p, top_i = jax.lax.top_k(probs, K)                                 # (B, T, K)
    chosen = jax.nn.one_hot(top_i, E, dtype=F32)                           # (B, T, K, E)
    gates = jnp.einsum("btk,btke->bte", top_p / top_p.sum(-1, keepdims=True), chosen)

    h = jax.nn.silu(jnp.einsum("btd,edf->btef", x, lp["w_gate"])) * jnp.einsum("btd,edf->btef", x, lp["w_up"])
    out = jnp.einsum("btef,efd,bte->btd", h, lp["w_down"], gates.astype(x.dtype))

    # Fraction of routing slots each expert receives, times its mean router probability.
    # Minimised (value 1) when tokens spread evenly over the experts.
    frac = chosen.sum(axis=2).mean(axis=(0, 1)) / K
    aux = E * jnp.sum(frac * probs.mean(axis=(0, 1)))
    return out, aux


# ---------------------------------------------------------------------- the model
def init_cache(cfg: ModelConfig, batch: int, max_len: int | None = None) -> dict:
    S = max_len or cfg.max_seq_len
    shape = (cfg.n_layers, batch, S, cfg.n_kv_heads, cfg.head_dim)
    dt = jnp.dtype(cfg.dtype)
    return {"k": jnp.zeros(shape, dt), "v": jnp.zeros(shape, dt)}


def _trunk(params: dict, tokens: jax.Array, cfg: ModelConfig, cache: dict | None = None, start_pos=0):
    """Everything but the output layer: (final hidden states, output matrix, aux, cache)."""
    B, T = tokens.shape
    dt = jnp.dtype(cfg.dtype)
    H, KV, Dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    p = jax.tree.map(lambda a: a.astype(dt), params)

    x = p["embed"][tokens]
    positions = start_pos + jnp.arange(T)
    cos, sin = rope_tables(positions, Dh, cfg.rope_theta)

    def layer(x, xs):
        lp, ck, cv = xs
        h = rms_norm(x, lp["attn_norm"], cfg.norm_eps)
        q = apply_rope((h @ lp["wq"]).reshape(B, T, H, Dh), cos, sin)
        k = apply_rope((h @ lp["wk"]).reshape(B, T, KV, Dh), cos, sin)
        v = (h @ lp["wv"]).reshape(B, T, KV, Dh)
        if ck is None:
            k_all, v_all, k_pos = k, v, positions
        else:
            ck = jax.lax.dynamic_update_slice(ck, k.astype(ck.dtype), (0, start_pos, 0, 0))
            cv = jax.lax.dynamic_update_slice(cv, v.astype(cv.dtype), (0, start_pos, 0, 0))
            k_all, v_all, k_pos = ck, cv, jnp.arange(ck.shape[1])
        if ck is None and cfg.attention == "splash":
            a = splash_attention(q, k, v)
        else:
            a = attention(q, k_all, v_all, positions, k_pos)
        x = x + a @ lp["wo"]

        h = rms_norm(x, lp["mlp_norm"], cfg.norm_eps)
        if cfg.n_experts:
            y, aux = moe(h, lp, cfg)
        else:
            y, aux = swiglu(h, lp["w_gate"], lp["w_up"], lp["w_down"]), jnp.zeros((), F32)
        return x + y, (aux, ck, cv)

    if cache is None:
        xs = (p["layers"], None, None)
    else:
        xs = (p["layers"], cache["k"], cache["v"])
    # With remat, only each layer's input is kept for the backward pass; everything inside
    # the layer (attention scores, MLP activations) is recomputed when gradients need it.
    body = jax.checkpoint(layer) if cfg.remat and cache is None else layer
    x, (aux, new_k, new_v) = jax.lax.scan(body, x, xs)

    x = rms_norm(x, p["final_norm"], cfg.norm_eps)
    head = p["embed"].T if cfg.tie_embeddings else p["lm_head"]
    new_cache = None if cache is None else {"k": new_k, "v": new_v}
    return x, head, aux.mean(), new_cache


def forward(params: dict, tokens: jax.Array, cfg: ModelConfig, cache: dict | None = None, start_pos=0):
    """tokens: (B, T) int. Returns (logits (B, T, vocab) float32, MoE aux loss, updated cache).

    With a cache, the tokens are taken to sit at positions start_pos .. start_pos + T - 1
    and attend to everything already in the cache.
    """
    x, head, aux, new_cache = _trunk(params, tokens, cfg, cache, start_pos)
    return (x @ head).astype(F32), aux, new_cache


def token_logprobs(params: dict, tokens: jax.Array, targets: jax.Array, cfg: ModelConfig):
    """log p(targets[b, t] | tokens[b, :t+1]), shape (B, T), and the MoE aux loss.

    With cfg.loss_chunk, the logits are made and reduced a chunk of positions at a time and
    recomputed in the backward pass, so the (B, T, vocab) float32 array (gigabytes with a 32k
    vocabulary) never exists all at once.
    """
    x, head, aux, _ = _trunk(params, tokens, cfg)
    B, T, D = x.shape

    def one(xc, tc):
        logits = (xc @ head).astype(F32)
        picked = jnp.take_along_axis(logits, tc[..., None], axis=-1)[..., 0]
        return picked - jax.nn.logsumexp(logits, axis=-1)

    c = cfg.loss_chunk
    if not c or c >= T or T % c:
        return one(x, targets), aux
    xs = x.reshape(B, T // c, c, D).swapaxes(0, 1)
    ts = targets.reshape(B, T // c, c).swapaxes(0, 1)
    lp = jax.lax.map(jax.checkpoint(lambda a: one(*a)), (xs, ts))     # (T // c, B, c)
    return lp.swapaxes(0, 1).reshape(B, T), aux




def loss_fn(params: dict, tokens: jax.Array, targets: jax.Array, mask: jax.Array, cfg: ModelConfig):
    """Mean next-token cross-entropy over positions where mask is 1, plus the MoE balance loss."""
    lp, aux = token_logprobs(params, tokens, targets, cfg)
    nll = -lp
    mask = mask.astype(F32)
    ce = jnp.sum(nll * mask) / jnp.maximum(mask.sum(), 1.0)
    total = ce + cfg.moe_aux_weight * aux if cfg.n_experts else ce
    return total, {"ce": ce, "aux": aux}
