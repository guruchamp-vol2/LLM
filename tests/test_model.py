import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.model import ModelConfig, count_params, forward, init_cache, init_params, loss_fn  # noqa: E402
from aio.model.transformer import apply_rope, rope_tables  # noqa: E402

TINY = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)
TINY_MOE = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32,
                       n_experts=4, experts_per_token=2)


def _tokens(seed=0, B=2, T=16, V=64):
    return jax.random.randint(jax.random.PRNGKey(seed), (B, T), 0, V)


@pytest.mark.parametrize("cfg", [TINY, TINY_MOE], ids=["dense", "moe"])
def test_shapes_and_param_count(cfg):
    params = init_params(cfg, jax.random.PRNGKey(0))
    logits, aux, cache = forward(params, _tokens(), cfg)
    assert logits.shape == (2, 16, 64) and logits.dtype == jnp.float32
    assert cache is None and bool(jnp.all(jnp.isfinite(logits)))
    D, L, F, V, E = cfg.d_model, cfg.n_layers, cfg.ff_dim, cfg.vocab_size, max(cfg.n_experts, 1)
    attn = D * D * 2 + D * (cfg.n_kv_heads * cfg.head_dim) * 2
    mlp = 3 * D * F * E + (D * cfg.n_experts if cfg.n_experts else 0)
    assert count_params(params) == V * D + L * (attn + mlp + 2 * D) + D


def test_config_checks():
    with pytest.raises(ValueError):
        ModelConfig(d_model=30, n_heads=4)
    with pytest.raises(ValueError):
        ModelConfig(n_heads=8, n_kv_heads=3)
    assert ModelConfig(d_model=512).ff_dim == 1408  # 8/3 * 512 = 1365 -> next multiple of 64


@pytest.mark.parametrize("cfg", [TINY, TINY_MOE], ids=["dense", "moe"])
def test_causal(cfg):
    """Changing later tokens must not change the predictions for earlier positions."""
    params = init_params(cfg, jax.random.PRNGKey(1))
    a = _tokens(2)
    b = a.at[:, 10:].set((a[:, 10:] + 7) % 64)
    la, _, _ = forward(params, a, cfg)
    lb, _, _ = forward(params, b, cfg)
    np.testing.assert_allclose(la[:, :10], lb[:, :10], atol=1e-5)
    assert not np.allclose(la[:, 10:], lb[:, 10:], atol=1e-3)


@pytest.mark.parametrize("cfg", [TINY, ModelConfig(**{**TINY.__dict__, "n_kv_heads": 1}),
                                 ModelConfig(**{**TINY.__dict__, "n_kv_heads": 4})],
                         ids=["gqa", "multi_query", "multi_head"])
def test_kv_cache_matches_full_forward(cfg):
    """Feeding a prompt, then one token at a time with the cache, gives the same logits."""
    params = init_params(cfg, jax.random.PRNGKey(3))
    toks = _tokens(4, B=2, T=12)
    full, _, _ = forward(params, toks, cfg)

    step = jax.jit(forward, static_argnames="cfg")
    cache = init_cache(cfg, batch=2, max_len=16)
    out, _, cache = step(params, toks[:, :5], cfg, cache, 0)
    pieces = [out]
    for t in range(5, 12):
        out, _, cache = step(params, toks[:, t : t + 1], cfg, cache, t)
        pieces.append(out)
    np.testing.assert_allclose(jnp.concatenate(pieces, axis=1), full, atol=1e-4)


def test_rope_depends_only_on_distance():
    """After RoPE, q.k depends only on how far apart the positions are."""
    q = jax.random.normal(jax.random.PRNGKey(5), (1, 1, 1, 16))
    k = jax.random.normal(jax.random.PRNGKey(6), (1, 1, 1, 16))

    def dot(pq, pk):
        cq, sq = rope_tables(jnp.array([pq]), 16, 10_000.0)
        ck, sk = rope_tables(jnp.array([pk]), 16, 10_000.0)
        return float(jnp.sum(apply_rope(q, cq, sq) * apply_rope(k, ck, sk)))

    assert abs(dot(7, 3) - dot(104, 100)) < 1e-4
    assert abs(dot(7, 3) - dot(7, 5)) > 1e-3
    # rotation keeps vector length
    c, s = rope_tables(jnp.array([42]), 16, 10_000.0)
    assert abs(float(jnp.linalg.norm(apply_rope(q, c, s)) - jnp.linalg.norm(q))) < 1e-4


@pytest.mark.parametrize("cfg", [TINY, TINY_MOE], ids=["dense", "moe"])
def test_tiny_model_learns(cfg):
    """A tiny model memorises a fixed batch: loss falls from ~ln(64) = 4.16 to near 0."""
    params = init_params(cfg, jax.random.PRNGKey(7))
    toks = _tokens(8, B=4, T=17)
    x, y, mask = toks[:, :-1], toks[:, 1:], jnp.ones((4, 16))
    opt = optax.adamw(3e-3)
    state = opt.init(params)

    @jax.jit
    def train_step(params, state):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y, mask, cfg)
        updates, state = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state, aux["ce"]

    first = None
    for _ in range(300):
        params, state, ce = train_step(params, state)
        first = float(ce) if first is None else first
    assert 3.8 < first < 4.6
    assert float(ce) < 0.2


def test_loss_mask():
    params = init_params(TINY, jax.random.PRNGKey(9))
    toks = _tokens(10, B=1, T=9)
    x, y = toks[:, :-1], toks[:, 1:]
    full, _ = loss_fn(params, x, y, jnp.ones((1, 8)), TINY)
    half, _ = loss_fn(params, x, y, jnp.array([[1, 1, 1, 1, 0, 0, 0, 0]]), TINY)
    logits, _, _ = forward(params, x, TINY)
    nll = -jax.nn.log_softmax(logits)[0, jnp.arange(8), y[0]]
    assert abs(float(full) - float(nll.mean())) < 1e-5
    assert abs(float(half) - float(nll[:4].mean())) < 1e-5


def test_moe_balance_loss():
    params = init_params(TINY_MOE, jax.random.PRNGKey(11))
    _, aux, _ = forward(params, _tokens(12), TINY_MOE)
    assert 0.9 < float(aux) < 2.0  # ~1 when routing is roughly balanced at init


def test_bfloat16_compute():
    cfg = ModelConfig(**{**TINY.__dict__, "dtype": "bfloat16"})
    params = init_params(cfg, jax.random.PRNGKey(13))
    lo, _, _ = forward(params, _tokens(), cfg)
    hi, _, _ = forward(params, _tokens(), TINY)
    assert lo.dtype == jnp.float32 and params["embed"].dtype == jnp.float32
    assert float(jnp.max(jnp.abs(lo - hi))) < 0.1


def test_remat_gives_same_loss_and_grads():
    import dataclasses

    cfg = TINY
    params = init_params(cfg, jax.random.PRNGKey(0))
    tok = jax.random.randint(jax.random.PRNGKey(1), (2, 16), 0, cfg.vocab_size)
    mask = jnp.ones_like(tok)

    def lg(c):
        return jax.value_and_grad(lambda p: loss_fn(p, tok[:, :-1], tok[:, 1:], mask[:, 1:], c)[0])(params)

    (l1, g1), (l2, g2) = lg(cfg), lg(dataclasses.replace(cfg, remat=True))
    assert np.allclose(l1, l2, atol=1e-6)
    for a, b in zip(jax.tree.leaves(g1), jax.tree.leaves(g2)):
        assert np.allclose(a, b, atol=1e-5)
