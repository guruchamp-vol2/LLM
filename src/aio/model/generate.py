"""Text generation with the key/value cache: temperature and top-k sampling."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .transformer import ModelConfig, forward, init_cache


@partial(jax.jit, static_argnames="cfg")
def _prefill(params, tokens, cache, cfg: ModelConfig):
    logits, _, cache = forward(params, tokens, cfg, cache, 0)
    return logits[:, -1], cache


@partial(jax.jit, static_argnames="cfg")
def _step(params, token, cache, pos, cfg: ModelConfig):
    logits, _, cache = forward(params, token[:, None], cfg, cache, pos)
    return logits[:, -1], cache


def sample_logits(logits: jax.Array, key: jax.Array, temperature: float = 1.0, top_k: int = 0) -> jax.Array:
    if temperature <= 0:
        return jnp.argmax(logits, axis=-1)
    logits = logits / temperature
    if top_k > 0:
        kth = jax.lax.top_k(logits, top_k)[0][:, -1:]
        logits = jnp.where(logits < kth, -jnp.inf, logits)
    return jax.random.categorical(key, logits, axis=-1)


def generate(params, cfg: ModelConfig, prompt: list[int], max_new_tokens: int, key: jax.Array,
             temperature: float = 0.8, top_k: int = 50, stop_id: int | None = None) -> list[int]:
    """Continue one prompt. Returns only the new tokens (stops early at stop_id)."""
    total = min(len(prompt) + max_new_tokens, cfg.max_seq_len)
    cache = init_cache(cfg, batch=1, max_len=total)
    logits, cache = _prefill(params, jnp.array([prompt], jnp.int32), cache, cfg)
    out: list[int] = []
    for pos in range(len(prompt), total):
        key, sub = jax.random.split(key)
        tok = sample_logits(logits, sub, temperature, top_k)
        t = int(tok[0])
        out.append(t)
        if t == stop_id or pos + 1 >= total:
            break
        logits, cache = _step(params, tok.astype(jnp.int32), cache, pos, cfg)
    return out


def generate_many(params, cfg: ModelConfig, prompt: list[int], n: int, max_new_tokens: int, key: jax.Array,
                  temperature: float = 0.8, top_k: int = 50, stop_ids: tuple[int, ...] = (),
                  stop_check=None) -> list[list[int]]:
    """n samples of the same prompt in one batch (for pass@k). Each sample stops at a stop
    id, or when stop_check(tokens) says so (e.g. the decoded text hit a stop string).
    The batch ends once every sample has stopped."""
    total = min(len(prompt) + max_new_tokens, cfg.max_seq_len)
    cache = init_cache(cfg, batch=1, max_len=total)
    logits, cache = _prefill(params, jnp.array([prompt], jnp.int32), cache, cfg)
    # Prefill once, then copy the cache to all n samples.
    cache = jax.tree_util.tree_map(lambda x: jnp.repeat(x, n, axis=1), cache)
    logits = jnp.repeat(logits, n, axis=0)
    outs: list[list[int]] = [[] for _ in range(n)]
    done = [False] * n
    for pos in range(len(prompt), total):
        key, sub = jax.random.split(key)
        tok = sample_logits(logits, sub, temperature, top_k)
        for i, t in enumerate(tok.tolist()):
            if done[i]:
                continue
            if t in stop_ids:
                done[i] = True
                continue
            outs[i].append(t)
            if stop_check is not None and stop_check(outs[i]):
                done[i] = True
        if all(done) or pos + 1 >= total:
            break
        logits, cache = _step(params, tok.astype(jnp.int32), cache, pos, cfg)
    return outs
