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
