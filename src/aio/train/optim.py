"""AdamW with linear warmup, cosine decay and global-norm gradient clipping, in plain JAX.

Weight decay applies to weight matrices and embeddings, not to norm gains (1-D arrays).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class OptimConfig:
    lr: float = 3e-4
    min_lr_ratio: float = 0.1   # cosine decays to lr * min_lr_ratio
    warmup_steps: int = 100
    total_steps: int = 1000
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    clip_norm: float = 1.0      # 0 = no clipping


def lr_at(step, cfg: OptimConfig):
    """Learning rate at a step (works on Python ints and traced JAX values)."""
    step = jnp.asarray(step, jnp.float32)
    warm = cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    decay_steps = max(cfg.total_steps - cfg.warmup_steps, 1)
    frac = jnp.clip((step - cfg.warmup_steps) / decay_steps, 0.0, 1.0)
    cos = cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + jnp.cos(math.pi * frac)))
    return jnp.where(step < cfg.warmup_steps, warm, cos)


def init_opt_state(params) -> dict:
    zeros = lambda p: jnp.zeros_like(p, dtype=jnp.float32)  # noqa: E731
    return {"m": jax.tree.map(zeros, params), "v": jax.tree.map(zeros, params), "step": jnp.zeros((), jnp.int32)}


def global_norm(tree) -> jax.Array:
    return jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32))) for x in jax.tree.leaves(tree)))


def adamw_update(params, grads, state: dict, cfg: OptimConfig):
    """One AdamW step. Returns (new params, new state, {"lr", "grad_norm"})."""
    gnorm = global_norm(grads)
    if cfg.clip_norm > 0:
        scale = jnp.minimum(1.0, cfg.clip_norm / (gnorm + 1e-6))
        grads = jax.tree.map(lambda g: g * scale, grads)

    step = state["step"]
    t = (step + 1).astype(jnp.float32)
    lr = lr_at(step, cfg)
    m = jax.tree.map(lambda m, g: cfg.b1 * m + (1 - cfg.b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: cfg.b2 * v + (1 - cfg.b2) * g * g, state["v"], grads)
    bc1, bc2 = 1 - cfg.b1**t, 1 - cfg.b2**t

    def upd(p, m, v):
        u = (m / bc1) / (jnp.sqrt(v / bc2) + cfg.eps)
        if p.ndim >= 2:
            u = u + cfg.weight_decay * p
        return (p - lr * u).astype(p.dtype)

    new_params = jax.tree.map(upd, params, m, v)
    return new_params, {"m": m, "v": v, "step": step + 1}, {"lr": lr, "grad_norm": gnorm}
