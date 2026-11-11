"""Training objectives. Each maps (params, batch, model config, reference params) to
(loss, metrics). The trainer picks one with TrainConfig.objective.

- "lm":  next-token cross-entropy. Pretraining batches are plain token arrays (train on
         every token); SFT batches are {"tokens", "mask"} (train on assistant tokens only).
- "dpo": Direct Preference Optimization (Rafailov et al. 2023). Raises the likelihood of
         the chosen answer relative to the rejected one, measured against a frozen
         reference model (the SFT model), so the policy can't drift arbitrarily far:
             loss = -log sigmoid(beta * [(log p(chosen) - log p_ref(chosen))
                                          - (log p(rejected) - log p_ref(rejected))])
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..model.transformer import ModelConfig, loss_fn, token_logprobs


def lm_objective(params, batch, cfg: ModelConfig, ref=None, **_):
    if isinstance(batch, dict):
        toks, mask = batch["tokens"], batch["mask"][:, 1:]
    else:
        toks, mask = batch, jnp.ones((batch.shape[0], batch.shape[1] - 1), jnp.float32)
    loss, aux = loss_fn(params, toks[:, :-1], toks[:, 1:], mask, cfg)
    return loss, {"ce": aux["ce"]}


def sequence_logprob(params, tokens, mask, cfg: ModelConfig):
    """Sum of log p(token) over masked target positions, per sequence. tokens: (B, T+1)."""
    tok_lp, _ = token_logprobs(params, tokens[:, :-1], tokens[:, 1:], cfg)
    return jnp.sum(tok_lp * mask[:, 1:], axis=-1)


def dpo_objective(params, batch, cfg: ModelConfig, ref=None, beta: float = 0.1, **_):
    if ref is None:
        raise ValueError("DPO needs reference parameters (set init_from to the SFT run)")
    pc = sequence_logprob(params, batch["chosen"], batch["chosen_mask"], cfg)
    pr = sequence_logprob(params, batch["rejected"], batch["rejected_mask"], cfg)
    rc = jax.lax.stop_gradient(sequence_logprob(ref, batch["chosen"], batch["chosen_mask"], cfg))
    rr = jax.lax.stop_gradient(sequence_logprob(ref, batch["rejected"], batch["rejected_mask"], cfg))
    margin = beta * ((pc - rc) - (pr - rr))
    loss = -jnp.mean(jax.nn.log_sigmoid(margin))
    return loss, {"ce": loss, "reward_acc": jnp.mean((margin > 0).astype(jnp.float32)), "margin": jnp.mean(margin)}


OBJECTIVES = {"lm": lm_objective, "dpo": dpo_objective}
