"""Multiple-choice evaluation by likelihood (the standard way to test base models).

Each item has a context, several candidate continuations and the index of the right one.
The model scores every candidate by the log-probability of its tokens given the context,
and picks the highest. Two scores are reported:

- acc:      raw sum of log-probs (favours short answers)
- acc_norm: log-prob per byte of the answer (fairer when answers differ in length;
            HellaSwag is usually reported this way)

Chance level is 1 / number of choices, so a tiny model scores close to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..model.transformer import ModelConfig, forward


@dataclass
class MCItem:
    context: str
    choices: list[str]
    answer: int


@partial(jax.jit, static_argnames="cfg")
def _choice_logprobs(params, tokens, mask, cfg: ModelConfig):
    """Sum of log p over masked target positions. tokens, mask: (B, T+1)."""
    logits, _, _ = forward(params, tokens[:, :-1], cfg)
    logp = jax.nn.log_softmax(logits, axis=-1)
    lp = jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
    return jnp.sum(lp * mask[:, 1:], axis=-1)


def score_items(params, cfg: ModelConfig, encode, items: list[MCItem], prefix: list[int] | None = None,
                batch_size: int = 32, pad_id: int = 0) -> dict:
    """encode: text -> token ids. prefix: tokens put before every context (e.g. end-of-text).

    Contexts that are too long are cut from the left, so the answer always fits.
    """
    prefix = prefix or []
    rows, owners, n_bytes = [], [], []
    for i, it in enumerate(items):
        ctx = prefix + encode(it.context)
        for c in it.choices:
            ans = encode(c)
            seq = (ctx + ans)[-(cfg.max_seq_len + 1):]
            n_ans = min(len(ans), len(seq) - 1)
            rows.append((seq, n_ans))
            owners.append(i)
            n_bytes.append(max(len(c.encode("utf-8")), 1))

    width = max(len(s) for s, _ in rows)
    width = min(width, cfg.max_seq_len + 1)
    # Pad the batch dimension to a multiple of batch_size and the length to a fixed width,
    # so the jitted function compiles once.
    scores = np.zeros(len(rows))
    for b in range(0, len(rows), batch_size):
        chunk = rows[b : b + batch_size]
        toks = np.full((batch_size, width), pad_id, dtype=np.int32)
        mask = np.zeros((batch_size, width), dtype=np.float32)
        for r, (seq, n_ans) in enumerate(chunk):
            toks[r, : len(seq)] = seq
            mask[r, len(seq) - n_ans : len(seq)] = 1.0
        out = np.asarray(_choice_logprobs(params, jnp.asarray(toks), jnp.asarray(mask), cfg))
        scores[b : b + len(chunk)] = out[: len(chunk)]

    owners = np.array(owners)
    n_bytes = np.array(n_bytes, dtype=np.float64)
    correct = correct_norm = 0
    for i, it in enumerate(items):
        sel = np.where(owners == i)[0]
        correct += int(np.argmax(scores[sel]) == it.answer)
        correct_norm += int(np.argmax(scores[sel] / n_bytes[sel]) == it.answer)
    n = len(items)
    chance = float(np.mean([1 / len(it.choices) for it in items])) if items else 0.0
    return {"n": n, "acc": correct / max(n, 1), "acc_norm": correct_norm / max(n, 1), "chance": round(chance, 4)}
