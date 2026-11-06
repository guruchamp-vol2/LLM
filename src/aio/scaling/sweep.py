"""Plan and run a scaling sweep: several model sizes at each of several compute budgets.

Each run is a normal training run in its own folder, so an interrupted sweep resumes
where it stopped, and finished runs are skipped.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

from ..model.transformer import ModelConfig
from ..train.optim import OptimConfig
from ..train.trainer import TrainConfig


def non_embedding_params(m: ModelConfig) -> int:
    attn = m.d_model * m.n_heads * m.head_dim * 2 + m.d_model * m.n_kv_heads * m.head_dim * 2
    mlp = 3 * m.d_model * m.ff_dim
    return m.n_layers * (attn + mlp + 2 * m.d_model) + m.d_model


def shape_for(width: int, vocab_size: int, seq_len: int, head_dim: int = 32) -> ModelConfig:
    """A model of the given width with depth ~ width / 32 (min 2 layers) and 2:1 grouped-query attention."""
    n_heads = max(width // head_dim, 2)
    return ModelConfig(vocab_size=vocab_size, d_model=width, n_layers=max(width // 32, 2), n_heads=n_heads,
                       n_kv_heads=max(n_heads // 2, 1), max_seq_len=seq_len)


def plan(base: TrainConfig, budgets: list[float], widths: list[int], out_dir: str | Path,
         min_steps: int = 50, max_tokens: float | None = None) -> list[TrainConfig]:
    """For each compute budget (FLOPs) and width, set the step count so 6 N D = budget.

    N counts non-embedding parameters (the usual choice at small scale, where embeddings
    would otherwise dominate). Runs needing fewer than min_steps steps are left out, and so
    are runs needing more than max_tokens (e.g. the dataset size, to avoid repeating data,
    which would bend the fitted curves).
    """
    out = Path(out_dir)
    tokens_per_step = base.batch_size * base.seq_len
    runs = []
    for c in budgets:
        for w in widths:
            m = shape_for(w, base.model.vocab_size, base.seq_len)
            n = non_embedding_params(m)
            steps = int(c / (6 * n * tokens_per_step))
            if steps < min_steps or (max_tokens is not None and steps * tokens_per_step > max_tokens):
                continue
            warm = max(10, steps // 20)
            # Smaller models tolerate larger learning rates.
            lr = min(3e-3, 1e-3 * math.sqrt(256 / w))
            runs.append(replace(
                base, model=m, steps=steps, out_dir=str(out / f"c{c:.0e}_w{w}"),
                optim=replace(base.optim, lr=lr, warmup_steps=warm, total_steps=steps),
                eval_every=steps, ckpt_every=max(steps // 4, 1), log_every=max(steps // 10, 1),
            ))
    return runs


def run_all(runs: list[TrainConfig], results_path: str | Path, log=print) -> list[dict]:
    """Train each planned run (skipping finished ones) and append results as JSON lines."""
    from ..train.trainer import train  # imported here so planning needs no JAX

    results_path = Path(results_path)
    done = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["out_dir"]] = r
    out = []
    for i, cfg in enumerate(runs, 1):
        if cfg.out_dir in done:
            out.append(done[cfg.out_dir])
            continue
        log(f"[{i}/{len(runs)}] {cfg.out_dir}: width {cfg.model.d_model}, {cfg.model.n_layers} layers, {cfg.steps} steps")
        res = train(cfg, log=lambda s: None)
        rec = {"out_dir": cfg.out_dir, "n_params": non_embedding_params(cfg.model),
               "tokens": cfg.steps * cfg.batch_size * cfg.seq_len, "val_loss": res["val_loss"],
               "d_model": cfg.model.d_model, "n_layers": cfg.model.n_layers, "steps": cfg.steps}
        rec["flops"] = 6 * rec["n_params"] * rec["tokens"]
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"    val loss {rec['val_loss']:.4f}")
        out.append(rec)
    return out
