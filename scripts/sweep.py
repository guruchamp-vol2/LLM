"""Scaling sweep: train several model sizes at several compute budgets, then fit scaling laws.

    python scripts/sweep.py --config configs/laptop-tiny.json --out C:/llm-data/sweeps/laptop \
        --budgets 1e13 3e13 1e14 --widths 64 96 128 192 256 --dry-run

--dry-run prints the plan and a rough time estimate without training. Re-running the same
command resumes: finished runs are read from results.jsonl and skipped.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.scaling import fit_isoflop, fit_parametric, non_embedding_params, plan, run_all  # noqa: E402
from aio.train import TrainConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="base training config (data, batch, seq_len, vocab)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", type=float, nargs="+", required=True, help="compute budgets in FLOPs (6 N D)")
    ap.add_argument("--widths", type=int, nargs="+", default=[64, 96, 128, 192, 256])
    ap.add_argument("--min-steps", type=int, default=50)
    ap.add_argument("--flops-per-s", type=float, default=1.2e11,
                    help="measured training speed (all FLOPs incl. embeddings) for the time estimate")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = TrainConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    runs = plan(base, args.budgets, args.widths, args.out, args.min_steps)
    total_h = 0.0
    print(f"{len(runs)} runs:")
    for r in runs:
        tokens = r.steps * r.batch_size * r.seq_len
        n = non_embedding_params(r.model)
        # wall time is driven by all parameters (embedding and output layer included)
        n_all = n + r.model.vocab_size * r.model.d_model
        h = 6 * n_all * tokens / args.flops_per_s / 3600
        total_h += h
        print(f"  {Path(r.out_dir).name:>14}: {n / 1e6:6.2f}M non-embedding params, {r.model.n_layers} layers, "
              f"{r.steps:>6} steps, {tokens / 1e6:7.1f}M tokens, ~{h * 60:5.0f} min")
    print(f"estimated total: {total_h:.1f} hours")
    if args.dry_run:
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = run_all(runs, out / "results.jsonl")
    n = [r["n_params"] for r in results]
    d = [r["tokens"] for r in results]
    y = [r["val_loss"] for r in results]
    summary = {"runs": len(results)}
    try:
        f = fit_parametric(n, d, y)
        summary["parametric"] = vars(f)
        print(f"\nL(N, D) = {f.E:.3f} + {f.A:.1f} / N^{f.alpha:.3f} + {f.B:.1f} / D^{f.beta:.3f}   (rmse {f.rmse:.4f})")
        for c in (1e18, 1e20, 1e22):
            nn, dd = f.optimal(c)
            print(f"  budget {c:.0e} FLOPs -> {nn / 1e6:,.1f}M params, {dd / 1e9:,.2f}B tokens ({dd / nn:.0f} tokens/param)")
    except ValueError as e:
        print("parametric fit skipped:", e)
    try:
        iso = fit_isoflop(n, d, y)
        summary["isoflop"] = vars(iso)
        print(f"\nIsoFLOP: N_opt = {iso.k:.3g} * C^{iso.a:.3f}  from budgets {iso.budgets}")
    except ValueError as e:
        print("isoflop fit skipped:", e)
    (out / "fit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
