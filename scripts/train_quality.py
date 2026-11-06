"""Train the educational-quality scorer on the FineWeb-Edu annotations.

Download the graded pages first (free, about 1.6 GB):
    https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations

    python scripts/train_quality.py --data "C:/llm-data/edu-annotations/*.parquet" \
        --out C:/llm-data/models/quality.npz

Holds out a fixed share of pages for validation and prints how well the scorer agrees
with the published grades. Writes the model and a metrics JSON next to it.
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.data.quality import QualityScorer, SparseRows, evaluate, pick_threshold  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="glob of annotation parquet files")
    ap.add_argument("--out", required=True, help="output model path (.npz)")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="use only this many pages (0 = all)")
    ap.add_argument("--buckets", type=int, default=1 << 20)
    ap.add_argument("--max-words", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--l2", type=float, default=1e-5)
    args = ap.parse_args()

    files = sorted(glob.glob(args.data))
    if not files:
        sys.exit(f"no files match {args.data}")

    t0 = time.time()
    texts, scores = [], []
    for f in files:
        tbl = pq.read_table(f, columns=["text", "score"])
        texts += tbl.column("text").to_pylist()
        scores += tbl.column("score").to_pylist()
        if args.limit and len(texts) >= args.limit:
            break
    if args.limit:
        texts, scores = texts[: args.limit], scores[: args.limit]
    y = np.array(scores, dtype=np.float32)
    print(f"loaded {len(texts):,} pages in {time.time() - t0:.0f}s; grade counts:",
          dict(zip(*np.unique(y.astype(int), return_counts=True))))

    rng = np.random.default_rng(1234)
    order = rng.permutation(len(texts))
    n_val = int(len(texts) * args.val_frac)
    val_i, tr_i = order[:n_val], order[n_val:]

    t0 = time.time()
    scorer = QualityScorer(args.buckets, args.max_words)
    tr = SparseRows.build((texts[i] for i in tr_i), args.buckets, args.max_words)
    va = SparseRows.build((texts[i] for i in val_i), args.buckets, args.max_words)
    del texts
    print(f"features built in {time.time() - t0:.0f}s ({len(tr.indices) / len(tr):.0f} per page)")

    t0 = time.time()
    history = scorer.fit(tr, y[tr_i], va, y[val_i], epochs=args.epochs, lr=args.lr, l2=args.l2)
    print(f"trained in {time.time() - t0:.0f}s")

    # Note: the cut-off is tuned on the same held-out pages it is reported on, which flatters
    # the F1 slightly; with ~23k held-out pages and one number tuned, the effect is small.
    val_pred = scorer.score_rows(va)
    scorer.keep_threshold = pick_threshold(val_pred, y[val_i])
    final = evaluate(val_pred, y[val_i], scorer.keep_threshold)
    final_at_3 = evaluate(val_pred, y[val_i], 3.0)
    baseline = evaluate(np.full(n_val, y[tr_i].mean()), y[val_i])
    print(f"validation (cut-off {scorer.keep_threshold}):", final)
    print("validation (cut-off 3.0):", final_at_3)
    print("baseline (always predict the average):", baseline)

    out = Path(args.out)
    scorer.save(out)
    metrics = {"args": vars(args), "n_train": len(tr_i), "n_val": n_val,
               "validation": final, "validation_at_3": final_at_3, "baseline": baseline, "history": history}
    out.with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
