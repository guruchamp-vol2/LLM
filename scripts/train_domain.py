"""Train the topic (domain) classifier on the WebOrganizer topic annotations.

Download the labelled pages first (free, about 300 MB, train/validation/test splits):
    https://huggingface.co/datasets/WebOrganizer/TopicAnnotations-Llama-3.1-405B-FP8

    python scripts/train_domain.py --data C:/llm-data/topic-annotations --out C:/llm-data/models/domain.npz

Picks the best epoch on the validation split and reports final numbers on the test split.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.data.domain import DomainClassifier, evaluate  # noqa: E402
from aio.data.quality import SparseRows  # noqa: E402


def slug(description: str) -> str:
    """'Science, Math & Technology\\n - Includes: ...' -> 'science_math_technology'."""
    name = description.split("\n")[0].replace("Hardare", "Hardware")  # typo in the source taxonomy
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load(path: Path, limit: int = 0):
    t = pq.read_table(path, columns=["text", "top_choice_index", "top_choice", "choice_probs"])
    if limit:
        t = t.slice(0, limit)
    names = dict(zip(t.column("top_choice_index").to_pylist(), t.column("top_choice").to_pylist()))
    probs = np.array(t.column("choice_probs").to_pylist(), dtype=np.float32)
    return t.column("text").to_pylist(), probs, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train-/validation-/test-*.parquet")
    ap.add_argument("--out", required=True, help="output model path (.npz)")
    ap.add_argument("--limit", type=int, default=0, help="use only this many training pages (0 = all)")
    ap.add_argument("--buckets", type=int, default=1 << 18)
    ap.add_argument("--max-words", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--l2", type=float, default=1e-6)
    args = ap.parse_args()

    d = Path(args.data)
    splits = {}
    for name in ("train", "validation", "test"):
        files = sorted(d.glob(f"{name}-*.parquet"))
        if not files:
            sys.exit(f"no {name}-*.parquet in {d}")
        splits[name] = load(files[0], args.limit if name == "train" else 0)

    names = splits["train"][2]
    labels = [slug(names[i]) for i in range(len(names))]
    print(f"{len(labels)} topics:", ", ".join(labels))

    t0 = time.time()
    rows = {k: SparseRows.build(v[0], args.buckets, args.max_words) for k, v in splits.items()}
    print(f"features built in {time.time() - t0:.0f}s; pages:", {k: len(v) for k, v in rows.items()})

    clf = DomainClassifier(labels, args.buckets, args.max_words)
    t0 = time.time()
    history = clf.fit(rows["train"], splits["train"][1], rows["validation"], splits["validation"][1],
                      epochs=args.epochs, lr=args.lr, l2=args.l2)
    print(f"trained in {time.time() - t0:.0f}s")

    test = evaluate(clf.proba_rows(rows["test"]), splits["test"][1])
    prior = splits["train"][1].mean(axis=0)
    baseline = evaluate(np.tile(prior, (len(rows["test"]), 1)), splits["test"][1])
    print("test:", test)
    print("baseline (always guess the most common topic):", baseline)

    out = Path(args.out)
    clf.save(out)
    out.with_suffix(".json").write_text(json.dumps(
        {"args": vars(args), "labels": labels, "test": test, "baseline": baseline, "history": history}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
