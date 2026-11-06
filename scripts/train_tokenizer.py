"""Train the BPE tokenizer on a sample of cleaned documents.

    python scripts/train_tokenizer.py --input "C:/llm-data/clean/final/*.jsonl.gz" \
        --vocab-size 32000 --max-docs 200000 --out C:/llm-data/tokenizer.json
"""

import argparse
import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.data.io import read_jsonl  # noqa: E402
from aio.tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="glob of cleaned JSONL files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=32_000)
    ap.add_argument("--max-docs", type=int, default=200_000)
    ap.add_argument("--min-quality", type=float, help="only learn from documents scoring at least this")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        sys.exit(f"no files match {args.input}")

    def docs():
        n = 0
        for f in files:
            for d in read_jsonl(f):
                if args.min_quality is not None and d.get("quality", 0) < args.min_quality:
                    continue
                yield d["text"]
                n += 1
                if n >= args.max_docs:
                    return

    t = time.time()
    tok = BPETokenizer.train(docs(), args.vocab_size, verbose=True)
    tok.save(args.out)
    print(f"vocab {tok.vocab_size} trained in {time.time() - t:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
