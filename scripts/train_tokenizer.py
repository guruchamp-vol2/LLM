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
    ap.add_argument("--input", required=True, nargs="+", help="globs of cleaned JSONL files, optionally GLOB=MAX_DOCS")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=32_000)
    ap.add_argument("--max-docs", type=int, default=200_000)
    ap.add_argument("--min-quality", type=float, help="only learn from documents scoring at least this")
    args = ap.parse_args()

    sources = []
    for spec in args.input:
        pattern, _, cap = spec.rpartition("=") if "=" in spec.rsplit("/", 1)[-1] else (spec, "", "")
        files = sorted(glob.glob(pattern))
        if not files:
            sys.exit(f"no files match {pattern}")
        sources.append((files, int(cap) if cap else args.max_docs))

    def one_source(files, cap):
        n = 0
        for f in files:
            for d in read_jsonl(f):
                if args.min_quality is not None and d.get("quality", 0) < args.min_quality:
                    continue
                yield d["text"]
                n += 1
                if n >= cap:
                    return

    def docs():
        its = [one_source(f, c) for f, c in sources]
        while its:
            for i, it in list(enumerate(its)):
                try:
                    yield next(it)
                except StopIteration:
                    its[i] = None
            its = [it for it in its if it is not None]

    t = time.time()
    tok = BPETokenizer.train(docs(), args.vocab_size, verbose=True)
    tok.save(args.out)
    print(f"vocab {tok.vocab_size} trained in {time.time() - t:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
