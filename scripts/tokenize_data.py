"""Tokenize cleaned documents into training and validation token shards.

    python scripts/tokenize_data.py --input "C:/llm-data/clean/final/*.jsonl.gz" \
        --tokenizer C:/llm-data/tokenizer.json --out C:/llm-data/tokens

Writes <out>/train and <out>/val. A document goes to val when a hash of its text falls in
the val share, so the split is stable across re-runs and machines.
"""

import argparse
import glob
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.data.io import read_jsonl  # noqa: E402
from aio.data.tokens import TokenShardWriter  # noqa: E402
from aio.tokenizer import BPETokenizer  # noqa: E402

_tok = None


def _init(path: str) -> None:
    global _tok
    _tok = BPETokenizer.load(path)


def _encode(text: str) -> list[int]:
    # Special-token strings inside web text are treated as plain text.
    return _tok.encode_ordinary(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="glob of cleaned JSONL files")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--min-quality", type=float, help="skip documents scoring below this")
    ap.add_argument("--workers", type=int, default=0, help="0 = one per CPU core")
    ap.add_argument("--tokens-per-shard", type=int, default=50_000_000)
    args = ap.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        sys.exit(f"no files match {args.input}")
    tok = BPETokenizer.load(args.tokenizer)
    eot = tok.special_tokens["<|endoftext|>"]
    out = Path(args.out)
    cut = int(args.val_frac * 2**32)

    def texts():
        for f in files:
            for d in read_jsonl(f):
                if args.min_quality is not None and d.get("quality", 0) < args.min_quality:
                    continue
                yield d["text"]

    t = time.time()
    with TokenShardWriter(out / "train", tok.vocab_size, eot, args.tokens_per_shard) as tr, \
         TokenShardWriter(out / "val", tok.vocab_size, eot, args.tokens_per_shard) as va, \
         Pool(args.workers or None, initializer=_init, initargs=(args.tokenizer,)) as pool:
        it = texts()
        # Pair each text with its tokens, keeping order so the split hash can be computed here.
        buf = []
        for text in it:
            buf.append(text)
            if len(buf) == 2000:
                for txt, ids in zip(buf, pool.map(_encode, buf, chunksize=50)):
                    (va if zlib.crc32(txt.encode()) < cut else tr).add(ids)
                buf = []
                n = tr.n_docs + va.n_docs
                if n % 20_000 == 0:
                    print(f"{n:,} docs, {time.time() - t:.0f}s", flush=True)
        for txt, ids in zip(buf, pool.map(_encode, buf, chunksize=50)):
            (va if zlib.crc32(txt.encode()) < cut else tr).add(ids)
    print(f"done in {time.time() - t:.0f}s")
    for name in ("train", "val"):
        idx = (out / name / "index.json").read_text()
        print(name, idx[:200].replace("\n", " "))


if __name__ == "__main__":
    main()
