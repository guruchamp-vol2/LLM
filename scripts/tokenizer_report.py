"""Compare tokenizers on held-out text: bytes per token (higher = more text per token,
so more data fits in the same context and compute), plus a few spot checks.

    python scripts/tokenizer_report.py --tokenizers C:/llm-data/tokenizer-8k.json C:/llm-data/tokenizer-32k.json \
        --sample "web=C:/llm-data/clean-edu/000_00000-00006.jsonl.gz" "code=C:/llm-data/clean-code/github/data_train-00008-*.jsonl.gz"
"""

import argparse
import glob
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.data.io import read_jsonl  # noqa: E402
from aio.post.chat import tool_call_text  # noqa: E402
from aio.tokenizer import BPETokenizer  # noqa: E402

SPOT = {
    "tool call": tool_call_text({"name": "calculator", "arguments": {"expression": "1234 * 5678"}}),
    "python": 'def mean(xs):\n    """Average of a list."""\n    if not xs:\n        return 0.0\n    return sum(xs) / len(xs)\n',
    "maths": "Solve for x: 3x^2 - 12x + 9 = 0, so x = 1 or x = 3.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizers", nargs="+", required=True)
    ap.add_argument("--sample", nargs="+", required=True, help="NAME=GLOB of held-out JSONL")
    ap.add_argument("--docs", type=int, default=2000, help="documents per sample")
    args = ap.parse_args()

    toks = {Path(t).stem: BPETokenizer.load(t) for t in args.tokenizers}
    samples = {}
    for spec in args.sample:
        name, pattern = spec.split("=", 1)
        files = sorted(glob.glob(pattern))
        if not files:
            sys.exit(f"no files match {pattern}")
        docs = itertools.islice((d["text"] for f in files for d in read_jsonl(f)), args.docs)
        samples[name] = list(docs)

    print(f"{'bytes/token':<14}" + "".join(f"{n:>18}" for n in toks))
    for name, docs in samples.items():
        nbytes = sum(len(d.encode("utf-8")) for d in docs)
        row = f"{name:<14}"
        for t in toks.values():
            ids = [len(t.encode_ordinary(d)) for d in docs]
            assert all(t.decode(t.encode_ordinary(d)) == d for d in docs[:50]), "round trip failed"
            row += f"{nbytes / sum(ids):>18.2f}"
        print(row)
    print()
    for label, text in SPOT.items():
        for n, t in toks.items():
            ids = t.encode_ordinary(text)
            pieces = [t.decode([i]) for i in ids]
            print(f"{label} [{n}] {len(ids)} tokens: {pieces[:24]}{' ...' if len(pieces) > 24 else ''}")
        print()


if __name__ == "__main__":
    main()
