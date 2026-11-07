"""Build an SFT mix of synthetic calculator conversations and ordinary chat.

    python scripts/make_tool_mix.py --tokenizer C:/llm-data/tokenizer-8k.json \
        --chat "C:/llm-data/chat/smol-smoltalk/train-00000-of-00004.parquet" \
        --n-tool 20000 --n-chat 20000 --max-len 513 --out C:/llm-data/chat/tool-mix.npz

The ordinary chat keeps the model's general behaviour while it learns when to call the
calculator.
"""

import argparse
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.agent.synth import calc_conversations  # noqa: E402
from aio.post.chat import render, save_conversations  # noqa: E402
from aio.tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--chat", required=True, help="smol-smoltalk parquet file(s), comma-separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-tool", type=int, default=20_000)
    ap.add_argument("--n-chat", type=int, default=20_000)
    ap.add_argument("--max-len", type=int, default=513)
    ap.add_argument("--max-digits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    convs = calc_conversations(args.n_tool, seed=args.seed, max_digits=args.max_digits)
    chat = []
    for f in args.chat.split(","):
        for msgs in pq.read_table(f, columns=["messages"]).column("messages").to_pylist():
            c = [{"role": m["role"], "content": m["content"]} for m in msgs
                 if m["role"] in ("system", "user", "assistant") and m["content"]]
            if any(m["role"] == "assistant" for m in c) and len(render(tok, c)[0]) <= args.max_len:
                chat.append(c)
            if len(chat) >= args.n_chat:
                break
        if len(chat) >= args.n_chat:
            break
    mix = convs + chat
    random.Random(args.seed).shuffle(mix)
    n = save_conversations(args.out, tok, mix)
    ex, _ = render(tok, convs[0])
    print(f"{len(convs):,} tool + {len(chat):,} chat = {n:,} conversations -> {args.out}")
    print(f"a tool conversation is {len(ex)} tokens with this tokenizer")


if __name__ == "__main__":
    main()
