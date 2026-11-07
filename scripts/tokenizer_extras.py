"""Text the tokenizer should see besides pretraining data: chat turns, coding answers
and tool calls (JSON). Without these, tool calls split into dozens of one-character
tokens and the model can't learn them (lesson from the calculator tool).

    python scripts/tokenizer_extras.py --out C:/llm-data/tok-extra

Writes <out>/extras-00000.jsonl.gz with one {"text": ...} per item, for train_tokenizer.py.
"""

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.agent.synth import calc_conversations  # noqa: E402
from aio.data.io import ShardWriter  # noqa: E402
from aio.post.chat import tool_call_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--chat", default="C:/llm-data/chat/smol-smoltalk/train-00000-of-00004.parquet")
    ap.add_argument("--code-sft", default="C:/llm-data/chat/Magicoder-OSS-Instruct-75K/*.parquet")
    ap.add_argument("--n-chat", type=int, default=40_000)
    ap.add_argument("--n-code", type=int, default=20_000)
    ap.add_argument("--n-tools", type=int, default=10_000)
    args = ap.parse_args()
    rng = random.Random(0)
    n = 0
    with ShardWriter(args.out, prefix="extras", docs_per_shard=10**9) as w:
        convs = pq.read_table(args.chat, columns=["messages"]).column("messages").to_pylist()[: args.n_chat]
        for c in convs:
            w.write({"text": "\n\n".join(m["content"] for m in c if m.get("content"))})
            n += 1
        for f in sorted(glob.glob(args.code_sft)):
            t = pq.read_table(f, columns=["problem", "solution"]).to_pylist()
            for r in t[: args.n_code]:
                w.write({"text": f"{r['problem']}\n\n{r['solution']}"})
                n += 1
        # Tool calls exactly as the chat template writes them, for each tool we have.
        for c in calc_conversations(args.n_tools, seed=1):
            for m in c:
                if m.get("tool_call"):
                    w.write({"text": tool_call_text(m["tool_call"])})
                    n += 1
        words = "python function list sort numbers file read json api error weather news price telescope".split()
        for _ in range(args.n_tools):
            q = " ".join(rng.sample(words, rng.randint(2, 4)))
            w.write({"text": tool_call_text({"name": rng.choice(["search", "wikipedia"]), "arguments": {"query": q}})})
            code = rng.choice(["print(sum(range(10)))", "import math\nprint(math.sqrt(2))", "print(sorted([3, 1, 2]))"])
            w.write({"text": tool_call_text({"name": "python", "arguments": {"code": code}})})
            n += 2
    print(f"wrote {n:,} items to {args.out}")


if __name__ == "__main__":
    main()
