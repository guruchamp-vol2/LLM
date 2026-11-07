"""Convert free chat datasets into our SFT / DPO .npz files.

    # SFT conversations (HuggingFaceTB/smol-smoltalk, Apache-2.0)
    python scripts/prepare_chat.py --format smoltalk --input "C:/llm-data/chat/smol-smoltalk/train-*.parquet" \
        --tokenizer C:/llm-data/tokenizer-8k.json --out C:/llm-data/chat/sft-train.npz

    # DPO pairs (HuggingFaceH4/ultrafeedback_binarized, MIT)
    python scripts/prepare_chat.py --format ultrafeedback --input "C:/llm-data/chat/ultrafeedback/train_prefs-*.parquet" \
        --tokenizer C:/llm-data/tokenizer-8k.json --out C:/llm-data/chat/dpo-train.npz

    # Single-turn instruction data (coding SFT). Columns are found automatically:
    #   OpenCoder-LLM/opc-sft-stage1/2 (MIT): instruction -> output
    #   ise-uiuc/Magicoder-OSS-Instruct-75K (MIT): problem -> solution
    #   bigcode/self-oss-instruct-sc2-exec-filter-50k (ODC-By): instruction -> response
    python scripts/prepare_chat.py --format instruct --input "C:/llm-data/chat/magicoder/*.parquet" ...

--max-len drops examples longer than that many tokens, so short-context models train on
whole conversations instead of cut-off ones. Examples that contain a HumanEval/MBPP
problem are dropped (--evals), so the coding benchmarks stay honest.
"""

import argparse
import glob
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.data.decontam import Decontaminator, benchmark_texts  # noqa: E402
from aio.post.chat import render, save_conversations, save_pairs  # noqa: E402
from aio.tokenizer import BPETokenizer  # noqa: E402

# (prompt column, response column) pairs we know about, tried in order.
INSTRUCT_COLUMNS = [("instruction", "output"), ("problem", "solution"), ("instruction", "response")]


def _clean(messages) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in messages
            if m["role"] in ("system", "user", "assistant") and m["content"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", required=True, choices=["smoltalk", "ultrafeedback", "instruct"])
    ap.add_argument("--evals", default="C:/llm-data/evals", help="benchmark files for decontamination")
    ap.add_argument("--input", required=True, help="glob of parquet files")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many examples (0 = all)")
    ap.add_argument("--max-len", type=int, default=0, help="drop examples longer than this many tokens (0 = keep all)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        sys.exit(f"no files match {args.input}")
    tok = BPETokenizer.load(args.tokenizer)
    decon = Decontaminator(benchmark_texts(args.evals))
    seen = dropped = 0

    def fits(messages) -> bool:
        if len(decon) and decon.contaminated("\n".join(m["content"] for m in messages)):
            return False
        return not args.max_len or len(render(tok, messages)[0]) <= args.max_len

    def raw_conversations(f):
        if args.format == "smoltalk":
            yield from pq.read_table(f, columns=["messages"]).column("messages").to_pylist()
            return
        names = pq.ParquetFile(f).schema_arrow.names
        cols = next(((p, r) for p, r in INSTRUCT_COLUMNS if p in names and r in names), None)
        if cols is None:
            sys.exit(f"{f}: no known prompt/response columns in {names}")
        t = pq.read_table(f, columns=list(cols))
        for p, r in zip(t.column(cols[0]).to_pylist(), t.column(cols[1]).to_pylist()):
            yield [{"role": "user", "content": (p or "").strip()}, {"role": "assistant", "content": (r or "").strip()}]

    def conversations():
        nonlocal seen, dropped
        for f in files:
            for msgs in raw_conversations(f):
                conv = _clean(msgs)
                seen += 1
                if not any(m["role"] == "assistant" for m in conv) or not fits(conv):
                    dropped += 1
                    continue
                yield conv
                if args.limit and seen - dropped >= args.limit:
                    return

    def pairs():
        nonlocal seen, dropped
        for f in files:
            t = pq.read_table(f, columns=["chosen", "rejected"])
            for ch, rj in zip(t.column("chosen").to_pylist(), t.column("rejected").to_pylist()):
                ch, rj = _clean(ch), _clean(rj)
                seen += 1
                ok = (len(ch) >= 2 and len(rj) >= 2 and ch[-1]["role"] == rj[-1]["role"] == "assistant"
                      and ch[:-1] == rj[:-1] and ch[-1]["content"] != rj[-1]["content"])
                if not ok or not fits(ch) or not fits(rj):
                    dropped += 1
                    continue
                yield {"prompt": ch[:-1], "chosen": ch[-1]["content"], "rejected": rj[-1]["content"]}
                if args.limit and seen - dropped >= args.limit:
                    return

    if args.format in ("smoltalk", "instruct"):
        n = save_conversations(args.out, tok, conversations())
    else:
        n = save_pairs(args.out, tok, pairs())
    print(f"read {seen:,}, kept {n:,}, dropped {dropped:,} -> {args.out}")


if __name__ == "__main__":
    main()
