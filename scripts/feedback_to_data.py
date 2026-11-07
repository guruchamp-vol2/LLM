"""Turn the chat feedback log into training data for the next round of SFT and DPO.

    python scripts/feedback_to_data.py --feedback C:/llm-data/feedback.jsonl \
        --tokenizer C:/llm-data/tokenizer-8k.json --out C:/llm-data/chat/feedback

Writes <out>-sft.npz (good answers and corrections) and <out>-dpo.npz (correction preferred
over the bad answer), plus <out>-review.jsonl to read before training: feedback can be
wrong or malicious, so check it first. Mix the SFT file with regular chat data when
training, so the model doesn't forget everything else.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.memory import FeedbackLog, to_pairs, to_sft  # noqa: E402
from aio.post.chat import save_conversations, save_pairs  # noqa: E402
from aio.tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True, help="output path prefix")
    args = ap.parse_args()

    entries = FeedbackLog(args.feedback).entries()
    if not entries:
        sys.exit("no feedback yet")
    tok = BPETokenizer.load(args.tokenizer)
    sft, pairs = to_sft(entries), to_pairs(entries)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}-review.jsonl", "w", encoding="utf-8") as f:
        for c in sft:
            f.write(json.dumps({"type": "sft", "conversation": c}, ensure_ascii=False) + "\n")
        for p in pairs:
            f.write(json.dumps({"type": "dpo", **p}, ensure_ascii=False) + "\n")
    n_sft = save_conversations(f"{out}-sft.npz", tok, sft) if sft else 0
    n_dpo = save_pairs(f"{out}-dpo.npz", tok, pairs) if pairs else 0
    good = sum(e["rating"] == "good" for e in entries)
    print(f"{len(entries)} feedback entries ({good} good, {len(entries) - good} bad) -> "
          f"{n_sft} SFT conversations, {n_dpo} DPO pairs. Review {out}-review.jsonl before training.")


if __name__ == "__main__":
    main()
