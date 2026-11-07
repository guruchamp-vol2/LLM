"""Learning from mistakes: a feedback log, an instant memory of corrections, and a way to
turn feedback into training data.

Three layers, from fastest to slowest:

1. Memory (instant): a correction is stored in a small search index. When a similar
   question comes up again, the correction is shown to the model as a note, so the fix
   applies straight away, with no retraining.
2. Feedback log (permanent record): every rating and correction is appended to a JSONL
   file with the conversation it belongs to.
3. Retraining (periodic, e.g. weekly): the log becomes SFT conversations (good answers
   and corrections) and DPO pairs (correction preferred over the bad answer), which the
   normal trainer learns from.

Feedback is never trained on automatically: someone should read the exported data first,
because anyone who can leave feedback could otherwise teach the model false things.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from ..retrieval.index import SearchIndex


class FeedbackLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, messages: list[dict], reply: str, rating: str, correction: str | None = None) -> dict:
        """messages: the conversation up to (not including) the model's reply."""
        if rating not in ("good", "bad"):
            raise ValueError("rating must be 'good' or 'bad'")
        entry = {"time": dt.datetime.now().isoformat(timespec="seconds"), "messages": messages,
                 "reply": reply, "rating": rating, "correction": correction}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _text_only(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") in ("system", "user", "assistant") and "content" in m]


def to_sft(entries: list[dict]) -> list[list[dict]]:
    """Good replies as they were, and bad replies replaced by their correction."""
    out = []
    for e in entries:
        answer = e["reply"] if e["rating"] == "good" else e.get("correction")
        if answer:
            out.append(_text_only(e["messages"]) + [{"role": "assistant", "content": answer}])
    return out


def to_pairs(entries: list[dict]) -> list[dict]:
    """DPO pairs: the correction is preferred over the bad reply."""
    return [{"prompt": _text_only(e["messages"]), "chosen": e["correction"], "rejected": e["reply"]}
            for e in entries if e["rating"] == "bad" and e.get("correction") and e["correction"] != e["reply"]]


class Memory:
    """Corrections the model should see next time a similar question comes up."""

    def __init__(self, path: str | Path, min_score: float = 1.0) -> None:
        self.index = SearchIndex(path)
        self.min_score = min_score

    def add(self, question: str, answer: str) -> None:
        self.index.add([{"url": f"memory:{SearchIndex.doc_id({'text': question + answer})}",
                         "title": question[:100], "text": f"Q: {question}\nA: {answer}", "source": "correction"}])

    def recall(self, question: str, k: int = 2) -> list[str]:
        return [h.doc["text"] for h in self.index.search(question, k) if h.score >= self.min_score]

    def note(self, question: str) -> dict | None:
        """A system message with relevant past corrections, or None if there are none."""
        notes = self.recall(question)
        if not notes:
            return None
        return {"role": "system", "content": "Corrections from earlier conversations:\n" + "\n\n".join(notes)}
