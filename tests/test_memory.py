import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.memory import FeedbackLog, Memory, to_pairs, to_sft  # noqa: E402

Q = [{"role": "user", "content": "What is the capital of Australia?"}]


def test_feedback_log_and_training_data(tmp_path):
    log = FeedbackLog(tmp_path / "fb.jsonl")
    log.record(Q, "Sydney.", "bad", correction="Canberra.")
    log.record([{"role": "user", "content": "2+2?"}], "4.", "good")
    log.record([{"role": "user", "content": "Hi"}], "Bye.", "bad")  # bad, but no correction given
    with pytest.raises(ValueError):
        log.record(Q, "x", "meh")
    entries = FeedbackLog(tmp_path / "fb.jsonl").entries()
    assert len(entries) == 3 and entries[0]["correction"] == "Canberra."

    sft = to_sft(entries)
    assert [c[-1]["content"] for c in sft] == ["Canberra.", "4."]  # uncorrected bad reply is not trained on
    pairs = to_pairs(entries)
    assert pairs == [{"prompt": Q, "chosen": "Canberra.", "rejected": "Sydney."}]


def test_tool_messages_are_dropped_from_training_prompts(tmp_path):
    msgs = Q + [{"role": "assistant", "tool_call": {"name": "search", "arguments": {}}}, {"role": "tool", "content": "r"}]
    [conv] = to_sft([{"messages": msgs, "reply": "Canberra.", "rating": "good"}])
    assert conv == Q + [{"role": "assistant", "content": "Canberra."}]


def test_memory_recalls_relevant_corrections_only(tmp_path):
    mem = Memory(tmp_path / "mem")
    mem.add("What is the capital of Australia?", "Canberra, not Sydney.")
    mem.add("How many legs does a spider have?", "Eight.")
    note = mem.note("capital city of Australia")
    assert note["role"] == "system" and "Canberra" in note["content"] and "spider" not in note["content"]
    assert mem.note("tell me about volcanoes") is None
    assert Memory(tmp_path / "mem").recall("spider legs") == ["Q: How many legs does a spider have?\nA: Eight."]
