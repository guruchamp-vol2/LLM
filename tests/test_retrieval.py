import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.retrieval import SearchIndex, search_tool  # noqa: E402

DOCS = [
    {"url": "https://e.org/paris", "title": "Paris", "text": "Paris is the capital and largest city of France."},
    {"url": "https://e.org/berlin", "title": "Berlin", "text": "Berlin is the capital of Germany. It has many museums."},
    {"url": "https://e.org/tomato", "title": "Growing tomatoes",
     "text": "Tomatoes need sun, warmth and regular watering. Plant tomatoes after the last frost."},
    {"url": "https://e.org/python", "title": "Python", "text": "Python is a programming language created by Guido van Rossum."},
]


def test_ranking(tmp_path):
    idx = SearchIndex(tmp_path)
    assert idx.add(DOCS, day="2026-11-06") == 4
    assert idx.search("capital of France")[0].doc["title"] == "Paris"
    assert idx.search("how to grow tomatoes")[0].doc["title"] == "Growing tomatoes"
    assert idx.search("Germany museums")[0].doc["title"] == "Berlin"
    assert idx.search("zebra quantum") == []


def test_dedup_and_persistence(tmp_path):
    idx = SearchIndex(tmp_path)
    idx.add(DOCS, day="2026-11-06")
    assert idx.add(DOCS[:2], day="2026-11-07") == 0  # same URLs: skipped
    assert idx.add([{"url": "https://e.org/new", "text": "Fresh news about Paris today."}], day="2026-11-07") == 1
    again = SearchIndex(tmp_path)  # reload from disk
    assert len(again) == 5
    assert {h.doc["date"] for h in again.search("Paris")} == {"2026-11-06", "2026-11-07"}


def test_search_tool_output(tmp_path):
    idx = SearchIndex(tmp_path)
    idx.add(DOCS, day="2026-11-06")
    out = search_tool(idx)("who created Python")
    assert out.startswith("[1] Python (https://e.org/python, 2026-11-06): ")
    assert "Guido" in out
    assert search_tool(idx)("") == "error: empty query"
    long_doc = {"url": "u", "text": "filler " * 200 + "the needle is here " + "filler " * 200}
    idx.add([long_doc], day="2026-11-06")
    snip = idx.search("needle")[0].snippet("needle")
    assert "needle" in snip and len(snip) < 320
