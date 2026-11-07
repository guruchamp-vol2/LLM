"""A small search index (BM25) that the model can query as a tool.

BM25 is the classic keyword-ranking formula behind most search engines before neural
search: documents score higher when they contain the query's rarer words, more often,
in a shorter text. It needs no GPU and no training, and runs fine on a 1 GB VPS.

The index lives on disk as a folder of JSON-lines files, one per day, plus an
inverted index rebuilt in memory when loaded:

    index_dir/
        docs-2026-11-06.jsonl   {"id", "url", "title", "text", "date", "source"}
        ...

Adding documents only appends to today's file, so a daily collector can add new pages
without rewriting old ones, and old days can be dropped by deleting files.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_WORD = re.compile(r"\w+")
# Very common English words carry no meaning for ranking.
_STOP = set("""a an and are as at be but by for from has have he her his i in is it its of on or our she
that the their them they this to was we were what when where which who will with you your""".split())


def terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) < 40]


@dataclass
class Hit:
    score: float
    doc: dict

    def snippet(self, query: str, width: int = 300) -> str:
        """The part of the text around the first query word, so results stay short."""
        text = self.doc["text"]
        q = terms(query)
        low = text.lower()
        pos = min([p for p in (low.find(w) for w in q) if p >= 0], default=0)
        start = max(0, pos - width // 3)
        cut = text[start : start + width].strip()
        return ("..." if start else "") + cut + ("..." if start + width < len(text) else "")


class SearchIndex:
    def __init__(self, index_dir: str | Path, k1: float = 1.2, b: float = 0.75) -> None:
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.k1, self.b = k1, b
        self.docs: list[dict] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(doc, count)]
        self.lengths: list[int] = []
        self.seen: set[str] = set()
        self.latest: dict[str, int] = {}  # url -> position of its newest version
        for f in sorted(self.dir.glob("docs-*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._index(json.loads(line))

    # ------------------------------------------------------------------ building
    @staticmethod
    def doc_id(doc: dict) -> str:
        """Same URL and same text = same document. A changed page (same URL, new text)
        gets a new id, is stored as a new version, and replaces the old one in search."""
        url, text = doc.get("url", ""), doc["text"]
        key = url + "\n" + hashlib.sha1(text.encode("utf-8")).hexdigest() if url else text[:2000]
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    def _index(self, doc: dict) -> None:
        key = self.doc_id(doc)
        if doc["id"] in self.seen or key in self.seen:
            return
        self.seen.update((doc["id"], key))  # stored ids from older versions of this code differ
        i = len(self.docs)
        self.docs.append(doc)
        if doc.get("url"):
            self.latest[doc["url"]] = i
        counts = Counter(terms(doc.get("title", "") + " " + doc["text"]))
        self.lengths.append(sum(counts.values()))
        for t, c in counts.items():
            self.postings[t].append((i, c))

    def add(self, docs: Iterable[dict], day: str | None = None) -> int:
        """Add documents (each needs "text"; "url", "title", "source" are optional).
        Documents already in the index (same URL and text, or same text when there is no
        URL) are skipped; a known URL with new text is added as its newest version.
        Returns how many were added."""
        day = day or dt.date.today().isoformat()
        new = 0
        with open(self.dir / f"docs-{day}.jsonl", "a", encoding="utf-8") as f:
            for d in docs:
                if not d.get("text", "").strip():
                    continue
                doc = {"id": self.doc_id(d), "url": d.get("url", ""), "title": d.get("title", ""),
                       "text": d["text"], "date": d.get("date", day), "source": d.get("source", "")}
                if doc["id"] in self.seen:
                    continue
                self._index(doc)
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                new += 1
        return new

    # ------------------------------------------------------------------ searching
    def _current(self, i: int) -> bool:
        url = self.docs[i].get("url")
        return not url or self.latest.get(url) == i

    def __len__(self) -> int:
        """Number of current documents (older versions of changed pages don't count)."""
        return sum(1 for i in range(len(self.docs)) if self._current(i))

    def search(self, query: str, k: int = 5) -> list[Hit]:
        n = len(self.docs)
        if not n:
            return []
        avg = sum(self.lengths) / n
        scores: dict[int, float] = defaultdict(float)
        for t in set(terms(query)):
            plist = self.postings.get(t)
            if not plist:
                continue
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, c in plist:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / avg)
                scores[i] += idf * c * (self.k1 + 1) / (c + norm)
        ranked = [(i, s) for i, s in sorted(scores.items(), key=lambda kv: -kv[1]) if self._current(i)]
        return [Hit(s, self.docs[i]) for i, s in ranked[:k]]


def search_tool(index: SearchIndex, k: int = 3, width: int = 300):
    """A tool function for the agent loop: search(query) -> short numbered results.
    k results with snippets of about `width` characters (keep small for short-context models)."""

    def search(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            return "error: empty query"
        hits = index.search(query[:300], k)
        if not hits:
            return "no results"
        lines = []
        for n, h in enumerate(hits, 1):
            src = h.doc.get("url") or h.doc.get("source") or "local"
            lines.append(f"[{n}] {h.doc.get('title') or '(untitled)'} ({src}, {h.doc.get('date', '')}): {h.snippet(query, width)}")
        return "\n".join(lines)

    return search
