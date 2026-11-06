"""Deduplication at three levels: URL, exact document, near-duplicate document (MinHash),
plus line-level dedup across the corpus to strip repeated headers, footers and menus.
"""

from __future__ import annotations

import hashlib
import re
import struct
from collections import Counter
from urllib.parse import urlsplit, urlunsplit

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)


def normalize_url(url: str) -> str:
    p = urlsplit(url.strip())
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = p.path.rstrip("/") or "/"
    return urlunsplit(("", host, path, p.query, ""))  # ignore scheme and #fragment


def text_hash(text: str) -> bytes:
    norm = " ".join(text.lower().split())
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).digest()


def line_hash(line: str) -> int:
    norm = " ".join(line.lower().split())
    return struct.unpack("<Q", hashlib.blake2b(norm.encode("utf-8"), digest_size=8).digest())[0]


class ExactDeduper:
    def __init__(self) -> None:
        self.seen: set[bytes] = set()

    def is_dup(self, key: bytes | str) -> bool:
        if isinstance(key, str):
            key = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
        if key in self.seen:
            return True
        self.seen.add(key)
        return False


class MinHashDeduper:
    """Streaming near-duplicate detection with MinHash + LSH.

    Defaults match FineWeb: word 5-grams, 112 hashes in 14 bands of 8, which flags
    pairs above roughly 75% Jaccard similarity. The first document seen is kept.
    """

    def __init__(self, num_bands: int = 14, rows: int = 8, ngram: int = 5, seed: int = 1) -> None:
        self.num_bands, self.rows, self.ngram = num_bands, rows, ngram
        rng = np.random.default_rng(seed)
        n = num_bands * rows
        # multiply-shift hashing: ((a*x + b) mod 2^64) >> 32, a odd
        self.a = (rng.integers(0, 2**63, n, dtype=np.uint64) * np.uint64(2) + np.uint64(1))
        self.b = rng.integers(0, 2**63, n, dtype=np.uint64)
        self.tables: list[set[bytes]] = [set() for _ in range(num_bands)]

    def _shingles(self, text: str) -> np.ndarray:
        words = _WORD.findall(text.lower())
        if len(words) < self.ngram:
            grams = [" ".join(words)]
        else:
            grams = {" ".join(words[i : i + self.ngram]) for i in range(len(words) - self.ngram + 1)}
        return np.fromiter(
            (int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "little") for g in grams),
            dtype=np.uint64,
        )

    def signature(self, text: str) -> np.ndarray:
        x = self._shingles(text)
        with np.errstate(over="ignore"):
            h = (x[None, :] * self.a[:, None] + self.b[:, None]) >> np.uint64(32)
        return h.min(axis=1).astype(np.uint32)

    def is_dup(self, text: str) -> bool:
        sig = self.signature(text)
        keys = [sig[i * self.rows : (i + 1) * self.rows].tobytes() for i in range(self.num_bands)]
        dup = any(k in t for k, t in zip(keys, self.tables))
        if not dup:
            for k, t in zip(keys, self.tables):
                t.add(k)
        return dup


def count_lines(texts) -> Counter:
    """Count in how many documents each (normalized) line appears."""
    counts: Counter = Counter()
    for text in texts:
        counts.update({line_hash(ln) for ln in text.split("\n") if ln.strip()})
    return counts


def remove_common_lines(text: str, counts: Counter, max_docs: int) -> tuple[str, int]:
    """Drop lines that appear in more than `max_docs` documents (boilerplate)."""
    kept, removed = [], 0
    for ln in text.split("\n"):
        if ln.strip() and counts.get(line_hash(ln), 0) > max_docs:
            removed += 1
            continue
        kept.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip(), removed
