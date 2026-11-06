"""Byte-level BPE tokenizer, written from scratch.

- Works on raw UTF-8 bytes, so any text (any language, emoji, code) can be encoded.
- Text is first split with a GPT-4-style regex so merges never cross word/number/space
  boundaries in odd ways.
- Training counts unique chunks once and updates pair counts incrementally, using a
  heap to find the most frequent pair, so it scales to large corpora.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import regex as re

# Same split pattern family as GPT-4 (cl100k): contractions, letters, 1-3 digit numbers,
# punctuation runs, newlines, and whitespace.
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)

# Reserved tokens for chat, tools and other modalities (the "all-in-one" layer).
DEFAULT_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|image|>",
    "<|audio|>",
]

Pair = tuple[int, int]


def _pairs(ids: list[int]) -> Iterable[Pair]:
    return zip(ids, ids[1:])


def _merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    a, b = pair
    out: list[int] = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(
        self,
        merges: list[Pair] | None = None,
        special_tokens: list[str] | None = None,
        pattern: str = SPLIT_PATTERN,
    ) -> None:
        self.pattern = pattern
        self._split = re.compile(pattern)
        self.merges: list[Pair] = list(merges or [])
        self.ranks: dict[Pair, int] = {p: i for i, p in enumerate(self.merges)}

        # Token id -> bytes.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]

        specials = DEFAULT_SPECIAL_TOKENS if special_tokens is None else special_tokens
        base = 256 + len(self.merges)
        self.special_tokens: dict[str, int] = {s: base + i for i, s in enumerate(specials)}
        self.special_ids: dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        self._special_split = (
            re.compile("(" + "|".join(re.escape(s) for s in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._cache: dict[bytes, list[int]] = {}

    # ------------------------------------------------------------------ training
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: list[str] | None = None,
        pattern: str = SPLIT_PATTERN,
        verbose: bool = False,
    ) -> "BPETokenizer":
        specials = DEFAULT_SPECIAL_TOKENS if special_tokens is None else special_tokens
        num_merges = vocab_size - 256 - len(specials)
        if num_merges < 0:
            raise ValueError("vocab_size too small for 256 bytes + special tokens")

        split = re.compile(pattern)
        chunk_counts: Counter[bytes] = Counter()
        for text in texts:
            for chunk in split.findall(text):
                chunk_counts[chunk.encode("utf-8")] += 1

        words: list[list[int]] = [list(c) for c in chunk_counts]
        freqs: list[int] = list(chunk_counts.values())

        pair_counts: dict[Pair, int] = {}
        where: dict[Pair, set[int]] = {}
        for wi, w in enumerate(words):
            f = freqs[wi]
            for p in _pairs(w):
                pair_counts[p] = pair_counts.get(p, 0) + f
                where.setdefault(p, set()).add(wi)

        # Max-heap via negative counts; stale entries are skipped when popped.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[Pair] = []
        while len(merges) < num_merges and heap:
            neg, pair = heapq.heappop(heap)
            current = pair_counts.get(pair, 0)
            if current <= 0 or current != -neg:
                continue
            new_id = 256 + len(merges)
            merges.append(pair)

            delta: dict[Pair, int] = {}
            for wi in where.pop(pair, ()):
                old = words[wi]
                new = _merge(old, pair, new_id)
                if len(new) == len(old):
                    continue  # stale index entry, pair no longer in this word
                f = freqs[wi]
                for p in _pairs(old):
                    delta[p] = delta.get(p, 0) - f
                for p in _pairs(new):
                    delta[p] = delta.get(p, 0) + f
                    where.setdefault(p, set()).add(wi)
                words[wi] = new

            for p, d in delta.items():
                if d == 0:
                    continue
                c = pair_counts.get(p, 0) + d
                if c > 0:
                    pair_counts[p] = c
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)
            pair_counts.pop(pair, None)

            if verbose and len(merges) % 1000 == 0:
                print(f"merges: {len(merges)}/{num_merges}")

        if len(merges) < num_merges:
            import warnings

            warnings.warn(
                f"corpus ran out of pairs: learned {len(merges)}/{num_merges} merges; "
                "vocab will be smaller than requested"
            )
        return cls(merges, specials, pattern)

    # ------------------------------------------------------------------ encoding
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk)
        while len(ids) >= 2:
            best = min(_pairs(ids), key=lambda p: self.ranks.get(p, float("inf")))
            rank = self.ranks.get(best)
            if rank is None:
                break
            ids = _merge(ids, best, 256 + rank)
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special-token strings as plain text."""
        out: list[int] = []
        for chunk in self._split.findall(text):
            out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        if not allow_special or self._special_split is None:
            return self.encode_ordinary(text)
        out: list[int] = []
        for part in self._special_split.split(text):
            if not part:
                continue
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            else:
                out.extend(self.encode_ordinary(part))
        return out

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        parts = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in self.special_ids:
                parts.append(self.special_ids[i].encode("utf-8"))
            else:
                raise ValueError(f"unknown token id {i}")
        return b"".join(parts)

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ save / load
    def save(self, path: str | Path) -> None:
        data = {
            "version": 1,
            "pattern": self.pattern,
            "merges": [list(p) for p in self.merges],
            "special_tokens": list(self.special_tokens),
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            [tuple(p) for p in data["merges"]],
            data["special_tokens"],
            data["pattern"],
        )
