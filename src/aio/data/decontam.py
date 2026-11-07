"""Decontamination: drop training documents that contain benchmark problems.

If test problems (or their solutions) leak into training data, benchmark scores measure
memory instead of skill. Like GPT-3 and most model reports since, we flag a document
when it shares any long n-gram (default 13 words) with a benchmark item.

Words are lower-cased runs of letters/digits/underscore, so formatting differences
(spacing, line breaks, quotes) don't hide a copy.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

_WORD = re.compile(r"\w+")


def _informative(gram: list[str]) -> bool:
    """Test data is full of runs like "1 2 3 ... 13", "a b c ... m" or "0 0 0 ...", which
    also fill ordinary code. Only runs with several real words can identify a problem."""
    return sum(len(t) >= 3 and not t.isdigit() for t in gram) >= len(gram) // 2


def _ngrams(text: str, n: int) -> Iterable[str]:
    w = _WORD.findall(text.lower())
    for i in range(len(w) - n + 1):
        if _informative(w[i : i + n]):
            yield " ".join(w[i : i + n])


def _grams(text: str, n: int) -> Iterable[bytes]:
    for s in _ngrams(text, n):
        yield hashlib.blake2b(s.encode(), digest_size=8).digest()


class Decontaminator:
    def __init__(self, benchmark_texts: Iterable[str], n: int = 13) -> None:
        self.n = n
        self.grams: set[bytes] = set()
        for t in benchmark_texts:
            self.grams.update(_grams(t, n))

    def contaminated(self, text: str) -> bool:
        return any(g in self.grams for g in _grams(text, self.n))

    def matches(self, text: str) -> list[str]:
        """The shared n-grams themselves (for checking what a filter is catching)."""
        return [s for s in _ngrams(text, self.n)
                if hashlib.blake2b(s.encode(), digest_size=8).digest() in self.grams]

    def __len__(self) -> int:
        return len(self.grams)


def benchmark_texts(eval_dir: str) -> list[str]:
    """Prompts and tests of the coding benchmarks we report. Reference solutions are left
    out: short solutions share 13-word runs with lots of ordinary code (loops over a
    list...), which would throw away good files, while a leaked problem almost always
    comes with its docstring or tests."""
    from pathlib import Path

    from ..eval.code import load

    texts = []
    for task in ("humaneval", "mbpp"):
        p = Path(eval_dir) / f"{task}.parquet"
        if p.exists():
            for prob in load(task, str(p)):
                texts += [prob.prompt, prob.test]
    return [t for t in texts if t]
