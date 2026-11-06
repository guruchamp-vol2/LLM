"""Educational-quality scorer: predicts a 0-5 score for how useful a page is to learn from.

Trained on the free FineWeb-Edu annotations (pages graded 0-5 by a large model), but the
scorer itself is ours: a linear model over hashed word and word-pair features, in NumPy.
It is cheap enough to score billions of pages on CPUs.

    scorer = QualityScorer.load("quality.npz")
    scorer.score("Photosynthesis is the process by which plants ...")  # -> e.g. 3.4
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_WORD = re.compile(r"\w+")
_UNI_MIX = np.uint64(0x9E3779B97F4A7C15)
_BI_MUL = np.uint64(1_000_003)
_BI_MIX = np.uint64(0xC2B2AE3D27D4EB4F)


def features(text: str, n_buckets: int, max_words: int) -> np.ndarray:
    """Sorted unique bucket ids of the words and word pairs in the first max_words words."""
    words = _WORD.findall(text.lower())[:max_words]
    if not words:
        return np.zeros(0, dtype=np.int32)
    # crc32 is stable across runs and machines, unlike Python's hash().
    h = np.fromiter((zlib.crc32(w.encode()) for w in words), dtype=np.uint64, count=len(words))
    uni = h ^ _UNI_MIX
    bi = (h[:-1] * _BI_MUL + h[1:]) ^ _BI_MIX  # wraps around on overflow, which is fine
    ids = np.concatenate([uni, bi]) % np.uint64(n_buckets)
    return np.unique(ids).astype(np.int32)


@dataclass
class SparseRows:
    """Documents as rows of feature ids (CSR layout without values)."""

    indptr: np.ndarray   # int64, length n_rows + 1
    indices: np.ndarray  # int32 bucket ids

    @classmethod
    def build(cls, texts: Iterable[str], n_buckets: int, max_words: int) -> "SparseRows":
        parts, lengths = [], [0]
        for t in texts:
            f = features(t, n_buckets, max_words)
            parts.append(f)
            lengths.append(len(f))
        indptr = np.cumsum(np.array(lengths, dtype=np.int64))
        indices = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
        return cls(indptr, indices.astype(np.int32, copy=False))

    def __len__(self) -> int:
        return len(self.indptr) - 1

    def batch(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For the given rows: (feature ids, row position of each id, value of each id).

        Each document's features are binary and scaled to unit length, so long and short
        pages get comparable scores.
        """
        starts, ends = self.indptr[rows], self.indptr[rows + 1]
        lens = ends - starts
        idx = np.concatenate([self.indices[s:e] for s, e in zip(starts, ends)]) if len(rows) else np.zeros(0, np.int32)
        pos = np.repeat(np.arange(len(rows)), lens)
        val = np.repeat(1.0 / np.sqrt(np.maximum(lens, 1)), lens).astype(np.float32)
        return idx, pos, val


class QualityScorer:
    VERSION = 1

    def __init__(self, n_buckets: int = 1 << 20, max_words: int = 600) -> None:
        self.n_buckets, self.max_words = n_buckets, max_words
        self.w = np.zeros(n_buckets, dtype=np.float32)
        self.b = 0.0
        # Score at or above which a page counts as "educational" (grade 3+).
        # Picked on held-out data by pick_threshold(); a linear model's scores bunch up
        # near the average, so this is usually below 3.
        self.keep_threshold = 3.0

    # ------------------------------------------------------------------ scoring
    def _raw(self, rows: SparseRows, sel: np.ndarray) -> np.ndarray:
        idx, pos, val = rows.batch(sel)
        return self.b + np.bincount(pos, weights=self.w[idx] * val, minlength=len(sel))

    def score_rows(self, rows: SparseRows, batch_size: int = 4096) -> np.ndarray:
        out = np.empty(len(rows), dtype=np.float32)
        for i in range(0, len(rows), batch_size):
            sel = np.arange(i, min(i + batch_size, len(rows)))
            out[sel] = self._raw(rows, sel)
        return np.clip(out, 0.0, 5.0)

    def score(self, text: str) -> float:
        f = features(text, self.n_buckets, self.max_words)
        if len(f) == 0:
            return float(np.clip(self.b, 0.0, 5.0))
        return float(np.clip(self.b + self.w[f].sum() / np.sqrt(len(f)), 0.0, 5.0))

    def score_many(self, texts: Sequence[str]) -> np.ndarray:
        return self.score_rows(SparseRows.build(texts, self.n_buckets, self.max_words))

    # ------------------------------------------------------------------ training
    def fit(
        self,
        rows: SparseRows,
        y: np.ndarray,
        val_rows: SparseRows | None = None,
        val_y: np.ndarray | None = None,
        epochs: int = 4,
        batch_size: int = 512,
        lr: float = 0.005,
        l2: float = 1e-5,
        seed: int = 0,
        log=print,
    ) -> list[dict]:
        """Mean-squared-error regression with Adam. Keeps the epoch with the best validation error."""
        y = np.asarray(y, dtype=np.float32)
        rng = np.random.default_rng(seed)
        self.b = float(y.mean())
        m = np.zeros_like(self.w)
        v = np.zeros_like(self.w)
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        history, best = [], (np.inf, self.w.copy(), self.b)

        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(rows))
            for i in range(0, len(order), batch_size):
                sel = order[i : i + batch_size]
                idx, pos, val = rows.batch(sel)
                pred = self.b + np.bincount(pos, weights=self.w[idx] * val, minlength=len(sel))
                err = (pred - y[sel]) / len(sel)  # d(MSE/2)/d(pred)
                g = np.bincount(idx, weights=err[pos] * val, minlength=self.n_buckets).astype(np.float32)
                g += l2 * self.w
                t += 1
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                step = lr * np.sqrt(1 - b2**t) / (1 - b1**t)
                self.w -= step * m / (np.sqrt(v) + eps)
                self.b -= lr * float(err.sum())

            rec = {"epoch": epoch, "train": evaluate(self.score_rows(rows), y)}
            if val_rows is not None and val_y is not None:
                rec["val"] = evaluate(self.score_rows(val_rows), val_y)
                if rec["val"]["mse"] < best[0]:
                    best = (rec["val"]["mse"], self.w.copy(), self.b)
            history.append(rec)
            log(rec)

        if val_rows is not None and best[0] < np.inf:
            _, self.w, self.b = best
        return history

    # ------------------------------------------------------------------ save / load
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, w=self.w, b=np.float64(self.b), n_buckets=self.n_buckets,
            max_words=self.max_words, keep_threshold=self.keep_threshold, version=self.VERSION,
        )

    @classmethod
    def load(cls, path: str | Path) -> "QualityScorer":
        z = np.load(path)
        if int(z["version"]) != cls.VERSION:
            raise ValueError(f"unsupported quality model version {int(z['version'])}")
        s = cls(int(z["n_buckets"]), int(z["max_words"]))
        s.w, s.b = z["w"].astype(np.float32), float(z["b"])
        s.keep_threshold = float(z["keep_threshold"])
        return s


def _prf(pred: np.ndarray, y: np.ndarray, threshold: float, grade: float = 3.0) -> tuple[float, float, float]:
    """Precision, recall and F1 of "pred >= threshold" at finding pages graded >= grade."""
    keep_p, keep_y = pred >= threshold, y >= grade
    tp = float(np.sum(keep_p & keep_y))
    precision = tp / max(float(keep_p.sum()), 1.0)
    recall = tp / max(float(keep_y.sum()), 1.0)
    return precision, recall, 2 * precision * recall / max(precision + recall, 1e-12)


def pick_threshold(pred: np.ndarray, y: np.ndarray, grade: float = 3.0) -> float:
    """The score cut-off that best separates pages graded >= grade from the rest (max F1)."""
    pred, y = np.asarray(pred, np.float64), np.asarray(y, np.float64)
    cands = np.round(np.arange(0.5, 5.0, 0.05), 2)
    return float(max(cands, key=lambda th: _prf(pred, y, th, grade)[2]))


def evaluate(pred: np.ndarray, y: np.ndarray, threshold: float = 3.0) -> dict:
    """Error, correlation, and how well "score >= threshold" finds pages graded 3 or higher."""
    pred, y = np.asarray(pred, np.float64), np.asarray(y, np.float64)
    precision, recall, f1 = _prf(pred, y, threshold)
    corr = float(np.corrcoef(pred, y)[0, 1]) if pred.std() > 0 and y.std() > 0 else 0.0
    return {
        "mse": round(float(np.mean((pred - y) ** 2)), 4),
        "pearson": round(corr, 4),
        "exact_acc": round(float(np.mean(np.clip(np.round(pred), 0, 5) == y)), 4),
        "threshold": threshold,
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }
