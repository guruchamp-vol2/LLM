"""Topic (domain) classifier: tags each page with one of a fixed set of topics.

Knowing the topic lets us choose the training mix, e.g. more science and code, less
celebrity news, instead of taking the web as it comes.

Trained on the free WebOrganizer topic annotations (24 topics, labelled by a large model),
but the classifier is ours: a softmax over the same hashed word and word-pair features as
the quality scorer, in NumPy.

    clf = DomainClassifier.load("domain.npz")
    clf.predict("The mitochondria is the powerhouse of the cell ...")  # -> ("science_math_tech", 0.91)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .quality import SparseRows, features


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class DomainClassifier:
    VERSION = 1

    def __init__(self, labels: Sequence[str], n_buckets: int = 1 << 18, max_words: int = 600) -> None:
        self.labels = list(labels)
        self.n_buckets, self.max_words = n_buckets, max_words
        k = len(self.labels)
        self.W = np.zeros((n_buckets, k), dtype=np.float32)
        self.b = np.zeros(k, dtype=np.float32)

    # ------------------------------------------------------------------ prediction
    def _logits(self, rows: SparseRows, sel: np.ndarray) -> np.ndarray:
        idx, pos, val = rows.batch(sel)
        k = len(self.labels)
        keys = (pos[:, None] * k + np.arange(k)).ravel()
        z = np.bincount(keys, weights=(self.W[idx] * val[:, None]).ravel(), minlength=len(sel) * k)
        return z.reshape(len(sel), k) + self.b

    def proba_rows(self, rows: SparseRows, batch_size: int = 2048) -> np.ndarray:
        out = np.empty((len(rows), len(self.labels)), dtype=np.float32)
        for i in range(0, len(rows), batch_size):
            sel = np.arange(i, min(i + batch_size, len(rows)))
            out[sel] = _softmax(self._logits(rows, sel))
        return out

    def proba(self, text: str) -> np.ndarray:
        f = features(text, self.n_buckets, self.max_words)
        z = self.b.astype(np.float64)
        if len(f):
            z = z + self.W[f].sum(axis=0) / np.sqrt(len(f))
        return _softmax(z[None, :])[0]

    def predict(self, text: str) -> tuple[str, float]:
        p = self.proba(text)
        i = int(p.argmax())
        return self.labels[i], float(p[i])

    def predict_many(self, texts: Sequence[str]) -> list[tuple[str, float]]:
        p = self.proba_rows(SparseRows.build(texts, self.n_buckets, self.max_words))
        return [(self.labels[i], float(p[j, i])) for j, i in enumerate(p.argmax(axis=1))]

    # ------------------------------------------------------------------ training
    def fit(
        self,
        rows: SparseRows,
        targets: np.ndarray,
        val_rows: SparseRows | None = None,
        val_targets: np.ndarray | None = None,
        epochs: int = 8,
        batch_size: int = 256,
        lr: float = 0.01,
        l2: float = 1e-6,
        seed: int = 0,
        log=print,
    ) -> list[dict]:
        """Cross-entropy against soft targets (one probability per topic, per page).

        Uses sparse Adam: only the feature rows present in a batch are updated, which keeps
        each step cheap even with millions of weights. Keeps the epoch with the best
        validation loss.
        """
        P = np.asarray(targets, dtype=np.float32)
        P = P / P.sum(axis=1, keepdims=True)
        k = len(self.labels)
        rng = np.random.default_rng(seed)
        self.b = np.log(P.mean(axis=0) + 1e-6).astype(np.float32)
        mW, vW = np.zeros_like(self.W), np.zeros_like(self.W)
        mb, vb = np.zeros_like(self.b), np.zeros_like(self.b)
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        history, best = [], (np.inf, None, None)

        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(rows))
            for i in range(0, len(order), batch_size):
                sel = order[i : i + batch_size]
                idx, pos, val = rows.batch(sel)
                keys = (pos[:, None] * k + np.arange(k)).ravel()
                z = np.bincount(keys, weights=(self.W[idx] * val[:, None]).ravel(), minlength=len(sel) * k)
                dz = (_softmax(z.reshape(len(sel), k) + self.b) - P[sel]) / len(sel)

                # Sum gradient contributions per distinct feature id.
                u, inv = np.unique(idx, return_inverse=True)
                gkeys = (inv[:, None] * k + np.arange(k)).ravel()
                g = np.bincount(gkeys, weights=(dz[pos] * val[:, None]).ravel(), minlength=len(u) * k)
                g = g.reshape(len(u), k).astype(np.float32) + l2 * self.W[u]
                gb = dz.sum(axis=0).astype(np.float32)

                t += 1
                step = lr * np.sqrt(1 - b2**t) / (1 - b1**t)
                mW[u] = b1 * mW[u] + (1 - b1) * g
                vW[u] = b2 * vW[u] + (1 - b2) * g * g
                self.W[u] -= step * mW[u] / (np.sqrt(vW[u]) + eps)
                mb[:] = b1 * mb + (1 - b1) * gb
                vb[:] = b2 * vb + (1 - b2) * gb * gb
                self.b -= step * mb / (np.sqrt(vb) + eps)

            rec = {"epoch": epoch}
            if val_rows is not None and val_targets is not None:
                rec["val"] = evaluate(self.proba_rows(val_rows), val_targets)
                if rec["val"]["loss"] < best[0]:
                    best = (rec["val"]["loss"], self.W.copy(), self.b.copy())
            history.append(rec)
            log(rec)

        if best[1] is not None:
            _, self.W, self.b = best
        return history

    # ------------------------------------------------------------------ save / load
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, W=self.W.astype(np.float16), b=self.b, labels=np.array(self.labels),
            n_buckets=self.n_buckets, max_words=self.max_words, version=self.VERSION,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DomainClassifier":
        z = np.load(path)
        if int(z["version"]) != cls.VERSION:
            raise ValueError(f"unsupported domain model version {int(z['version'])}")
        c = cls([str(x) for x in z["labels"]], int(z["n_buckets"]), int(z["max_words"]))
        c.W, c.b = z["W"].astype(np.float32), z["b"].astype(np.float32)
        return c


def evaluate(proba: np.ndarray, targets: np.ndarray) -> dict:
    """Agreement with the labeller's top topic, plus per-topic F1 averaged over topics."""
    P = np.asarray(targets, dtype=np.float64)
    P = P / P.sum(axis=1, keepdims=True)
    pred, true = proba.argmax(axis=1), P.argmax(axis=1)
    top2 = np.argsort(-proba, axis=1)[:, :2]
    f1s = []
    for c in range(P.shape[1]):
        tp = float(np.sum((pred == c) & (true == c)))
        n_pred, n_true = float(np.sum(pred == c)), float(np.sum(true == c))
        if n_true == 0:
            continue
        prec, rec = tp / max(n_pred, 1.0), tp / n_true
        f1s.append(2 * prec * rec / max(prec + rec, 1e-12))
    return {
        "loss": round(float(-np.mean(np.sum(P * np.log(np.clip(proba, 1e-9, 1.0)), axis=1))), 4),
        "accuracy": round(float(np.mean(pred == true)), 4),
        "top2_accuracy": round(float(np.mean(np.any(top2 == true[:, None], axis=1))), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
    }
