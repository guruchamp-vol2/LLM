"""Scaling-law fits: predict loss from model size and training tokens, and pick the
compute-optimal size for a budget.

Two standard methods (Hoffmann et al. 2022, "Chinchilla"):

1. Parametric fit:   L(N, D) = E + A / N**alpha + B / D**beta
   N = parameters, D = training tokens. For fixed (alpha, beta) the formula is linear in
   (E, A, B), so we search a grid of exponents, solve each by least squares (keeping
   E, A, B >= 0), then refine the grid around the best point.

2. IsoFLOP fit: train several sizes at each compute budget C (FLOPs ~ 6 * N * D), fit a
   parabola of loss against log N per budget, and take its minimum as the best N for that
   C. Then fit N_opt = k * C**a across budgets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def flops(n_params: float, tokens: float) -> float:
    """Training compute, ~6 FLOPs per parameter per token (forward + backward)."""
    return 6.0 * n_params * tokens


@dataclass
class ParametricFit:
    E: float
    A: float
    alpha: float
    B: float
    beta: float
    rmse: float

    def loss(self, n, d):
        n, d = np.asarray(n, np.float64), np.asarray(d, np.float64)
        return self.E + self.A / n**self.alpha + self.B / d**self.beta

    def optimal(self, compute: float) -> tuple[float, float]:
        """(N, D) minimising predicted loss subject to 6 N D = compute (closed form)."""
        a, b = self.alpha, self.beta
        G = (a * self.A / (b * self.B)) ** (1 / (a + b))
        n = G * (compute / 6) ** (b / (a + b))
        return float(n), float(compute / (6 * n))


def _nnls3(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least squares with all 3 coefficients >= 0, by trying every active set (2**3 cases)."""
    best = (np.zeros(3), np.inf)
    for mask in range(1, 8):
        cols = [i for i in range(3) if mask >> i & 1]
        coef, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
        if np.any(coef < 0):
            continue
        full = np.zeros(3)
        full[cols] = coef
        err = float(np.sum((X @ full - y) ** 2))
        if err < best[1]:
            best = (full, err)
    return best


def fit_parametric(n, d, loss, alphas=None, betas=None, refine: int = 3) -> ParametricFit:
    n, d, y = (np.asarray(v, np.float64) for v in (n, d, loss))
    if len(y) < 5:
        raise ValueError("need at least 5 runs for a 5-parameter fit")
    alphas = np.linspace(0.05, 1.0, 40) if alphas is None else np.asarray(alphas)
    betas = np.linspace(0.05, 1.0, 40) if betas is None else np.asarray(betas)
    best = None
    for _ in range(refine + 1):
        for a in alphas:
            xa = n**-a
            for b in betas:
                X = np.stack([np.ones_like(y), xa, d**-b], axis=1)
                coef, err = _nnls3(X, y)
                if best is None or err < best[0]:
                    best = (err, a, b, coef)
        _, a, b, _ = best
        sa, sb = (alphas[1] - alphas[0]), (betas[1] - betas[0])
        alphas = np.linspace(max(a - sa, 1e-3), a + sa, 21)
        betas = np.linspace(max(b - sb, 1e-3), b + sb, 21)
    err, a, b, (E, A, B) = best
    return ParametricFit(float(E), float(A), float(a), float(B), float(b), float(np.sqrt(err / len(y))))


@dataclass
class IsoFlopFit:
    budgets: list[float]
    n_opt: list[float]
    k: float
    a: float

    def optimal_n(self, compute: float) -> float:
        return float(self.k * compute**self.a)


def fit_isoflop(n, d, loss) -> IsoFlopFit:
    """Group runs by compute budget (6 N D, rounded to 2 significant figures)."""
    n, d, y = (np.asarray(v, np.float64) for v in (n, d, loss))
    c = np.array([float(f"{x:.2g}") for x in 6 * n * d])
    budgets, n_opt = [], []
    for budget in sorted(set(c)):
        sel = c == budget
        if sel.sum() < 3:
            continue
        x = np.log(n[sel])
        p2, p1, _ = np.polyfit(x, y[sel], 2)
        if p2 <= 0:
            continue  # no minimum inside this budget's sizes
        best = -p1 / (2 * p2)
        if not x.min() <= best <= x.max():
            continue  # minimum lies outside the sizes tried: extrapolation, skip
        budgets.append(float(budget))
        n_opt.append(float(np.exp(best)))
    if len(budgets) < 2:
        raise ValueError("need at least 2 budgets with 3+ sizes and an interior minimum")
    a, logk = np.polyfit(np.log(budgets), np.log(n_opt), 1)
    return IsoFlopFit(budgets, n_opt, float(np.exp(logk)), float(a))
