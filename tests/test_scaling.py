import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.model import ModelConfig, count_params, init_params  # noqa: E402
from aio.scaling import fit_isoflop, fit_parametric, non_embedding_params, plan, shape_for  # noqa: E402
from aio.scaling.fit import ParametricFit  # noqa: E402
from aio.train import OptimConfig, TrainConfig  # noqa: E402

TRUE = ParametricFit(E=1.7, A=400.0, alpha=0.34, B=410.0, beta=0.28, rmse=0.0)


def _grid(noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n, d = [], []
    for c in [1e15, 3e15, 1e16, 3e16, 1e17]:
        for nn in np.geomspace(1e6, 1e8, 7):
            n.append(nn)
            d.append(c / (6 * nn))
    n, d = np.array(n), np.array(d)
    return n, d, TRUE.loss(n, d) + rng.normal(0, noise, len(n))


def test_parametric_fit_recovers_known_law():
    n, d, y = _grid()
    f = fit_parametric(n, d, y)
    assert f.rmse < 1e-3
    assert abs(f.alpha - TRUE.alpha) < 0.03 and abs(f.beta - TRUE.beta) < 0.03
    assert abs(f.E - TRUE.E) < 0.05
    # predictions far outside the data stay close
    assert abs(float(f.loss(1e9, 2e10)) - float(TRUE.loss(1e9, 2e10))) < 0.02


def test_parametric_optimal_is_minimum():
    n_opt, d_opt = TRUE.optimal(1e20)
    assert abs(6 * n_opt * d_opt - 1e20) / 1e20 < 1e-9
    for k in (0.7, 1.4):  # moving off the optimum along the budget makes it worse
        assert TRUE.loss(n_opt * k, d_opt / k) > TRUE.loss(n_opt, d_opt)


def test_parametric_fit_with_noise():
    n, d, y = _grid(noise=0.01, seed=1)
    f = fit_parametric(n, d, y)
    n_true, _ = TRUE.optimal(1e19)
    n_fit, _ = f.optimal(1e19)
    assert 0.5 < n_fit / n_true < 2.0  # right size to within 2x at 100x past the data


def test_isoflop_fit_matches_parametric_optimum():
    n, d, y = _grid()
    iso = fit_isoflop(n, d, y)
    assert len(iso.budgets) >= 3
    for c in iso.budgets:
        assert abs(np.log(iso.optimal_n(c) / TRUE.optimal(c)[0])) < 0.15
    a_true = TRUE.beta / (TRUE.alpha + TRUE.beta)
    assert abs(iso.a - a_true) < 0.03


def test_fit_needs_enough_points():
    with pytest.raises(ValueError):
        fit_parametric([1e6, 2e6], [1e9, 1e9], [3.0, 2.9])


def test_non_embedding_param_count():
    m = ModelConfig(vocab_size=1000, d_model=64, n_layers=2, n_heads=2, n_kv_heads=1)
    import jax
    total = count_params(init_params(m, jax.random.PRNGKey(0)))
    assert non_embedding_params(m) == total - m.vocab_size * m.d_model


def test_plan_hits_budget():
    base = TrainConfig(data_dir="d", out_dir="o", seq_len=128, batch_size=16,
                       model=ModelConfig(vocab_size=8192), optim=OptimConfig())
    runs = plan(base, [1e14, 1e15], [64, 128, 256], "sweep", min_steps=10)
    assert len(runs) >= 4 and len({r.out_dir for r in runs}) == len(runs)
    for r in runs:
        c = float(Path(r.out_dir).name.split("_")[0][1:])
        used = 6 * non_embedding_params(r.model) * r.steps * 16 * 128
        assert 0.9 < used / c <= 1.0 + 1e-9
        assert r.optim.total_steps == r.steps
    m = shape_for(256, 8192, 128)
    assert (m.n_layers, m.n_heads, m.n_kv_heads) == (8, 8, 4)
