import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aio.data.tokens import TokenDataset, TokenShardWriter  # noqa: E402
from aio.model import ModelConfig, forward, init_params  # noqa: E402
from aio.model.generate import generate  # noqa: E402
from aio.train import OptimConfig, TrainConfig, adamw_update, init_opt_state, lr_at, train  # noqa: E402
from aio.train import checkpoint  # noqa: E402

V = 32
TINY = ModelConfig(vocab_size=V, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)


def _write_counting_data(d: Path, n_docs=200, seed=0) -> Path:
    """Documents that count upward mod 30 from a random start: easy to learn."""
    rng = np.random.default_rng(seed)
    with TokenShardWriter(d, vocab_size=V, eot_id=31, tokens_per_shard=2000) as w:
        for _ in range(n_docs):
            s = int(rng.integers(0, 30))
            w.add([(s + i) % 30 for i in range(int(rng.integers(20, 60)))])
    return d


def _cfg(tmp: Path, **kw) -> TrainConfig:
    data = _write_counting_data(tmp / "train") if not (tmp / "train").exists() else tmp / "train"
    val = _write_counting_data(tmp / "val", 50, seed=1) if not (tmp / "val").exists() else tmp / "val"
    base = dict(data_dir=str(data), val_dir=str(val), out_dir=str(tmp / "run"), seq_len=32, batch_size=8,
                steps=40, log_every=10, eval_every=20, eval_batches=2, ckpt_every=20, model=TINY,
                optim=OptimConfig(lr=1e-2, warmup_steps=5, total_steps=40))
    base.update(kw)
    return TrainConfig(**base)


# ------------------------------------------------------------------ optimizer
def test_lr_schedule():
    c = OptimConfig(lr=1.0, min_lr_ratio=0.1, warmup_steps=10, total_steps=110)
    assert abs(float(lr_at(0, c)) - 0.1) < 1e-6
    assert abs(float(lr_at(9, c)) - 1.0) < 1e-6
    assert abs(float(lr_at(60, c)) - 0.55) < 1e-6  # halfway through the cosine
    assert abs(float(lr_at(110, c)) - 0.1) < 1e-6
    assert abs(float(lr_at(500, c)) - 0.1) < 1e-6


def test_adamw_minimises_and_skips_decay_on_vectors():
    params = {"w": jnp.array([[3.0, -2.0]]), "g": jnp.array([5.0])}
    cfg = OptimConfig(lr=0.1, warmup_steps=1, total_steps=500, weight_decay=0.0, min_lr_ratio=1.0)
    state = init_opt_state(params)
    loss = lambda p: jnp.sum((p["w"] - 1.0) ** 2) + jnp.sum((p["g"] - 2.0) ** 2)  # noqa: E731
    for _ in range(500):
        params, state, stats = adamw_update(params, jax.grad(loss)(params), state, cfg)
    assert float(loss(params)) < 1e-3 and int(state["step"]) == 500

    # With zero gradients, only matrices shrink from weight decay.
    p = {"w": jnp.ones((2, 2)), "g": jnp.ones(2)}
    zero = jax.tree.map(jnp.zeros_like, p)
    p2, _, _ = adamw_update(p, zero, init_opt_state(p), OptimConfig(lr=0.1, warmup_steps=1, weight_decay=0.5))
    assert float(p2["w"][0, 0]) < 1.0 and float(p2["g"][0]) == 1.0


def test_gradient_clipping():
    p = {"w": jnp.zeros((2, 2))}
    g = {"w": jnp.full((2, 2), 100.0)}
    _, state, stats = adamw_update(p, g, init_opt_state(p), OptimConfig(clip_norm=1.0))
    assert abs(float(stats["grad_norm"]) - 200.0) < 1e-3
    assert abs(float(jnp.linalg.norm(state["m"]["w"])) - 0.1) < 1e-4  # (1 - b1) * clipped grad (norm 1)


# ------------------------------------------------------------------ data and checkpoints
def test_token_shards_roundtrip(tmp_path):
    with TokenShardWriter(tmp_path, vocab_size=1000, eot_id=999, tokens_per_shard=10) as w:
        w.add([1, 2, 3])
        w.add([4, 5, 6, 7, 8, 9, 10, 11])
        w.add([12])
    ds = TokenDataset(tmp_path)
    assert ds.index["n_docs"] == 3 and ds.n_tokens == 3 + 8 + 1 + 3
    flat = np.concatenate([np.asarray(s) for s in ds.shards])
    assert flat.tolist() == [1, 2, 3, 999, 4, 5, 6, 7, 8, 9, 10, 11, 999, 12, 999]
    assert ds.shards[0].dtype == np.uint16
    a, b = ds.batch(5, 4, 6, seed=1), ds.batch(5, 4, 6, seed=1)
    assert a.shape == (4, 7) and np.array_equal(a, b)            # same step -> same batch
    assert not np.array_equal(a, ds.batch(6, 4, 6, seed=1))      # different step -> different batch
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_roundtrip_and_keep(tmp_path):
    state = {"a": jnp.arange(4.0), "b": {"c": jnp.ones((2, 3), jnp.int32)}}
    for s in (10, 20, 30, 40):
        checkpoint.save(tmp_path, s, jax.tree.map(lambda x: x * s, state), {"note": s}, keep=2)
    assert checkpoint.list_steps(tmp_path) == [30, 40]
    restored, meta = checkpoint.restore(tmp_path, 40, state)
    assert meta == {"step": 40, "note": 40}
    assert restored["b"]["c"].dtype == np.int32 and int(restored["b"]["c"][0, 0]) == 40
    with pytest.raises(ValueError):
        checkpoint.restore(tmp_path, 40, {"a": jnp.arange(4.0)})


# ------------------------------------------------------------------ training
def test_training_learns(tmp_path):
    res = train(_cfg(tmp_path), log=lambda s: None)
    assert res["step"] == 40
    assert res["val_loss"] < 1.5  # from ln(32) = 3.47 at the start
    lines = [json.loads(x) for x in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert lines[0]["step"] == 10 and "tokens_per_s" in lines[0]
    assert checkpoint.list_steps(tmp_path / "run" / "checkpoints") == [20, 40]


def test_resume_is_exact(tmp_path):
    """Stopping at step 15 and resuming gives the same weights as one uninterrupted run."""
    straight = train(_cfg(tmp_path, out_dir=str(tmp_path / "a"), steps=30), log=lambda s: None)
    train(_cfg(tmp_path, out_dir=str(tmp_path / "b"), steps=30), log=lambda s: None, stop_after=15)
    assert checkpoint.list_steps(tmp_path / "b" / "checkpoints") == [15]
    logs = []
    resumed = train(_cfg(tmp_path, out_dir=str(tmp_path / "b"), steps=30), log=logs.append)
    assert "resumed from step 15" in logs
    for x, y in zip(jax.tree.leaves(straight["state"]["params"]), jax.tree.leaves(resumed["state"]["params"])):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-6)


def test_grad_accum_matches_full_batch(tmp_path):
    a = train(_cfg(tmp_path, out_dir=str(tmp_path / "a"), steps=3, grad_accum=1), log=lambda s: None)
    b = train(_cfg(tmp_path, out_dir=str(tmp_path / "b"), steps=3, grad_accum=4), log=lambda s: None)
    for x, y in zip(jax.tree.leaves(a["state"]["params"]), jax.tree.leaves(b["state"]["params"])):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-5)


def test_data_parallel_on_4_fake_devices(tmp_path):
    """Same run on 1 and on 4 (simulated) devices ends with the same weights."""
    _cfg(tmp_path)  # writes data
    script = f"""
import sys, numpy as np, jax
sys.path.insert(0, {str(ROOT / 'src')!r}); sys.path.insert(0, {str(ROOT / 'tests')!r})
from pathlib import Path
from test_train import _cfg
from aio.train import train
assert len(jax.devices()) == 4
r = train(_cfg(Path({str(tmp_path)!r}), out_dir={str(tmp_path / 'dp')!r}, steps=5), log=lambda s: None)
np.save({str(tmp_path / 'dp.npy')!r}, np.asarray(r['state']['params']['embed']))
"""
    env = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=4", "JAX_PLATFORMS": "cpu"}
    subprocess.run([sys.executable, "-c", script], check=True, env=env, timeout=600)
    one = train(_cfg(tmp_path, out_dir=str(tmp_path / "single"), steps=5), log=lambda s: None)
    np.testing.assert_allclose(np.load(tmp_path / "dp.npy"), np.asarray(one["state"]["params"]["embed"]), atol=1e-5)


# ------------------------------------------------------------------ generation
def test_greedy_generation_matches_full_forward():
    params = init_params(TINY, jax.random.PRNGKey(0))
    prompt = [1, 2, 3, 4]
    new = generate(params, TINY, prompt, 6, jax.random.PRNGKey(1), temperature=0)
    seq = list(prompt)
    for _ in range(6):
        logits, _, _ = forward(params, jnp.array([seq]), TINY)
        seq.append(int(jnp.argmax(logits[0, -1])))
    assert new == seq[len(prompt):]


def test_trained_model_continues_counting(tmp_path):
    res = train(_cfg(tmp_path, steps=60, optim=OptimConfig(lr=1e-2, warmup_steps=5, total_steps=60)),
                log=lambda s: None)
    out = generate(res["state"]["params"], TINY, [5, 6, 7, 8, 9], 5, jax.random.PRNGKey(0), temperature=0)
    assert out == [10, 11, 12, 13, 14]
