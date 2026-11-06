import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.eval import MCItem, score_items  # noqa: E402
from aio.eval.tasks import load  # noqa: E402
from aio.model import ModelConfig, forward, init_params  # noqa: E402
from aio.train import OptimConfig, TrainConfig, train  # noqa: E402
from aio.data.tokens import TokenShardWriter  # noqa: E402

V = 40
CFG = ModelConfig(vocab_size=V, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=16)
encode = lambda s: [ord(c) - ord("a") + 1 for c in s if c.isalpha()]  # noqa: E731  a=1 .. z=26


def test_scores_match_manual_logprob():
    params = init_params(CFG, jax.random.PRNGKey(0))
    items = [MCItem("abc", ["d", "xy"], 0)]
    r = score_items(params, CFG, encode, items, batch_size=4)
    logits, _, _ = forward(params, jnp.array([[1, 2, 3, 24]]), CFG)
    lp = jax.nn.log_softmax(logits[0])
    manual_xy = float(lp[2, 24] + lp[3, 25])
    manual_d = float(jax.nn.log_softmax(forward(params, jnp.array([[1, 2, 3]]), CFG)[0][0])[2, 4])
    assert r["acc"] == float(manual_d > manual_xy)
    assert r["n"] == 1 and r["chance"] == 0.5


def test_trained_model_answers(tmp_path):
    """Train on 'abcd' / 'mnop' style sequences; it must pick the right continuation."""
    with TokenShardWriter(tmp_path / "d", vocab_size=V, eot_id=39, tokens_per_shard=10_000) as w:
        rng = np.random.default_rng(0)
        for _ in range(300):
            s = int(rng.integers(1, 20))
            w.add(list(range(s, s + 6)))
    cfg = TrainConfig(data_dir=str(tmp_path / "d"), out_dir=str(tmp_path / "r"), seq_len=16, batch_size=8,
                      steps=150, log_every=50, ckpt_every=150, model=CFG,
                      optim=OptimConfig(lr=1e-2, warmup_steps=10, total_steps=150))
    params = train(cfg, log=lambda s: None)["state"]["params"]
    letters = "abcdefghijklmnopqrstuvwxyz"
    items = []
    for s in range(0, 12):
        ctx = letters[s : s + 3]
        right = letters[s + 3]
        wrong = [letters[(s + 3 + k) % 20] for k in (5, 9, 13)]
        items.append(MCItem(ctx, [wrong[0], right, wrong[1], wrong[2]], 1))
    r = score_items(params, CFG, encode, items, prefix=[39])
    assert r["acc"] >= 0.9 and r["chance"] == 0.25


def test_task_loaders(tmp_path):
    pq.write_table(pa.Table.from_pylist([{"activity_label": "Cooking", "ctx_a": "A man cracks eggs.",
                                          "ctx_b": "he", "endings": ["whisks them.", "flies.", "sings.", "naps."],
                                          "label": "0"}]), tmp_path / "h.parquet")
    [h] = load("hellaswag", str(tmp_path / "h.parquet"))
    assert h.context.startswith("Cooking: A man cracks eggs. He") and h.answer == 0 and len(h.choices) == 4
    pq.write_table(pa.Table.from_pylist([{"question": "2+2?", "choices": {"text": ["3", "4"], "label": ["A", "B"]},
                                          "answerKey": "B"}]), tmp_path / "a.parquet")
    [a] = load("arc_easy", str(tmp_path / "a.parquet"))
    assert a.answer == 1 and a.choices == [" 3", " 4"]
    pq.write_table(pa.Table.from_pylist([{"question": "Q", "choices": ["w", "x", "y", "z"], "answer": 2,
                                          "subject": "s"}]), tmp_path / "m.parquet")
    [m] = load("mmlu", str(tmp_path / "m.parquet"))
    assert m.answer == 2
