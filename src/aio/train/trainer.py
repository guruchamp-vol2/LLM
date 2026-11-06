"""The pretraining loop.

- Resumes automatically from the newest checkpoint in out_dir/checkpoints. Batches are a
  function of (seed, step), so a resumed run continues exactly as if never interrupted.
- On SIGTERM or Ctrl+C (e.g. a preemptible TPU being reclaimed) it saves a checkpoint
  before exiting.
- Data parallel over all local devices: each batch is split across devices, the parameters
  are copied to each, and XLA averages the gradients. (FSDP sharding comes next.)
- Logs to stdout and out_dir/metrics.jsonl; evaluates on held-out tokens every eval_every steps.
"""

from __future__ import annotations

import dataclasses
import json
import math
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..data.tokens import TokenDataset
from ..model.transformer import ModelConfig, init_params, loss_fn
from . import checkpoint
from .optim import OptimConfig, adamw_update, init_opt_state


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    val_dir: str | None = None     # held-out token shards; None = no evaluation
    seq_len: int = 512
    batch_size: int = 16           # sequences per optimizer step (all devices together)
    grad_accum: int = 1            # split each batch into this many sequential micro-batches
    steps: int = 1000
    seed: int = 0
    log_every: int = 10
    eval_every: int = 200
    eval_batches: int = 20
    ckpt_every: int = 200
    keep_ckpts: int = 3
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = dict(d)
        model = ModelConfig(**d.pop("model", {}))
        optim = OptimConfig(**d.pop("optim", {}))
        return cls(model=model, optim=optim, **d)


def make_train_step(cfg: TrainConfig):
    mcfg, ocfg, A = cfg.model, cfg.optim, cfg.grad_accum

    def step(params, opt_state, batch):
        x, y = batch[:, :-1], batch[:, 1:]
        mask = jnp.ones(x.shape, jnp.float32)
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        if A == 1:
            (loss, aux), grads = grad_fn(params, x, y, mask, mcfg)
        else:
            # Average gradients over A micro-batches without holding all their activations.
            mb = lambda a: a.reshape(A, a.shape[0] // A, *a.shape[1:])  # noqa: E731

            def body(carry, xs):
                g_sum, l_sum, c_sum = carry
                (l, aux), g = grad_fn(params, *xs, mcfg)
                return (jax.tree.map(jnp.add, g_sum, g), l_sum + l, c_sum + aux["ce"]), None

            zeros = jax.tree.map(jnp.zeros_like, params)
            (g_sum, l_sum, c_sum), _ = jax.lax.scan(body, (zeros, 0.0, 0.0), (mb(x), mb(y), mb(mask)))
            grads = jax.tree.map(lambda g: g / A, g_sum)
            loss, aux = l_sum / A, {"ce": c_sum / A}
        params, opt_state, stats = adamw_update(params, grads, opt_state, ocfg)
        return params, opt_state, {"loss": loss, "ce": aux["ce"], **stats}

    return step


def make_eval_step(cfg: TrainConfig):
    def step(params, batch):
        x, y = batch[:, :-1], batch[:, 1:]
        _, aux = loss_fn(params, x, y, jnp.ones(x.shape, jnp.float32), cfg.model)
        return aux["ce"]

    return step


class _StopFlag:
    """Set by SIGTERM / Ctrl+C so the loop can save and exit cleanly."""

    def __init__(self) -> None:
        self.stop = False
        self._old = {}

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._old[sig] = signal.signal(sig, self._handle)
            except (ValueError, OSError):  # not the main thread / unsupported
                pass
        return self

    def _handle(self, signum, frame) -> None:
        if self.stop:  # second signal: give up immediately
            raise KeyboardInterrupt
        print(f"\nsignal {signum}: saving a checkpoint, then stopping (send again to abort)")
        self.stop = True

    def __exit__(self, *exc) -> None:
        for sig, old in self._old.items():
            signal.signal(sig, old)


def train(cfg: TrainConfig, log=print, stop_after: int | None = None) -> dict:
    """Run (or resume) training. stop_after: return after this many steps of this call (tests)."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    data = TokenDataset(cfg.data_dir)
    if data.vocab_size > cfg.model.vocab_size:
        raise ValueError(f"data vocab {data.vocab_size} > model vocab {cfg.model.vocab_size}")
    val = TokenDataset(cfg.val_dir) if cfg.val_dir else None

    devices = jax.devices()
    n_dev = len(devices)
    if cfg.batch_size % (n_dev * cfg.grad_accum):
        raise ValueError(f"batch_size {cfg.batch_size} must divide by devices x grad_accum = {n_dev * cfg.grad_accum}")
    mesh = Mesh(np.array(devices), ("data",))
    replicated = NamedSharding(mesh, P())
    # With grad_accum, keep micro-batches whole per device: split rows across devices.
    batch_sharding = NamedSharding(mesh, P("data"))

    params = init_params(cfg.model, jax.random.PRNGKey(cfg.seed))
    state = {"params": params, "opt": init_opt_state(params)}
    start = 0
    if (last := checkpoint.latest_step(ckpt_dir)) is not None:
        state, meta = checkpoint.restore(ckpt_dir, last, state)
        start = meta["step"]
        log(f"resumed from step {start}")
    state = jax.device_put(state, replicated)

    train_step = jax.jit(make_train_step(cfg), donate_argnums=(0, 1),
                         in_shardings=(replicated, replicated, batch_sharding),
                         out_shardings=(replicated, replicated, replicated))
    eval_step = jax.jit(make_eval_step(cfg), in_shardings=(replicated, batch_sharding))

    def evaluate(params) -> float:
        # Fixed held-out batches (seed is independent of the training seed).
        losses = [eval_step(params, jax.device_put(val.batch(i, cfg.batch_size, cfg.seq_len, seed=10_000), batch_sharding))
                  for i in range(cfg.eval_batches)]
        return float(np.mean([float(x) for x in losses]))

    def save(step: int) -> None:
        checkpoint.save(ckpt_dir, step, state, {"config": cfg.to_dict()}, keep=cfg.keep_ckpts)

    tokens_per_step = cfg.batch_size * cfg.seq_len
    log(f"devices: {n_dev} x {devices[0].platform}; params: {sum(x.size for x in jax.tree.leaves(params)) / 1e6:.1f}M; "
        f"tokens/step: {tokens_per_step:,}; data: {data.n_tokens:,} tokens")

    metrics_f = open(out / "metrics.jsonl", "a", encoding="utf-8")
    last_metrics: dict = {}
    step = start
    try:
        with _StopFlag() as flag:
            t0, t_tokens = time.time(), 0
            while step < cfg.steps:
                batch = jax.device_put(data.batch(step, cfg.batch_size, cfg.seq_len, cfg.seed), batch_sharding)
                state["params"], state["opt"], m = train_step(state["params"], state["opt"], batch)
                step += 1
                t_tokens += tokens_per_step

                if step % cfg.log_every == 0 or step == cfg.steps:
                    m = {k: float(v) for k, v in m.items()}
                    dt = time.time() - t0
                    rec = {"step": step, **{k: round(v, 5) for k, v in m.items()},
                           "tokens_per_s": round(t_tokens / max(dt, 1e-9)), "tokens_seen": step * tokens_per_step}
                    if not math.isfinite(m["loss"]):
                        raise FloatingPointError(f"loss became {m['loss']} at step {step}")
                    log(f"step {step:>6} | loss {m['ce']:.4f} | lr {m['lr']:.2e} | grad norm {m['grad_norm']:.2f} "
                        f"| {rec['tokens_per_s']:,} tok/s")
                    metrics_f.write(json.dumps(rec) + "\n")
                    metrics_f.flush()
                    last_metrics = rec
                    t0, t_tokens = time.time(), 0

                if val is not None and (step % cfg.eval_every == 0 or step == cfg.steps):
                    vl = evaluate(state["params"])
                    rec = {"step": step, "val_loss": round(vl, 5), "val_ppl": round(math.exp(min(vl, 50)), 2)}
                    log(f"step {step:>6} | validation loss {vl:.4f} | perplexity {rec['val_ppl']:.2f}")
                    metrics_f.write(json.dumps(rec) + "\n")
                    metrics_f.flush()
                    last_metrics |= rec
                    t0, t_tokens = time.time(), 0

                if step % cfg.ckpt_every == 0 or step == cfg.steps:
                    save(step)
                if flag.stop or (stop_after is not None and step - start >= stop_after):
                    if step % cfg.ckpt_every:
                        save(step)
                    break
    finally:
        metrics_f.close()
    return {"step": step, **last_metrics, "state": state}
