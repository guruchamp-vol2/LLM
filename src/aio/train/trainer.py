"""The pretraining loop.

- Resumes automatically from the newest checkpoint in out_dir/checkpoints. Batches are a
  function of (seed, step), so a resumed run continues exactly as if never interrupted.
- On SIGTERM or Ctrl+C (e.g. a preemptible TPU being reclaimed) it saves a checkpoint
  before exiting.
- Runs on all devices. sharding="dp" copies the weights to every device and splits the
  batch; sharding="fsdp" also splits the weights and optimizer state (see sharding.py).
- Multi-host (TPU pods): set distributed=True and start the same command on every host.
  Only process 0 logs and writes checkpoints, to its local disk; on resume it sends the
  checkpoint to the other hosts, so no shared (paid) storage is needed. Every host needs
  its own copy of the token shards.
- Logs to stdout and out_dir/metrics.jsonl; evaluates on held-out tokens every eval_every steps.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from ..data.tokens import TokenDataset
from ..model.transformer import ModelConfig, init_params, loss_fn
from . import checkpoint
from .sharding import (any_process, batch_sharding as make_batch_sharding, broadcast_from_main, make_mesh,
                       put_batch, put_global, state_shardings, to_host)
from .optim import OptimConfig, adamw_update, init_opt_state


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    val_dir: str | None = None     # held-out token shards; None = no evaluation
    seq_len: int = 512
    batch_size: int = 16           # sequences per optimizer step (all devices together)
    grad_accum: int = 1            # split each batch into this many sequential micro-batches
    sharding: str = "dp"           # "dp" (copy weights to each device) or "fsdp" (split them)
    distributed: bool = False      # True on multi-host TPU pods: call jax.distributed.initialize()
    steps: int = 1000
    seed: int = 0
    log_every: int = 10
    eval_every: int = 200
    eval_batches: int = 20
    ckpt_every: int = 200
    keep_ckpts: int = 3
    objective: str = "lm"          # "lm" (pretraining and SFT) or "dpo"
    dpo_beta: float = 0.1
    init_from: str | None = None   # checkpoints folder of an earlier run: start from its newest weights
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = {k: v for k, v in d.items() if not k.startswith("_")}  # "_comment" etc.
        model = ModelConfig(**d.pop("model", {}))
        optim = OptimConfig(**d.pop("optim", {}))
        return cls(model=model, optim=optim, **d)


def load_dataset(path: str):
    """Token shards (a folder with index.json), a weighted mixture of them
    ("DIR=WEIGHT,DIR=WEIGHT"), or a post-training .npz (SFT or DPO pairs)."""
    if "=" in path:
        from ..data.tokens import MixtureDataset

        return MixtureDataset(path)
    p = Path(path)
    if p.suffix == ".npz":
        from ..post.chat import ConversationDataset, PairDataset

        kind = str(np.load(p)["kind"])
        return ConversationDataset(p) if kind == "sft" else PairDataset(p)
    return TokenDataset(p)


def _objective(cfg: TrainConfig):
    from ..post.objectives import OBJECTIVES

    if cfg.objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {cfg.objective!r}; choose from {sorted(OBJECTIVES)}")
    fn = OBJECTIVES[cfg.objective]
    return lambda params, batch, ref: fn(params, batch, cfg.model, ref, beta=cfg.dpo_beta)


def make_train_step(cfg: TrainConfig):
    ocfg, A = cfg.optim, cfg.grad_accum
    objective = _objective(cfg)
    grad_fn = jax.value_and_grad(objective, has_aux=True)

    def step(params, opt_state, batch, ref):
        if A == 1:
            (loss, metrics), grads = grad_fn(params, batch, ref)
        else:
            # Average gradients over A micro-batches without holding all their activations.
            mb = jax.tree.map(lambda a: a.reshape(A, a.shape[0] // A, *a.shape[1:]), batch)

            def body(carry, micro):
                g_sum, l_sum, m_sum = carry
                (l, m), g = grad_fn(params, micro, ref)
                return (jax.tree.map(jnp.add, g_sum, g), l_sum + l, jax.tree.map(jnp.add, m_sum, m)), None

            zeros = jax.tree.map(jnp.zeros_like, params)
            _, m_shape = jax.eval_shape(objective, params, jax.tree.map(lambda a: a[0], mb), ref)
            m_zero = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), m_shape)
            (g_sum, l_sum, m_sum), _ = jax.lax.scan(body, (zeros, 0.0, m_zero), mb)
            grads = jax.tree.map(lambda g: g / A, g_sum)
            loss, metrics = l_sum / A, jax.tree.map(lambda m: m / A, m_sum)
        params, opt_state, stats = adamw_update(params, grads, opt_state, ocfg)
        return params, opt_state, {"loss": loss, **metrics, **stats}

    return step


def make_eval_step(cfg: TrainConfig):
    objective = _objective(cfg)

    def step(params, batch, ref):
        return objective(params, batch, ref)[0]

    return step


_distributed_ready = False


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
    global _distributed_ready
    if cfg.distributed and not _distributed_ready:
        jax.distributed.initialize()  # TPU VMs find their peers automatically
        _distributed_ready = True
    main = jax.process_index() == 0
    if not main:
        log = lambda *a, **k: None  # noqa: E731
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    if main:
        (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    data = load_dataset(cfg.data_dir)
    if data.vocab_size > cfg.model.vocab_size:
        raise ValueError(f"data vocab {data.vocab_size} > model vocab {cfg.model.vocab_size}")
    val = load_dataset(cfg.val_dir) if cfg.val_dir else None

    devices = jax.devices()
    n_dev = len(devices)
    if cfg.batch_size % (n_dev * cfg.grad_accum):
        raise ValueError(f"batch_size {cfg.batch_size} must divide by devices x grad_accum = {n_dev * cfg.grad_accum}")
    mesh = make_mesh(devices)
    replicated = NamedSharding(mesh, P())
    batch_sharding = make_batch_sharding(mesh)

    params = init_params(cfg.model, jax.random.PRNGKey(cfg.seed))
    ref = None
    if cfg.init_from:
        # Start from (and, for DPO, compare against) an earlier run's newest weights.
        src = checkpoint.latest_step(cfg.init_from) if main else None
        if main and src is None:
            raise FileNotFoundError(f"no checkpoints in {cfg.init_from}")
        if main:
            loaded, _ = checkpoint.restore(cfg.init_from, src, {"params": params, "opt": init_opt_state(params)})
            params = loaded["params"]
        params = broadcast_from_main(jax.device_get(params))
        log(f"initialised from {cfg.init_from} (step {src})")
        if cfg.objective == "dpo":
            ref = params
    if cfg.objective == "dpo" and ref is None:
        raise ValueError("objective 'dpo' needs init_from (the SFT run to compare against)")
    state = {"params": params, "opt": init_opt_state(params)}
    last = checkpoint.latest_step(ckpt_dir) if main else None
    last = int(broadcast_from_main(np.array(-1 if last is None else last)))
    start = 0
    if last >= 0:
        if main:
            state, meta = checkpoint.restore(ckpt_dir, last, state)
        state = broadcast_from_main(jax.device_get(state))
        start = last
        log(f"resumed from step {start}")
    shard = state_shardings(state, mesh, cfg.sharding)
    state = put_global(state, shard)

    if ref is not None:
        ref = put_global(ref, shard["params"])
    ref_shard = shard["params"] if ref is not None else None
    train_step = jax.jit(make_train_step(cfg), donate_argnums=(0, 1),
                         in_shardings=(shard["params"], shard["opt"], batch_sharding, ref_shard),
                         out_shardings=(shard["params"], shard["opt"], replicated))
    eval_step = jax.jit(make_eval_step(cfg), in_shardings=(shard["params"], batch_sharding, ref_shard))

    def evaluate(params) -> float:
        # Fixed held-out batches (seed is independent of the training seed).
        losses = [eval_step(params, put_batch(val.batch(i, cfg.batch_size, cfg.seq_len, seed=10_000), batch_sharding), ref)
                  for i in range(cfg.eval_batches)]
        return float(np.mean([float(x) for x in losses]))

    def save(step: int) -> None:
        host_state = to_host(state)  # every process takes part in gathering sharded arrays
        if main:
            checkpoint.save(ckpt_dir, step, host_state, {"config": cfg.to_dict()}, keep=cfg.keep_ckpts)

    tokens_per_step = cfg.batch_size * cfg.seq_len
    log(f"devices: {n_dev} x {devices[0].platform} on {jax.process_count()} host(s); params: {sum(x.size for x in jax.tree.leaves(params)) / 1e6:.1f}M; "
        f"tokens/step: {tokens_per_step:,}; data: {data.n_tokens:,} tokens")

    metrics_f = open(out / "metrics.jsonl", "a", encoding="utf-8") if main else open(os.devnull, "w")
    last_metrics: dict = {}
    step = start
    try:
        with _StopFlag() as flag:
            t0, t_tokens = time.time(), 0
            while step < cfg.steps:
                batch = put_batch(data.batch(step, cfg.batch_size, cfg.seq_len, cfg.seed), batch_sharding)
                state["params"], state["opt"], m = train_step(state["params"], state["opt"], batch, ref)
                step += 1
                t_tokens += tokens_per_step

                if step % cfg.log_every == 0 or step == cfg.steps:
                    m = {k: float(v) for k, v in m.items()}
                    dt = time.time() - t0
                    rec = {"step": step, **{k: round(v, 5) for k, v in m.items()},
                           "tokens_per_s": round(t_tokens / max(dt, 1e-9)), "tokens_seen": step * tokens_per_step}
                    if not math.isfinite(m["loss"]):
                        raise FloatingPointError(f"loss became {m['loss']} at step {step}")
                    extra = "".join(f" | {k} {m[k]:.3f}" for k in ("reward_acc", "margin") if k in m)
                    log(f"step {step:>6} | loss {m['ce']:.4f} | lr {m['lr']:.2e} | grad norm {m['grad_norm']:.2f}"
                        f"{extra} | {rec['tokens_per_s']:,} tok/s")
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
                if any_process(flag.stop) or (stop_after is not None and step - start >= stop_after):
                    if step % cfg.ckpt_every:
                        save(step)
                    break
    finally:
        metrics_f.close()
    return {"step": step, **last_metrics, "state": state}
