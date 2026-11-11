"""Kaggle TPU benchmark v2: which settings make training fastest, and how big a model fits.

    kaggle kernels push -p kaggle/bench --accelerator TpuV5E8

Runs configs/kaggle-pretrain.json (seq 2048, vocab 32k, real token data) for a few steps
under each setting in CASES and records tokens/s, model FLOPs utilisation (MFU) and peak
memory per chip in /kaggle/working/results.json. Cases run in order of importance and the
whole run stops at a deadline, so a hang can't burn free TPU hours.
"""

import glob
import json
import os
import re
import subprocess
import sys
import time

WORK = "/kaggle/working"
REPO = f"{WORK}/llm"
DEADLINE = time.time() + 55 * 60
PEAK_FLOPS = 8 * 197e12            # TPU v5e-8: 197 TFLOP/s bf16 per chip
SEQ, VOCAB = 2048, 32768
results = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": []}

FAST_XLA = ("--xla_tpu_enable_latency_hiding_scheduler=true --xla_tpu_enable_async_collective_fusion=true "
            "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
            "--xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true "
            "--xla_enable_async_all_gather=true")

# name, model overrides, run overrides, extra env
M300_64 = {"d_model": 1024, "n_layers": 24, "n_heads": 16, "n_kv_heads": 4}      # head_dim 64 (old config)
M300 = {"d_model": 1024, "n_layers": 24, "n_heads": 8, "n_kv_heads": 2}          # head_dim 128
SPLASH = {"attention": "splash", "loss_chunk": 512}
CASES = [
    ("300M old (xla attn, remat)", {**M300_64, "attention": "xla", "loss_chunk": 0, "remat": True}, {}, {}),
    ("300M splash remat", {**M300, **SPLASH, "remat": True}, {}, {}),
    ("300M splash no-remat", {**M300, **SPLASH, "remat": False}, {}, {}),
    ("300M splash remat batch128", {**M300, **SPLASH, "remat": True}, {"batch_size": 128}, {}),
    ("300M splash remat +xla flags", {**M300, **SPLASH, "remat": True}, {}, {"LIBTPU_INIT_ARGS": FAST_XLA}),
    ("700M d1536 splash remat", {"d_model": 1536, "n_layers": 24, "n_heads": 12, "n_kv_heads": 4, **SPLASH, "remat": True}, {}, {}),
    ("1.2B d2048 splash remat", {"d_model": 2048, "n_layers": 24, "n_heads": 16, "n_kv_heads": 4, **SPLASH, "remat": True}, {}, {}),
    ("300M splash remat dp", {**M300, **SPLASH, "remat": True}, {"sharding": "dp"}, {}),
    ("300M xla attn chunked-loss remat", {**M300, "attention": "xla", "loss_chunk": 512, "remat": True}, {}, {}),
]


def save() -> None:
    with open(f"{WORK}/results.json", "w") as f:
        json.dump(results, f, indent=2)


def sh(cmd: str, timeout: float = 10 * 60, env=None, log=None) -> int:
    left = DEADLINE - time.time()
    if left < 60:
        print(f"\n$ {cmd}\n(skipped: past the deadline)", flush=True)
        return 124
    print(f"\n$ {cmd}", flush=True)
    out = open(log, "w") if log else None
    try:
        return subprocess.run(cmd, shell=True, timeout=min(timeout, left), env=env,
                              stdout=out, stderr=subprocess.STDOUT if out else None).returncode
    except subprocess.TimeoutExpired:
        print("(timed out)", flush=True)
        return 124
    finally:
        if out:
            out.close()


def matmul_flops_per_token(m: dict) -> float:
    """Training FLOPs per token: 6 x (matmul weights incl. output layer) + attention scores."""
    d, L, H, KV = m["d_model"], m["n_layers"], m["n_heads"], m["n_kv_heads"]
    dh = d // H
    ff = 64 * -(-8 * d // 3 // 64)
    per_layer = d * H * dh * 2 + d * KV * dh * 2 + 3 * d * ff
    n = L * per_layer + VOCAB * d
    return 6 * n + 12 * L * d * SEQ / 2   # causal: half the score matrix is used


# ---------------------------------------------------------------- setup
sh(f"{sys.executable} -m pip install -q regex pyarrow")
jx = subprocess.run([sys.executable, "-c", "import jax; print(jax.__version__, jax.devices()[0].device_kind, len(jax.devices()))"],
                    capture_output=True, text=True, timeout=300)
results["jax"] = (jx.stdout or jx.stderr[-1000:]).strip()
print(results["jax"], flush=True)
save()
if "TPU" not in results["jax"]:
    sys.exit(f"no TPU visible: {results['jax']}")

sh(f"git clone -q --depth 1 https://github.com/guruchamp-vol2/LLM.git {REPO}")
os.chdir(REPO)
results["commit"] = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True, text=True).stdout.strip()
idx = [p for p in glob.glob("/kaggle/input/**/train/index.json", recursive=True) if "llm-tok32k" in p]
if not idx:
    sys.exit("no llm-tok32k dataset attached")
train_dir = os.path.dirname(idx[0])
results["data"] = train_dir
save()

# ---------------------------------------------------------------- cases
for name, model, run, env_extra in CASES:
    od = f"{WORK}/case"
    sh(f"rm -rf {od}")
    sets = {"data_dir": train_dir, "val_dir": "null", "out_dir": od, "steps": 30, "log_every": 10,
            "ckpt_every": 1000000, "keep_ckpts": 1, "batch_size": 64, "grad_accum": 1,
            "optim.total_steps": 30, "optim.warmup_steps": 5, **run}
    sets |= {f"model.{k}": v for k, v in model.items()}
    args = " ".join(f"--set {k}={json.dumps(v) if isinstance(v, bool) else v}" for k, v in sets.items())
    log = f"{WORK}/case.log"
    t = time.time()
    rc = sh(f"{sys.executable} scripts/train.py --config configs/kaggle-pretrain.json {args}",
            env={**os.environ, **env_extra}, log=log)
    text = open(log).read() if os.path.exists(log) else ""
    m = [json.loads(line) for line in open(f"{od}/metrics.jsonl")] if os.path.exists(f"{od}/metrics.jsonl") else []
    tps = [r["tokens_per_s"] for r in m if "tokens_per_s" in r]
    full = {**json.load(open("configs/kaggle-pretrain.json"))["model"], **model}
    params = re.search(r"params: ([\d.]+)M", text)
    rec = {"name": name, "exit_code": rc, "seconds": round(time.time() - t),
           "params_m": float(params.group(1)) if params else None,
           "tokens_per_s": tps[-1] if tps else None, "tokens_per_s_all": tps,
           "peak_mem_gb": max((r.get("peak_mem_gb", 0) for r in m), default=None),
           "loss": [r["ce"] for r in m if "ce" in r]}
    if rec["tokens_per_s"]:
        rec["mfu"] = round(rec["tokens_per_s"] * matmul_flops_per_token(full) / PEAK_FLOPS, 3)
    if rc:
        err = [l for l in text.splitlines() if "Error" in l or "error" in l or "RESOURCE" in l]
        rec["error"] = (err[-1] if err else text[-600:])[:600]
    print(json.dumps(rec), flush=True)
    results["cases"].append(rec)
    save()

sh(f"rm -rf {WORK}/case {REPO}")
results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
save()
print(json.dumps(results, indent=2))
