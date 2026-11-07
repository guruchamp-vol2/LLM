"""Kaggle TPU smoke test + throughput benchmark.

Pushed as a Kaggle script kernel (see kernel-metadata.json next to this file):
    kaggle kernels push -p kaggle/smoke

It clones the repo from GitHub, then:
1. trains configs/tpu-smoke.json for 200 steps, then resumes it to 400 (checks the TPU
   path end to end: devices, FSDP, bf16, checkpoints, resume),
2. times a few model sizes for 40 steps each, to learn tokens/second on this TPU,
3. writes everything to /kaggle/working/results.json (downloadable with
   `kaggle kernels output`).
"""

import glob
import json
import os
import subprocess
import sys
import time

WORK = "/kaggle/working"
REPO = f"{WORK}/llm"
results = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}


def sh(cmd: str, check: bool = True) -> int:
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check).returncode


def save() -> None:
    with open(f"{WORK}/results.json", "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------- environment
try:
    import jax  # noqa: F401
except ImportError:
    sh(f"{sys.executable} -m pip install -q -U 'jax[tpu]'")
sh(f"{sys.executable} -m pip install -q regex pyarrow")
out = subprocess.run([sys.executable, "-c", "import jax; print(jax.__version__); print(jax.devices())"],
                     capture_output=True, text=True)
print(out.stdout, out.stderr[-2000:], flush=True)
results["jax"] = out.stdout.strip()
save()

if not os.path.exists(REPO):
    sh(f"git clone -q --depth 1 https://github.com/guruchamp-vol2/LLM.git {REPO}")
os.chdir(REPO)
results["commit"] = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True, text=True).stdout.strip()

train_idx = [p for p in glob.glob("/kaggle/input/**/index.json", recursive=True) if "/train/" in p]
val_idx = [p for p in glob.glob("/kaggle/input/**/index.json", recursive=True) if "/val/" in p]
if not train_idx:
    sys.exit("token dataset not found under /kaggle/input")
train_dir, val_dir = os.path.dirname(train_idx[0]), os.path.dirname(val_idx[0]) if val_idx else None
results["data"] = {"train": train_dir, "val": val_dir}
print(results["data"], flush=True)


def metrics(out_dir: str) -> list[dict]:
    path = f"{out_dir}/metrics.jsonl"
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


# ---------------------------------------------------------------- 1. smoke test + resume
smoke = f"{WORK}/smoke"
common = (f"--set data_dir={train_dir} --set val_dir={val_dir or 'null'} --set out_dir={smoke} "
          f"--set distributed=false")
t = time.time()
rc1 = sh(f"{sys.executable} scripts/train.py --config configs/tpu-smoke.json {common}", check=False)
rc2 = sh(f"{sys.executable} scripts/train.py --config configs/tpu-smoke.json {common} --set steps=400", check=False)
m = metrics(smoke)
results["smoke"] = {
    "exit_codes": [rc1, rc2], "seconds": round(time.time() - t),
    "train_loss": [(r["step"], r["ce"]) for r in m if "ce" in r][::4],
    "val_loss": [(r["step"], r["val_loss"]) for r in m if "val_loss" in r],
    "tokens_per_s": [r["tokens_per_s"] for r in m if "tokens_per_s" in r][-5:],
    "checkpoints": sorted(os.listdir(f"{smoke}/checkpoints")) if os.path.exists(f"{smoke}/checkpoints") else [],
}
save()

# ---------------------------------------------------------------- 2. throughput by model size
sizes = [  # (d_model, layers, heads, kv_heads)
    (512, 8, 8, 4),
    (768, 12, 12, 4),
    (1024, 16, 16, 4),
]
results["throughput"] = []
for d, L, H, KV in sizes:
    od = f"{WORK}/bench_d{d}"
    cmd = (f"{sys.executable} scripts/train.py --config configs/tpu-smoke.json {common} --set out_dir={od} "
           f"--set val_dir=null --set steps=40 --set log_every=10 --set ckpt_every=100000 --set keep_ckpts=1 "
           f"--set seq_len=1024 --set batch_size=64 --set model.max_seq_len=1024 "
           f"--set model.d_model={d} --set model.n_layers={L} --set model.n_heads={H} --set model.n_kv_heads={KV} "
           f"--set optim.total_steps=40 --set optim.warmup_steps=5")
    t = time.time()
    rc = sh(cmd, check=False)
    tps = [r["tokens_per_s"] for r in metrics(od) if "tokens_per_s" in r]
    results["throughput"].append({"d_model": d, "layers": L, "exit_code": rc, "seconds": round(time.time() - t),
                                  "tokens_per_s_last": tps[-1] if tps else None, "tokens_per_s_all": tps})
    save()
    sh(f"rm -rf {od}/checkpoints", check=False)  # keep the output small

results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
save()
print(json.dumps(results, indent=2))
