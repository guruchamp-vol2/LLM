"""Main pretraining run on Kaggle's free TPU, one session at a time.

    kaggle kernels push -p kaggle/pretrain --accelerator TpuV5E8

Each session:
1. checks a TPU is visible (stops at once if not, so no hours are wasted),
2. clones the code from GitHub,
3. finds every token dataset attached as input (llm-tok32k-<source>-<n>) and builds a
   weighted mixture from MIX below,
4. copies the newest checkpoint from the previous session's output (this kernel's own
   output, attached as an input from the 2nd session on) and resumes,
5. trains until TIME_LIMIT, then saves and stops, so the session ends cleanly before
   Kaggle cuts it off. Re-push to continue; the run picks up where it stopped.

Outputs (/kaggle/working, kept by Kaggle after the session): run/checkpoints (newest only),
run/metrics.jsonl, status.json.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

CONFIG = "configs/kaggle-pretrain.json"
SELF = "llm-pretrain"            # this kernel's slug: its previous output holds the checkpoints
TIME_LIMIT = 8 * 3600            # Kaggle stops TPU sessions at 9 h; leave time to save and upload
# Share of training tokens per source. Each source's share is split across its datasets
# by size. Sources not attached are skipped (their share goes to the others).
MIX = {"edu": 0.40, "github": 0.25, "opc": 0.10, "math": 0.25}

WORK = "/kaggle/working"
REPO = f"{WORK}/llm"
RUN = f"{WORK}/run"
T0 = time.time()
status = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}


def save_status() -> None:
    with open(f"{WORK}/status.json", "w") as f:
        json.dump(status, f, indent=2)


def sh(cmd: str, check: bool = True, timeout: float | None = None) -> int:
    print(f"\n$ {cmd}", flush=True)
    try:
        return subprocess.run(cmd, shell=True, check=check, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print("(timed out)", flush=True)
        return 124


def devices() -> str:
    try:
        out = subprocess.run([sys.executable, "-c", "import jax; print(jax.__version__, jax.devices())"],
                             capture_output=True, text=True, timeout=300)
        return (out.stdout.strip() or out.stderr[-1500:]).strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


# ---------------------------------------------------------------- 1. TPU
sh(f"{sys.executable} -m pip install -q regex pyarrow", check=False)
status["jax"] = devices()
if "Tpu" not in status["jax"] and "TPU" not in status["jax"]:
    sh(f"{sys.executable} -m pip install -q -U 'jax[tpu]'", check=False)
    status["jax"] = devices()
print(status["jax"], flush=True)
save_status()
if "Tpu" not in status["jax"] and "TPU" not in status["jax"]:
    sys.exit(f"no TPU visible to JAX: {status['jax']}")

# ---------------------------------------------------------------- 2. code
if not os.path.exists(REPO):
    sh(f"git clone -q --depth 1 https://github.com/guruchamp-vol2/LLM.git {REPO}")
os.chdir(REPO)
status["commit"] = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True,
                                  text=True).stdout.strip()

# ---------------------------------------------------------------- 3. data mixture
parts = {}  # source -> [(train_dir, val_dir, tokens)]
for idx in glob.glob("/kaggle/input/**/train/index.json", recursive=True):
    m = re.search(r"llm-tok32k-([a-z]+)-\d+", idx)
    if not m:
        continue
    train_dir = os.path.dirname(idx)
    val_dir = os.path.join(os.path.dirname(train_dir), "val")
    tokens = json.load(open(idx))["n_tokens"]
    parts.setdefault(m.group(1), []).append((train_dir, val_dir if os.path.exists(val_dir) else None, tokens))
if not parts:
    sys.exit("no llm-tok32k-* datasets attached")
share = {s: MIX.get(s, 0.0) for s in parts if MIX.get(s, 0.0) > 0}
total = sum(share.values())
train_spec, val_spec, status["data"] = [], [], {}
for s, items in parts.items():
    if s not in share:
        continue
    n = sum(t for _, _, t in items)
    for tr, va, t in items:
        w = share[s] / total * t / n
        train_spec.append(f"{tr}={w:.6f}")
        if va:
            val_spec.append(f"{va}={w:.6f}")
    status["data"][s] = {"share": round(share[s] / total, 4), "tokens": n, "datasets": len(items)}
print(json.dumps(status["data"], indent=1), flush=True)
save_status()

# ---------------------------------------------------------------- 4. resume
os.makedirs(f"{RUN}/checkpoints", exist_ok=True)
prev = sorted(glob.glob(f"/kaggle/input/**/{SELF}/**/run/checkpoints/step_*", recursive=True)
              + glob.glob("/kaggle/input/**/run/checkpoints/step_*", recursive=True))
prev = sorted(set(prev), key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
if prev and not os.listdir(f"{RUN}/checkpoints"):
    src = prev[-1]
    print(f"resuming from {src}", flush=True)
    shutil.copytree(src, f"{RUN}/checkpoints/{os.path.basename(src)}")
    for f in ("metrics.jsonl",):
        old = os.path.join(os.path.dirname(os.path.dirname(src)), f)
        if os.path.exists(old):
            shutil.copy(old, f"{RUN}/{f}")
    status["resumed_from"] = os.path.basename(src)
save_status()

# ---------------------------------------------------------------- 5. train
left = TIME_LIMIT - (time.time() - T0)
cmd = (f"{sys.executable} scripts/train.py --config {CONFIG} --set out_dir={RUN} "
       f"--set 'data_dir={','.join(train_spec)}' "
       + (f"--set 'val_dir={','.join(val_spec)}' " if val_spec else "--set val_dir=null ")
       + f"--set time_limit_s={int(left)} --set keep_ckpts=1")
status["train_exit"] = sh(cmd, check=False)
metrics = [json.loads(line) for line in open(f"{RUN}/metrics.jsonl")] if os.path.exists(f"{RUN}/metrics.jsonl") else []
status["last_train"] = next((m for m in reversed(metrics) if "ce" in m), None)
status["last_val"] = next((m for m in reversed(metrics) if "val_loss" in m), None)
status["checkpoints"] = sorted(os.listdir(f"{RUN}/checkpoints"))
status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
status["hours"] = round((time.time() - T0) / 3600, 2)
save_status()
shutil.rmtree(REPO, ignore_errors=True)  # keep the output to checkpoints + logs
print(json.dumps(status, indent=2))
