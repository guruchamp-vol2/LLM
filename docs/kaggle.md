# Training on Kaggle's free TPUs ($0, no billing)

Kaggle notebooks come with a free weekly accelerator quota (TPU and GPU hours) and need
no credit card. Our JAX code runs there unchanged. Check the current limits on Kaggle's
own pages: session length, weekly hours and disk sizes change from time to time.

## One-time setup
1. Make a Kaggle account and **verify your phone number** (Settings). Without it there is
   no internet access and no accelerators in notebooks.
2. Put the code where a notebook can fetch it:
   - a GitHub repo (`git clone` works with internet on), or
   - zip the repo and upload it as a private **Kaggle Dataset**.
3. Upload the token shards (the `train/` and `val/` folders from `scripts/tokenize_data.py`)
   as a private Kaggle Dataset, e.g. `llm-tokens`. Uploading is free.

## Each session
New notebook → Settings → Accelerator: **TPU** → Internet: **on** → Add data: your
`llm-tokens` dataset (and the previous run's output, if resuming).

```python
!git clone https://github.com/<you>/<repo>.git llm    # or unzip the uploaded code
%cd llm
!pip install -q -U "jax[tpu]" regex pyarrow
import jax; print(jax.devices())                        # should list the TPU cores
```

Copy the previous checkpoints in, so the trainer resumes instead of starting over:

```python
!mkdir -p /kaggle/working/run && cp -r /kaggle/input/<previous-output>/run/checkpoints /kaggle/working/run/ || true
```

Then train. Point the config at Kaggle's paths (`--set` overrides any field):

```python
!python scripts/train.py --config configs/tpu-smoke.json \
    --set data_dir=/kaggle/input/llm-tokens/train --set val_dir=/kaggle/input/llm-tokens/val \
    --set out_dir=/kaggle/working/run --set distributed=false
```

`distributed=false`: a Kaggle TPU is one machine, so multi-host setup is not needed.

## Keeping checkpoints between sessions
Sessions have a time limit. Only files in `/kaggle/working` survive, and only when you
**Save Version** (Save & Run All, or Quick Save). The trainer saves every `ckpt_every` steps
and on shutdown signals, so:

1. Pick `steps` so a session ends before the time limit, or just let it run and stop it
   (Ctrl+C equivalent: the notebook's stop button). Re-running resumes.
2. Save a version. Its output then appears as a dataset you can attach to the next
   session, as in the copy step above.
3. Keep `keep_ckpts` small (2-3): working-dir space is limited.

## Good first jobs
1. `configs/tpu-smoke.json`: a few minutes. Checks devices, bf16, loss falling and resume.
   Write down tokens/s: it sets how big a sweep fits in the weekly hours.
2. The scaling sweep (`scripts/sweep.py`) with bigger widths than the laptop could handle,
   until each budget shows a U-shape (some size in the middle wins).
3. The compute-optimal model that the sweep's fit predicts.
