# All-in-One LLM: project notes

We're building an all-in-one LLM **from scratch**, with no corners cut. The goal is the best model possible for its size, later extended with tools, retrieval, memory, vision and voice. Stage order and status are in README.md.

## Constraints
- **Budget: $0.** Never suggest paid services or subscriptions.
- **Laptop:** Windows 11, CPU only (no NVIDIA GPU), 32 GB RAM, Python 3.11, PyTorch CPU installed. Used to build and test everything at small scale.
- **Small VPS:** 1 vCPU, 1 GB RAM, 25 GB disk. Only good for 24/7 downloads, test runs and hosting a demo, not training.
- **Google TPU Research Cloud:** accepted. TPUs are free, but storage, disks and egress are not.
- **Google Cloud free trial:** $300 credit, expires 2027-02-04. Not upgraded to paid. Don't buy the "Cloud TPU v5e Fixed Price Subscription."
- Still unknown: how long TRC access lasts, which TPU types, and whether the clock has started. Don't start the clock until the training code is ready.
- This folder is inside OneDrive. Keep datasets and checkpoints out of it (see .gitignore) and use a non-synced folder such as `C:\llm-data`.

## Decisions
- Byte-level BPE tokenizer with special tokens for chat, tools and modalities.
- The model and training code go in **JAX**, because it runs best on TPUs. The tokenizer and data code are plain Python.
- Everything is built small on the laptop first, then scaled on TPUs. Scaling laws (stage 5) decide the final size.
- Training must checkpoint and resume, since free or preemptible sessions get cut off.
- Data: our own cleaning pipeline, run over free open datasets such as FineWeb and FineWeb-Edu.

## Status (2026-11-05)
- Stage 1, tokenizer (`src/aio/tokenizer/bpe.py`): done, tests pass.
- Stage 2, data pipeline (`src/aio/data/`): extraction, filters, PII, dedup and the pipeline are written. Tests pass.
  - Known limitation: valid-looking version strings such as `1.2.3.4` are redacted as IPs.
  - Quality scorer (`quality.py`): done. A linear model over hashed words and word pairs (NumPy), trained on the free FineWeb-Edu annotations (467k pages graded 0-5, in `C:\llm-data\edu-annotations`). The model is at `C:\llm-data\models\quality.npz`, about 8 minutes to train. On held-out pages: Pearson 0.73, F1 0.60 at finding grade 3+ pages, with the tuned cut-off of 2.1 stored in the model. The pipeline stores a `quality` field per document, and `--min-quality` filters on it. To improve it later, rescore with our own transformer once stage 3 exists.
  - Topic classifier (`domain.py`): done. A softmax over the same hashed features with 24 topics, trained on the free WebOrganizer topic annotations (100k pages labelled by Llama-3.1-405B, in `C:\llm-data\topic-annotations`). The model is at `C:\llm-data\models\domain.npz`, about 12 minutes to train. On the test split: 71.5% top-1 accuracy, 86% top-2, macro F1 0.70. Short texts get low confidence; normal pages (~580 words) are fine. The pipeline stores `domain` and `domain_prob`, and counts per topic go in stats. To improve it later, add the 1M-page Llama-3.1-8B annotations (2.9 GB) as a first training stage, as WebOrganizer did.
  - Still open in stage 2: choosing the actual data mix (topic weights × quality cut-off). This is best decided with small-model experiments once stage 3/4 exist.
- Stage 3, transformer (`src/aio/model/transformer.py`): core done in plain JAX (no Flax). It's Llama-style: RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings, optional top-k Mixture of Experts with a balance loss, and a KV cache for generation. Layers are stacked and run in `lax.scan`. Params stay float32, and compute dtype is configurable (bfloat16 on TPU). 14 tests cover shapes, param count, causality, KV cache = full forward, RoPE relative-position property, a tiny model memorising a batch (dense and MoE), the loss mask and bf16.
  - Laptop speed: a 21.7M-param model (d=384, 6 layers, vocab 32k) trains at about 430 tokens/s on CPU, so it's for testing only.
  - MoE currently runs every expert on every token (exact but not compute-saving). Token dispatch comes with sharding in stage 4.
  - Optimizer is now our own AdamW (`src/aio/train/optim.py`); Optax is only used in model tests.
- Stage 4, training loop (`src/aio/train/`): done. 53 tests pass (whole repo).
  - `data/tokens.py`: token shards (memory-mapped uint16/uint32 .bin + index.json). Batches are random windows chosen by (seed, step), so resume is exact.
  - `train/optim.py`: AdamW, warmup + cosine, global-norm clipping, no decay on 1-D params.
  - `train/checkpoint.py`: atomic save (tmp dir + rename), keep last N, restore with shape checks.
  - `train/trainer.py`: auto-resume, save on SIGTERM/Ctrl+C (preemptible TPUs), grad accumulation, eval, `metrics.jsonl`. `sharding: "dp"` (copy weights) or `"fsdp"` (split weights + Adam state, `train/sharding.py`). Both are tested on 4 simulated devices against 1 device.
  - `model/generate.py`: KV-cache sampling (temperature, top-k).
  - Single-host only so far. Multi-host TPU pods need `jax.distributed.initialize()` and per-host data slices.
  - First laptop run (`configs/laptop-tiny.json`): 6.5M params, vocab 8192, 60k FineWeb pages cleaned to 57k (49M train tokens), 3000 steps x 8192 tokens at about 3.8k tok/s (about 2 hours). Data: `C:/llm-data/tokens-sample`, run: `C:/llm-data/runs/laptop-tiny`, tokenizer: `C:/llm-data/tokenizer-8k.json`.
    - Result: validation loss 4.42 (perplexity 83). Fluent-looking English that stays loosely on topic but has no real meaning. Samples are in `C:/llm-data/runs/laptop-tiny/samples.txt`.
- Stage 5, scaling laws (`src/aio/scaling/`, `scripts/sweep.py`): tools done and tested. `fit_parametric` fits L(N,D) = E + A/N^a + B/D^b; `fit_isoflop` gives the best N per budget; `plan`/`run_all` run a resumable size-by-budget sweep with `--dry-run` time estimates.
  - Sweep v1 (2026-11-06, `C:/llm-data/sweeps/laptop`, 15 runs, budgets 3e12-1e14, batch 32x256): **not usable.** The smallest model won at every budget, because larger models got only 77-350 optimizer steps (step-limited, not compute-limited). The fit gave E=0 and absurd tokens/param. Lesson: small budgets need small batches and a minimum step count.
  - Sweep v2 (`C:/llm-data/sweeps/laptop-v2`, done 2026-11-06): batch 8, at least 400 steps, budgets 1e13/3e13/1e14, widths 64-256, no repeated data. Larger models improved by 0.2-0.5 loss over v1, but **the smallest width still won at every budget**, so the optimum lies below the tested range. The fit (E=3.39, alpha=0.59, beta=0.26) is still not trustworthy for sizing (it predicts thousands of tokens/param). The smallest model with the most steps did slightly worse at batch 8 than at batch 32, because small batches add noise.
  - Conclusion: the laptop can't reach the regime where sizes trade off. The real sweep on TPU should use widths that put each budget's minimum inside the range (check with `--dry-run`, and look for a U-shape per budget before fitting).
  - The laptop only reaches tiny models (<6M non-embedding params), where the 8k embedding dominates. A sweep that sets the real model size has to run on the TPUs.
- TPU readiness: multi-host support in the trainer (`distributed: true`; process 0 saves to local disk and broadcasts on resume, so no paid shared storage). It's **untested across hosts**, because Windows has no multi-process CPU collectives. `configs/tpu-smoke.json` is the first thing to run on a pod. The $0 guide is `docs/tpu.md`.
- Stage 6, post-training (`src/aio/post/`): done. 
  - Chat template with role special tokens. Content is encoded with `encode_ordinary`, so no role injection, and loss is on assistant text + `<|end|>` only.
  - Objectives in the trainer: `objective: "lm"` (pretrain/SFT) or `"dpo"` (frozen reference = `init_from` model, `dpo_beta`). `init_from` starts from another run's newest checkpoint.
  - Data: `scripts/prepare_chat.py` (smol-smoltalk, Apache-2.0, for SFT; ultrafeedback_binarized, MIT, for DPO) writes `.npz`. Raw data is in `C:/llm-data/chat/`. With `--max-len 513`: 139k SFT conversations and 27k DPO pairs.
  - Laptop SFT (`configs/laptop-sft.json`, from laptop-tiny, 1500 steps, seq 512): val loss on assistant text 2.50 (ppl 12.2). It follows the chat format and stops at `<|end|>`, and answers "What is the capital of France?" with "The capital of France is Paris." Longer answers are fluent but confused (6.5M params).
  - Laptop DPO: `configs/laptop-dpo.json` (from laptop-sft). `scripts/chat.py` is a terminal chat (`--once` for one question).
  - Not built yet: a separate reward model (DPO doesn't need one).
- Stage 7, evaluation (`src/aio/eval/`, `scripts/eval_mc.py`): multiple-choice by log-likelihood (acc and per-byte acc_norm) for HellaSwag, ARC-Easy/Challenge and MMLU (files cached in `C:/llm-data/evals`).
  - Base laptop-tiny, 1000 items each, acc_norm: HellaSwag 0.297, ARC-E 0.295, ARC-C 0.224, MMLU 0.244 (chance 0.25). HellaSwag and ARC-E are a bit above chance; the others are at chance, as expected at this size.
  - Not built yet: LLM-as-judge, our own task suites, perplexity-by-domain report.
- Next: stage 8 pieces that can be built small (tool-call format and execution loop, retrieval), the reward model, and more eval. Bigger models wait for TPU.

## Commands
- Tests: `python -m pytest -q tests`
- Clean data: `python scripts/run_pipeline.py --input "raw/*.jsonl.gz" --output <out_dir>` (add `--quality-model C:/llm-data/models/quality.npz` to score pages)
- Train topic classifier: `python scripts/train_domain.py --data C:/llm-data/topic-annotations --out C:/llm-data/models/domain.npz` (pipeline flag: `--domain-model`)
- Train tokenizer: `python scripts/train_tokenizer.py --input "<clean>/final/*.jsonl.gz" --vocab-size 32000 --out <tok.json>`
- Tokenize: `python scripts/tokenize_data.py --input "<clean>/final/*.jsonl.gz" --tokenizer <tok.json> --out <tokens_dir>` (writes train/ and val/)
- Train model: `python scripts/train.py --config configs/laptop-tiny.json [--set steps=100 --set model.d_model=128]` (re-run the same command to resume)
- Scaling sweep v2: `python scripts/sweep.py --config configs/laptop-tiny.json --out C:/llm-data/sweeps/laptop-v2 --budgets 1e13 3e13 1e14 --widths 64 96 128 192 256 --batch-size 8 --min-steps 400 --max-tokens 49e6` (add `--dry-run` to see the plan only)
- Chat data: `python scripts/prepare_chat.py --format smoltalk|ultrafeedback --input "<parquet glob>" --tokenizer <tok.json> --out <file.npz> --max-len 513`
- SFT / DPO: `python scripts/train.py --config configs/laptop-sft.json` then `configs/laptop-dpo.json`
- Chat: `python scripts/chat.py --run C:/llm-data/runs/laptop-sft --tokenizer C:/llm-data/tokenizer-8k.json`
- Benchmarks: `python scripts/eval_mc.py --run <out_dir> --tokenizer <tok.json> --tasks hellaswag arc_easy --limit 1000`
- Generate: `python scripts/generate.py --run <out_dir> --tokenizer <tok.json> --prompt "Once upon a time"`
- Train quality scorer: `python scripts/train_quality.py --data "C:/llm-data/edu-annotations/*.parquet" --out C:/llm-data/models/quality.npz`
