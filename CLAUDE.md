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
- Stage 2, data pipeline (`src/aio/data/`): extraction, filters, PII, dedup and the pipeline are written. All 40 tests pass (whole repo).
  - Known limitation: valid-looking version strings such as `1.2.3.4` are redacted as IPs.
  - Quality scorer (`quality.py`): done. A linear model over hashed words and word pairs (NumPy), trained on the free FineWeb-Edu annotations (467k pages graded 0-5, in `C:\llm-data\edu-annotations`). The model is at `C:\llm-data\models\quality.npz`, about 8 minutes to train. On held-out pages: Pearson 0.73, F1 0.60 at finding grade 3+ pages, with the tuned cut-off of 2.1 stored in the model. The pipeline stores a `quality` field per document, and `--min-quality` filters on it. To improve it later, rescore with our own transformer once stage 3 exists.
  - Topic classifier (`domain.py`): done. A softmax over the same hashed features with 24 topics, trained on the free WebOrganizer topic annotations (100k pages labelled by Llama-3.1-405B, in `C:\llm-data	opic-annotations`). The model is at `C:\llm-data\models\domain.npz`, about 12 minutes to train. On the test split: 71.5% top-1 accuracy, 86% top-2, macro F1 0.70. Short texts get low confidence; normal pages (~580 words) are fine. The pipeline stores `domain` and `domain_prob`, and counts per topic go in stats. To improve it later, add the 1M-page Llama-3.1-8B annotations (2.9 GB) as a first training stage, as WebOrganizer did.
  - Still open in stage 2: choosing the actual data mix (topic weights × quality cut-off). This is best decided with small-model experiments once stage 3/4 exist.
- Stage 3, transformer (`src/aio/model/transformer.py`): core done in plain JAX (no Flax). It's Llama-style: RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings, optional top-k Mixture of Experts with a balance loss, and a KV cache for generation. Layers are stacked and run in `lax.scan`. Params stay float32, and compute dtype is configurable (bfloat16 on TPU). 14 tests cover shapes, param count, causality, KV cache = full forward, RoPE relative-position property, a tiny model memorising a batch (dense and MoE), the loss mask and bf16.
  - Laptop speed: a 21.7M-param model (d=384, 6 layers, vocab 32k) trains at about 430 tokens/s on CPU, so it's for testing only.
  - MoE currently runs every expert on every token (exact but not compute-saving). Token dispatch comes with sharding in stage 4.
  - Optax is used only in tests for now. Stage 4 can write our own AdamW.
- Next: stage 4, the training loop (data loader from tokenized shards, AdamW plus warmup/cosine schedule, gradient clipping, checkpoint/resume, logging, eval), then sharding for TPU.

## Commands
- Tests: `python -m pytest -q tests`
- Clean data: `python scripts/run_pipeline.py --input "raw/*.jsonl.gz" --output <out_dir>` (add `--quality-model C:/llm-data/models/quality.npz` to score pages)
- Train topic classifier: `python scripts/train_domain.py --data C:/llm-data/topic-annotations --out C:/llm-data/models/domain.npz` (pipeline flag: `--domain-model`)
- Train quality scorer: `python scripts/train_quality.py --data "C:/llm-data/edu-annotations/*.parquet" --out C:/llm-data/models/quality.npz`
