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
- Stage 2, data pipeline (`src/aio/data/`): extraction, filters, PII, dedup and the pipeline are written. All 18 tests pass.
  - Known limitation: valid-looking version strings such as `1.2.3.4` are redacted as IPs.
- Next: a quality classifier and domain classification (remaining parts of stage 2), then the JAX transformer (stage 3).

## Commands
- Tests: `python -m pytest -q tests`
- Clean data: `python scripts/run_pipeline.py --input "raw/*.jsonl.gz" --output <out_dir>`
