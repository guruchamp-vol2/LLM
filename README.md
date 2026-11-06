# All-in-One LLM

A language model built from scratch, aiming for an all-in-one AI: text, code, tools, retrieval, vision and voice.
Every part is written by us. The same code runs a small model on a CPU and a large one on a GPU cluster.

## Roadmap

| # | Stage | Status |
|---|-------|--------|
| 1 | Byte-level BPE tokenizer (GPT-4-style split, chat/tool/modality special tokens) | Done |
| 2 | Data pipeline: HTML extraction, PII/NSFW filters, URL/doc/line dedup, heuristic filters, quality classifier, domain mixing | Mostly done (classifiers built; final mix chosen after stage 4 experiments) |
| 3 | Transformer: RoPE, RMSNorm, SwiGLU, grouped-query attention, optional Mixture of Experts | Core done |
| 4 | Pretraining loop: mixed precision, checkpointing, data/optimizer sharding (FSDP) | Done (single host) |
| 5 | Scaling laws: small-model sweeps, fitted curves, compute-optimal sizing | Tools done; laptop sweep running, real sweep needs TPU |
| 6 | Post-training: SFT, reward model, DPO | |
| 7 | Evaluation: perplexity, MMLU-style benchmarks, LLM-as-judge, our own task suites | |
| 8 | All-in-one layer: tool use, retrieval, memory, vision and audio encoders | |

## Layout

```
src/aio/tokenizer/   BPE tokenizer
src/aio/model/       JAX transformer
src/aio/data/        data cleaning pipeline, quality scorer, topic classifier
scripts/             run_pipeline.py, train_quality.py, train_domain.py
tests/               tests (run: python -m pytest -q tests)
```
