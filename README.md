# All-in-One LLM

A language model built from scratch, aiming for an all-in-one AI: text, code, tools, retrieval, vision and voice.
Every part is written by us. The same code runs a small model on a CPU and a large one on a GPU cluster.

## Roadmap

| # | Stage | Status |
|---|-------|--------|
| 1 | Byte-level BPE tokenizer (GPT-4-style split, chat/tool/modality special tokens) | Done |
| 2 | Data pipeline: HTML extraction, PII/NSFW filters, URL/doc/line dedup, heuristic filters, quality classifier, domain mixing | Next |
| 3 | Transformer: RoPE, RMSNorm, SwiGLU, grouped-query attention, optional Mixture of Experts | |
| 4 | Pretraining loop: mixed precision, checkpointing, data/optimizer sharding (FSDP) | |
| 5 | Scaling laws: small-model sweeps, fitted curves, compute-optimal sizing | |
| 6 | Post-training: SFT, reward model, DPO | |
| 7 | Evaluation: perplexity, MMLU-style benchmarks, LLM-as-judge, our own task suites | |
| 8 | All-in-one layer: tool use, retrieval, memory, vision and audio encoders | |

## Layout

```
src/aio/tokenizer/   BPE tokenizer
tests/               tests (run: python -m pytest -q tests)
```
