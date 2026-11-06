"""Loaders for free benchmark datasets (Hugging Face parquet files) into MCItems.

| Task       | Dataset                          | Licence      | Choices |
|------------|----------------------------------|--------------|---------|
| hellaswag  | Rowan/hellaswag (validation)     | MIT          | 4       |
| arc_easy   | allenai/ai2_arc ARC-Easy (test)  | CC-BY-SA-4.0 | 3-5     |
| arc_challenge | allenai/ai2_arc ARC-Challenge | CC-BY-SA-4.0 | 3-5     |
| mmlu       | cais/mmlu all (test)             | MIT          | 4       |
"""

from __future__ import annotations

import re

import pyarrow.parquet as pq

from .mc import MCItem

URLS = {
    "hellaswag": ["https://huggingface.co/datasets/Rowan/hellaswag/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"],
    "arc_easy": ["https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Easy/test-00000-of-00001.parquet"],
    "arc_challenge": ["https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet"],
    "mmlu": ["https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet"],
}


def _hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load(task: str, path: str) -> list[MCItem]:
    rows = pq.read_table(path).to_pylist()
    items = []
    if task == "hellaswag":
        for r in rows:
            ctx = _hellaswag_clean(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize())
            items.append(MCItem(ctx, [" " + _hellaswag_clean(e) for e in r["endings"]], int(r["label"])))
    elif task in ("arc_easy", "arc_challenge"):
        for r in rows:
            labels = list(r["choices"]["label"])
            if r["answerKey"] not in labels:
                continue
            items.append(MCItem(f"Question: {r['question']}\nAnswer:", [" " + t for t in r["choices"]["text"]],
                                labels.index(r["answerKey"])))
    elif task == "mmlu":
        for r in rows:
            items.append(MCItem(f"Question: {r['question']}\nAnswer:", [" " + c for c in r["choices"]], int(r["answer"])))
    else:
        raise ValueError(f"unknown task {task!r}; choose from {sorted(URLS)}")
    return items
