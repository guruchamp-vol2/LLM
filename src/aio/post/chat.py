"""Chat format and post-training datasets.

A conversation is a list of {"role": "system" | "user" | "assistant", "content": str}.
It becomes tokens like this:

    <|system|> ...text... <|end|> <|user|> ...text... <|end|> <|assistant|> ...text... <|end|>

Message text is encoded with encode_ordinary, so a user typing "<|assistant|>" cannot
inject a role marker. The loss mask is 1 only on assistant text and its <|end|>, so the
model learns to answer, not to imitate users.

Datasets are saved as .npz files (flat token arrays plus offsets) and loaded whole: chat
datasets are small next to pretraining data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ..tokenizer import BPETokenizer

ROLE_TOKENS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}
END = "<|end|>"
PAD = "<|pad|>"


def render(tok: BPETokenizer, messages: list[dict], add_generation_prompt: bool = False) -> tuple[list[int], list[int]]:
    """Token ids and loss mask (1 = train on this token) for a conversation."""
    end_id = tok.special_tokens[END]
    ids: list[int] = []
    mask: list[int] = []
    for m in messages:
        role = m["role"]
        if role not in ROLE_TOKENS:
            raise ValueError(f"unknown role {role!r}")
        body = tok.encode_ordinary(m["content"]) + [end_id]
        ids.append(tok.special_tokens[ROLE_TOKENS[role]])
        mask.append(0)
        ids += body
        mask += [1 if role == "assistant" else 0] * len(body)
    if add_generation_prompt:
        ids.append(tok.special_tokens[ROLE_TOKENS["assistant"]])
        mask.append(0)
    return ids, mask


def _pack(seqs: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(s) for s in seqs])
    flat = np.fromiter((t for s in seqs for t in s), dtype=np.int32, count=int(offsets[-1]))
    return flat, offsets


def _rows(flat: np.ndarray, offsets: np.ndarray, idx: np.ndarray, width: int, fill) -> np.ndarray:
    """Rows idx of a ragged array, cut or padded to width."""
    out = np.full((len(idx), width), fill, dtype=flat.dtype)
    for r, i in enumerate(idx):
        s = flat[offsets[i] : offsets[i + 1]][:width]
        out[r, : len(s)] = s
    return out


# ---------------------------------------------------------------------- SFT
def save_conversations(path: str | Path, tok: BPETokenizer, conversations: Iterable[list[dict]]) -> int:
    seqs, masks = [], []
    for conv in conversations:
        ids, mask = render(tok, conv)
        if sum(mask):  # nothing to learn from a conversation without an assistant turn
            seqs.append(ids)
            masks.append(mask)
    tokens, offsets = _pack(seqs)
    mask_flat, _ = _pack(masks)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, kind="sft", tokens=tokens, mask=mask_flat.astype(np.uint8), offsets=offsets,
             pad_id=tok.special_tokens[PAD], vocab_size=tok.vocab_size)
    return len(seqs)


class ConversationDataset:
    def __init__(self, path: str | Path) -> None:
        z = np.load(path)
        if str(z["kind"]) != "sft":
            raise ValueError(f"{path} is not an SFT dataset")
        self.tokens, self.mask, self.offsets = z["tokens"], z["mask"], z["offsets"]
        self.pad_id, self.vocab_size = int(z["pad_id"]), int(z["vocab_size"])

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_tokens(self) -> int:
        return int(self.offsets[-1])

    def batch(self, step: int, batch_size: int, seq_len: int, seed: int = 0) -> dict:
        idx = np.random.default_rng((seed, step)).integers(0, len(self), batch_size)
        return {
            "tokens": _rows(self.tokens, self.offsets, idx, seq_len + 1, self.pad_id),
            "mask": _rows(self.mask, self.offsets, idx, seq_len + 1, 0).astype(np.float32),
        }


# ---------------------------------------------------------------------- preference pairs (DPO)
def save_pairs(path: str | Path, tok: BPETokenizer, items: Iterable[dict]) -> int:
    """items: {"prompt": [messages...], "chosen": str, "rejected": str}."""
    ch, ch_m, rj, rj_m = [], [], [], []
    for it in items:
        a, am = render(tok, it["prompt"] + [{"role": "assistant", "content": it["chosen"]}])
        b, bm = render(tok, it["prompt"] + [{"role": "assistant", "content": it["rejected"]}])
        ch.append(a), ch_m.append(am), rj.append(b), rj_m.append(bm)
    c, co = _pack(ch)
    cm, _ = _pack(ch_m)
    r, ro = _pack(rj)
    rm, _ = _pack(rj_m)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, kind="pairs", chosen=c, chosen_mask=cm.astype(np.uint8), chosen_offsets=co,
             rejected=r, rejected_mask=rm.astype(np.uint8), rejected_offsets=ro,
             pad_id=tok.special_tokens[PAD], vocab_size=tok.vocab_size)
    return len(ch)


class PairDataset:
    def __init__(self, path: str | Path) -> None:
        z = np.load(path)
        if str(z["kind"]) != "pairs":
            raise ValueError(f"{path} is not a preference-pair dataset")
        self.z = {k: z[k] for k in z.files}
        self.pad_id, self.vocab_size = int(z["pad_id"]), int(z["vocab_size"])

    def __len__(self) -> int:
        return len(self.z["chosen_offsets"]) - 1

    @property
    def n_tokens(self) -> int:
        return int(self.z["chosen_offsets"][-1] + self.z["rejected_offsets"][-1])

    def batch(self, step: int, batch_size: int, seq_len: int, seed: int = 0) -> dict:
        idx = np.random.default_rng((seed, step)).integers(0, len(self), batch_size)
        z, w = self.z, seq_len + 1
        out = {}
        for side in ("chosen", "rejected"):
            off = z[f"{side}_offsets"]
            out[side] = _rows(z[side], off, idx, w, self.pad_id)
            out[f"{side}_mask"] = _rows(z[f"{side}_mask"], off, idx, w, 0).astype(np.float32)
        return out
