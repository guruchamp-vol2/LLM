"""Token shards: tokenized training data as flat binary files, read with memory mapping.

Each shard is a .bin file of token ids (uint16 when the vocabulary fits, else uint32), with
documents joined end to end and separated by the end-of-text token. index.json lists
the shards. Reading is instant and uses no RAM up front, whatever the dataset size.

Batches are random windows chosen from a seed and the step number only, so a resumed
run sees exactly the batches it would have seen without the interruption.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


class TokenShardWriter:
    def __init__(self, out_dir: str | Path, vocab_size: int, eot_id: int, tokens_per_shard: int = 50_000_000) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = np.uint16 if vocab_size <= 65_536 else np.uint32
        self.vocab_size, self.eot_id, self.tokens_per_shard = vocab_size, eot_id, tokens_per_shard
        self.buf: list[np.ndarray] = []
        self.buf_len = 0
        self.shards: list[dict] = []
        self.n_docs = 0

    def add(self, ids: Iterable[int]) -> None:
        """Add one document's token ids; the end-of-text token is appended here."""
        arr = np.fromiter(ids, dtype=np.int64)
        arr = np.append(arr, self.eot_id).astype(self.dtype)
        self.buf.append(arr)
        self.buf_len += len(arr)
        self.n_docs += 1
        if self.buf_len >= self.tokens_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self.buf_len:
            return
        name = f"tokens-{len(self.shards):05d}.bin"
        tmp = self.out_dir / (name + ".tmp")
        np.concatenate(self.buf).tofile(tmp)
        tmp.replace(self.out_dir / name)  # only complete shards get their final name
        self.shards.append({"file": name, "tokens": self.buf_len})
        self.buf, self.buf_len = [], 0

    def close(self) -> dict:
        self._flush()
        index = {
            "dtype": np.dtype(self.dtype).name, "vocab_size": self.vocab_size, "eot_id": self.eot_id,
            "n_docs": self.n_docs, "n_tokens": sum(s["tokens"] for s in self.shards), "shards": self.shards,
        }
        (self.out_dir / "index.json").write_text(json.dumps(index, indent=2))
        return index

    def __enter__(self) -> "TokenShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class TokenDataset:
    def __init__(self, data_dir: str | Path) -> None:
        self.dir = Path(data_dir)
        self.index = json.loads((self.dir / "index.json").read_text())
        dtype = np.dtype(self.index["dtype"])
        self.shards = [np.memmap(self.dir / s["file"], dtype=dtype, mode="r") for s in self.index["shards"]]
        self.vocab_size, self.eot_id = self.index["vocab_size"], self.index["eot_id"]

    @property
    def n_tokens(self) -> int:
        return sum(len(s) for s in self.shards)

    def sample(self, rng: np.random.Generator, n: int, seq_len: int) -> np.ndarray:
        """n random windows of seq_len + 1 tokens (inputs plus the next-token targets)."""
        usable = [s for s in self.shards if len(s) > seq_len + 1]
        if not usable:
            raise ValueError(f"no shard has more than {seq_len + 1} tokens")
        sizes = np.array([len(s) - seq_len - 1 for s in usable], dtype=np.float64)
        which = rng.choice(len(usable), size=n, p=sizes / sizes.sum())
        out = np.empty((n, seq_len + 1), dtype=np.int32)
        for row, k in enumerate(which):
            start = int(rng.integers(0, len(usable[k]) - seq_len - 1))
            out[row] = usable[k][start : start + seq_len + 1]
        return out

    def batch(self, step: int, batch_size: int, seq_len: int, seed: int = 0) -> np.ndarray:
        return self.sample(np.random.default_rng((seed, step)), batch_size, seq_len)


class MixtureDataset:
    """Several token datasets sampled by weight: each row of a batch comes from source i
    with probability weight_i / sum(weights). This is how the data mix (e.g. 50% web,
    30% code, 20% maths) is set without rewriting any shards.

    Spec: "DIR=WEIGHT,DIR=WEIGHT,...", e.g. "C:/tok/edu/train=5,C:/tok/github/train=3".
    """

    def __init__(self, spec: str) -> None:
        self.parts: list[tuple[str, float]] = []
        for item in spec.split(","):
            path, _, w = item.strip().rpartition("=")
            if not path:
                raise ValueError(f"mixture items look like DIR=WEIGHT, got {item!r}")
            self.parts.append((path, float(w)))
        self.sources = [TokenDataset(p) for p, _ in self.parts]
        vocab = {s.vocab_size for s in self.sources}
        if len(vocab) != 1:
            raise ValueError(f"mixture sources use different tokenizers (vocab sizes {sorted(vocab)})")
        self.vocab_size, self.eot_id = self.sources[0].vocab_size, self.sources[0].eot_id
        w = np.array([w for _, w in self.parts], dtype=np.float64)
        if (w <= 0).any():
            raise ValueError("mixture weights must be positive")
        self.probs = w / w.sum()

    @property
    def n_tokens(self) -> int:
        return sum(s.n_tokens for s in self.sources)

    def sample(self, rng: np.random.Generator, n: int, seq_len: int) -> np.ndarray:
        which = rng.choice(len(self.sources), size=n, p=self.probs)
        out = np.empty((n, seq_len + 1), dtype=np.int32)
        for k in range(len(self.sources)):
            rows = np.nonzero(which == k)[0]
            if len(rows):
                out[rows] = self.sources[k].sample(rng, len(rows), seq_len)
        return out

    def batch(self, step: int, batch_size: int, seq_len: int, seed: int = 0) -> np.ndarray:
        return self.sample(np.random.default_rng((seed, step)), batch_size, seq_len)

    def epochs(self, tokens_seen: int) -> list[float]:
        """How many times each source has been seen after tokens_seen training tokens
        (more than ~4 epochs of the same data stops helping)."""
        return [tokens_seen * p / max(s.n_tokens, 1) for p, s in zip(self.probs, self.sources)]
