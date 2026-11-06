"""Read and write documents as JSON Lines shards (optionally gzip-compressed).

Small shards let the laptop, the server and the TPU machines each work on separate
pieces, and let interrupted jobs resume at shard boundaries.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator


def _open(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with _open(Path(path), "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class ShardWriter:
    def __init__(self, out_dir: str | Path, prefix: str = "shard", docs_per_shard: int = 50_000) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix, self.docs_per_shard = prefix, docs_per_shard
        self.index, self.count, self.f = 0, 0, None
        self.paths: list[Path] = []

    def _next(self) -> None:
        if self.f:
            self.f.close()
            self._tmp.replace(self.paths[-1])  # rename only when complete
        path = self.out_dir / f"{self.prefix}-{self.index:05d}.jsonl.gz"
        self._tmp = path.with_name(path.name + ".tmp")
        self.f = gzip.open(self._tmp, "wt", encoding="utf-8")
        self.paths.append(path)
        self.index += 1
        self.count = 0

    def write(self, doc: dict) -> None:
        if self.f is None or self.count >= self.docs_per_shard:
            self._next()
        self.f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        self.count += 1

    def write_all(self, docs: Iterable[dict]) -> None:
        for d in docs:
            self.write(d)

    def close(self) -> None:
        if self.f:
            self.f.close()
            self._tmp.replace(self.paths[-1])
            self.f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
