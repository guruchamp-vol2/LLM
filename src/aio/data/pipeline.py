"""The data cleaning pipeline.

Pass 1 (streaming, per document):
    HTML extraction -> URL dedup -> URL safety -> quality filters -> text safety
    -> PII redaction -> exact dedup -> near-duplicate dedup (MinHash)
Pass 2 (corpus-wide):
    count lines across all documents, strip lines repeated in many documents
    (headers, footers, menus), then re-check length.

Every removed document is counted by reason, so we can see what each step does.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from .dedup import ExactDeduper, MinHashDeduper, count_lines, normalize_url, remove_common_lines, text_hash
from .extract import html_to_text
from .filters import QualityConfig, SafetyConfig, quality_filter, text_safety_filter, url_filter
from .io import ShardWriter, read_jsonl
from .pii import redact_pii


@dataclass
class PipelineConfig:
    quality: QualityConfig = field(default_factory=QualityConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    near_dedup: bool = True
    # a line in more than this many documents counts as boilerplate
    line_max_docs: int = 3
    docs_per_shard: int = 50_000


class Pipeline:
    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        self.cfg = cfg or PipelineConfig()
        self.stats: Counter = Counter()
        self.urls = ExactDeduper()
        self.exact = ExactDeduper()
        self.near = MinHashDeduper() if self.cfg.near_dedup else None

    def _drop(self, reason: str) -> None:
        self.stats[f"removed:{reason}"] += 1

    def process_doc(self, doc: dict) -> dict | None:
        self.stats["input"] += 1
        url = doc.get("url")
        text = doc.get("text")
        title = doc.get("title", "")
        if text is None and doc.get("html") is not None:
            title, text = html_to_text(doc["html"])
        if not text:
            return self._drop("empty")

        if url and self.urls.is_dup(normalize_url(url)):
            return self._drop("dup_url")
        if reason := url_filter(url, self.cfg.safety):
            return self._drop(reason)
        if reason := quality_filter(text, self.cfg.quality):
            return self._drop(reason)
        if reason := text_safety_filter(text.split(), self.cfg.safety):
            return self._drop(reason)

        text, n_pii = redact_pii(text)
        self.stats["pii_redactions"] += n_pii

        if self.exact.is_dup(text_hash(text)):
            return self._drop("dup_exact")
        if self.near is not None and self.near.is_dup(text):
            return self._drop("dup_near")

        self.stats["kept_pass1"] += 1
        out = {"text": text}
        if url:
            out["url"] = url
        if title:
            out["title"] = title
        for k in ("id", "source", "date", "lang"):
            if k in doc:
                out[k] = doc[k]
        return out

    def process(self, docs: Iterable[dict]) -> Iterator[dict]:
        for d in docs:
            if (out := self.process_doc(d)) is not None:
                yield out

    # ------------------------------------------------------------------ full run
    def run(self, inputs: list[str | Path], out_dir: str | Path) -> Counter:
        """Clean input JSONL shards into out_dir/final. Returns removal statistics."""
        out_dir = Path(out_dir)
        stage1 = out_dir / "stage1"

        def all_docs():
            for path in inputs:
                yield from read_jsonl(path)

        with ShardWriter(stage1, "stage1", self.cfg.docs_per_shard) as w:
            w.write_all(self.process(all_docs()))
            stage1_paths = list(w.paths)

        # Pass 2: corpus-wide line dedup.
        counts = count_lines(d["text"] for p in stage1_paths for d in read_jsonl(p))
        with ShardWriter(out_dir / "final", "data", self.cfg.docs_per_shard) as w:
            for p in stage1_paths:
                for d in read_jsonl(p):
                    text, n = remove_common_lines(d["text"], counts, self.cfg.line_max_docs)
                    self.stats["boilerplate_lines_removed"] += n
                    if len(text.split()) < self.cfg.quality.min_words:
                        self._drop("too_short_after_line_dedup")
                        continue
                    d["text"] = text
                    w.write(d)
                    self.stats["kept_final"] += 1
        return self.stats
