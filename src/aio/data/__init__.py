"""Data pipeline. Names load lazily, so light users (the web collector on the 1 GB VPS)
only need the Python standard library, not NumPy."""

import importlib

_WHERE = {
    "html_to_text": "extract",
    "QualityConfig": "filters", "SafetyConfig": "filters", "quality_filter": "filters", "url_filter": "filters",
    "redact_pii": "pii",
    "MinHashDeduper": "dedup", "ExactDeduper": "dedup", "normalize_url": "dedup",
    "Pipeline": "pipeline", "PipelineConfig": "pipeline",
    "read_jsonl": "io", "ShardWriter": "io",
    "QualityScorer": "quality",
    "DomainClassifier": "domain",
}

__all__ = list(_WHERE)


def __getattr__(name):
    if name not in _WHERE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{_WHERE[name]}", __name__), name)
    globals()[name] = value
    return value
