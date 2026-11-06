"""Safety and quality filters for web documents.

Each filter returns None if the document passes, or a short reason string if it is
removed, so the pipeline can report exactly why data was dropped.
Thresholds follow the Gopher (DeepMind) and C4 (Google) papers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlsplit

# --------------------------------------------------------------------------- safety

# Minimal defaults. Load a full list (e.g. a public adult-domain blocklist) with
# `load_list` for real runs.
DEFAULT_BLOCKED_DOMAINS: set[str] = set()
DEFAULT_URL_BANNED_WORDS = {"porn", "xxx", "nsfw", "hentai", "sex-cam", "escort"}


def load_list(path: str) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {ln.strip().lower() for ln in f if ln.strip() and not ln.startswith("#")}


@dataclass
class SafetyConfig:
    blocked_domains: set[str] = field(default_factory=lambda: set(DEFAULT_BLOCKED_DOMAINS))
    url_banned_words: set[str] = field(default_factory=lambda: set(DEFAULT_URL_BANNED_WORDS))
    text_banned_words: set[str] = field(default_factory=set)
    max_banned_word_ratio: float = 0.005


def url_filter(url: str | None, cfg: SafetyConfig) -> str | None:
    if not url:
        return None
    host = (urlsplit(url).hostname or "").lower()
    parts = host.split(".")
    # match the domain and every parent domain (a.b.example.com -> example.com)
    for i in range(len(parts) - 1):
        if ".".join(parts[i:]) in cfg.blocked_domains:
            return "blocked_domain"
    low = url.lower()
    if any(w in low for w in cfg.url_banned_words):
        return "url_banned_word"
    return None


def text_safety_filter(words: list[str], cfg: SafetyConfig) -> str | None:
    if not cfg.text_banned_words or not words:
        return None
    hits = sum(1 for w in words if w.lower().strip(".,!?;:\"'()") in cfg.text_banned_words)
    return "banned_words" if hits / len(words) > cfg.max_banned_word_ratio else None


# --------------------------------------------------------------------------- quality

STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
BULLETS = ("-", "*", "•", "‣", "◦", "·")


@dataclass
class QualityConfig:
    min_words: int = 50
    max_words: int = 100_000
    min_mean_word_len: float = 3.0
    max_mean_word_len: float = 10.0
    max_symbol_ratio: float = 0.1
    max_bullet_line_frac: float = 0.9
    max_ellipsis_line_frac: float = 0.3
    min_alpha_word_frac: float = 0.8
    # The stop-word check only makes sense for English. Turn off for multilingual data.
    require_stop_words: bool = True
    min_stop_words: int = 2
    max_dup_line_frac: float = 0.3
    max_dup_para_frac: float = 0.3
    max_dup_line_char_frac: float = 0.2
    # most-frequent n-gram share of characters, n = 2, 3, 4
    top_ngram_char_frac: dict = field(default_factory=lambda: {2: 0.20, 3: 0.18, 4: 0.16})
    # share of characters inside repeated n-grams, n = 5..10
    dup_ngram_char_frac: dict = field(
        default_factory=lambda: {5: 0.15, 6: 0.14, 7: 0.13, 8: 0.12, 9: 0.11, 10: 0.10}
    )


def _dup_fraction(items: list[str]) -> tuple[float, float]:
    """(fraction of items that are repeats, fraction of chars in repeated items)."""
    items = [x for x in items if x.strip()]
    if not items:
        return 0.0, 0.0
    seen: set[str] = set()
    dup_n = dup_chars = 0
    for x in items:
        if x in seen:
            dup_n += 1
            dup_chars += len(x)
        seen.add(x)
    return dup_n / len(items), dup_chars / max(1, sum(len(x) for x in items))


def _top_ngram_frac(words: list[str], n: int, total_chars: int) -> float:
    if len(words) < n:
        return 0.0
    counts = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    gram, c = counts.most_common(1)[0]
    if c <= 1:
        return 0.0
    return c * sum(len(w) for w in gram) / total_chars


def _dup_ngram_frac(words: list[str], n: int, total_chars: int) -> float:
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    covered = [False] * len(words)
    for i, g in enumerate(grams):
        if counts[g] > 1:
            for j in range(i, i + n):
                covered[j] = True
    return sum(len(w) for w, c in zip(words, covered) if c) / total_chars


def quality_filter(text: str, cfg: QualityConfig) -> str | None:
    words = text.split()
    n = len(words)
    if n < cfg.min_words:
        return "too_short"
    if n > cfg.max_words:
        return "too_long"
    total_chars = sum(len(w) for w in words)
    mean_len = total_chars / n
    if not cfg.min_mean_word_len <= mean_len <= cfg.max_mean_word_len:
        return "mean_word_length"
    if (text.count("#") + text.count("...") + text.count("…")) / n > cfg.max_symbol_ratio:
        return "symbol_ratio"

    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines:
        if sum(ln.lstrip().startswith(BULLETS) for ln in lines) / len(lines) > cfg.max_bullet_line_frac:
            return "bullet_lines"
        if sum(ln.rstrip().endswith(("...", "…")) for ln in lines) / len(lines) > cfg.max_ellipsis_line_frac:
            return "ellipsis_lines"
    if sum(any(c.isalpha() for c in w) for w in words) / n < cfg.min_alpha_word_frac:
        return "non_alpha_words"
    if cfg.require_stop_words:
        if len({w.lower() for w in words} & STOP_WORDS) < cfg.min_stop_words:
            return "no_stop_words"
    if "lorem ipsum" in text.lower():
        return "lorem_ipsum"

    line_frac, line_chars = _dup_fraction(lines)
    if line_frac > cfg.max_dup_line_frac or line_chars > cfg.max_dup_line_char_frac:
        return "dup_lines"
    para_frac, _ = _dup_fraction(text.split("\n\n"))
    if para_frac > cfg.max_dup_para_frac:
        return "dup_paragraphs"
    for k, limit in cfg.top_ngram_char_frac.items():
        if _top_ngram_frac(words, k, total_chars) > limit:
            return f"top_{k}gram"
    for k, limit in cfg.dup_ngram_char_frac.items():
        if _dup_ngram_frac(words, k, total_chars) > limit:
            return f"dup_{k}gram"
    return None
