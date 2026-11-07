"""Live search tools: look things up at question time instead of keeping a copy.

This is how chat assistants stay current (ChatGPT search, Perplexity): when a question
needs fresh or rare facts, they query a search service right then. Our free version
uses Wikipedia's public search API (Wikimedia's REST API, which welcomes automated use
from clients that identify themselves with a User-Agent).
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
from typing import Callable

_TAG = re.compile(r"<[^>]+>")


def _plain(s: str) -> str:
    return " ".join(html.unescape(_TAG.sub("", s or "")).split())


def wikipedia_search_tool(fetch: Callable[[str], str] | None = None, lang: str = "en", k: int = 3):
    """A tool for the agent loop: wikipedia(query) -> numbered results with a short excerpt."""
    if fetch is None:
        from .collect import PoliteFetcher

        fetch = PoliteFetcher(delay=1.0)

    def wikipedia(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            return "error: empty query"
        url = (f"https://api.wikimedia.org/core/v1/wikipedia/{lang}/search/page?"
               f"q={urllib.parse.quote(query.strip()[:300])}&limit={k}")
        try:
            pages = json.loads(fetch(url)).get("pages", [])
        except Exception as e:
            return f"error: search failed ({e})"
        if not pages:
            return "no results"
        lines = []
        for n, p in enumerate(pages[:k], 1):
            link = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(p.get('key', ''))}"
            desc = _plain(p.get("description") or "")
            excerpt = _plain(p.get("excerpt") or "")
            body = "; ".join(x for x in (desc, excerpt) if x)
            lines.append(f"[{n}] {p.get('title', '?')} ({link}): {body}")
        return "\n".join(lines)

    return wikipedia
