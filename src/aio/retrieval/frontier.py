"""Crawl frontier: which known pages to re-check, and when. A small version of what big
search engines do.

Big crawlers don't re-read the whole web every day. They keep a list of known pages,
each with its own revisit interval, and spend a fixed daily budget on the pages most
overdue:

- A page that changed when last checked is checked again sooner (interval halves).
- A page that hadn't changed waits longer (interval doubles), between min and max.
- Each check is a conditional request (ETag / Last-Modified). An unchanged page answers
  "304 Not Modified" and costs almost nothing.
- New pages are discovered from sitemaps, which sites publish for crawlers, often with
  each page's last-change date.

State is one JSON file, so the daily job can stop and start any time.
"""

from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..data.extract import html_to_text
from ..data.pii import redact_pii
from .index import SearchIndex

DAY = 86_400.0
_CHANGEFREQ = {"always": 3600, "hourly": 3600, "daily": DAY, "weekly": 7 * DAY, "monthly": 30 * DAY,
               "yearly": 180 * DAY, "never": 180 * DAY}
_SM = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


@dataclass
class PageState:
    url: str
    interval: float = DAY          # seconds between checks
    next_due: float = 0.0          # unix time of the next check (0 = as soon as possible)
    etag: str = ""
    last_modified: str = ""
    content_hash: str = ""
    checks: int = 0
    changes: int = 0
    errors: int = 0
    source: str = ""


@dataclass
class Frontier:
    path: Path
    min_interval: float = 3600.0
    max_interval: float = 60 * DAY
    pages: dict[str, PageState] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, **kw) -> "Frontier":
        f = cls(Path(path), **kw)
        if f.path.exists():
            for d in json.loads(f.path.read_text(encoding="utf-8")):
                f.pages[d["url"]] = PageState(**d)
        return f

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(p) for p in self.pages.values()]), encoding="utf-8")
        tmp.replace(self.path)

    # ------------------------------------------------------------------ discovery
    def add(self, url: str, interval: float = DAY, source: str = "") -> bool:
        if url in self.pages:
            return False
        self.pages[url] = PageState(url=url, interval=self._clamp(interval), source=source)
        return True

    def add_from_sitemap(self, xml: str, source: str = "") -> tuple[int, list[str]]:
        """Add the pages listed in a sitemap. Returns (pages added, nested sitemap URLs):
        a sitemap index only lists other sitemaps, which the caller fetches next."""
        root = ET.fromstring(xml)
        nested = []
        for sm in root.iter(_SM + "sitemap"):
            loc = sm.findtext(_SM + "loc")
            if loc and loc.strip():
                nested.append(loc.strip())
        added = 0
        for u in root.iter(_SM + "url"):
            loc = u.find(_SM + "loc")
            if loc is None or not (loc.text or "").strip():
                continue
            freq = (u.findtext(_SM + "changefreq") or "").strip().lower()
            added += self.add(loc.text.strip(), _CHANGEFREQ.get(freq, 7 * DAY), source)
        return added, nested

    # ------------------------------------------------------------------ scheduling
    def _clamp(self, s: float) -> float:
        return max(self.min_interval, min(self.max_interval, s))

    def due(self, now: float | None = None, budget: int = 1000) -> list[PageState]:
        """The `budget` most overdue pages."""
        now = time.time() if now is None else now
        ready = [p for p in self.pages.values() if p.next_due <= now]
        ready.sort(key=lambda p: p.next_due)
        return ready[:budget]

    def record(self, page: PageState, changed: bool, now: float | None = None, error: bool = False) -> None:
        now = time.time() if now is None else now
        first = page.checks == 0
        page.checks += 1
        if first and not error:
            pass  # first visit: we learn nothing about how often it changes yet
        elif error:
            page.errors += 1
            page.interval = self._clamp(page.interval * 2)  # back off from failing pages
        elif changed:
            page.changes += 1
            page.interval = self._clamp(page.interval / 2)
        else:
            page.interval = self._clamp(page.interval * 2)
        page.next_due = now + page.interval


def crawl(frontier: Frontier, index: SearchIndex, fetcher, budget: int = 1000, now: float | None = None,
          day: str | None = None, log=print) -> dict:
    """Re-check the most overdue pages; add new or changed ones to the search index."""
    stats = {"checked": 0, "not_modified": 0, "unchanged": 0, "changed": 0, "errors": 0}
    for page in frontier.due(now, budget):
        stats["checked"] += 1
        try:
            status, body, etag, lm = fetcher.conditional(page.url, page.etag, page.last_modified)
        except Exception as e:  # disallowed, timeout, 404, ...
            frontier.record(page, changed=False, now=now, error=True)
            stats["errors"] += 1
            log(f"error {page.url}: {e}")
            continue
        if status == 304:
            frontier.record(page, changed=False, now=now)
            stats["not_modified"] += 1
            continue
        title, text = html_to_text(body)
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        page.etag, page.last_modified = etag, lm
        if not text or h == page.content_hash:
            frontier.record(page, changed=False, now=now)
            stats["unchanged"] += 1
            continue
        page.content_hash = h
        text, _ = redact_pii(text)
        index.add([{"url": page.url, "title": title, "text": text, "source": page.source}], day=day)
        frontier.record(page, changed=True, now=now)
        stats["changed"] += 1
    frontier.save()
    return stats
