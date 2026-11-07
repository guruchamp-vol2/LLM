"""Daily collector: fetch new pages from a list of sources into the search index.

Sources are listed in a JSON file you control (see configs/sources.json):

    [{"type": "rss", "url": "https://export.arxiv.org/rss/cs.CL", "name": "arXiv cs.CL", "full_text": false},
     {"type": "wikipedia_featured", "lang": "en"}]

- rss: RSS or Atom feeds. Each item's title and summary are indexed; with
  "full_text": true the linked page is fetched too and cleaned with our HTML extractor.
- wikipedia_featured: Wikipedia's daily feed (today's featured article, most-read
  articles, in-the-news items), via the free Wikimedia REST API.

Rules it follows:
- robots.txt is checked before fetching any page, and disallowed pages are skipped.
- At most one request per host every `delay` seconds.
- Personal information is redacted with the same code as the training-data pipeline.
- Only `keep_days` of index files are kept, so the index fits on a small server.

Check each site's terms before adding it. Feeds are published for automated reading;
full article pages often are not.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

from ..data.extract import html_to_text
from ..data.pii import redact_pii
from .index import SearchIndex

USER_AGENT = "aio-llm-collector/0.1 (personal research project; polite, robots.txt-aware)"

Fetch = Callable[[str], str]


class PoliteFetcher:
    """HTTP GET with robots.txt checks and a per-host delay."""

    def __init__(self, delay: float = 2.0, timeout: float = 20.0, user_agent: str = USER_AGENT) -> None:
        self.delay, self.timeout, self.ua = delay, timeout, user_agent
        self.last: dict[str, float] = {}
        self.robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        root = f"{parts.scheme}://{parts.netloc}"
        if root not in self.robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                rp.parse(self._get(root + "/robots.txt").splitlines())
            except Exception:  # no robots.txt (or unreachable): treat as allowed, like most crawlers
                rp = None
            self.robots[root] = rp
        rp = self.robots[root]
        return True if rp is None else rp.can_fetch(self.ua, url)

    def _get(self, url: str) -> str:
        host = urllib.parse.urlsplit(url).netloc
        wait = self.last.get(host, 0) + self.delay - time.time()
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": self.ua})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read(5_000_000)  # cap: 5 MB per response
                charset = r.headers.get_content_charset() or "utf-8"
        finally:
            self.last[host] = time.time()
        return raw.decode(charset, errors="replace")

    def __call__(self, url: str) -> str:
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        return self._get(url)

    def conditional(self, url: str, etag: str = "", last_modified: str = "") -> tuple[int, str, str, str]:
        """GET that tells the server what we already have. Returns (status, body, etag,
        last_modified). Status 304 means "not modified": nothing is downloaded, which is
        how big crawlers re-check millions of pages cheaply."""
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        host = urllib.parse.urlsplit(url).netloc
        wait = self.last.get(host, 0) + self.delay - time.time()
        if wait > 0:
            time.sleep(wait)
        headers = {"User-Agent": self.ua}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read(5_000_000)
                charset = r.headers.get_content_charset() or "utf-8"
                return (r.status, raw.decode(charset, errors="replace"),
                        r.headers.get("ETag", ""), r.headers.get("Last-Modified", ""))
        except urllib.error.HTTPError as e:
            if e.code == 304:
                return 304, "", etag, last_modified
            raise
        finally:
            self.last[host] = time.time()


# ---------------------------------------------------------------------- sources
def _text(el, *names) -> str:
    for n in names:
        found = el.find(n)
        if found is not None and (found.text or "").strip():
            return found.text.strip()
    return ""


def parse_feed(xml: str) -> list[dict]:
    """Items of an RSS 2.0 or Atom feed: {"url", "title", "summary", "date"}."""
    root = ET.fromstring(xml)
    atom = "{http://www.w3.org/2005/Atom}"
    items = []
    for it in root.iter("item"):  # RSS
        items.append({"url": _text(it, "link"), "title": _text(it, "title"),
                      "summary": _text(it, "description"), "date": _text(it, "pubDate")})
    for it in root.iter(atom + "entry"):  # Atom
        link = it.find(atom + "link")
        items.append({"url": link.get("href", "") if link is not None else "", "title": _text(it, atom + "title"),
                      "summary": _text(it, atom + "summary", atom + "content"), "date": _text(it, atom + "updated")})
    return items


_ARXIV_PREFIX = re.compile(r"^arXiv:\S+\s+Announce Type:\s*\S+\s*(Abstract:)?\s*")


def _clean_summary(s: str) -> str:
    """Drop boilerplate some feeds put before the real summary (e.g. arXiv's header)."""
    return _ARXIV_PREFIX.sub("", s.strip())


def _strip_html(s: str) -> str:
    if "<" not in s:
        return s
    return html_to_text(f"<html><body><p>{s}</p></body></html>")[1] or s


def collect_rss(src: dict, fetch: Fetch) -> list[dict]:
    out = []
    for it in parse_feed(fetch(src["url"])):
        text = _clean_summary(_strip_html(it["summary"]))
        if src.get("full_text") and it["url"]:
            try:
                _, page = html_to_text(fetch(it["url"]))
                text = page or text
            except Exception:
                pass  # disallowed or failed: keep the summary
        if text:
            out.append({"url": it["url"], "title": _strip_html(it["title"]), "text": text,
                        "source": src.get("name", src["url"])})
    return out


def collect_wikipedia_featured(src: dict, fetch: Fetch, day: dt.date) -> list[dict]:
    lang = src.get("lang", "en")
    url = f"https://api.wikimedia.org/feed/v1/wikipedia/{lang}/featured/{day:%Y/%m/%d}"
    data = json.loads(fetch(url))
    out = []

    def page(p: dict, why: str) -> None:
        text = p.get("extract") or ""
        if text:
            out.append({"url": p.get("content_urls", {}).get("desktop", {}).get("page", ""),
                        "title": p.get("titles", {}).get("normalized", p.get("title", "")),
                        "text": text, "source": f"Wikipedia ({why})"})

    if "tfa" in data:
        page(data["tfa"], "featured article")
    for a in data.get("mostread", {}).get("articles", [])[:20]:
        page(a, "most read")
    for n in data.get("news", []):
        story = _strip_html(n.get("story", ""))
        for a in n.get("links", [])[:3]:
            page(a, "in the news")
        if story:
            out.append({"url": "", "title": "In the news", "text": story, "source": "Wikipedia (in the news)"})
    return out


# ---------------------------------------------------------------------- the daily run
def run_daily(sources: list[dict], index: SearchIndex, fetch: Fetch | None = None, day: dt.date | None = None,
              keep_days: int = 60, log=print, frontier=None, budget: int = 1000) -> dict:
    """Collect from all sources, then (with a frontier) re-check the `budget` most overdue
    tracked pages. Old index days beyond keep_days are dropped."""
    fetch = fetch or PoliteFetcher()
    day = day or dt.date.today()
    stats = {"added": 0, "errors": 0}
    for src in sources:
        name = src.get("name", src.get("url", src["type"]))
        try:
            if src["type"] == "sitemap":
                if frontier is None:
                    raise ValueError("sitemap sources need a frontier (--frontier)")
                n = add_sitemap(src["url"], frontier, fetch, source=name)
                log(f"{name}: {n} new pages to track")
                continue
            if src["type"] == "rss":
                docs = collect_rss(src, fetch)
            elif src["type"] == "wikipedia_featured":
                docs = collect_wikipedia_featured(src, fetch, day)
            else:
                raise ValueError(f"unknown source type {src['type']!r}")
            for d in docs:
                d["text"], _ = redact_pii(d["text"])
            n = index.add(docs, day=day.isoformat())
            stats["added"] += n
            log(f"{name}: {len(docs)} items, {n} new")
        except Exception as e:  # one bad source must not stop the others
            stats["errors"] += 1
            log(f"{name}: failed ({e})")

    if frontier is not None and hasattr(fetch, "conditional"):
        from .frontier import crawl

        c = crawl(frontier, index, fetch, budget=budget, day=day.isoformat(), log=log)
        stats["added"] += c["changed"]
        stats["crawl"] = c
        log(f"re-checked {c['checked']} tracked pages: {c['changed']} new or changed, "
            f"{c['not_modified'] + c['unchanged']} unchanged, {c['errors']} errors")

    cutoff = day - dt.timedelta(days=keep_days)
    dropped_urls = set()
    for f in Path(index.dir).glob("docs-*.jsonl"):
        try:
            old = dt.date.fromisoformat(f.stem[5:]) < cutoff
        except ValueError:
            continue
        if old:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    dropped_urls.add(json.loads(line).get("url", ""))
            f.unlink()
    if frontier is not None and dropped_urls:
        # A tracked page whose only copy was just dropped must be downloaded again in
        # full, or it would vanish from search while staying "unchanged".
        for url in dropped_urls & set(frontier.pages):
            p = frontier.pages[url]
            p.etag = p.last_modified = p.content_hash = ""
            p.next_due = 0.0
        frontier.save()
    return stats


def add_sitemap(url: str, frontier, fetch: Fetch, source: str = "", max_sitemaps: int = 20) -> int:
    """Add the pages of a sitemap (following a sitemap index) to the frontier."""
    todo, done, added = [url], 0, 0
    while todo and done < max_sitemaps:
        n, nested = frontier.add_from_sitemap(fetch(todo.pop(0)), source=source)
        added += n
        todo += nested
        done += 1
    frontier.save()
    return added
