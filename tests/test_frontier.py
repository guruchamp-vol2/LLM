import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.retrieval import SearchIndex  # noqa: E402
from aio.retrieval.collect import run_daily  # noqa: E402
from aio.retrieval.frontier import DAY, Frontier, crawl  # noqa: E402
from aio.retrieval.live import wikipedia_search_tool  # noqa: E402

SITEMAP_INDEX = """<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<sitemap><loc>https://site.example/sitemap-1.xml</loc></sitemap></sitemapindex>"""
SITEMAP = """<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url><loc>https://site.example/news</loc><changefreq>daily</changefreq></url>
<url><loc>https://site.example/about</loc><changefreq>yearly</changefreq></url>
<url><loc>https://site.example/post</loc></url></urlset>"""


def page(title, body):
    return f"<html><head><title>{title}</title></head><body><p>{body}</p></body></html>"


class FakeSite:
    """Serves pages with ETags, answers 304 when nothing changed, counts full downloads."""

    def __init__(self, pages):
        self.pages = dict(pages)
        self.full = 0

    def __call__(self, url):
        return self.pages[url]

    def conditional(self, url, etag="", last_modified=""):
        if url not in self.pages:
            raise OSError("404")
        tag = '"%d"' % hash(self.pages[url])
        if etag == tag:
            return 304, "", etag, last_modified
        self.full += 1
        return 200, self.pages[url], tag, ""


def test_sitemap_discovery_and_intervals(tmp_path):
    f = Frontier(tmp_path / "f.json")
    added, nested = f.add_from_sitemap(SITEMAP_INDEX)
    assert (added, nested) == (0, ["https://site.example/sitemap-1.xml"])
    added, _ = f.add_from_sitemap(SITEMAP, source="site")
    assert added == 3 and f.add_from_sitemap(SITEMAP)[0] == 0  # no duplicates
    assert f.pages["https://site.example/news"].interval == DAY
    assert f.pages["https://site.example/about"].interval == f.max_interval  # "yearly", capped at 60 days
    assert f.pages["https://site.example/post"].interval == 7 * DAY   # no changefreq: weekly


def test_adaptive_schedule(tmp_path):
    f = Frontier(tmp_path / "f.json")
    f.add("u", interval=4 * DAY)
    p = f.pages["u"]
    f.record(p, changed=True, now=0)             # first visit: interval unchanged
    assert p.interval == 4 * DAY and p.next_due == 4 * DAY
    f.record(p, changed=True, now=0)             # changed again: check twice as often
    assert p.interval == 2 * DAY
    f.record(p, changed=False, now=0)            # unchanged: back off
    f.record(p, changed=False, now=0)
    assert p.interval == 8 * DAY
    for _ in range(20):
        f.record(p, changed=False, now=0)
    assert p.interval == f.max_interval          # capped
    assert f.due(now=0) == [] and f.due(now=10**9) == [p]


def test_crawl_uses_304_and_replaces_changed_pages(tmp_path):
    site = FakeSite({"https://s/a": page("A", "first version of page a"),
                     "https://s/b": page("B", "page b never changes")})
    f = Frontier(tmp_path / "f.json")
    for u in site.pages:
        f.add(u, interval=DAY)
    idx = SearchIndex(tmp_path / "idx")
    s1 = crawl(f, idx, site, now=0, day="2026-11-06", log=lambda m: None)
    assert s1["changed"] == 2 and site.full == 2

    site.pages["https://s/a"] = page("A", "second version of page a with news")
    s2 = crawl(f, idx, site, now=2 * DAY, day="2026-11-08", log=lambda m: None)
    assert s2 == {"checked": 2, "not_modified": 1, "unchanged": 0, "changed": 1, "errors": 0}
    assert site.full == 3                         # b answered 304: not downloaded again
    hits = [h for h in idx.search("version of page a") if h.doc["url"] == "https://s/a"]
    assert len(hits) == 1 and "second version" in hits[0].doc["text"]  # only the newest version
    assert len(idx) == 2
    assert f.pages["https://s/a"].interval < f.pages["https://s/b"].interval

    again = Frontier.load(tmp_path / "f.json")    # state survives restarts
    assert again.pages["https://s/b"].etag and again.pages["https://s/a"].changes == 1


def test_errors_back_off(tmp_path):
    f = Frontier(tmp_path / "f.json")
    f.add("https://s/missing", interval=DAY)
    s = crawl(f, SearchIndex(tmp_path / "idx"), FakeSite({}), now=0, log=lambda m: None)
    assert s["errors"] == 1 and f.pages["https://s/missing"].errors == 1


def test_daily_run_with_sitemap_and_dropped_days(tmp_path):
    site = FakeSite({"https://site.example/sitemap.xml": SITEMAP_INDEX,
                     "https://site.example/sitemap-1.xml": SITEMAP,
                     "https://site.example/news": page("News", "today's news"),
                     "https://site.example/about": page("About", "about this site"),
                     "https://site.example/post": page("Post", "a blog post")})
    idx = SearchIndex(tmp_path / "idx")
    f = Frontier(tmp_path / "f.json")
    src = [{"type": "sitemap", "url": "https://site.example/sitemap.xml", "name": "Site"}]
    stats = run_daily(src, idx, site, day=dt.date(2026, 8, 1), frontier=f, log=lambda m: None)
    assert stats["crawl"]["changed"] == 3 and len(f.pages) == 3

    # 90 days later the August file is dropped; tracked pages from it are queued for a full
    # download again instead of silently vanishing from search.
    stats = run_daily([], idx, site, day=dt.date(2026, 10, 30), frontier=f, budget=0, log=lambda m: None)
    assert not list((tmp_path / "idx").glob("docs-2026-08-01.jsonl"))
    assert all(p.etag == "" and p.next_due == 0 for p in f.pages.values())


def test_wikipedia_tool():
    resp = {"pages": [{"key": "James_Webb_Space_Telescope", "title": "James Webb Space Telescope",
                       "description": "Space telescope launched in 2021",
                       "excerpt": "The <span class=\"searchmatch\">James</span> Webb Space Telescope is"}]}
    seen = []

    def fetch(url):
        seen.append(url)
        return json.dumps(resp)

    out = wikipedia_search_tool(fetch)("James Webb telescope")
    assert "q=James%20Webb%20telescope" in seen[0]
    assert out == ("[1] James Webb Space Telescope (https://en.wikipedia.org/wiki/James_Webb_Space_Telescope): "
                   "Space telescope launched in 2021; The James Webb Space Telescope is")
    assert wikipedia_search_tool(lambda u: json.dumps({"pages": []}))("zzz") == "no results"
    assert wikipedia_search_tool(fetch)("  ") == "error: empty query"
