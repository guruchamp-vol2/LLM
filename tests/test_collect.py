import datetime as dt
import json
import sys
import urllib.robotparser
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.retrieval import SearchIndex  # noqa: E402
from aio.retrieval.collect import PoliteFetcher, parse_feed, run_daily  # noqa: E402

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>Feed</title>
<item><title>New model released</title><link>https://news.example/a</link>
<description>&lt;p&gt;A lab released a &lt;b&gt;small&lt;/b&gt; language model. Contact press@lab.example.&lt;/p&gt;</description>
<pubDate>Fri, 06 Nov 2026 08:00:00 GMT</pubDate></item>
<item><title>Second story</title><link>https://news.example/b</link><description>Plain text summary.</description></item>
</channel></rss>"""

ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>A</title>
<entry><title>Atom entry</title><link href="https://atom.example/1"/><summary>Atom summary text.</summary>
<updated>2026-11-06T00:00:00Z</updated></entry></feed>"""

WIKI = {
    "tfa": {"titles": {"normalized": "Octopus"}, "extract": "The octopus is a soft-bodied, eight-limbed mollusc.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Octopus"}}},
    "mostread": {"articles": [{"titles": {"normalized": "Comet"}, "extract": "A comet is an icy small body.",
                               "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Comet"}}}]},
    "news": [{"story": "<b>Scientists</b> announce a discovery.", "links": []}],
}


def fake_fetch(pages):
    def fetch(url):
        if url not in pages:
            raise OSError(f"404 {url}")
        return pages[url]
    return fetch


def test_parse_rss_and_atom():
    rss = parse_feed(RSS)
    assert [i["title"] for i in rss] == ["New model released", "Second story"]
    assert rss[0]["url"] == "https://news.example/a"
    atom = parse_feed(ATOM)
    assert atom == [{"url": "https://atom.example/1", "title": "Atom entry", "summary": "Atom summary text.",
                     "date": "2026-11-06T00:00:00Z"}]


def test_daily_run_indexes_redacts_and_dedups(tmp_path):
    day = dt.date(2026, 11, 6)
    pages = {"https://feed.example/rss": RSS, "https://atom.example/feed": ATOM,
             "https://api.wikimedia.org/feed/v1/wikipedia/en/featured/2026/11/06": json.dumps(WIKI)}
    sources = [{"type": "rss", "url": "https://feed.example/rss", "name": "News"},
               {"type": "rss", "url": "https://atom.example/feed"},
               {"type": "wikipedia_featured"},
               {"type": "rss", "url": "https://broken.example/rss", "name": "Broken"}]
    idx = SearchIndex(tmp_path)
    logs = []
    stats = run_daily(sources, idx, fake_fetch(pages), day=day, log=logs.append)
    assert stats == {"added": 6, "errors": 1}           # the broken feed doesn't stop the others
    assert any("Broken: failed" in line for line in logs)
    hit = idx.search("small language model")[0].doc
    assert hit["title"] == "New model released" and "<b>" not in hit["text"]
    assert "press@lab.example" not in hit["text"] and "[EMAIL]" in hit["text"]
    assert idx.search("octopus mollusc")[0].doc["source"] == "Wikipedia (featured article)"
    again = run_daily(sources, idx, fake_fetch(pages), day=day, log=lambda s: None)
    assert again["added"] == 0                          # same items tomorrow are not duplicated


def test_old_days_are_dropped(tmp_path):
    idx = SearchIndex(tmp_path)
    idx.add([{"url": "old", "text": "old news"}], day="2026-08-01")
    idx.add([{"url": "new", "text": "new news"}], day="2026-11-05")
    run_daily([], idx, fake_fetch({}), day=dt.date(2026, 11, 6), keep_days=60, log=lambda s: None)
    assert sorted(f.name for f in tmp_path.glob("docs-*.jsonl")) == ["docs-2026-11-05.jsonl"]


def test_robots_txt_is_respected():
    f = PoliteFetcher(delay=0)
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(["User-agent: *", "Disallow: /private/"])
    f.robots["https://site.example"] = rp
    assert f.allowed("https://site.example/public/page")
    assert not f.allowed("https://site.example/private/page")
    with pytest.raises(PermissionError):
        f("https://site.example/private/page")
