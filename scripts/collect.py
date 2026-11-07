"""Daily collection into the search index. Meant to run once a day on the VPS (cron).

    python scripts/collect.py --sources configs/sources.json --index C:/llm-data/search-index \
        --frontier C:/llm-data/frontier.json --budget 2000

Each run:
1. reads every feed in the sources file (RSS/Atom, Wikipedia's daily feed),
2. adds pages from any "sitemap" sources to the frontier (the list of tracked pages),
3. re-checks the `--budget` most overdue tracked pages with conditional requests: pages
   that change get checked more often, pages that don't, less often,
4. drops index days older than --keep-days.

Example crontab line on the Linux VPS (runs at 06:00 every day):
    0 6 * * * cd /home/llm/llm && python3 scripts/collect.py --sources configs/sources.json --index /home/llm/search-index --frontier /home/llm/frontier.json >> /home/llm/collect.log 2>&1

Search from the command line:
    python scripts/collect.py --index C:/llm-data/search-index --search "new language models"
    python scripts/collect.py --wikipedia "James Webb Space Telescope"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.retrieval.collect import PoliteFetcher, run_daily  # noqa: E402
from aio.retrieval.frontier import Frontier  # noqa: E402
from aio.retrieval.index import SearchIndex, search_tool  # noqa: E402
from aio.retrieval.live import wikipedia_search_tool  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", help="search index folder")
    ap.add_argument("--sources", help="JSON list of sources (see configs/sources.json)")
    ap.add_argument("--frontier", help="JSON file tracking pages to re-check (enables sitemaps and re-checks)")
    ap.add_argument("--budget", type=int, default=1000, help="tracked pages to re-check per run")
    ap.add_argument("--keep-days", type=int, default=60)
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between requests to the same site")
    ap.add_argument("--search", help="search the local index instead of collecting")
    ap.add_argument("--wikipedia", help="search Wikipedia live instead of collecting")
    args = ap.parse_args()

    if args.wikipedia:
        print(wikipedia_search_tool()(args.wikipedia))
        return
    if not args.index:
        sys.exit("--index is required")
    index = SearchIndex(args.index)
    if args.search:
        print(search_tool(index, k=5)(args.search))
        return
    if not args.sources:
        sys.exit("--sources is required to collect")
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    frontier = Frontier.load(args.frontier) if args.frontier else None
    stats = run_daily(sources, index, PoliteFetcher(delay=args.delay), keep_days=args.keep_days,
                      frontier=frontier, budget=args.budget)
    tracked = f"; tracking {len(frontier.pages):,} pages" if frontier else ""
    print(f"added {stats['added']} new or changed documents ({stats['errors']} sources failed); "
          f"index has {len(index):,}{tracked}")


if __name__ == "__main__":
    main()
