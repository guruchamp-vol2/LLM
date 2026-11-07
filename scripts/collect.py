"""Daily collection into the search index. Meant to run once a day on the VPS (cron).

    python scripts/collect.py --sources configs/sources.json --index C:/llm-data/search-index

Example crontab line on the Linux VPS (runs at 06:00 every day):
    0 6 * * * cd /home/llm/llm && python3 scripts/collect.py --sources configs/sources.json --index /home/llm/search-index >> /home/llm/collect.log 2>&1

Search it from the command line:
    python scripts/collect.py --index C:/llm-data/search-index --search "new language models"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.retrieval.collect import PoliteFetcher, run_daily  # noqa: E402
from aio.retrieval.index import SearchIndex, search_tool  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="search index folder")
    ap.add_argument("--sources", help="JSON list of sources (see configs/sources.json)")
    ap.add_argument("--keep-days", type=int, default=60)
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between requests to the same site")
    ap.add_argument("--search", help="search the index instead of collecting")
    args = ap.parse_args()

    index = SearchIndex(args.index)
    if args.search:
        print(search_tool(index, k=5)(args.search))
        return
    if not args.sources:
        sys.exit("--sources is required to collect")
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    stats = run_daily(sources, index, PoliteFetcher(delay=args.delay), keep_days=args.keep_days)
    print(f"added {stats['added']} new documents ({stats['errors']} sources failed); index has {len(index):,}")


if __name__ == "__main__":
    main()
