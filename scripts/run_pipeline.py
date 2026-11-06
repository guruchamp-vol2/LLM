"""Clean raw documents into training-ready shards.

Input: JSONL (or .jsonl.gz) files, one document per line, with "text" or "html",
and optionally "url".

    python scripts/run_pipeline.py --input "raw/*.jsonl.gz" --output data/clean
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.data import Pipeline, PipelineConfig  # noqa: E402
from aio.data.filters import load_list  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="glob of input JSONL files")
    ap.add_argument("--output", required=True)
    ap.add_argument("--multilingual", action="store_true", help="disable the English stop-word check")
    ap.add_argument("--blocked-domains", help="file with one blocked domain per line")
    ap.add_argument("--no-near-dedup", action="store_true")
    ap.add_argument("--quality-model", help="trained quality scorer (.npz) from scripts/train_quality.py")
    ap.add_argument("--min-quality", type=float, help="drop documents scoring below this")
    ap.add_argument("--domain-model", help="trained topic classifier (.npz) from scripts/train_domain.py")
    args = ap.parse_args()

    inputs = sorted(glob.glob(args.input))
    if not inputs:
        sys.exit(f"no files match {args.input}")

    cfg = PipelineConfig(near_dedup=not args.no_near_dedup, quality_model=args.quality_model,
                         min_quality=args.min_quality, domain_model=args.domain_model)
    if args.multilingual:
        cfg.quality.require_stop_words = False
    if args.blocked_domains:
        cfg.safety.blocked_domains |= load_list(args.blocked_domains)

    t = time.time()
    stats = Pipeline(cfg).run(inputs, args.output)
    stats["seconds"] = round(time.time() - t, 1)
    print(json.dumps(dict(sorted(stats.items())), indent=2))
    Path(args.output, "stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
