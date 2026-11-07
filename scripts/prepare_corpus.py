"""Download and clean pretraining sources into JSONL shards (input for tokenize_data.py).

    # GitHub code (codeparrot/github-code-clean, permissive licences only), 4 shards:
    python scripts/prepare_corpus.py --source github-code --shards 0-3 --out C:/llm-data/clean-code/github

    # OpenCoder's high-quality annealing corpus (algorithmic code, synthetic snippets, QA):
    python scripts/prepare_corpus.py --source opc-annealing --out C:/llm-data/clean-code/opc-annealing

    # Educational web text (FineWeb-Edu sample-10BT, 14 files of ~0.7M pages) and maths pages:
    python scripts/prepare_corpus.py --source fineweb-edu --shards 0-1 --out C:/llm-data/clean-edu
    python scripts/prepare_corpus.py --source finemath --shards 0-7 --out C:/llm-data/clean-math

    # Parquet files you already downloaded (cleaned the same way as their source):
    python scripts/prepare_corpus.py --source fineweb-edu --local "C:/llm-data/raw-edu/*.parquet" --out ...

Downloads one parquet file at a time into --tmp, cleans it, writes shards to --out, and
deletes the download, so disk use stays small. Finished input files are skipped on
re-runs (a `.done` marker per input), so it can be stopped and restarted.

Every document goes through:
- filters: for code, licence + quality heuristics (aio/data/code.py); text sources are
  already quality-filtered by their authors, so only a minimum length;
- secret and PII redaction;
- exact dedup (within this run);
- decontamination against HumanEval/MBPP (aio/data/decontam.py).
"""

import argparse
import glob
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.data.code import CodeFilterConfig, clean_code, code_filter  # noqa: E402
from aio.data.decontam import Decontaminator, benchmark_texts  # noqa: E402
from aio.data.dedup import ExactDeduper, text_hash  # noqa: E402
from aio.data.io import ShardWriter  # noqa: E402
from aio.data.pii import redact_pii  # noqa: E402

HF = "https://huggingface.co/datasets"
SOURCES = {
    "github-code": {  # Apache-2.0 collection; each file keeps its own licence (we keep permissive ones)
        "files": lambda i: f"{HF}/codeparrot/github-code-clean/resolve/main/data/train-{i:05d}-of-00880.parquet",
        "n": 880, "kind": "code", "text": "code", "lang": "language", "license": "license", "path": "path",
    },
    "opc-annealing": {  # ODC-By. Stored as JSON lines; we read the Hub's parquet copy.
        "parquet_api": "OpenCoder-LLM/opc-annealing-corpus",
        "kind": "code", "text": "text", "lang": "lang",
    },
    "fineweb-edu": {  # ODC-By; pages graded 3+ for educational value by the FineWeb team
        "files": lambda i: f"{HF}/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/{i:03d}_00000.parquet",
        "n": 14, "kind": "text", "text": "text", "extra": ["url", "score"],
    },
    "finemath": {  # ODC-By; maths pages graded 4+ (finemath-4plus)
        "files": lambda i: f"{HF}/HuggingFaceTB/finemath/resolve/main/finemath-4plus/train-{i:05d}-of-00064.parquet",
        "n": 64, "kind": "text", "text": "text", "extra": ["url", "score"],
    },
}

# Languages we keep from GitHub (the ones people ask coding assistants about most).
LANGS = {"Python", "JavaScript", "TypeScript", "Java", "C", "C++", "C#", "GO", "Rust", "Shell", "SQL", "PHP",
         "Ruby", "HTML", "CSS", "Markdown", "Dockerfile", "Makefile", "Lua", "Scala", "Haskell", "Julia",
         "PowerShell", "TeX"}


def parquet_urls(src: dict, shards: str | None) -> list[str]:
    if "files" in src:
        lo, hi = (int(x) for x in (shards or f"0-{src['n'] - 1}").split("-"))
        return [src["files"](i) for i in range(lo, hi + 1)]
    # The Hub keeps a parquet copy of every dataset: {config: {split: [urls]}}.
    api = f"https://huggingface.co/api/datasets/{src['parquet_api']}/parquet"
    with urllib.request.urlopen(api, timeout=60) as r:
        listing = json.load(r)
    urls = [u for splits in listing.values() for us in splits.values() for u in us]
    if not urls:
        sys.exit(f"no parquet files listed at {api}")
    return sorted(urls)


def input_tag(url: str) -> str:
    """A unique, file-name-safe name per input file (used for shard names and .done markers)."""
    for marker in ("/resolve/main/", "/parquet/"):
        if marker in url:
            return url.split(marker, 1)[1].replace("/", "_").removesuffix(".parquet")
    return url.rsplit("/", 1)[1].removesuffix(".parquet")


def download(url: str, tmp: Path, tag: str) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / f"{tag}.parquet"
    part = path.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "aio-llm-data/0.1"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            part.replace(path)
            return path
        except Exception as e:  # flaky network: retry with back-off
            print(f"  download failed ({e}), retry {attempt + 1}/5", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"could not download {url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--shards", help="input file range, e.g. 0-3 (github-code, fineweb-edu, finemath)")
    ap.add_argument("--local", help="glob of already-downloaded parquet files to clean instead")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default="C:/llm-data/tmp-download")
    ap.add_argument("--evals", default="C:/llm-data/evals", help="benchmark files for decontamination")
    ap.add_argument("--min-chars", type=int, default=200, help="text sources: drop shorter documents")
    ap.add_argument("--all-licenses", action="store_true", help="keep non-permissive licences too")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    src = SOURCES[args.source]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    decon = Decontaminator(benchmark_texts(args.evals))
    if not len(decon):
        sys.exit(f"no benchmark files in {args.evals}: run scripts/eval_code.py --check-references once first")
    lic_col = src.get("license")
    cfg = CodeFilterConfig(licenses=None if args.all_licenses or not lic_col else CodeFilterConfig().licenses)
    dedup = ExactDeduper()

    inputs = sorted(glob.glob(args.local)) if args.local else parquet_urls(src, args.shards)
    for url in inputs:
        url = url.replace("\\", "/")
        tag = input_tag(url)
        name = tag
        done = out / f"{tag}.done"
        if done.exists():
            continue
        t0 = time.time()
        local = Path(url) if args.local else download(url, Path(args.tmp), tag)
        stats: Counter = Counter()
        cols = [c for c in (src["text"], src.get("lang"), lic_col, src.get("path")) if c] + src.get("extra", [])
        with ShardWriter(out, prefix=tag, docs_per_shard=100_000) as w:
            for batch in pq.ParquetFile(local).iter_batches(batch_size=5000, columns=cols):
                for row in batch.to_pylist():
                    stats["in"] += 1
                    text = row[src["text"]] or ""
                    lang = (row.get(src["lang"]) if src.get("lang") else "") or ""
                    if src["kind"] == "code":
                        if lic_col and lang not in LANGS:
                            stats["drop_language"] += 1
                            continue
                        why = code_filter(text, lang, row.get(lic_col, "") if lic_col else "", cfg)
                        if why:
                            stats["drop_" + why] += 1
                            continue
                        text, red = clean_code(text)
                    else:
                        text = text.strip()
                        if len(text) < args.min_chars:
                            stats["drop_too_short"] += 1
                            continue
                        text, n_pii = redact_pii(text)
                        red = {"secrets": 0, "pii": n_pii}
                    if dedup.is_dup(text_hash(text)):
                        stats["drop_duplicate"] += 1
                        continue
                    if decon.contaminated(text):
                        stats["drop_benchmark"] += 1
                        continue
                    stats["secrets"] += red["secrets"]
                    stats["pii"] += red["pii"]
                    stats["kept"] += 1
                    stats["chars"] += len(text)
                    doc = {"text": text, "source": args.source}
                    if lang:
                        doc["lang"] = lang
                    if src.get("path"):
                        doc["path"] = row.get(src["path"], "")
                    for k in src.get("extra", []):
                        doc[k] = row.get(k)
                    w.write(doc)
        if not args.local:
            local.unlink()
        (out / f"{tag}.stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        done.write_text("ok", encoding="utf-8")
        print(f"{name}: kept {stats['kept']:,}/{stats['in']:,} documents, {stats['chars'] / 1e6:.0f}M chars, "
              f"{stats['drop_benchmark']} benchmark copies dropped, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
