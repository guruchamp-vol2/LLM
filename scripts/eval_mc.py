"""Score a trained model on multiple-choice benchmarks (HellaSwag, ARC, MMLU).

    python scripts/eval_mc.py --run C:/llm-data/runs/laptop-tiny --tokenizer C:/llm-data/tokenizer-8k.json \
        --tasks hellaswag arc_easy --limit 500

Benchmark files download once into --cache (free, a few MB each). Results print and are
saved to <run>/eval_mc.json.
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.eval.mc import score_items  # noqa: E402
from aio.eval.tasks import URLS, load  # noqa: E402
from aio.model import init_params  # noqa: E402
from aio.tokenizer import BPETokenizer  # noqa: E402
from aio.train import TrainConfig, checkpoint  # noqa: E402
from aio.train.optim import init_opt_state  # noqa: E402


def fetch(task: str, cache: Path) -> Path:
    path = cache / f"{task}.parquet"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(URLS[task][0], tmp)
        tmp.replace(path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tasks", nargs="+", default=["hellaswag", "arc_easy", "arc_challenge", "mmlu"], choices=sorted(URLS))
    ap.add_argument("--limit", type=int, default=0, help="first N items per task (0 = all)")
    ap.add_argument("--cache", default="C:/llm-data/evals")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = TrainConfig.from_dict(json.loads((run / "config.json").read_text(encoding="utf-8")))
    step = checkpoint.latest_step(run / "checkpoints")
    if step is None:
        sys.exit(f"no checkpoints in {run / 'checkpoints'}")
    p0 = init_params(cfg.model, jax.random.PRNGKey(0))
    state, _ = checkpoint.restore(run / "checkpoints", step, {"params": p0, "opt": init_opt_state(p0)})
    tok = BPETokenizer.load(args.tokenizer)
    eot = tok.special_tokens["<|endoftext|>"]

    results = {"run": str(run), "step": step}
    for task in args.tasks:
        items = load(task, str(fetch(task, Path(args.cache))))
        if args.limit:
            items = items[: args.limit]
        t = time.time()
        r = score_items(state["params"], cfg.model, tok.encode_ordinary, items, prefix=[eot],
                        pad_id=tok.special_tokens["<|pad|>"])
        r["seconds"] = round(time.time() - t, 1)
        results[task] = r
        print(f"{task:>14}: acc {r['acc']:.3f}  acc_norm {r['acc_norm']:.3f}  (chance {r['chance']:.3f}, n={r['n']})")
    (run / "eval_mc.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
