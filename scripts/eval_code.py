"""Grade a trained model on coding benchmarks by running its code (HumanEval, MBPP).

    python scripts/eval_code.py --run C:/llm-data/runs/laptop-tiny --tokenizer C:/llm-data/tokenizer-8k.json \
        --tasks humaneval mbpp --samples 1 --limit 20

    # pass@1 and pass@10 the standard way: 20 samples per problem at temperature 0.8
    python scripts/eval_code.py ... --samples 20 --temperature 0.8 --ks 1 10

    # chat model: prompt goes in a user message, first code block of the answer is graded
    python scripts/eval_code.py ... --chat

    # check the grader itself: the reference solutions must all pass
    python scripts/eval_code.py --tasks humaneval mbpp --check-references

The model's code runs in a separate process (see aio/agent/sandbox.py), which is NOT a
full sandbox. Run it on Kaggle/TPU machines or a throwaway container, not on a machine
with files you care about, once models are big enough to write code that does things.
Results go to <run>/eval_code.json.
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.eval.code import STOP, URLS, check, evaluate, extract_code, load, truncate  # noqa: E402


def fetch(task: str, cache: Path) -> Path:
    path = cache / f"{task}.parquet"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(URLS[task], tmp)
        tmp.replace(path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--tokenizer")
    ap.add_argument("--tasks", nargs="+", default=["humaneval", "mbpp"], choices=sorted(URLS))
    ap.add_argument("--limit", type=int, default=0, help="first N problems per task (0 = all)")
    ap.add_argument("--samples", type=int, default=1, help="programs sampled per problem")
    ap.add_argument("--ks", type=int, nargs="+", default=[1])
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (use with --samples 1)")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--chat", action="store_true", help="the run is a chat model")
    ap.add_argument("--check-references", action="store_true", help="grade the reference solutions instead")
    ap.add_argument("--cache", default="C:/llm-data/evals")
    args = ap.parse_args()

    if args.check_references:
        for task in args.tasks:
            probs = load(task, str(fetch(task, Path(args.cache))))[: args.limit or None]
            full = task == "mbpp"  # MBPP references are whole programs
            ok = sum(check(p, p.solution, args.timeout, full_program=full) for p in probs)
            print(f"{task}: {ok}/{len(probs)} reference solutions pass")
        return
    if not (args.run and args.tokenizer):
        sys.exit("--run and --tokenizer are required")

    import jax

    from aio.model import init_params
    from aio.model.generate import generate_many
    from aio.post.chat import render
    from aio.tokenizer import BPETokenizer
    from aio.train import TrainConfig, checkpoint
    from aio.train.optim import init_opt_state

    run = Path(args.run)
    cfg = TrainConfig.from_dict(json.loads((run / "config.json").read_text(encoding="utf-8")))
    step = checkpoint.latest_step(run / "checkpoints")
    if step is None:
        sys.exit(f"no checkpoints in {run / 'checkpoints'}")
    p0 = init_params(cfg.model, jax.random.PRNGKey(0))
    state, _ = checkpoint.restore(run / "checkpoints", step, {"params": p0, "opt": init_opt_state(p0)})
    params = state["params"]
    tok = BPETokenizer.load(args.tokenizer)
    sp = tok.special_tokens
    key = [jax.random.PRNGKey(0)]
    temp = args.temperature if args.samples > 1 or args.temperature > 0 else 0.0

    def sample(problem, n):
        if args.chat:
            msgs = [{"role": "user", "content": "Complete this Python code. Answer with one code block.\n\n"
                     + problem.prompt}]
            ids = render(tok, msgs, add_generation_prompt=True)[0]
            stop_ids, stop_check = (sp["<|end|>"], sp["<|endoftext|>"]), None
        else:
            ids = [sp["<|endoftext|>"]] + tok.encode_ordinary(problem.prompt)
            stop_ids = (sp["<|endoftext|>"],)
            stop_check = lambda t: any(s in tok.decode(t) for s in STOP)  # noqa: E731
        key[0], sub = jax.random.split(key[0])
        outs = generate_many(params, cfg.model, ids[-(cfg.model.max_seq_len - 64):], n, args.max_new, sub,
                             temperature=temp, top_k=args.top_k, stop_ids=stop_ids, stop_check=stop_check)
        texts = [tok.decode(o) for o in outs]
        if args.chat:  # whole program from the answer; keep the prompt's imports
            return [problem.prompt.split("def ")[0] + extract_code(t) for t in texts]
        return [truncate(t) for t in texts]

    results = {"run": str(run), "step": step, "samples": args.samples, "temperature": temp, "chat": args.chat}
    for task in args.tasks:
        probs = load(task, str(fetch(task, Path(args.cache))))[: args.limit or None]
        t = time.time()
        r = evaluate(probs, sample, n=args.samples, ks=tuple(args.ks), timeout=args.timeout,
                     full_program=args.chat, log=print)
        r["seconds"] = round(time.time() - t, 1)
        results[task] = r
        scores = "  ".join(f"{k} {v:.3f}" for k, v in r.items() if k.startswith("pass@"))
        print(f"{task:>10}: {scores}  (n={r['n_problems']}, {r['seconds']} s)")
    (run / "eval_code.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
