"""Check how well a chat model uses its tools, on questions it never saw in training.

    python scripts/eval_tools.py --run C:/llm-data/runs/laptop-tools --tokenizer C:/llm-data/tokenizer-8k.json
    python scripts/eval_tools.py --run C:/llm-data/runs/laptop-search --tokenizer C:/llm-data/tokenizer-8k.json \
        --index C:/llm-data/search-index

Calculator: new random sums (a seed not used in training). Success means the model calls
the calculator with the right expression and its final answer contains the exact result.

Search (with --index): questions about held-out pages (the "test" split), which the
model never saw. Success means it searches, the page is in the results, and the answer
quotes the page's opening sentence.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.agent import DEFAULT_TOOLS, calculator, run  # noqa: E402
from aio.agent.synth import calc_conversation, search_conversations  # noqa: E402
from aio.model import init_params  # noqa: E402
from aio.retrieval import SearchIndex, search_tool  # noqa: E402
from aio.tokenizer import BPETokenizer  # noqa: E402
from aio.train import TrainConfig, checkpoint  # noqa: E402
from aio.train.optim import init_opt_state  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--index", help="search index, to also test the search tool")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--show", type=int, default=3, help="print this many examples per test")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = TrainConfig.from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    step = checkpoint.latest_step(run_dir / "checkpoints")
    p0 = init_params(cfg.model, jax.random.PRNGKey(0))
    state, _ = checkpoint.restore(run_dir / "checkpoints", step, {"params": p0, "opt": init_opt_state(p0)})
    params, tok = state["params"], BPETokenizer.load(args.tokenizer)
    tools = dict(DEFAULT_TOOLS)
    if args.index:
        index = SearchIndex(args.index)
        tools["search"] = search_tool(index, k=2, width=200)
    key = jax.random.PRNGKey(0)
    results = {"run": str(run_dir), "step": step}

    def ask(question: str):
        nonlocal key
        key, sub = jax.random.split(key)
        return run(params, cfg.model, tok, [{"role": "user", "content": question}], tools, sub, max_new=80)

    # Calculator
    rng = random.Random(4242)
    ok = called = 0
    for i in range(args.n):
        conv = calc_conversation(rng, max_digits=4)
        expr = conv[1]["tool_call"]["arguments"]["expression"]
        new = ask(conv[0]["content"])
        calls = [m["tool_call"] for m in new if m.get("tool_call")]
        called += bool(calls)
        good = any(c["name"] == "calculator" and c["arguments"].get("expression") == expr for c in calls) \
            and calculator(expr) in new[-1]["content"]
        ok += good
        if i < args.show:
            print(f"Q: {conv[0]['content']}\n   calls: {calls}\n   A: {new[-1]['content'][:150]}  [{'ok' if good else 'wrong'}]")
    results["calculator"] = {"n": args.n, "correct": ok, "called_tool": called}
    print(f"calculator: {ok}/{args.n} exactly right ({called}/{args.n} called the tool)\n")

    # Search
    if args.index:
        convs = search_conversations(index, args.n, seed=99, split="test")
        ok = called = 0
        for i, conv in enumerate(convs):
            new = ask(conv[0]["content"])
            calls = [m["tool_call"] for m in new if m.get("tool_call")]
            called += any(c["name"] == "search" for c in calls)
            want = conv[3]["content"].split(" (Source:")[0][:60]
            good = any(c["name"] == "search" for c in calls) and want in new[-1]["content"]
            ok += good
            if i < args.show:
                print(f"Q: {conv[0]['content']}\n   calls: {calls}\n   A: {new[-1]['content'][:150]}  [{'ok' if good else 'wrong'}]")
        results["search"] = {"n": len(convs), "correct": ok, "called_tool": called}
        print(f"search: {ok}/{len(convs)} answered from the right page ({called}/{len(convs)} searched)")

    # Ordinary questions should not trigger tools.
    plain = ["What is the capital of France?", "Write a short poem about the sea.", "Give me a tip for sleeping better."]
    stray = sum(any(m.get("tool_call") for m in ask(q)) for q in plain)
    results["plain_questions_using_tools"] = f"{stray}/{len(plain)}"
    print(f"ordinary questions that wrongly used a tool: {stray}/{len(plain)}")
    (run_dir / "eval_tools.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
