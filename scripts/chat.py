"""Chat with a trained (SFT or DPO) model in the terminal.

    python scripts/chat.py --run C:/llm-data/runs/laptop-tools --tokenizer C:/llm-data/tokenizer-8k.json \
        --tools --index C:/llm-data/search-index --memory C:/llm-data/memory --feedback C:/llm-data/feedback.jsonl

Commands while chatting:
    /good             the last answer was good (saved to the feedback log)
    /bad              the last answer was bad
    /fix <answer>     the last answer was wrong; this is the right one. It is saved to the
                      log, and added to memory so similar questions get it as a note at once
    /reset            start a new conversation
    /quit             exit

--once "question" answers one message and exits (for scripts).
"""

import argparse
import json
import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.agent import DEFAULT_TOOLS, run as agent_run  # noqa: E402
from aio.memory import FeedbackLog, Memory  # noqa: E402
from aio.model import init_params  # noqa: E402
from aio.retrieval import SearchIndex, search_tool  # noqa: E402
from aio.tokenizer import BPETokenizer  # noqa: E402
from aio.train import TrainConfig, checkpoint  # noqa: E402
from aio.train.optim import init_opt_state  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training out_dir of an SFT/DPO run")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--system", default="", help="optional system message")
    ap.add_argument("--tools", action="store_true", help="let the model call the calculator (and search, with --index)")
    ap.add_argument("--index", help="search index folder (from scripts/collect.py) for the search tool")
    ap.add_argument("--memory", help="folder for remembered corrections")
    ap.add_argument("--feedback", help="JSONL file for /good /bad /fix")
    ap.add_argument("--max-new", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--once", help="answer this one message and exit")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = TrainConfig.from_dict(json.loads((run / "config.json").read_text(encoding="utf-8")))
    step = checkpoint.latest_step(run / "checkpoints")
    if step is None:
        sys.exit(f"no checkpoints in {run / 'checkpoints'}")
    p0 = init_params(cfg.model, jax.random.PRNGKey(0))
    state, _ = checkpoint.restore(run / "checkpoints", step, {"params": p0, "opt": init_opt_state(p0)})
    params = state["params"]
    tok = BPETokenizer.load(args.tokenizer)

    tools = {}
    if args.tools:
        tools = dict(DEFAULT_TOOLS)
        if args.index:
            tools["search"] = search_tool(SearchIndex(args.index))
    memory = Memory(args.memory) if args.memory else None
    feedback = FeedbackLog(args.feedback) if args.feedback else None
    key = jax.random.PRNGKey(0)

    system = [{"role": "system", "content": args.system}] if args.system else []
    history: list[dict] = []
    last = None  # (conversation before the reply, reply text)

    def reply(text: str) -> list[dict]:
        nonlocal key, last
        history.append({"role": "user", "content": text})
        note = memory.note(text) if memory else None
        context = system + ([note] if note else []) + history
        key, sub = jax.random.split(key)
        new = agent_run(params, cfg.model, tok, context, tools, sub, max_new=args.max_new,
                        temperature=args.temperature)
        last = (list(history), new[-1]["content"])
        history.extend(new)
        return new

    def show(new: list[dict]) -> None:
        for m in new:
            if m.get("tool_call"):
                print(f"  [tool] {m['tool_call']['name']}({json.dumps(m['tool_call']['arguments'])})")
            elif m["role"] == "tool":
                print(f"  [result] {m['content'][:200]}")
            else:
                print(f"model> {m['content']}")

    if args.once:
        show(reply(args.once))
        return
    print(f"model step {step}. Commands: /good /bad /fix <answer> /reset /quit")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/quit":
            break
        if text == "/reset":
            history.clear()
            last = None
            print("(new conversation)")
        elif text in ("/good", "/bad") or text.startswith("/fix"):
            if last is None:
                print("(nothing to rate yet)")
                continue
            msgs, answer = last
            correction = text[4:].strip() if text.startswith("/fix") else None
            if text.startswith("/fix") and not correction:
                print("(usage: /fix <the right answer>)")
                continue
            if feedback:
                feedback.record(msgs, answer, "good" if text == "/good" else "bad", correction)
            if correction and memory:
                question = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
                memory.add(question, correction)
            print("(saved" + (" to memory" if correction and memory else "") + ")" if feedback or memory
                  else "(not saved: start with --feedback and/or --memory)")
        elif text:
            show(reply(text))


if __name__ == "__main__":
    main()
