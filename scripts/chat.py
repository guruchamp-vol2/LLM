"""Chat with a trained (SFT or DPO) model in the terminal.

    python scripts/chat.py --run C:/llm-data/runs/sft-tiny --tokenizer C:/llm-data/tokenizer-8k.json

Type a message and press Enter. "/reset" starts over, "/quit" exits. With --once "question"
it answers one message and exits (handy for scripts).
"""

import argparse
import json
import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.model import init_params  # noqa: E402
from aio.model.generate import generate  # noqa: E402
from aio.post.chat import render  # noqa: E402
from aio.tokenizer import BPETokenizer  # noqa: E402
from aio.train import TrainConfig, checkpoint  # noqa: E402
from aio.train.optim import init_opt_state  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training out_dir of an SFT/DPO run")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--system", default="", help="optional system message")
    ap.add_argument("--max-new", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
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
    end = tok.special_tokens["<|end|>"]
    key = jax.random.PRNGKey(0)

    history = [{"role": "system", "content": args.system}] if args.system else []

    def reply(text: str) -> str:
        nonlocal key
        history.append({"role": "user", "content": text})
        ids, _ = render(tok, history, add_generation_prompt=True)
        ids = ids[-(cfg.model.max_seq_len - args.max_new):]  # keep the most recent context
        key, sub = jax.random.split(key)
        out = generate(params, cfg.model, ids, args.max_new, sub, args.temperature, args.top_k, stop_id=end)
        answer = tok.decode([t for t in out if t != end]).strip()
        history.append({"role": "assistant", "content": answer})
        return answer

    if args.once:
        print(reply(args.once))
        return
    print(f"model step {step}. /reset to start over, /quit to exit.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/quit":
            break
        if text == "/reset":
            history[:] = [m for m in history if m["role"] == "system"]
            print("(conversation cleared)")
            continue
        if text:
            print(f"model> {reply(text)}")


if __name__ == "__main__":
    main()
