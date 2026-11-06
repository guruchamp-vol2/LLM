"""Generate text from a training run's newest checkpoint.

    python scripts/generate.py --run C:/llm-data/runs/laptop-tiny --tokenizer C:/llm-data/tokenizer.json \
        --prompt "The history of science"
"""

import argparse
import json
import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.model import init_params  # noqa: E402
from aio.model.generate import generate  # noqa: E402
from aio.tokenizer import BPETokenizer  # noqa: E402
from aio.train import TrainConfig, checkpoint  # noqa: E402
from aio.train.optim import init_opt_state  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training out_dir")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", action="append", required=True, help="can be given several times")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = TrainConfig.from_dict(json.loads((run / "config.json").read_text()))
    step = checkpoint.latest_step(run / "checkpoints")
    if step is None:
        sys.exit(f"no checkpoints in {run / 'checkpoints'}")
    params = init_params(cfg.model, jax.random.PRNGKey(0))
    state, _ = checkpoint.restore(run / "checkpoints", step, {"params": params, "opt": init_opt_state(params)})

    tok = BPETokenizer.load(args.tokenizer)
    eot = tok.special_tokens["<|endoftext|>"]
    key = jax.random.PRNGKey(args.seed)
    print(f"checkpoint step {step}\n")
    for prompt in args.prompt:
        key, sub = jax.random.split(key)
        ids = [eot] + tok.encode_ordinary(prompt)  # documents start right after end-of-text
        new = generate(state["params"], cfg.model, ids, args.max_new, sub, args.temperature, args.top_k, stop_id=eot)
        print(f"--- {prompt}{tok.decode([i for i in new if i != eot])}\n")


if __name__ == "__main__":
    main()
