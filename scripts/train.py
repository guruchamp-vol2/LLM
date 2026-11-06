"""Pretrain a model. Resumes automatically if out_dir already has checkpoints.

    python scripts/train.py --config configs/laptop-tiny.json

Any config field can be overridden: --set steps=2000 --set model.d_model=256
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aio.train import TrainConfig, train  # noqa: E402


def _apply(d: dict, key: str, raw: str) -> None:
    *parents, last = key.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    try:
        d[last] = json.loads(raw)
    except json.JSONDecodeError:
        d[last] = raw  # plain string


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON file with TrainConfig fields")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a field")
    args = ap.parse_args()

    d = json.loads(Path(args.config).read_text())
    for item in args.set:
        k, _, v = item.partition("=")
        _apply(d, k, v)
    cfg = TrainConfig.from_dict(d)
    res = train(cfg)
    res.pop("state")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
