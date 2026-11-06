"""Save and restore training state (any pytree of arrays plus JSON metadata).

Layout: <dir>/step_00001000/{arrays.npz, meta.json}. Each checkpoint is written to a
temporary folder and renamed when complete, so a crash mid-save never leaves a broken
"latest" checkpoint. Only the newest `keep` checkpoints are kept.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import jax
import numpy as np

_STEP_DIR = re.compile(r"^step_(\d{8})$")


def _flatten(tree) -> dict[str, np.ndarray]:
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(path): np.asarray(jax.device_get(x)) for path, x in leaves}


def save(ckpt_dir: str | Path, step: int, state, meta: dict | None = None, keep: int = 3) -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final = ckpt_dir / f"step_{step:08d}"
    tmp = ckpt_dir / f".tmp_step_{step:08d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    np.savez(tmp / "arrays.npz", **_flatten(state))
    (tmp / "meta.json").write_text(json.dumps({"step": step, **(meta or {})}, indent=2))
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    for old in list_steps(ckpt_dir)[:-keep] if keep > 0 else []:
        shutil.rmtree(ckpt_dir / f"step_{old:08d}", ignore_errors=True)
    return final


def list_steps(ckpt_dir: str | Path) -> list[int]:
    d = Path(ckpt_dir)
    if not d.exists():
        return []
    return sorted(int(m.group(1)) for p in d.iterdir() if p.is_dir() and (m := _STEP_DIR.match(p.name)))


def latest_step(ckpt_dir: str | Path) -> int | None:
    steps = list_steps(ckpt_dir)
    return steps[-1] if steps else None


def restore(ckpt_dir: str | Path, step: int, template):
    """Load a checkpoint into the structure (and dtypes) of `template`. Returns (state, meta)."""
    path = Path(ckpt_dir) / f"step_{step:08d}"
    arrays = np.load(path / "arrays.npz")
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    names = [jax.tree_util.keystr(p) for p, _ in leaves]
    missing = set(names) - set(arrays.files)
    extra = set(arrays.files) - set(names)
    if missing or extra:
        raise ValueError(f"checkpoint does not match: missing {sorted(missing)[:5]}, unexpected {sorted(extra)[:5]}")
    out = []
    for name, (_, ref) in zip(names, leaves):
        a = arrays[name]
        if a.shape != np.shape(ref):
            raise ValueError(f"{name}: checkpoint shape {a.shape} != expected {np.shape(ref)}")
        out.append(a.astype(np.asarray(ref).dtype))
    meta = json.loads((path / "meta.json").read_text())
    return jax.tree_util.tree_unflatten(treedef, out), meta
