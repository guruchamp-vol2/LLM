"""How parameters and optimizer state are laid out across devices.

"dp"   (data parallel): every device holds a full copy of the weights; the batch is split.
"fsdp" (fully sharded data parallel): each weight is also split across devices along one
       dimension, so per-device memory for weights and Adam state falls by the device
       count. XLA gathers each weight when it is used and scatters the gradients back.

Both use one mesh axis, "data". The batch is always split along it.
"""

from __future__ import annotations

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def make_mesh(devices=None) -> Mesh:
    return Mesh(np.array(devices if devices is not None else jax.devices()), ("data",))


def _fsdp_spec(path, shape: tuple[int, ...], n: int) -> P:
    """Split the largest dimension that divides evenly by n. Stacked layer weights
    (first dim = layer) never split the layer dim, because lax.scan walks along it."""
    stacked = any(getattr(k, "key", None) == "layers" for k in path)
    dims = [i for i in range(1 if stacked else 0, len(shape)) if shape[i] % n == 0]
    if not dims or n == 1:
        return P()
    best = max(dims, key=lambda i: shape[i])
    spec = [None] * len(shape)
    spec[best] = "data"
    return P(*spec)


def state_shardings(state, mesh: Mesh, mode: str = "dp"):
    """A pytree of NamedShardings matching `state` (params, or {"params", "opt"})."""
    if mode not in ("dp", "fsdp"):
        raise ValueError(f"unknown sharding mode {mode!r}")
    n = mesh.devices.size

    def one(path, x):
        if mode == "dp" or np.ndim(x) == 0:
            return NamedSharding(mesh, P())
        return NamedSharding(mesh, _fsdp_spec(path, np.shape(x), n))

    return jax.tree_util.tree_map_with_path(one, state)


def batch_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P("data"))
