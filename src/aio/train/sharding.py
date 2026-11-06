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


# ---------------------------------------------------------------------- multi-host helpers
# On one machine these are plain device_put / device_get. On a TPU pod (several machines,
# each driving its own chips) every process runs the same program, and arrays are global:
# each process only holds the pieces that live on its own chips.

def is_multihost() -> bool:
    return jax.process_count() > 1


def put_global(tree, shardings):
    """Place host arrays (identical on every process) onto devices with the given shardings."""
    if not is_multihost():
        return jax.device_put(tree, shardings)

    def one(x, s):
        x = np.asarray(x)
        return jax.make_array_from_callback(x.shape, s, lambda idx: x[idx])

    return jax.tree.map(one, tree, shardings)


def put_batch(batch: np.ndarray, sharding):
    """Every process builds the same global batch (cheap: a few memory-mapped reads) and
    contributes only the rows that belong to its own devices."""
    if not is_multihost():
        return jax.device_put(batch, sharding)
    return jax.make_array_from_callback(batch.shape, sharding, lambda idx: batch[idx])


def to_host(tree):
    """Full numpy copies of (possibly sharded) arrays. All processes must call this together."""
    if not is_multihost():
        return jax.device_get(tree)
    from jax.experimental import multihost_utils

    return multihost_utils.process_allgather(tree, tiled=True)


def broadcast_from_main(tree):
    """Process 0's values, on every process (e.g. a checkpoint only process 0 can read)."""
    if not is_multihost():
        return tree
    from jax.experimental import multihost_utils

    return multihost_utils.broadcast_one_to_all(tree)


def any_process(flag: bool) -> bool:
    """True if the flag is set on any process, so all processes stop at the same step."""
    if not is_multihost():
        return flag
    from jax.experimental import multihost_utils

    return bool(np.any(multihost_utils.process_allgather(np.array(flag))))
