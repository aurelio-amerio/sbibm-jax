# src/sbibm_jax/data/process.py
"""Joint/conditional collate reproducing GenSBI's tokenized processing.

Each scalar feature becomes a length-1 token via a trailing [..., None] so a
graph transformer can index it as a node. Normalization (optional) is applied
post-tokenization with trailing-dim stats, matching GenSBI's process_*_norm.

NumPy throughout (not jnp): grain `.map(collate)` runs in worker subprocesses
under `mp_prefetch`, and numpy arrays pickle cleanly across the process
boundary where jax device arrays are a footgun. GenSBI's hot path likewise uses
np.concatenate. The loader therefore yields numpy arrays; wrap in jnp.asarray
downstream for jax. `dtype` defaults to np.float32 (jnp.float32 IS np.float32,
so a jnp dtype passed from TaskDataset works too).
"""

import numpy as np


def _stat_array(values, dtype):
    """metadata stat (native-reduced, e.g. (1, dim)) -> trailing-dim for tokens."""
    a = np.asarray(np.array(values), dtype=dtype)
    return a[..., None]  # (1, dim) -> (1, dim, 1); (1,1,1) -> (1,1,1,1)


def make_collate(*, kind, data_kind, normalize=False, stats=None, dtype=np.float32):
    """Return a collate fn mapping a {'thetas','xs'} batch to model-ready arrays.

    kind="conditional" -> (theta, x); kind="joint" -> concat([theta, x], axis=1).
    Joint is vector-only. `stats` is the metadata stats dict (native-reduced).
    """
    if kind == "joint" and data_kind != "vector":
        raise ValueError(
            f"kind='joint' is vector-only; task data_kind={data_kind!r} supports "
            "kind='conditional' only."
        )
    if kind not in ("joint", "conditional"):
        raise ValueError(f"Unknown kind {kind!r}.")

    if normalize:
        if stats is None:
            raise ValueError("normalize=True requires stats.")
        tm = _stat_array(stats["theta_mean"], dtype)
        ts = _stat_array(stats["theta_std"], dtype)
        xm = _stat_array(stats["x_mean"], dtype)
        xs_ = _stat_array(stats["x_std"], dtype)

    def collate(batch):
        theta = np.asarray(batch["thetas"], dtype=dtype)[..., None]
        x = np.asarray(batch["xs"], dtype=dtype)[..., None]
        if normalize:
            theta = (theta - tm) / ts
            x = (x - xm) / xs_
        if kind == "conditional":
            return theta, x
        return np.concatenate((theta, x), axis=1)

    return collate
