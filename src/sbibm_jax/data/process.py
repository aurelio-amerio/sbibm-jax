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
make_collate_jax is the jnp twin for the online path (OnlineTaskDataset),
where collation runs in the main process after the pickle boundary.
"""

import numpy as np
import jax.numpy as jnp


def _stat_array(values, dtype):
    """metadata stat (native-reduced, e.g. (1, dim)) -> trailing-dim for tokens."""
    a = np.asarray(np.array(values), dtype=dtype)
    return a[..., None]  # (1, dim) -> (1, dim, 1); (1,1,1) -> (1,1,1,1)


def make_collate(
    *, kind, x_kind, theta_kind="vector", normalize=False, stats=None,
    dtype=np.float32,
):
    """Return a collate fn mapping a {'thetas','xs'} batch to model-ready arrays.

    kind="conditional" -> (theta, x); kind="joint" -> concat([theta, x], axis=1).
    Joint is vector-only (both x and theta). `stats` is the metadata stats dict
    (native-reduced).
    """
    if kind not in ("joint", "conditional"):
        raise ValueError(f"Unknown kind {kind!r}.")
    if kind == "joint" and (x_kind != "vector" or theta_kind != "vector"):
        raise ValueError(
            f"kind='joint' is vector-only; got x_kind={x_kind!r}, "
            f"theta_kind={theta_kind!r}. Use kind='conditional'."
        )

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


def _stat_array_jax(values, dtype):
    """jnp twin of _stat_array: metadata stat -> trailing-dim for tokens."""
    a = jnp.asarray(np.asarray(values), dtype=dtype)
    return a[..., None]  # (1, dim) -> (1, dim, 1); (1,1,1) -> (1,1,1,1)


def make_collate_jax(
    *, kind, x_kind, theta_kind="vector", normalize=False, stats=None,
    dtype=jnp.float32,
):
    """jnp twin of make_collate, for the online (main-process) path.

    Same semantics as make_collate, but the returned collate yields jax
    arrays. Used by OnlineTaskDataset, where collation runs in the consumer
    process after grain's pickle boundary — so jnp is safe there, unlike in
    mp_prefetch workers (see module docstring).
    """
    if kind not in ("joint", "conditional"):
        raise ValueError(f"Unknown kind {kind!r}.")
    if kind == "joint" and (x_kind != "vector" or theta_kind != "vector"):
        raise ValueError(
            f"kind='joint' is vector-only; got x_kind={x_kind!r}, "
            f"theta_kind={theta_kind!r}. Use kind='conditional'."
        )

    if normalize:
        if stats is None:
            raise ValueError("normalize=True requires stats.")
        tm = _stat_array_jax(stats["theta_mean"], dtype)
        ts = _stat_array_jax(stats["theta_std"], dtype)
        xm = _stat_array_jax(stats["x_mean"], dtype)
        xs_ = _stat_array_jax(stats["x_std"], dtype)

    def collate(batch):
        theta = jnp.asarray(batch["thetas"], dtype=dtype)[..., None]
        x = jnp.asarray(batch["xs"], dtype=dtype)[..., None]
        if normalize:
            theta = (theta - tm) / ts
            x = (x - xm) / xs_
        if kind == "conditional":
            return theta, x
        return jnp.concatenate((theta, x), axis=1)

    return collate
