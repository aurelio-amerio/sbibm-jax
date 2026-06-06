"""Chunked JAX generation of (theta, x) with stable per-task seeding.

Public surface:
    derive_task_keys(name, *, master_seed) -> dict[str, jax.PRNGKey]
        Stable per-task fold-in + split into independent train/val/test keys.

    iter_chunks(task, key, n, *, resample_invalid, chunk_size, dtype,
                max_factor, stats) -> Iterator[(theta_chunk, x_chunk)]
        Streams valid chunks of (theta, x_flat) totalling exactly n rows.
        Stats are written into the provided dict.

    generate_samples(task, key, n, **kwargs) -> (thetas, xs_flat, stats)
        Materializing convenience wrapper around iter_chunks (for tests).

Validity policy:
    resample_invalid=False (default): assert all rows are finite; raise loudly
        if any NaN/Inf appears. Used by analytical tasks.
    resample_invalid=True: drop non-finite (theta, x) rows and keep drawing
        until exactly n valid rows are produced; cap total raw draws at
        max_factor * n; raise if cap is exceeded. Used by ODE / PEtab tasks.
        This is pure rejection sampling — no imputation — so kept rows remain
        i.i.d. draws from the (prior ∩ {sim succeeds}) measure.
"""

import logging
import math
import zlib
from typing import Iterator, Tuple

import jax
import numpy as np

from sbibm_jax.hf import config

log = logging.getLogger(__name__)


def derive_task_keys(
    name: str,
    *,
    master_seed: int = config.DEFAULT_MASTER_SEED,
) -> dict:
    """Return stable, independent train/val/test PRNG keys for `name`.

    Stability across runs is guaranteed by zlib.crc32 (deterministic), unlike
    Python's salted hash().
    """
    master = jax.random.PRNGKey(master_seed)
    per_task = jax.random.fold_in(master, int(zlib.crc32(name.encode())))
    k_train, k_val, k_test = jax.random.split(per_task, 3)
    return {"train": k_train, "validation": k_val, "test": k_test}


def _draw_chunk(task, theta_key, sim_key, chunk_size, dtype):
    thetas = task.get_prior(theta_key, num_samples=chunk_size)
    sim = task.get_simulator(sim_key, max_calls=None)
    xs = sim(sim_key, thetas)
    thetas_np = np.asarray(thetas, dtype=dtype)
    xs_np = np.asarray(xs, dtype=dtype)
    return thetas_np, xs_np


def iter_chunks(
    task,
    key: jax.Array,
    n: int,
    *,
    resample_invalid: bool,
    chunk_size: int,
    dtype,
    max_factor: float,
    stats: dict,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Stream (theta_chunk, x_flat_chunk) totalling exactly `n` valid rows.

    Writes summary counters into `stats` once exhausted.
    """
    yielded = 0
    drawn = 0
    rejected = 0
    cap = int(math.ceil(max_factor * n)) if resample_invalid else n

    while yielded < n:
        key, sub = jax.random.split(key)
        theta_key, sim_key = jax.random.split(sub)
        size = min(chunk_size, max(n - yielded, chunk_size))

        thetas_np, xs_np = _draw_chunk(task, theta_key, sim_key, size, dtype)
        drawn += size

        finite_mask = (
            np.isfinite(thetas_np).all(axis=1) & np.isfinite(xs_np).all(axis=1)
        )
        bad = int((~finite_mask).sum())

        if not resample_invalid:
            if bad:
                raise ValueError(
                    f"Task {task.name!r} produced {bad} non-finite row(s) under "
                    "the default validity policy. Set hf_resample_invalid=True "
                    "on the task to drop them via rejection sampling."
                )
            take = min(n - yielded, thetas_np.shape[0])
            yield thetas_np[:take], xs_np[:take]
            yielded += take
        else:
            rejected += bad
            valid_t = thetas_np[finite_mask]
            valid_x = xs_np[finite_mask]
            if valid_t.shape[0] == 0:
                if drawn > cap:
                    raise ValueError(
                        f"Task {task.name!r}: rejection-sampling cap exceeded "
                        f"({drawn}/{cap} draws for n={n}); "
                        f"rejection_rate={rejected / max(drawn, 1):.3f}."
                    )
                continue
            take = min(n - yielded, valid_t.shape[0])
            yield valid_t[:take], valid_x[:take]
            yielded += take
            if drawn > cap:
                raise ValueError(
                    f"Task {task.name!r}: rejection-sampling cap exceeded "
                    f"({drawn}/{cap} draws for n={n}); "
                    f"rejection_rate={rejected / max(drawn, 1):.3f}."
                )

    stats["total_drawn"] = drawn
    stats["rejected"] = rejected
    stats["rejection_rate"] = rejected / max(drawn, 1)
    if resample_invalid and rejected:
        log.info(
            "[hf.generate] task=%s n=%d rejected=%d rate=%.4f",
            task.name, n, rejected, stats["rejection_rate"],
        )


def generate_samples(
    task,
    key: jax.Array,
    n: int,
    *,
    resample_invalid: bool = False,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    dtype=config.DEFAULT_DTYPE,
    max_factor: float = config.DEFAULT_MAX_FACTOR,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Materializing wrapper around iter_chunks.

    Returns concatenated arrays and the populated stats dict. Used by tests and
    for tasks where n is small enough to fit in memory; the streaming pipeline
    (build.py) uses iter_chunks directly to bound RAM at the chunk size.
    """
    stats: dict = {}
    chunks = list(iter_chunks(
        task, key, n,
        resample_invalid=resample_invalid,
        chunk_size=chunk_size,
        dtype=dtype,
        max_factor=max_factor,
        stats=stats,
    ))
    thetas = np.concatenate([c[0] for c in chunks], axis=0)
    xs = np.concatenate([c[1] for c in chunks], axis=0)
    return thetas, xs, stats
