"""build_dataset(task_name, **opts): the end-to-end pipeline entry point."""

from typing import Optional

import numpy as np
from datasets import Dataset

from sbibm_jax import get_task
from sbibm_jax.hf import config
from sbibm_jax.hf.generate import derive_task_keys, iter_chunks
from sbibm_jax.hf.reference import load_reference
from sbibm_jax.hf.registry import get_exporter
from sbibm_jax.hf.stats import StatsAccumulator, resolve_stats_axes


def _build_split(exporter, key, n: int) -> Dataset:
    """Stream (theta, x) rows of one split through Dataset.from_generator."""
    stats: dict = {}

    def row_generator():
        for theta_chunk, x_flat_chunk in iter_chunks(
            exporter.task, key, n,
            resample_invalid=exporter.resample_invalid,
            chunk_size=exporter.chunk_size,
            dtype=exporter.dtype,
            max_factor=exporter.max_factor,
            stats=stats,
        ):
            x_chunk = exporter.shape_x(x_flat_chunk)
            for i in range(theta_chunk.shape[0]):
                yield {"xs": x_chunk[i], "thetas": theta_chunk[i]}

    return Dataset.from_generator(row_generator, features=exporter.features())


def _compute_train_stats(task, exporter, train_dataset) -> dict:
    """Accumulate normalization stats by iterating the built train split.

    Cache-independent (unlike accumulating inside Dataset.from_generator's
    generator, which is skipped on a cache hit). x is stored native-shaped.
    """
    theta_axes, x_axes = resolve_stats_axes(task)
    acc = StatsAccumulator(theta_axes, x_axes)
    for batch in train_dataset.with_format("numpy").iter(
        batch_size=exporter.chunk_size
    ):
        acc.update(np.asarray(batch["thetas"]), np.asarray(batch["xs"]))
    return acc.result()


def build_dataset(
    task_name: str,
    *,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    max_factor: float = config.DEFAULT_MAX_FACTOR,
    dtype=config.DEFAULT_DTYPE,
    master_seed: int = config.DEFAULT_MASTER_SEED,
    task_kwargs: Optional[dict] = None,
) -> dict:
    """Build the train / validation / test (+ optional reference) bundle.

    Returns a dict with keys "train", "validation", "test", "reference", and
    "stats". "reference" may be None if the task ships no reference CSVs.
    "stats" contains per-feature normalization statistics (mean/std) derived by
    iterating the materialized train split (cache-independent).
    """
    task = get_task(task_name, **(task_kwargs or {}))
    exporter = get_exporter(
        task,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        chunk_size=chunk_size,
        max_factor=max_factor,
        dtype=dtype,
    )
    keys = derive_task_keys(task.name, master_seed=master_seed)
    train = _build_split(exporter, keys["train"], exporter.train_size)
    return {
        "train": train,
        "validation": _build_split(exporter, keys["validation"], exporter.val_size),
        "test": _build_split(exporter, keys["test"], exporter.test_size),
        "reference": load_reference(task, exporter),
        "stats": _compute_train_stats(task, exporter, train),
    }
