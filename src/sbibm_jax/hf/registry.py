"""Data-kind registry: hf_data_kind hint -> exporter class."""

from typing import Type

from sbibm_jax.hf import config
from sbibm_jax.hf.exporter import (
    DatasetExporter,
    ImageExporter,
    TimeSeriesExporter,
    VectorExporter,
)
from sbibm_jax.tasks.task import Task

DATA_KIND_REGISTRY: dict[str, Type[DatasetExporter]] = {
    "vector": VectorExporter,
    "image": ImageExporter,
    "timeseries": TimeSeriesExporter,
}


def get_exporter(
    task: Task,
    *,
    train_size: int | None = None,
    val_size: int | None = None,
    test_size: int | None = None,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    max_factor: float = config.DEFAULT_MAX_FACTOR,
    dtype=config.DEFAULT_DTYPE,
) -> DatasetExporter:
    """Build the exporter for `task`, honouring its hf_* hint attributes.

    Hint attributes (all optional, read via getattr with safe defaults):
        hf_data_kind:        "vector" | "image" | "timeseries" (default "vector")
        hf_data_shape:       tuple[int, ...]  (default (task.dim_x,))
        hf_resample_invalid: bool             (default False)
        hf_split_sizes:      dict             (default config.DEFAULT_SPLIT_SIZES)

    Explicit train/val/test_size keyword arguments override the task hint and
    the global default. Unknown data_kind raises ValueError.
    """
    data_kind = getattr(task, "hf_data_kind", "vector")
    if data_kind not in DATA_KIND_REGISTRY:
        raise ValueError(
            f"Unknown data_kind {data_kind!r} for task {task.name!r}; "
            f"known kinds: {sorted(DATA_KIND_REGISTRY)}."
        )

    data_shape = getattr(task, "hf_data_shape", (task.dim_x,))
    resample_invalid = getattr(task, "hf_resample_invalid", False)
    task_split_sizes = getattr(task, "hf_split_sizes", config.DEFAULT_SPLIT_SIZES)

    ts = train_size if train_size is not None else task_split_sizes["train"]
    vs = val_size if val_size is not None else task_split_sizes["validation"]
    es = test_size if test_size is not None else task_split_sizes["test"]

    cls = DATA_KIND_REGISTRY[data_kind]
    kwargs = dict(
        train_size=ts,
        val_size=vs,
        test_size=es,
        chunk_size=chunk_size,
        max_factor=max_factor,
        dtype=dtype,
        resample_invalid=resample_invalid,
    )
    if cls is VectorExporter:
        return cls(task, **kwargs)
    return cls(task, data_shape=tuple(data_shape), **kwargs)
