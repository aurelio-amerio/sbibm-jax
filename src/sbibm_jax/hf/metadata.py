"""Auto-generate metadata.json from Task attributes."""

import json
from pathlib import Path
from typing import Iterable, Optional

from sbibm_jax import get_task
from sbibm_jax.hf.reference import load_reference
from sbibm_jax.hf.registry import get_exporter


def make_metadata(
    task_names: Iterable[str],
    *,
    output_path: Optional[Path] = None,
    split_sizes: Optional[dict] = None,
) -> dict:
    """Build a metadata dict (and optionally write metadata.json).

    Schema per task:
        dim_parameters: int
        dim_data:       int
        data_kind:      "vector" | "image" | "timeseries"
        data_shape:     list[int]
        splits:         dict[str, int]
        has_reference:  bool
        num_observations: int
    """
    meta: dict = {}
    for name in task_names:
        task = get_task(name)
        if split_sizes is None:
            # No override: get_exporter resolves the task's hf_split_sizes
            # (falling back to config.DEFAULT_SPLIT_SIZES).
            exporter = get_exporter(task)
        else:
            # Explicit override (e.g. CLI --train-size) wins over the hint.
            exporter = get_exporter(
                task,
                train_size=split_sizes["train"],
                val_size=split_sizes["validation"],
                test_size=split_sizes["test"],
            )
        # Record resolved sizes so metadata matches the uploaded dataset.
        meta[name] = {
            "dim_parameters": int(task.dim_parameters),
            "dim_data": int(task.dim_data),
            "data_kind": exporter.data_kind,
            "data_shape": list(exporter.data_shape),
            "splits": {
                "train": exporter.train_size,
                "validation": exporter.val_size,
                "test": exporter.test_size,
            },
            "has_reference": load_reference(task, exporter) is not None,
            "num_observations": int(task.num_observations),
        }

    if output_path is not None:
        Path(output_path).write_text(json.dumps(meta, indent=4))

    return meta


def merge_metadata(remote: dict, local: dict) -> dict:
    """Merge freshly-built ``local`` entries over ``remote``.

    Tasks present in ``local`` overwrite their own entries; every other task
    already documented in ``remote`` is preserved. Pure — no I/O, no mutation
    of the inputs.
    """
    return {**remote, **local}
