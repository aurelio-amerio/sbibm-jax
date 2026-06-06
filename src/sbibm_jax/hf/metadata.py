"""Auto-generate metadata.json from Task attributes."""

import json
from pathlib import Path
from typing import Iterable, Optional

from sbibm_jax import get_task
from sbibm_jax.hf import config
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
    if split_sizes is None:
        split_sizes = dict(config.DEFAULT_SPLIT_SIZES)

    meta: dict = {}
    for name in task_names:
        task = get_task(name)
        exporter = get_exporter(
            task,
            train_size=split_sizes["train"],
            val_size=split_sizes["validation"],
            test_size=split_sizes["test"],
        )
        meta[name] = {
            "dim_parameters": int(task.dim_parameters),
            "dim_data": int(task.dim_data),
            "data_kind": exporter.data_kind,
            "data_shape": list(exporter.data_shape),
            "splits": dict(split_sizes),
            "has_reference": load_reference(task, exporter) is not None,
            "num_observations": int(task.num_observations),
        }

    if output_path is not None:
        Path(output_path).write_text(json.dumps(meta, indent=4))

    return meta
