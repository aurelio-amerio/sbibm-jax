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
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    stats_by_task: Optional[dict] = None,
) -> dict:
    """Build a metadata dict (and optionally write metadata.json).

    Split sizes are resolved through the same per-dimension path as
    ``build_dataset``: an explicit ``train/val/test_size`` wins, otherwise
    ``get_exporter`` falls back to the task's ``hf_split_sizes`` (then the
    global default). Passing only some sizes leaves the rest at the task cap,
    so the recorded ``splits`` always match what the uploaded dataset contains.

    Schema per task:
        dim_theta:      int
        dim_x:          int
        data_kind:      "vector" | "image" | "timeseries"
        data_shape:     list[int]
        splits:         dict[str, int]
        has_reference:  bool
        num_observations: int
        stats:          dict | None
    """
    meta: dict = {}
    for name in task_names:
        task = get_task(name)
        exporter = get_exporter(
            task,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
        # Record resolved sizes so metadata matches the uploaded dataset.
        meta[name] = {
            "dim_theta": int(task.dim_theta),
            "dim_x": int(task.dim_x),
            "data_kind": exporter.data_kind,
            "data_shape": list(exporter.data_shape),
            "splits": {
                "train": exporter.train_size,
                "validation": exporter.val_size,
                "test": exporter.test_size,
            },
            "has_reference": load_reference(task, exporter) is not None,
            "num_observations": int(task.num_observations),
            "stats": (stats_by_task or {}).get(name),
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
