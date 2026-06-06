"""Top-level orchestration: build_dataset(task_name, **opts).

NOTE: This is a stub. The real implementation is added in Task 13. Tests in
Task 12 monkeypatch this symbol via sbibm_jax.hf.upload.build_dataset, so the
stub never actually runs.
"""

from typing import Any


def build_dataset(task_name: str, **opts: Any):
    raise NotImplementedError(
        "build_dataset is a stub until Task 13 implements the orchestration."
    )
