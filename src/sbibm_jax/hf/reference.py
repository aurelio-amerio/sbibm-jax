"""Optional per-task reference block loader.

For tasks that ship reference posterior CSVs under
files/num_observation_<i>/, builds a HuggingFace Dataset matching the original
SBI-benchmarks-data schema (reference_samples, observations, true_parameters).
Returns None when the files are absent so the caller can skip the _posterior
config without erroring.
"""

from typing import Optional

import numpy as np
from datasets import Dataset

from sbibm_jax.hf.exporter import DatasetExporter
from sbibm_jax.tasks.task import Task


def load_reference(task: Task, exporter: DatasetExporter) -> Optional[Dataset]:
    """Load the reference block for `task`, reshaping observations via `exporter`.

    Returns None if any required CSV is missing (e.g. for GRF, Beer).
    """
    observations = []
    reference_samples = []
    true_parameters = []

    for i in range(1, task.num_observations + 1):
        try:
            obs = np.asarray(task.get_observation(i), dtype=np.float32)
            ref = np.asarray(
                task.get_reference_posterior_samples(i), dtype=np.float32,
            )
            true_p = np.asarray(task.get_true_parameters(i), dtype=np.float32)
        except FileNotFoundError:
            return None

        # Reshape the observation `x` from (1, dim_x) into the exporter's
        # native storage shape (drop the leading sample axis after reshape).
        obs_flat = obs.reshape(1, task.dim_x)
        obs_shaped = exporter.shape_x(obs_flat)[0]

        observations.append(obs_shaped)
        reference_samples.append(ref)
        true_parameters.append(true_p.reshape(-1))

    return Dataset.from_dict({
        "reference_samples": np.stack(reference_samples).astype(np.float32),
        "observations": np.stack(observations).astype(np.float32),
        "true_parameters": np.stack(true_parameters).astype(np.float32),
    })
