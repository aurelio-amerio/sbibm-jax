"""Gravitational Waves task: file-backed time-series inference (no simulator).

Unlike the other tasks, gravitational_waves has no simulator yet: its data is a
fixed corpus of pre-generated (theta, x) rows published by
scripts/make_gw_dataset.py. get_prior / get_simulator / the reference sampler
raise NotImplementedError; consume the dataset via
sbibm_jax.data.TaskDataset("gravitational_waves").

theta = 2 parameters; x = a (8192, 2) two-channel strain time series.
"""

from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp

from sbibm_jax.tasks.simulator import Simulator
from sbibm_jax.tasks.task import Task

_MSG = (
    "The gravitational_waves {what} is not available yet. This is a "
    "file-backed task: load the published dataset via "
    "sbibm_jax.data.TaskDataset('gravitational_waves'). The simulator/prior "
    "will be added in a future rework."
)


class GravitationalWaves(Task):
    def __init__(self):
        super().__init__(
            dim_theta=2,
            dim_x=8192 * 2,
            name=Path(__file__).parent.name,
            name_display="Gravitational Waves",
            num_observations=1,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            path=Path(__file__).parent.absolute(),
        )
        # HF export hints: (T, C) time series stored via TimeSeriesExporter.
        self.hf_x_kind = "timeseries"
        self.hf_x_shape = (8192, 2)
        # per-feature theta -> (1, 2); per-channel x -> (1, 1, 2).
        self.hf_stats_axes = {"theta": (0,), "x": (0, 1)}
        # File-backed: skipped by make_dataset.py (the simulator is a mock);
        # uploaded by scripts/make_gw_dataset.py.
        self.hf_external = True

    def get_prior(
        self, key: jax.random.PRNGKey, num_samples: int = 1
    ) -> jnp.ndarray:
        raise NotImplementedError(_MSG.format(what="prior"))

    def get_simulator(
        self, key: jax.random.PRNGKey, max_calls: Optional[int] = None
    ) -> Simulator:
        raise NotImplementedError(_MSG.format(what="simulator"))

    def get_labels_parameters(self) -> List[str]:
        return ["theta_1", "theta_2"]

    def unflatten_data(self, data: jnp.ndarray) -> jnp.ndarray:
        return data.reshape(-1, 8192, 2)

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        raise NotImplementedError(_MSG.format(what="reference posterior"))
