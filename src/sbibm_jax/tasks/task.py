"""Base Task class for sbibm-jax benchmark tasks."""

from abc import abstractmethod
from pathlib import Path
from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from sbibm_jax.utils.io import get_array_from_csv, save_array_to_csv


class Task:
    def __init__(
        self,
        dim_x: int,
        dim_theta: int,
        name: str,
        num_observations: int,
        num_posterior_samples: int,
        path: Path,
        name_display: Optional[str] = None,
        num_reference_posterior_samples: Optional[int] = None,
        observation_seeds: Optional[List[int]] = None,
    ):
        """Base class for benchmark tasks.

        Args:
            dim_x: Dimensionality of data.
            dim_theta: Dimensionality of parameters.
            name: Name of task (matches folder name).
            num_observations: Number of different observations.
            num_posterior_samples: Number of posterior samples to generate.
            path: Path to task folder.
            name_display: Display name with proper casing/spacing.
            num_reference_posterior_samples: Number of reference posterior
                samples. Defaults to num_posterior_samples.
            observation_seeds: Seeds used to generate observations. Defaults
                to a sequence starting at 1000000.
        """
        self.dim_x = dim_x
        self.dim_theta = dim_theta
        self.name = name
        self.num_observations = num_observations
        self.num_posterior_samples = num_posterior_samples
        self.path = path

        self.name_display = name_display if name_display is not None else name
        self.num_reference_posterior_samples = (
            num_reference_posterior_samples
            if num_reference_posterior_samples is not None
            else num_posterior_samples
        )
        self.observation_seeds = (
            observation_seeds
            if observation_seeds is not None
            else [i + 1000000 for i in range(self.num_observations)]
        )

    @abstractmethod
    def get_prior(self, key: jax.random.PRNGKey, num_samples: int = 1) -> jnp.ndarray:
        """Sample from the prior distribution.

        Args:
            key: JAX PRNG key.
            num_samples: Number of samples to draw.

        Returns:
            Array of shape (num_samples, dim_theta).
        """
        raise NotImplementedError

    def get_prior_dist(self):
        """Get the prior distribution object (numpyro distribution)."""
        return self.prior_dist

    @abstractmethod
    def get_simulator(
        self, key: jax.random.PRNGKey, max_calls: Optional[int] = None
    ) -> "Simulator":
        """Get simulator callable.

        Args:
            key: JAX PRNG key for stochastic simulators.
            max_calls: Maximum number of simulator calls.

        Returns:
            Simulator instance.
        """
        raise NotImplementedError

    def get_labels_data(self) -> List[str]:
        """Get list of data dimension labels."""
        return [f"data_{i+1}" for i in range(self.dim_x)]

    def get_labels_parameters(self) -> List[str]:
        """Get list of parameter labels."""
        return [f"parameter_{i+1}" for i in range(self.dim_theta)]

    def get_observation(self, num_observation: int) -> jnp.ndarray:
        """Load observed data for a given observation number.

        Args:
            num_observation: Observation number (1-indexed).

        Returns:
            Array of shape (1, dim_x).
        """
        path = (
            self.path
            / "files"
            / f"num_observation_{num_observation}"
            / "observation.csv"
        )
        return get_array_from_csv(path)

    def get_reference_posterior_samples(
        self, num_observation: int
    ) -> jnp.ndarray:
        """Load reference posterior samples for a given observation.

        Args:
            num_observation: Observation number (1-indexed).

        Returns:
            Array of shape (num_reference_posterior_samples, dim_theta).
        """
        path = (
            self.path
            / "files"
            / f"num_observation_{num_observation}"
            / "reference_posterior_samples.csv.bz2"
        )
        return get_array_from_csv(path)

    def get_true_parameters(self, num_observation: int) -> jnp.ndarray:
        """Load true parameters for a given observation.

        Args:
            num_observation: Observation number (1-indexed).

        Returns:
            Array of shape (1, dim_theta).
        """
        path = (
            self.path
            / "files"
            / f"num_observation_{num_observation}"
            / "true_parameters.csv"
        )
        return get_array_from_csv(path)

    def flatten_data(self, data: jnp.ndarray) -> jnp.ndarray:
        """Flatten data into 2D array."""
        return data.reshape(-1, self.dim_x)

    def unflatten_data(self, data: jnp.ndarray) -> jnp.ndarray:
        """Unflatten data. Override for tasks with structured output."""
        return data.reshape(-1, self.dim_x)

    def save_data(self, path, data: jnp.ndarray) -> None:
        """Save data array to CSV."""
        save_array_to_csv(path, data, self.get_labels_data())

    def save_parameters(self, path, parameters: jnp.ndarray) -> None:
        """Save parameters array to CSV."""
        save_array_to_csv(path, parameters, self.get_labels_parameters())

    @abstractmethod
    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Sample reference posterior for given observation.

        Args:
            key: JAX PRNG key.
            num_samples: Number of samples.
            num_observation: Observation number.
            observation: Direct observation array (alternative to num_observation).

        Returns:
            Samples from reference posterior.
        """
        raise NotImplementedError
