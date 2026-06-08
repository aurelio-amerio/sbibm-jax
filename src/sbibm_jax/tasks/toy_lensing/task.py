"""Toy Gravitational Lensing task: image inference from lensed star fields."""

from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from sbibm_jax.tasks.simulator import Simulator
from sbibm_jax.tasks.task import Task


class ToyLensing(Task):
    def __init__(self, resolution: int = 32):
        """Toy Gravitational Lensing image-inference task.

        Parameters theta = (radius, width) control a ring-shaped lensing arc
        embedded in a star-field background. The observation is a noisy
        (resolution x resolution) image, flattened to resolution^2 pixels.

        Args:
            resolution: Side length N of the (N, N) output image.
        """
        self.resolution = resolution
        super().__init__(
            dim_theta=2,
            dim_x=resolution * resolution,
            name=Path(__file__).parent.name,
            name_display="Toy Gravitational Lensing",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            path=Path(__file__).parent.absolute(),
        )

        # HF export hints: stored as (H, W) images via ImageExporter.
        self.hf_data_kind = "image"
        self.hf_data_shape = (resolution, resolution)
        # Cap HF generation at 100k train (expensive simulator / large image
        # rows); consumers subsample smaller budgets by indexing the prefix.
        self.hf_split_sizes = {
            "train": 100_000, "validation": 10_000, "test": 10_000,
        }

        self.prior_dist = dist.Independent(
            dist.Uniform(
                low=jnp.array([0.1, 0.01]),
                high=jnp.array([1.1, 0.31]),
            ),
            1,
        )

    def get_prior(
        self, key: jax.random.PRNGKey, num_samples: int = 1
    ) -> jnp.ndarray:
        return self.prior_dist.sample(key, (num_samples,))

    def get_simulator(
        self, key: jax.random.PRNGKey, max_calls: Optional[int] = None
    ) -> Simulator:
        # key is unused here; it is consumed by the returned Simulator.__call__
        N = self.resolution
        n_lines = 20
        line_amplitude = 0.8
        line_width = 0.01
        noise_std = 0.3

        x = jnp.linspace(-2, 2, N)
        X, Y = jnp.meshgrid(x, x)

        def generate_single(skey, params):
            r = params[0]
            w = params[1]
            k_pos, k_lines, k_noise = jax.random.split(skey, 3)

            pos = jax.random.uniform(k_pos, (2,), minval=-1.0, maxval=1.0)
            x0, y0 = pos[0], pos[1]

            R = jnp.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
            mu = jnp.exp(-(R - r) ** 2 / (w ** 2 * 2))

            xr = jax.random.uniform(k_lines, (n_lines, 2))

            def line(xr_i):
                return line_amplitude * jnp.exp(
                    -(X * xr_i[0] + Y * (1 - xr_i[0]) - xr_i[1]) ** 2 / line_width ** 2
                )

            mu = mu + jnp.sum(jax.vmap(line)(xr), axis=0)
            std_mu = jnp.std(mu)
            mu = (mu - jnp.mean(mu)) / jnp.where(std_mu > 0, std_mu, 1.0)
            img = mu + jax.random.normal(k_noise, (N, N)) * noise_std
            return img

        def simulator(key, parameters):
            num_samples = parameters.shape[0]
            keys = jax.random.split(key, num_samples)
            return jax.vmap(generate_single)(keys, parameters)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def get_labels_parameters(self) -> List[str]:
        return ["radius", "width"]

    def unflatten_data(self, data: jnp.ndarray) -> jnp.ndarray:
        return data.reshape(-1, self.resolution, self.resolution)

    def _observation_keys(self, num_observation: int):
        """Return (k_theta, k_sim) derived from the observation seed."""
        seed = self.observation_seeds[num_observation - 1]
        return jax.random.split(jax.random.PRNGKey(seed))

    def get_true_parameters(self, num_observation: int) -> jnp.ndarray:
        """Deterministically sampled true parameters for an observation.

        Args:
            num_observation: Observation number (1-indexed).

        Returns:
            Array of shape (1, 2).
        """
        k_theta, _ = self._observation_keys(num_observation)
        return self.get_prior(k_theta, num_samples=1)

    def get_observation(self, num_observation: int) -> jnp.ndarray:
        """Deterministically generated observation image.

        Args:
            num_observation: Observation number (1-indexed).

        Returns:
            Array of shape (1, dim_x).
        """
        k_theta, k_sim = self._observation_keys(num_observation)
        z_o = self.get_prior(k_theta, num_samples=1)
        simulator = self.get_simulator(k_sim)
        return simulator(k_sim, z_o)

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        raise NotImplementedError(
            "toy_lensing has no tractable reference posterior."
        )
