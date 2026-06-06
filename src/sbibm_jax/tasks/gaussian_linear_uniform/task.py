"""Gaussian Linear Uniform task: inference of mean under uniform prior."""

from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from sbibm_jax.tasks.simulator import Simulator
from sbibm_jax.tasks.task import Task


class GaussianLinearUniform(Task):
    def __init__(
        self, dim: int = 10, prior_bound: float = 1.0, simulator_scale: float = 0.1
    ):
        """Gaussian Linear Uniform task.

        Inference of mean under uniform prior.

        Args:
            dim: Dimensionality of parameters and data.
            prior_bound: Prior is uniform in [-prior_bound, +prior_bound].
            simulator_scale: Standard deviation of simulator noise.
        """
        super().__init__(
            dim_parameters=dim,
            dim_data=dim,
            name=Path(__file__).parent.name,
            name_display="Gaussian Linear Uniform",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            path=Path(__file__).parent.absolute(),
        )

        self.prior_params = {
            "low": -prior_bound * jnp.ones((self.dim_parameters,)),
            "high": +prior_bound * jnp.ones((self.dim_parameters,)),
        }
        self.prior_dist = dist.Independent(
            dist.Uniform(
                low=self.prior_params["low"],
                high=self.prior_params["high"],
            ),
            1,
        )

        self.simulator_params = {
            "precision_matrix": jnp.linalg.inv(
                simulator_scale * jnp.eye(self.dim_parameters)
            ),
        }

    def get_prior(
        self, key: jax.random.PRNGKey, num_samples: int = 1
    ) -> jnp.ndarray:
        return self.prior_dist.sample(key, (num_samples,))

    def get_simulator(
        self, key: jax.random.PRNGKey, max_calls: Optional[int] = None
    ) -> Simulator:
        sim_precision = self.simulator_params["precision_matrix"]

        def simulator(key, parameters):
            sim_dist = dist.MultivariateNormal(
                loc=parameters,
                precision_matrix=sim_precision,
            )
            return sim_dist.sample(key)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Sample reference posterior with rejection sampling.

        Uses closed-form Gaussian posterior truncated to prior bounds.
        """
        assert not (num_observation is None and observation is None)
        assert not (num_observation is not None and observation is not None)

        if num_observation is not None:
            observation = self.get_observation(num_observation=num_observation)

        sampling_dist = dist.MultivariateNormal(
            loc=observation.reshape(-1),
            precision_matrix=self.simulator_params["precision_matrix"],
        )

        # Rejection sampling: keep samples within prior bounds
        samples = []
        while len(samples) < num_samples:
            key, subkey = jax.random.split(key)
            sample = sampling_dist.sample(subkey)
            log_prob = self.prior_dist.log_prob(sample)
            if jnp.isfinite(log_prob):
                samples.append(sample)

        return jnp.stack(samples)
