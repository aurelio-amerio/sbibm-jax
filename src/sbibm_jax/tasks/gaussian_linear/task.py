"""Gaussian Linear task: inference of mean under Gaussian prior."""

from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from sbibm_jax.tasks.simulator import Simulator
from sbibm_jax.tasks.task import Task


class GaussianLinear(Task):
    def __init__(
        self, dim: int = 10, prior_scale: float = 0.1, simulator_scale: float = 0.1
    ):
        """Gaussian Linear task.

        Inference of mean under Gaussian prior.

        Args:
            dim: Dimensionality of parameters and data.
            prior_scale: Standard deviation of prior.
            simulator_scale: Standard deviation of simulator noise.
        """
        super().__init__(
            dim_theta=dim,
            dim_x=dim,
            name=Path(__file__).parent.name,
            name_display="Gaussian Linear",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            path=Path(__file__).parent.absolute(),
        )

        self.prior_params = {
            "loc": jnp.zeros((self.dim_theta,)),
            "precision_matrix": jnp.linalg.inv(
                prior_scale * jnp.eye(self.dim_theta)
            ),
        }
        self.prior_dist = dist.MultivariateNormal(
            loc=self.prior_params["loc"],
            precision_matrix=self.prior_params["precision_matrix"],
        )

        self.simulator_params = {
            "precision_matrix": jnp.linalg.inv(
                simulator_scale * jnp.eye(self.dim_theta)
            ),
        }

    def get_prior(
        self, key: jax.random.PRNGKey, num_samples: int = 1
    ) -> jnp.ndarray:
        return self.prior_dist.sample(key, (num_samples,))

    def get_simulator(
        self, key: jax.random.PRNGKey, max_calls: Optional[int] = None
    ) -> Simulator:
        sim_dist_precision = self.simulator_params["precision_matrix"]

        def simulator(key, parameters):
            sim_dist = dist.MultivariateNormal(
                loc=parameters,
                precision_matrix=sim_dist_precision,
            )
            return sim_dist.sample(key)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _get_reference_posterior(
        self,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> dist.MultivariateNormal:
        """Compute closed-form posterior distribution.

        Args:
            num_observation: Observation number.
            observation: Direct observation array.

        Returns:
            Posterior MultivariateNormal distribution.
        """
        assert not (num_observation is None and observation is None)
        assert not (num_observation is not None and observation is not None)

        if num_observation is not None:
            observation = self.get_observation(num_observation=num_observation)

        N = 1
        covariance_matrix = jnp.linalg.inv(
            self.prior_params["precision_matrix"]
            + N * self.simulator_params["precision_matrix"]
        )
        loc = covariance_matrix @ (
            N * self.simulator_params["precision_matrix"] @ observation.reshape(-1)
            + self.prior_params["precision_matrix"] @ self.prior_params["loc"]
        )

        return dist.MultivariateNormal(
            loc=loc, covariance_matrix=covariance_matrix
        )

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Sample reference posterior (closed form).

        Args:
            key: JAX PRNG key.
            num_samples: Number of samples.
            num_observation: Observation number.
            observation: Direct observation array.

        Returns:
            Samples of shape (num_samples, dim_theta).
        """
        posterior = self._get_reference_posterior(
            num_observation=num_observation,
            observation=observation,
        )
        return posterior.sample(key, (num_samples,))


if __name__ == "__main__":
    task = GaussianLinear()
    key = jax.random.PRNGKey(0)
    samples = task.get_prior(key, num_samples=5)
    print(f"Prior samples shape: {samples.shape}")
