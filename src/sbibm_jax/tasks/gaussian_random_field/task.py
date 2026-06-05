"""Gaussian Random Field task: field inference via a coloured-noise simulator."""

from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from sbibm_jax.tasks.simulator import Simulator
from sbibm_jax.tasks.task import Task


class GaussianRandomField(Task):
    def __init__(self, field_size: int = 32):
        """Gaussian Random Field field-inference task.

        Parameters theta = (log_std, alpha) control a 2D Gaussian random
        field generated from a power-law power spectrum. The field (an
        N x N image, flattened to N*N) is the inference target; theta are
        the conditioning parameters.

        Args:
            field_size: Side length N of the (N, N) field.
        """
        self.field_size = field_size
        super().__init__(
            dim_parameters=2,
            dim_data=field_size * field_size,
            name=Path(__file__).parent.name,
            name_display="Gaussian Random Field",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
        )

        self.prior_dist = dist.Independent(
            dist.Normal(
                loc=jnp.array([0.0, 3.0]),
                scale=jnp.array([0.3, 0.5]),
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
        raise NotImplementedError

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        raise NotImplementedError

    def unflatten_data(self, data: jnp.ndarray) -> jnp.ndarray:
        return data.reshape(-1, self.field_size, self.field_size)


if __name__ == "__main__":
    task = GaussianRandomField()
    key = jax.random.PRNGKey(0)
    print("Prior samples shape:", task.get_prior(key, num_samples=5).shape)
