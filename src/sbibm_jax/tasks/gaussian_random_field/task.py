"""Gaussian Random Field task: field inference via a coloured-noise simulator.
"""

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

        # HF export hints: stored as (H, W) images via ImageExporter.
        self.hf_data_kind = "image"
        self.hf_data_shape = (field_size, field_size)

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
        N = self.field_size

        # Base k-grid (d=1); knorm scales linearly with (|alpha| + 1e-7).
        k0 = jnp.fft.fftfreq(N, d=1.0)
        kx, ky = jnp.meshgrid(k0, k0, indexing="ij")
        knorm_base = jnp.sqrt(kx**2 + ky**2)

        def generate_single(skey, params):
            log_std = params[0]
            alpha = params[1]
            knorm = knorm_base * (jnp.abs(alpha) + 1e-7)

            ka, kb = jax.random.split(skey)
            a = jax.random.normal(ka, (N, N))
            b = jax.random.normal(kb, (N, N))
            fftfield = a + 1j * b

            # sqrt(P(k)) = knorm**(-alpha/2) * exp(log_std); DC mode -> 0.
            safe_knorm = jnp.where(knorm > 0, knorm, 1.0)
            power_k = jnp.where(
                knorm > 0,
                safe_knorm ** (-alpha / 2.0) * jnp.exp(log_std),
                0.0,
            )
            field = jnp.real(jnp.fft.ifftn(fftfield * power_k))
            return field

        def simulator(key, parameters):
            num_samples = parameters.shape[0]
            keys = jax.random.split(key, num_samples)
            return jax.vmap(generate_single)(keys, parameters)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _get_observation_parameters(self, num_observation: int) -> jnp.ndarray:
        """Conditioning parameters theta_o for an observation.

        Derived deterministically from the observation seed (forward-compatible
        with later-generated observation files). Returns shape (1, 2).
        """
        seed = self.observation_seeds[num_observation - 1]
        key = jax.random.PRNGKey(seed)
        return self.get_prior(key, num_samples=1)

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Sample the conditional likelihood p(field | theta_o).

        This is the exact reference: run the simulator at a fixed theta_o.
        theta_o comes from the observation seed (num_observation) or is passed
        directly as `observation` (the role-inverted conditioning parameters).
        Returns shape (num_samples, dim_data) in field space.
        """
        assert (num_observation is None) != (observation is None), (
            "Provide exactly one of num_observation or observation."
        )
        if num_observation is not None:
            theta_o = self._get_observation_parameters(num_observation)
        else:
            theta_o = jnp.atleast_2d(observation)

        simulator = self.get_simulator(key)
        thetas = jnp.broadcast_to(
            theta_o.reshape(1, -1), (num_samples, self.dim_parameters)
        )
        return simulator(key, thetas)

    def unflatten_data(self, data: jnp.ndarray) -> jnp.ndarray:
        return data.reshape(-1, self.field_size, self.field_size)


if __name__ == "__main__":
    task = GaussianRandomField()
    key = jax.random.PRNGKey(0)
    print("Prior samples shape:", task.get_prior(key, num_samples=5).shape)
