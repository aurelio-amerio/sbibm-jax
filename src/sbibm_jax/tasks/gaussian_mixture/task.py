"""Gaussian Mixture task: inference of mean under uniform prior."""

from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from sbibm_jax.tasks.simulator import Simulator
from sbibm_jax.tasks.task import Task


class GaussianMixture(Task):
    def __init__(self, dim: int = 2, prior_bound: float = 10.0):
        """Gaussian Mixture task.

        Inference of mean under uniform prior with a 2-component
        Gaussian mixture likelihood.

        Args:
            dim: Dimensionality of parameters and data.
            prior_bound: Prior is uniform in [-prior_bound, +prior_bound].
        """
        super().__init__(
            dim_parameters=dim,
            dim_data=dim,
            name=Path(__file__).parent.name,
            name_display="Gaussian Mixture",
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
            "mixture_locs_factor": jnp.array([1.0, 1.0]),
            "mixture_scales": jnp.array([1.0, 0.1]),
            "mixture_weights": jnp.array([0.5, 0.5]),
        }

    def get_prior(
        self, key: jax.random.PRNGKey, num_samples: int = 1
    ) -> jnp.ndarray:
        return self.prior_dist.sample(key, (num_samples,))

    def get_simulator(
        self, key: jax.random.PRNGKey, max_calls: Optional[int] = None
    ) -> Simulator:
        mixture_weights = self.simulator_params["mixture_weights"]
        mixture_locs_factor = self.simulator_params["mixture_locs_factor"]
        mixture_scales = self.simulator_params["mixture_scales"]

        def simulator(key, parameters):
            num_samples = parameters.shape[0]
            k1, k2 = jax.random.split(key)

            idx = dist.Categorical(probs=mixture_weights).sample(
                k1, (num_samples,)
            )

            loc = mixture_locs_factor[idx, None] * parameters
            scale = mixture_scales[idx, None] * jnp.ones_like(parameters)

            return dist.Independent(
                dist.Normal(loc=loc, scale=scale), 1
            ).sample(k2)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Sample reference posterior with rejection sampling."""
        assert not (num_observation is None and observation is None)
        assert not (num_observation is not None and observation is not None)

        if num_observation is not None:
            observation = self.get_observation(num_observation=num_observation)

        samples = []
        while len(samples) < num_samples:
            k1, k2, key = jax.random.split(key, 3)

            idx = dist.Categorical(
                probs=self.simulator_params["mixture_weights"]
            ).sample(k1)

            sample = dist.Normal(
                loc=self.simulator_params["mixture_locs_factor"][idx] * observation,
                scale=self.simulator_params["mixture_scales"][idx],
            ).sample(k2)

            log_prob = self.prior_dist.log_prob(sample.reshape(-1))
            if jnp.isfinite(log_prob):
                samples.append(sample.reshape(-1))

        return jnp.stack(samples)
