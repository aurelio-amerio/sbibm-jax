"""Two Moons task."""

import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from sbibm_jax.tasks.simulator import Simulator
from sbibm_jax.tasks.task import Task


class TwoMoons(Task):
    def __init__(self):
        """Two Moons task."""

        observation_seeds = [
            1000011, 1000001, 1000002, 1000003, 1000013,
            1000005, 1000006, 1000007, 1000008, 1000009,
        ]

        super().__init__(
            dim_parameters=2,
            dim_data=2,
            name=Path(__file__).parent.name,
            name_display="Two Moons",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
        )

        prior_bound = 1.0
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
            "a_low": -math.pi / 2.0,
            "a_high": +math.pi / 2.0,
            "base_offset": 0.25,
            "r_loc": 0.1,
            "r_scale": 0.01,
        }

    def get_prior(
        self, key: jax.random.PRNGKey, num_samples: int = 1
    ) -> jnp.ndarray:
        return self.prior_dist.sample(key, (num_samples,))

    def get_simulator(
        self, key: jax.random.PRNGKey, max_calls: Optional[int] = None
    ) -> Simulator:
        sp = self.simulator_params

        def simulator(key, parameters):
            num_samples = parameters.shape[0]
            k1, k2 = jax.random.split(key)

            a = dist.Uniform(low=sp["a_low"], high=sp["a_high"]).sample(
                k1, (num_samples, 1)
            )
            r = dist.Normal(loc=sp["r_loc"], scale=sp["r_scale"]).sample(
                k2, (num_samples, 1)
            )

            p = jnp.concatenate(
                [jnp.cos(a) * r + sp["base_offset"], jnp.sin(a) * r],
                axis=1,
            )
            return TwoMoons._map_fun(parameters, p)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    @staticmethod
    def _map_fun(parameters: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        ang = jnp.array(-math.pi / 4.0)
        c = jnp.cos(ang)
        s = jnp.sin(ang)
        z0 = (c * parameters[:, 0] - s * parameters[:, 1]).reshape(-1, 1)
        z1 = (s * parameters[:, 0] + c * parameters[:, 1]).reshape(-1, 1)
        return p + jnp.concatenate([-jnp.abs(z0), z1], axis=1)

    @staticmethod
    def _map_fun_inv(parameters: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        ang = jnp.array(-math.pi / 4.0)
        c = jnp.cos(ang)
        s = jnp.sin(ang)
        z0 = (c * parameters[:, 0] - s * parameters[:, 1]).reshape(-1, 1)
        z1 = (s * parameters[:, 0] + c * parameters[:, 1]).reshape(-1, 1)
        return x - jnp.concatenate([-jnp.abs(z0), z1], axis=1)

    def _likelihood(
        self,
        parameters: jnp.ndarray,
        data: jnp.ndarray,
        log: bool = True,
    ) -> jnp.ndarray:
        if parameters.ndim == 1:
            parameters = parameters.reshape(1, -1)

        p = self._map_fun_inv(parameters, data).squeeze(0)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        u = p[:, 0] - self.simulator_params["base_offset"]
        v = p[:, 1]

        r = jnp.sqrt(u**2 + v**2)
        L = -0.5 * (
            (r - self.simulator_params["r_loc"]) / self.simulator_params["r_scale"]
        ) ** 2 - 0.5 * jnp.log(
            2 * jnp.array(math.pi) * self.simulator_params["r_scale"] ** 2
        )

        # Mask out negative u values
        L = jnp.where(u < 0.0, -jnp.inf, L)

        return L if log else jnp.exp(L)

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Sample reference posterior using closed-form solution."""
        if observation is None:
            observation = self.get_observation(num_observation)

        ang = jnp.array(-math.pi / 4.0)
        c = jnp.cos(-ang)
        s = jnp.sin(-ang)

        samples = []
        while len(samples) < num_samples:
            k1, k2, key = jax.random.split(key, 3)

            sim = self.get_simulator(k1)
            p = sim(k1, jnp.zeros((1, 2)))

            q = jnp.zeros(2)
            q = q.at[0].set(p[0, 0] - observation[0, 0])
            q = q.at[1].set(observation[0, 1] - p[0, 1])

            # Randomly flip sign
            flip = jax.random.uniform(k2) < 0.5
            q = q.at[0].set(jnp.where(flip, -q[0], q[0]))

            sample = jnp.array([[c * q[0] - s * q[1], s * q[0] + c * q[1]]])
            log_prob = self.prior_dist.log_prob(sample.reshape(-1))

            if jnp.isfinite(log_prob):
                samples.append(sample)

        return jnp.concatenate(samples)
