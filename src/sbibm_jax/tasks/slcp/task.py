"""SLCP (Simple Likelihood Complex Posterior) task."""

from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from sbibm_jax.tasks.simulator import Simulator
from sbibm_jax.tasks.task import Task
from sbibm_jax.utils.io import get_array_from_csv


class SLCP(Task):
    def __init__(self, distractors: bool = False):
        """SLCP task.

        Args:
            distractors: If True, uses the distractors variant with 100D data.
        """
        self.num_data = 4
        self.distractors = distractors

        if not self.distractors:
            dim_data = 2 * self.num_data
            name = "slcp"
            name_display = "SLCP"
        else:
            dim_data = 100
            name = "slcp_distractors"
            name_display = "SLCP Distractors"

        observation_seeds = [
            1000000, 1000001, 1000002, 1000003, 1000004,
            1000005, 1000010, 1000012, 1000008, 1000009,
        ]

        super().__init__(
            dim_parameters=5,
            dim_data=dim_data,
            name=name,
            name_display=name_display,
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=observation_seeds,
        )

        self.prior_params = {
            "low": jnp.full((self.dim_parameters,), -3.0),
            "high": jnp.full((self.dim_parameters,), 3.0),
        }
        self.prior_dist = dist.Independent(
            dist.Uniform(
                low=self.prior_params["low"],
                high=self.prior_params["high"],
            ),
            1,
        )

        self._noise_dist = None
        self._permutation_idx = None

    def get_prior(
        self, key: jax.random.PRNGKey, num_samples: int = 1
    ) -> jnp.ndarray:
        return self.prior_dist.sample(key, (num_samples,))

    def get_simulator(
        self, key: jax.random.PRNGKey, max_calls: Optional[int] = None
    ) -> Simulator:
        num_data = self.num_data
        distractors = self.distractors

        if distractors:
            noise_dist, permutation_idx = self._get_noise_dist()

        def simulator(key, parameters):
            num_samples = parameters.shape[0]

            m = jnp.stack(
                [parameters[:, 0], parameters[:, 1]], axis=1
            )

            s1 = parameters[:, 2] ** 2
            s2 = parameters[:, 3] ** 2
            rho = jnp.tanh(parameters[:, 4])

            eps = 1e-6
            S = jnp.zeros((num_samples, 2, 2))
            S = S.at[:, 0, 0].set(s1**2 + eps)
            S = S.at[:, 0, 1].set(rho * s1 * s2)
            S = S.at[:, 1, 0].set(rho * s1 * s2)
            S = S.at[:, 1, 1].set(s2**2 + eps)

            k1, k2 = jax.random.split(key)
            data_dist = dist.MultivariateNormal(
                loc=jnp.broadcast_to(m[:, None, :], (num_samples, num_data, 2)),
                covariance_matrix=jnp.broadcast_to(
                    S[:, None, :, :], (num_samples, num_data, 2, 2)
                ),
            )
            data = data_dist.sample(k1)

            if not distractors:
                return data.reshape(num_samples, -1)
            else:
                data_flat = data.reshape(num_samples, 8)
                noise = noise_dist.sample(k2, (num_samples,))
                data_and_noise = jnp.concatenate([data_flat, noise], axis=1)
                return data_and_noise[:, permutation_idx]

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def get_observation(self, num_observation: int) -> jnp.ndarray:
        if not self.distractors:
            path = (
                self.path
                / "files"
                / f"num_observation_{num_observation}"
                / "observation.csv"
            )
        else:
            path = (
                self.path
                / "files"
                / f"num_observation_{num_observation}"
                / "observation_distractors.csv"
            )
        return get_array_from_csv(path)

    def unflatten_data(self, data: jnp.ndarray) -> jnp.ndarray:
        if not self.distractors:
            return data.reshape(-1, self.num_data, 2)
        else:
            raise NotImplementedError("Unflatten not supported for distractors variant")

    def _get_noise_dist(self):
        """Generate or return cached noise distribution for distractors."""
        if self._noise_dist is not None:
            return self._noise_dist, self._permutation_idx

        noise_dim = 92
        n_noise_comps = 20

        rng = np.random.RandomState(42)

        loc = jnp.array(
            np.array([15 * rng.normal(size=noise_dim) for _ in range(n_noise_comps)])
        )

        cholesky_factors = [
            np.tril(rng.normal(size=(noise_dim, noise_dim)))
            + np.diag(np.exp(rng.normal(size=noise_dim)))
            for _ in range(n_noise_comps)
        ]
        scale_tril = jnp.array(3 * np.array(cholesky_factors))

        mix = dist.Categorical(probs=jnp.ones(n_noise_comps) / n_noise_comps)
        self._noise_mix = mix
        self._noise_locs = loc
        self._noise_scale_tril = scale_tril

        permutation_idx = jnp.array(rng.permutation(noise_dim + 8))


        class NoiseDist:
            def __init__(self, mix, locs, scale_tril):
                self.mix = mix
                self.locs = locs
                self.scale_tril = scale_tril
                self.n_comps = locs.shape[0]

            def sample(self, key, shape=()):
                k1, k2 = jax.random.split(key)
                batch_size = shape[0] if shape else 1
                # Sample component indices
                idx = self.mix.sample(k1, (batch_size,))
                # Sample from corresponding MVN
                selected_loc = self.locs[idx]  # (batch, noise_dim)
                selected_scale = self.scale_tril[idx]  # (batch, noise_dim, noise_dim)

                # Sample from MVN using scale_tril
                z = jax.random.normal(k2, (batch_size, selected_loc.shape[-1]))
                samples = selected_loc + jnp.einsum("bij,bj->bi", selected_scale, z)
                return samples

        self._noise_dist = NoiseDist(mix, loc, scale_tril)
        self._permutation_idx = permutation_idx

        return self._noise_dist, self._permutation_idx

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        raise NotImplementedError(
            "Reference posterior sampling not yet implemented for SLCP. "
            "Use get_reference_posterior_samples() to load pre-computed samples."
        )
