"""Bernoulli GLM task."""

from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from sbibm_jax.tasks.simulator import Simulator
from sbibm_jax.tasks.task import Task
from sbibm_jax.utils.io import get_array_from_csv


class BernoulliGLM(Task):
    def __init__(self, summary="sufficient"):
        """Bernoulli GLM task.

        Args:
            summary: "sufficient" for 10D summary stats, "raw" for 100D raw spikes.
        """
        self.summary = summary
        if self.summary == "sufficient":
            dim_data = 10
            name = "bernoulli_glm"
            name_display = "Bernoulli GLM"
            self.raw = False
        elif self.summary == "raw":
            dim_data = 100
            self.raw = True
            name = "bernoulli_glm_raw"
            name_display = "Bernoulli GLM Raw"
        else:
            raise NotImplementedError(f"Unknown summary type: {summary}")

        super().__init__(
            dim_parameters=10,
            dim_data=dim_data,
            name=name,
            name_display=name_display,
            num_simulations=[1000, 10000, 100000, 1000000],
            num_posterior_samples=10000,
            num_observations=10,
            path=Path(__file__).parent.absolute(),
        )

        self.stimulus = {
            "dt": 1,
            "duration": 100,
            "seed": 42,
        }


        M = self.dim_parameters - 1
        D = jnp.diag(jnp.ones(M)) - jnp.diag(jnp.ones(M - 1), -1)
        F = D @ D + jnp.diag(1.0 * jnp.arange(M) / M) ** 0.5
        Binv = jnp.zeros((M + 1, M + 1))
        Binv = Binv.at[0, 0].set(0.5)
        Binv = Binv.at[1:, 1:].set(F.T @ F)

        self.prior_params = {
            "loc": jnp.zeros((M + 1,)),
            "precision_matrix": Binv,
        }
        self.prior_dist = dist.MultivariateNormal(
            loc=self.prior_params["loc"],
            precision_matrix=self.prior_params["precision_matrix"],
        )


        self._design_matrix = None
        self._stimulus_I = None

    def _load_design_matrix(self) -> jnp.ndarray:
        """Load or generate design matrix."""
        if self._design_matrix is not None:
            return self._design_matrix


        npz_path = self.path / "files" / "design_matrix.npz"
        pt_path = self.path / "files" / "design_matrix.pt"

        if npz_path.exists():
            self._design_matrix = jnp.array(
                np.load(npz_path)["data"].astype(np.float32)
            )
        elif pt_path.exists():
            import torch
            self._design_matrix = jnp.array(
                torch.load(pt_path, weights_only=True).numpy()
            )
        else:

            self._design_matrix = self._generate_design_matrix()

        return self._design_matrix

    def _load_stimulus_I(self) -> jnp.ndarray:
        """Load stimulus current."""
        if self._stimulus_I is not None:
            return self._stimulus_I

        npz_path = self.path / "files" / "stimulus_I.npz"
        pt_path = self.path / "files" / "stimulus_I.pt"

        if npz_path.exists():
            self._stimulus_I = jnp.array(
                np.load(npz_path)["data"].astype(np.float32)
            )
        elif pt_path.exists():
            import torch
            self._stimulus_I = jnp.array(
                torch.load(pt_path, weights_only=True).numpy()
            )
        else:
            self._stimulus_I = self._generate_stimulus_I()

        return self._stimulus_I

    def _generate_stimulus_I(self) -> jnp.ndarray:
        """Generate Gaussian white noise stimulus."""
        rng = np.random.RandomState(self.stimulus["seed"])
        n_timesteps = self.stimulus["duration"] // self.stimulus["dt"]
        return jnp.array(rng.randn(n_timesteps).astype(np.float32))

    def _generate_design_matrix(self) -> jnp.ndarray:
        """Generate design matrix from stimulus."""
        stimulus_I = self._generate_stimulus_I()
        n_timesteps = len(stimulus_I)
        stimulus_np = np.array(stimulus_I)

        design = np.zeros((n_timesteps, self.dim_parameters - 1), dtype=np.float32)
        for j in range(self.dim_parameters - 1):
            design[j:, j] = stimulus_np[:n_timesteps - j]


        design_with_offset = np.concatenate(
            [np.ones((n_timesteps, 1), dtype=np.float32), design], axis=1
        )
        return jnp.array(design_with_offset)

    def get_prior(
        self, key: jax.random.PRNGKey, num_samples: int = 1
    ) -> jnp.ndarray:
        return self.prior_dist.sample(key, (num_samples,))

    def get_simulator(
        self, key: jax.random.PRNGKey, max_calls: Optional[int] = None
    ) -> Simulator:
        design_matrix = self._load_design_matrix()
        stimulus_I = self._load_stimulus_I()
        raw = self.raw

        def simulator(key, parameters):
            num_samples = parameters.shape[0]

            def simulate_single(key, params):
                psi = design_matrix @ params
                z = jax.nn.sigmoid(psi)
                y = (jax.random.uniform(key, shape=z.shape) < z).astype(jnp.float32)


                num_spikes = jnp.sum(y).reshape(1)

                # STA (spike-triggered average)
                # Equivalent to F.conv1d(y, stimulus_I, padding=8).squeeze()[-9:]

                y_padded = jnp.pad(y, (8, 8))
                conv_out = jnp.array([
                    jnp.sum(y_padded[i:i+len(stimulus_I)] * stimulus_I)
                    for i in range(len(y_padded) - len(stimulus_I) + 1)
                ])
                sta = conv_out[-9:]

                summary = jnp.concatenate([num_spikes, sta])

                if raw:
                    return y
                else:
                    return summary

            keys = jax.random.split(key, num_samples)
            return jax.vmap(simulate_single)(keys, parameters)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def get_observation(self, num_observation: int) -> jnp.ndarray:
        if not self.raw:
            path = (
                self.path / "files"
                / f"num_observation_{num_observation}"
                / "observation.csv"
            )
        else:
            path = (
                self.path / "files"
                / f"num_observation_{num_observation}"
                / "observation_raw.csv"
            )
        return get_array_from_csv(path)

    def flatten_data(self, data: jnp.ndarray) -> jnp.ndarray:
        return data.reshape(-1, self.dim_data)

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        raise NotImplementedError(
            "Reference posterior sampling not yet implemented for BernoulliGLM. "
            "Use get_reference_posterior_samples() to load pre-computed samples."
        )
