"""Spherical GRF task: HEALPix Gaussian random field, polynomial Cl.

theta = (logA, n, alpha) parameterizes the angular power spectrum as a
log-log polynomial (positive by construction):

    ln C_ell = logA + n * x + 0.5 * alpha * x**2,  x = ln(max(ell,1)/ell0)

The Cl are sufficient statistics for a GRF, so an exact reference
posterior exists (anafast + Gaussian spectrum likelihood); this is the
benchmark's correctness-check task. Design doc:
docs/superpowers/specs/2026-07-14-spherical-grf-task-design.md
"""

from pathlib import Path
from typing import Optional

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from sbibm_jax.tasks.simulator import Simulator
from sbibm_jax.tasks.task import Task

PRIOR_LOW = (-2.0, -3.0, -0.5)
PRIOR_HIGH = (2.0, 0.0, 0.5)


def cl_target(
    theta: jnp.ndarray, lmax: int, ell0: float = 64.0
) -> jnp.ndarray:
    """Angular power spectrum C_ell for theta = (logA, n, alpha).

    Log-log polynomial in x = ln(max(ell,1)/ell0); C_0 = C_1 = 0.
    Returns shape (lmax + 1,).
    """
    theta = jnp.asarray(theta)
    log_a, n, alpha = theta[0], theta[1], theta[2]
    ell = jnp.arange(lmax + 1)
    x = jnp.log(jnp.maximum(ell, 1) / ell0)
    cl = jnp.exp(log_a + n * x + 0.5 * alpha * x**2)
    return cl.at[:2].set(0.0)


def _cl_target_np(
    theta: np.ndarray, lmax: int, ell0: float = 64.0
) -> np.ndarray:
    """float64 NumPy twin of cl_target for the healpy simulate path.

    Computed outside JAX so the result (and hence the seed-derived
    canonical observations) is independent of the global
    jax_enable_x64 flag, which e.g. importing jax_healpy flips.
    """
    theta = np.asarray(theta, dtype=np.float64)
    log_a, n, alpha = theta[0], theta[1], theta[2]
    ell = np.arange(lmax + 1)
    x = np.log(np.maximum(ell, 1) / ell0)
    cl = np.exp(log_a + n * x + 0.5 * alpha * x**2)
    cl[:2] = 0.0
    return cl


class SphericalGRF(Task):
    def __init__(
        self,
        nside: int = 64,
        noise_std: float = 0.0,
        backend: str = "healpy",
        name: Optional[str] = None,
        name_display: Optional[str] = None,
    ):
        """Spherical GRF task.

        Args:
            nside: HEALPix resolution (power of two, 4..1024).
            noise_std: Std of optional i.i.d. Gaussian pixel noise
                added to every map (0 disables it). The reference
                likelihood accounts for it via N_ell = std^2*4pi/npix.
            backend: "healpy" (default; NumPy, ground truth) or "jax"
                (jax-healpy, optional [jaxhp] extra; jit/GPU).
            name: Optional task name override (registry alias).
            name_display: Optional human-readable label override.
        """
        if (
            not isinstance(nside, int)
            or nside < 4
            or nside > 1024
            or (nside & (nside - 1)) != 0
        ):
            raise ValueError(
                f"nside must be a power of two in [4, 1024], got {nside}."
            )
        if backend not in ("healpy", "jax"):
            raise ValueError(
                f"backend must be 'healpy' or 'jax', got {backend!r}."
            )
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}.")

        self.nside = nside
        self.npix = 12 * nside * nside
        self.lmax = 3 * nside - 1
        self.ell0 = 64.0
        self.noise_std = float(noise_std)
        self.backend = backend

        super().__init__(
            dim_theta=3,
            dim_x=self.npix,
            name=name or Path(__file__).parent.name,
            name_display=name_display or "Spherical GRF",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            path=Path(__file__).parent.absolute(),
        )

        # HF export hints: flat RING-ordered maps via HealpixExporter,
        # global scalar x stats (the field is isotropic). Generation
        # runs on the jax backend (hf_backend, read by hf.build).
        self.hf_x_kind = "healpix"
        self.hf_x_shape = (self.npix,)
        self.hf_stats_axes = {"theta": (0,), "x": (0, 1)}
        if nside <= 64:
            self.hf_split_sizes = {
                "train": 100_000, "validation": 10_000, "test": 10_000,
            }
        else:
            self.hf_split_sizes = {
                "train": 30_000, "validation": 5_000, "test": 5_000,
            }

        self.prior_params = {
            "low": jnp.array(PRIOR_LOW),
            "high": jnp.array(PRIOR_HIGH),
        }
        self.prior_dist = dist.Independent(
            dist.Uniform(
                low=self.prior_params["low"],
                high=self.prior_params["high"],
            ),
            1,
        )

    def get_prior(
        self, key: jax.random.PRNGKey, num_samples: int = 1
    ) -> jnp.ndarray:
        return self.prior_dist.sample(key, (num_samples,))

    def _seed_words(self, subkey) -> np.ndarray:
        """uint32 words of a JAX key, for seeding NumPy RNGs."""
        return np.asarray(
            jax.random.key_data(subkey), dtype=np.uint32
        ).ravel()

    def _simulate_one_np(self, subkey, theta_np: np.ndarray) -> np.ndarray:
        """One RING map (npix,) float32 via healpy, seeded from subkey."""
        words = self._seed_words(subkey)
        cl = _cl_target_np(theta_np, self.lmax, self.ell0)
        # healpy's synfast draws from NumPy's *global* RNG (no rng arg),
        # so seed it per row. Not thread-safe; consumers use process
        # workers (grain spawn), never threads.
        np.random.seed(words)
        m = hp.synfast(cl, self.nside, lmax=self.lmax, new=True)
        if self.noise_std > 0:
            rng = np.random.default_rng(
                np.concatenate([words, np.uint32([0x5EED])])
            )
            m = m + self.noise_std * rng.standard_normal(m.shape)
        return m.astype(np.float32)

    def _healpy_simulator(self):
        """Batch simulator closure on the healpy backend.

        Used by get_simulator(backend="healpy") and — always, whatever
        self.backend is — for observation generation, so observed maps
        are backend-independent.
        """
        def simulator(key, parameters):
            params_np = np.asarray(parameters)
            keys = jax.random.split(key, params_np.shape[0])
            maps = np.empty(
                (params_np.shape[0], self.npix), dtype=np.float32
            )
            for i in range(params_np.shape[0]):
                maps[i] = self._simulate_one_np(keys[i], params_np[i])
            return jnp.asarray(maps)

        return simulator

    def get_simulator(
        self, key: jax.random.PRNGKey, max_calls: Optional[int] = None
    ) -> Simulator:
        if self.backend == "jax":
            from sbibm_jax.tasks.spherical_grf.jax_backend import (
                make_jax_simulator,
            )
            return Simulator(
                task=self,
                simulator=make_jax_simulator(self),
                max_calls=max_calls,
            )
        return Simulator(
            task=self,
            simulator=self._healpy_simulator(),
            max_calls=max_calls,
        )

    def _config_files_dir(self) -> Path:
        return self.path / "files" / f"nside_{self.nside}"

    def _generate_observation(self, num_observation: int):
        """Seed-derived (theta_o (1,3), observation (1,npix)).

        Always uses the healpy backend so observed maps are identical
        whatever self.backend is. theta_o is drawn with an explicit
        float32 jax.random.uniform (bit-identical to the numpyro prior
        sample under default x32) so the result does not depend on the
        global jax_enable_x64 flag (which importing jax_healpy flips).
        """
        seed = self.observation_seeds[num_observation - 1]
        key_theta, key_sim = jax.random.split(jax.random.PRNGKey(seed))
        low = jnp.asarray(self.prior_params["low"], dtype=jnp.float32)
        high = jnp.asarray(self.prior_params["high"], dtype=jnp.float32)
        theta_o = jax.random.uniform(
            key_theta, (1, 3), minval=low, maxval=high,
            dtype=jnp.float32,
        )
        obs = self._healpy_simulator()(key_sim, theta_o)
        return theta_o, obs

    def _load_canonical(self, filename: str):
        """np.load handle for a canonical-config file, else None."""
        path = self._config_files_dir() / filename
        if self.noise_std == 0.0 and path.exists():
            return np.load(path)
        return None

    def get_observation(self, num_observation: int) -> jnp.ndarray:
        data = self._load_canonical("observations.npz")
        if data is not None:
            return jnp.asarray(
                data["observations"][num_observation - 1]
            ).reshape(1, -1)
        _, obs = self._generate_observation(num_observation)
        return obs

    def get_true_parameters(self, num_observation: int) -> jnp.ndarray:
        data = self._load_canonical("observations.npz")
        if data is not None:
            return jnp.asarray(
                data["true_parameters"][num_observation - 1]
            ).reshape(1, -1)
        theta_o, _ = self._generate_observation(num_observation)
        return theta_o

    def get_reference_posterior_samples(
        self, num_observation: int
    ) -> jnp.ndarray:
        data = self._load_canonical("reference_posterior_samples.npz")
        if data is not None:
            return jnp.asarray(data["samples"][num_observation - 1])
        raise FileNotFoundError(
            f"No precomputed reference posterior for task "
            f"{self.name!r} (nside={self.nside}, "
            f"noise_std={self.noise_std}). Precomputed references ship "
            f"only for the canonical noiseless nside 64/128 configs; "
            f"use _sample_reference_posterior(...) to sample it live."
        )

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        assert (num_observation is None) != (observation is None), (
            "Provide exactly one of num_observation or observation."
        )
        from sbibm_jax.tasks.spherical_grf.reference_posterior import (
            sample_reference_posterior,
        )
        if observation is None:
            observation = self.get_observation(num_observation)
        samples, _ = sample_reference_posterior(
            key,
            observation,
            nside=self.nside,
            noise_std=self.noise_std,
            low=self.prior_params["low"],
            high=self.prior_params["high"],
            num_samples=num_samples,
        )
        return samples

    def unflatten_data(self, data: jnp.ndarray) -> jnp.ndarray:
        return data.reshape(-1, self.npix)
