"""Tests for the spherical_grf task (HEALPix GRF, polynomial Cl)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sbibm_jax import get_available_tasks, get_task
from sbibm_jax.tasks.simulator import SimulationBudgetExceeded
from sbibm_jax.tasks.spherical_grf.task import SphericalGRF, cl_target


class TestClTarget:
    def test_shape_and_monopole_dipole_zero(self):
        cl = cl_target(jnp.array([0.0, -1.0, 0.1]), lmax=47)
        assert cl.shape == (48,)
        assert cl[0] == 0.0 and cl[1] == 0.0

    @pytest.mark.parametrize(
        "theta",
        [
            [-2.0, -3.0, -0.5],
            [2.0, 0.0, 0.5],
            [-2.0, 0.0, -0.5],
            [2.0, -3.0, 0.5],
        ],
    )
    def test_positive_at_prior_corners(self, theta):
        cl = cl_target(jnp.array(theta), lmax=383)
        assert bool(jnp.all(cl[2:] > 0.0))
        assert bool(jnp.all(jnp.isfinite(cl)))

    def test_pivot_value_is_amplitude(self):
        # At ell = ell0 the log-polynomial reduces to logA.
        log_a = 0.7
        cl = cl_target(jnp.array([log_a, -1.3, 0.2]), lmax=191, ell0=64.0)
        assert np.isclose(float(cl[64]), np.exp(log_a), rtol=1e-5)


class TestConstructor:
    def test_defaults(self):
        task = SphericalGRF()
        assert task.nside == 64
        assert task.npix == 12 * 64 * 64
        assert task.dim_x == task.npix
        assert task.dim_theta == 3
        assert task.lmax == 3 * 64 - 1
        assert task.noise_std == 0.0
        assert task.backend == "healpy"
        assert task.name == "spherical_grf"

    @pytest.mark.parametrize("bad_nside", [0, 3, 48, 2048, -64])
    def test_invalid_nside_raises(self, bad_nside):
        with pytest.raises(ValueError, match="nside"):
            SphericalGRF(nside=bad_nside)

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend"):
            SphericalGRF(backend="torch")

    def test_hf_hints(self):
        task = SphericalGRF()
        assert task.hf_x_kind == "healpix"
        assert task.hf_x_shape == (task.npix,)
        assert task.hf_stats_axes == {"theta": (0,), "x": (0, 1)}
        assert task.hf_split_sizes == {
            "train": 100_000, "validation": 10_000, "test": 10_000,
        }

    def test_hf_split_sizes_128(self):
        task = SphericalGRF(nside=128)
        assert task.hf_split_sizes == {
            "train": 30_000, "validation": 5_000, "test": 5_000,
        }


class TestPrior:
    def test_samples_within_box(self):
        task = SphericalGRF(nside=8)
        theta = task.get_prior(jax.random.PRNGKey(0), num_samples=500)
        assert theta.shape == (500, 3)
        low = np.array([-2.0, -3.0, -0.5])
        high = np.array([2.0, 0.0, 0.5])
        assert np.all(np.asarray(theta) >= low)
        assert np.all(np.asarray(theta) <= high)

    def test_prior_params_exposed(self):
        task = SphericalGRF(nside=8)
        np.testing.assert_allclose(
            np.asarray(task.prior_params["low"]), [-2.0, -3.0, -0.5]
        )
        np.testing.assert_allclose(
            np.asarray(task.prior_params["high"]), [2.0, 0.0, 0.5]
        )


class TestRegistry:
    def test_get_task_default(self):
        task = get_task("spherical_grf")
        assert isinstance(task, SphericalGRF)
        assert task.nside == 64

    def test_get_task_128_alias(self):
        task = get_task("spherical_grf_128")
        assert task.nside == 128
        assert task.name == "spherical_grf_128"
        assert task.name_display == "Spherical GRF 128"

    def test_available_tasks_contains_both(self):
        names = get_available_tasks()
        assert "spherical_grf" in names
        assert "spherical_grf_128" in names


class TestHealpySimulator:
    def test_shapes_and_dtype(self):
        task = SphericalGRF(nside=8)
        key = jax.random.PRNGKey(0)
        sim = task.get_simulator(key)
        theta = task.get_prior(key, num_samples=3)
        x = sim(key, theta)
        assert x.shape == (3, task.npix)
        assert x.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(x)))

    def test_deterministic_given_key(self):
        task = SphericalGRF(nside=8)
        key = jax.random.PRNGKey(7)
        theta = task.get_prior(key, num_samples=2)
        x1 = task.get_simulator(key)(key, theta)
        x2 = task.get_simulator(key)(key, theta)
        np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))

    def test_rows_differ(self):
        task = SphericalGRF(nside=8)
        key = jax.random.PRNGKey(1)
        theta = jnp.tile(jnp.array([[0.0, -1.0, 0.0]]), (2, 1))
        x = task.get_simulator(key)(key, theta)
        assert not np.allclose(np.asarray(x[0]), np.asarray(x[1]))

    def test_noise_increases_variance(self):
        key = jax.random.PRNGKey(2)
        theta = jnp.array([[-2.0, 0.0, 0.0]])
        clean = SphericalGRF(nside=8)
        noisy = SphericalGRF(nside=8, noise_std=2.0)
        x_clean = clean.get_simulator(key)(key, theta)
        x_noisy = noisy.get_simulator(key)(key, theta)
        v_clean = float(np.var(np.asarray(x_clean)))
        v_noisy = float(np.var(np.asarray(x_noisy)))
        # noise_std=2 adds variance 4 on top of Cl-level ~exp(-2).
        assert v_noisy > v_clean + 2.0

    def test_budget_enforced(self):
        task = SphericalGRF(nside=8)
        key = jax.random.PRNGKey(3)
        sim = task.get_simulator(key, max_calls=2)
        theta = task.get_prior(key, num_samples=3)
        with pytest.raises(SimulationBudgetExceeded):
            sim(key, theta)

    def test_single_row_parameters(self):
        task = SphericalGRF(nside=8)
        key = jax.random.PRNGKey(4)
        x = task.get_simulator(key)(key, jnp.array([0.0, -1.0, 0.0]))
        assert x.shape == (1, task.npix)


@pytest.mark.slow
class TestSpectrumMC:
    def test_mean_anafast_matches_cl(self):
        import healpy as hp

        nside, n_maps = 32, 200
        task = SphericalGRF(nside=nside)
        theta = jnp.tile(jnp.array([[0.0, -1.0, 0.1]]), (n_maps, 1))
        key = jax.random.PRNGKey(0)
        x = np.asarray(task.get_simulator(key)(key, theta), dtype=np.float64)
        cl_hat = np.mean(
            [hp.anafast(m, lmax=task.lmax) for m in x], axis=0
        )
        cl_true = np.asarray(cl_target(theta[0], task.lmax, task.ell0))
        ratio = cl_hat[2:] / cl_true[2:]
        assert np.max(np.abs(ratio - 1.0)) < 0.2
        assert np.mean(np.abs(ratio - 1.0)) < 0.03

    def test_mean_anafast_with_noise_matches_cl_plus_nl(self):
        import healpy as hp

        nside, n_maps, noise_std = 16, 200, 10.0
        task = SphericalGRF(nside=nside, noise_std=noise_std)
        theta = jnp.tile(jnp.array([[-2.0, 0.0, 0.0]]), (n_maps, 1))
        key = jax.random.PRNGKey(1)
        x = np.asarray(task.get_simulator(key)(key, theta), dtype=np.float64)
        cl_hat = np.mean(
            [hp.anafast(m, lmax=task.lmax) for m in x], axis=0
        )
        nl = noise_std**2 * 4.0 * np.pi / task.npix
        cl_true = np.asarray(cl_target(theta[0], task.lmax, task.ell0))
        ratio = cl_hat[2:] / (cl_true[2:] + nl)
        assert np.max(np.abs(ratio - 1.0)) < 0.2
        assert np.mean(np.abs(ratio - 1.0)) < 0.03
