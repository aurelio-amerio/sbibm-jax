"""Tests for the spherical_grf task (HEALPix GRF, polynomial Cl)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sbibm_jax import get_available_tasks, get_task
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
