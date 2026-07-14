"""Tests for the spherical_grf task (HEALPix GRF, polynomial Cl)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sbibm_jax import get_available_tasks, get_task
from sbibm_jax.tasks.simulator import SimulationBudgetExceeded
from sbibm_jax.tasks.spherical_grf.task import SphericalGRF, cl_target
from sbibm_jax.tasks.spherical_grf import reference_posterior as refpost


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


def _has_jax_healpy() -> bool:
    try:
        import jax_healpy  # noqa: F401
        return True
    except ImportError:
        return False


class TestJaxBackendGuard:
    def test_missing_extra_raises_informative(self, monkeypatch):
        import sys

        # A None entry in sys.modules makes `import jax_healpy` raise
        # ImportError even when the package is installed.
        monkeypatch.setitem(sys.modules, "jax_healpy", None)
        task = SphericalGRF(nside=8, backend="jax")
        with pytest.raises(ImportError, match=r"\[jaxhp\]"):
            task.get_simulator(jax.random.PRNGKey(0))


@pytest.mark.skipif(
    not _has_jax_healpy(), reason="[jaxhp] extra not installed"
)
class TestJaxSimulator:
    def test_shapes_and_dtype(self):
        task = SphericalGRF(nside=8, backend="jax")
        key = jax.random.PRNGKey(0)
        sim = task.get_simulator(key)
        theta = task.get_prior(key, num_samples=3)
        x = sim(key, theta)
        assert x.shape == (3, task.npix)
        assert x.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(x)))

    def test_deterministic_given_key(self):
        task = SphericalGRF(nside=8, backend="jax")
        key = jax.random.PRNGKey(5)
        theta = task.get_prior(key, num_samples=2)
        x1 = task.get_simulator(key)(key, theta)
        x2 = task.get_simulator(key)(key, theta)
        np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))

    def test_synalm_variance(self):
        # Mean |alm|^2 over many draws approximates Cl.
        from sbibm_jax.tasks.spherical_grf.jax_backend import (
            _alm_index_arrays, synalm,
        )

        lmax = 23
        cl = np.asarray(cl_target(jnp.array([0.0, -1.0, 0.0]), lmax))
        l_arr, m_arr = _alm_index_arrays(lmax)
        keys = jax.random.split(jax.random.PRNGKey(0), 500)
        alms = np.stack([
            np.asarray(synalm(k, jnp.asarray(cl), jnp.asarray(l_arr),
                              jnp.asarray(m_arr)))
            for k in keys
        ])
        est = np.mean(np.abs(alms) ** 2, axis=0)
        sel = l_arr >= 2
        ratio = est[sel] / cl[l_arr[sel]]
        assert np.abs(np.mean(ratio) - 1.0) < 0.05


@pytest.mark.slow
@pytest.mark.skipif(
    not _has_jax_healpy(), reason="[jaxhp] extra not installed"
)
class TestBackendParity:
    def test_mean_spectra_agree(self):
        import healpy as hp

        nside, n_maps = 32, 200
        theta_row = jnp.array([0.0, -1.0, 0.1])
        theta = jnp.tile(theta_row[None, :], (n_maps, 1))
        key = jax.random.PRNGKey(0)

        def mean_spectrum(task):
            x = np.asarray(
                task.get_simulator(key)(key, theta), dtype=np.float64
            )
            return np.mean(
                [hp.anafast(m, lmax=task.lmax) for m in x], axis=0
            )

        cl_hp = mean_spectrum(SphericalGRF(nside=nside))
        cl_jx = mean_spectrum(SphericalGRF(nside=nside, backend="jax"))
        cl_true = np.asarray(cl_target(theta_row, 3 * nside - 1))

        # Each backend against the analytic truth...
        for cl_hat in (cl_hp, cl_jx):
            ratio = cl_hat[2:] / cl_true[2:]
            assert np.max(np.abs(ratio - 1.0)) < 0.2
            assert np.mean(np.abs(ratio - 1.0)) < 0.03
        # ...and against each other (independent MC noise, ~sqrt(2)x).
        ratio = cl_jx[2:] / cl_hp[2:]
        assert np.max(np.abs(ratio - 1.0)) < 0.3
        assert np.mean(np.abs(ratio - 1.0)) < 0.05


class TestLogDensity:
    def _make(self, nside=8, noise_std=0.0):
        task = SphericalGRF(nside=nside, noise_std=noise_std)
        theta_o, obs = task._generate_observation(1)
        cl_hat = refpost.compute_cl_hat(obs, task.lmax)
        logdens = refpost.make_logdensity(
            cl_hat, task.noise_std, task.npix, task.lmax, task.ell0,
            task.prior_params["low"], task.prior_params["high"],
        )
        return task, theta_o, logdens

    def test_finite_and_differentiable(self):
        _, _, logdens = self._make()
        z = jnp.zeros(3)
        val = logdens(z)
        grad = jax.grad(logdens)(z)
        assert bool(jnp.isfinite(val))
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_finite_with_noise(self):
        _, _, logdens = self._make(noise_std=1.0)
        assert bool(jnp.isfinite(logdens(jnp.array([0.5, -0.5, 1.0]))))

    def test_higher_at_truth_than_far_away(self):
        task, theta_o, logdens = self._make()
        low = task.prior_params["low"]
        high = task.prior_params["high"]
        u = (theta_o[0] - low) / (high - low)
        z_true = jnp.log(u) - jnp.log1p(-u)  # logit
        z_far = jnp.array([6.0, -6.0, 6.0])  # extreme box corner
        assert float(logdens(z_true)) > float(logdens(z_far))


class TestObservationGeneration:
    def test_deterministic_and_shaped(self):
        task = SphericalGRF(nside=8)
        t1, o1 = task._generate_observation(3)
        t2, o2 = task._generate_observation(3)
        assert t1.shape == (1, 3) and o1.shape == (1, task.npix)
        np.testing.assert_array_equal(np.asarray(o1), np.asarray(o2))

    def test_distinct_observations(self):
        task = SphericalGRF(nside=8)
        _, o1 = task._generate_observation(1)
        _, o2 = task._generate_observation(2)
        assert not np.allclose(np.asarray(o1), np.asarray(o2))

    def test_backend_independent(self):
        t_hp, o_hp = SphericalGRF(nside=8)._generate_observation(1)
        t_jx, o_jx = SphericalGRF(
            nside=8, backend="jax"
        )._generate_observation(1)
        np.testing.assert_array_equal(np.asarray(o_hp), np.asarray(o_jx))
        np.testing.assert_array_equal(np.asarray(t_hp), np.asarray(t_jx))

    def test_get_observation_noncanonical_falls_back(self):
        # nside=8 is not a canonical config -> seed-derived generation.
        task = SphericalGRF(nside=8)
        obs = task.get_observation(1)
        theta = task.get_true_parameters(1)
        assert obs.shape == (1, task.npix)
        assert theta.shape == (1, 3)

    def test_reference_samples_noncanonical_raises(self):
        task = SphericalGRF(nside=8)
        with pytest.raises(
            FileNotFoundError, match="_sample_reference_posterior"
        ):
            task.get_reference_posterior_samples(1)


@pytest.mark.slow
class TestReferencePosteriorSmoke:
    def test_truth_within_credible_box(self):
        task = SphericalGRF(nside=16)
        theta_o, _ = task._generate_observation(1)
        samples = task._sample_reference_posterior(
            jax.random.PRNGKey(0), num_samples=2000, num_observation=1
        )
        assert samples.shape == (2000, 3)
        s = np.asarray(samples)
        truth = np.asarray(theta_o)[0]
        lo = np.quantile(s, 0.005, axis=0)
        hi = np.quantile(s, 0.995, axis=0)
        assert np.all(truth >= lo) and np.all(truth <= hi)
        # Posterior should be a lot tighter than the prior box on logA.
        assert np.std(s[:, 0]) < 0.4


class TestCanonicalFiles:
    @pytest.mark.parametrize("task_name", ["spherical_grf"])
    def test_shipped_observations_match_generation(self, task_name):
        task = get_task(task_name)
        npz = task.path / "files" / f"nside_{task.nside}"
        if not (npz / "observations.npz").exists():
            pytest.skip("canonical npz not generated yet")
        for n in (1, 5, 10):
            theta_gen, obs_gen = task._generate_observation(n)
            np.testing.assert_array_equal(
                np.asarray(task.get_observation(n)), np.asarray(obs_gen)
            )
            np.testing.assert_array_equal(
                np.asarray(task.get_true_parameters(n)),
                np.asarray(theta_gen),
            )

    @pytest.mark.parametrize("task_name", ["spherical_grf"])
    def test_shipped_reference_shape_and_support(self, task_name):
        task = get_task(task_name)
        npz = task.path / "files" / f"nside_{task.nside}"
        if not (npz / "reference_posterior_samples.npz").exists():
            pytest.skip("canonical npz not generated yet")
        s = np.asarray(task.get_reference_posterior_samples(1))
        assert s.shape == (10000, 3)
        assert np.all(s >= np.asarray(task.prior_params["low"]))
        assert np.all(s <= np.asarray(task.prior_params["high"]))


class TestReferenceCalibration:
    """Truth-vs-posterior calibration of the shipped references.

    If the likelihood is unbiased, z = (theta_true - post_mean) /
    post_std is ~N(0, 1) per observation, so the mean z over the 10
    canonical observations is ~N(0, 1/sqrt(10)); |mean z| < 0.8 is a
    2.5-sigma gate. The pre-fix references (likelihood summed to
    3*nside-1, aliasing-biased band included) failed this at mean
    z(n) = +2.5 (nside 64) and mean z(alpha) = +3.4 (nside 128).
    """

    @pytest.mark.parametrize(
        "task_name", ["spherical_grf", "spherical_grf_128"]
    )
    def test_truth_z_scores_unbiased(self, task_name):
        task = get_task(task_name)
        npz = task.path / "files" / f"nside_{task.nside}"
        if not (npz / "reference_posterior_samples.npz").exists():
            pytest.skip("canonical npz not generated yet")
        zs = []
        for n in range(1, task.num_observations + 1):
            s = np.asarray(
                task.get_reference_posterior_samples(n),
                dtype=np.float64,
            )
            truth = np.asarray(
                task.get_true_parameters(n), dtype=np.float64
            )[0]
            zs.append((truth - s.mean(0)) / s.std(0, ddof=1))
        mean_z = np.mean(zs, axis=0)
        assert np.all(np.abs(mean_z) < 0.8), (
            f"reference posteriors biased: mean z (logA, n, alpha) "
            f"= {mean_z}"
        )
