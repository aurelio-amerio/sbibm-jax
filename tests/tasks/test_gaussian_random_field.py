"""Tests for the Gaussian Random Field field-inference task."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sbibm_jax.tasks.gaussian_random_field.task import GaussianRandomField


class TestPrior:
    def test_shape(self):
        task = GaussianRandomField(field_size=16)
        key = jax.random.PRNGKey(42)
        samples = task.get_prior(key, num_samples=50)
        assert samples.shape == (50, 2)

    def test_single_sample(self):
        task = GaussianRandomField(field_size=16)
        sample = task.get_prior(jax.random.PRNGKey(0), num_samples=1)
        assert sample.shape == (1, 2)

    def test_different_keys_give_different_samples(self):
        task = GaussianRandomField(field_size=16)
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        s1 = task.get_prior(k1, num_samples=5)
        s2 = task.get_prior(k2, num_samples=5)
        assert not jnp.allclose(s1, s2)

    def test_metadata(self):
        task = GaussianRandomField(field_size=16)
        assert task.dim_parameters == 2
        assert task.dim_data == 16 * 16
        assert task.name == "gaussian_random_field"


class TestSimulator:
    def test_shape_flattened(self):
        task = GaussianRandomField(field_size=16)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
        theta = task.get_prior(k1, num_samples=20)
        sim = task.get_simulator(k2)
        data = sim(k3, theta)
        assert data.shape == (20, 16 * 16)

    def test_unflatten_to_image(self):
        task = GaussianRandomField(field_size=16)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(1), 3)
        theta = task.get_prior(k1, num_samples=4)
        sim = task.get_simulator(k2)
        data = sim(k3, theta)
        images = task.unflatten_data(data)
        assert images.shape == (4, 16, 16)

    def test_fields_are_real_and_finite(self):
        task = GaussianRandomField(field_size=16)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(2), 3)
        theta = task.get_prior(k1, num_samples=32)
        sim = task.get_simulator(k2)
        data = sim(k3, theta)
        assert jnp.isrealobj(data)
        assert bool(jnp.all(jnp.isfinite(data)))

    def test_fields_are_zero_mean(self):
        # DC mode is zeroed, so each field's spatial mean is exactly ~0.
        task = GaussianRandomField(field_size=16)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(3), 3)
        theta = task.get_prior(k1, num_samples=16)
        sim = task.get_simulator(k2)
        images = task.unflatten_data(sim(k3, theta))
        means = images.mean(axis=(1, 2))
        assert jnp.allclose(means, 0.0, atol=1e-4)

    def test_deterministic_same_key(self):
        task = GaussianRandomField(field_size=16)
        k1, k2 = jax.random.split(jax.random.PRNGKey(4))
        theta = task.get_prior(k1, num_samples=8)
        sim = task.get_simulator(k1)
        d1 = sim(k2, theta)
        sim2 = task.get_simulator(k1)
        d2 = sim2(k2, theta)
        assert jnp.allclose(d1, d2)

    def test_log_std_scales_field_exactly(self):
        # With identical noise, raising log_std by c multiplies the field by
        # exp(c), because the field is linear in exp(log_std).
        task = GaussianRandomField(field_size=16)
        key = jax.random.PRNGKey(5)
        theta0 = jnp.array([[0.0, 3.0]])
        theta1 = jnp.array([[0.7, 3.0]])
        sim = task.get_simulator(key)
        f0 = sim(key, theta0)
        sim2 = task.get_simulator(key)
        f1 = sim2(key, theta1)
        assert jnp.allclose(f1, jnp.exp(0.7) * f0, atol=1e-3)

    def test_budget_exceeded(self):
        from sbibm_jax.tasks.simulator import SimulationBudgetExceeded

        task = GaussianRandomField(field_size=16)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(6), 3)
        theta = task.get_prior(k1, num_samples=20)
        sim = task.get_simulator(k2, max_calls=10)
        with pytest.raises(SimulationBudgetExceeded):
            sim(k3, theta)


def _radial_power_spectrum(images, N):
    """Mean |FFT|^2 over samples, radially binned by integer radius.

    Returns (k_base, power) over radii 1..N//2-1, where k_base is the
    d=1 grid frequency magnitude (so power ~ k_base**(-alpha)).
    """
    F = np.fft.fft2(np.asarray(images), axes=(-2, -1))
    power = np.mean(np.abs(F) ** 2, axis=0)  # (N, N)
    k0 = np.fft.fftfreq(N, d=1.0)
    kx, ky = np.meshgrid(k0, k0, indexing="ij")
    knorm = np.sqrt(kx**2 + ky**2)
    radius = np.round(knorm * N).astype(int)
    ks, ps = [], []
    for r in range(1, N // 2):
        mask = radius == r
        if mask.sum() == 0:
            continue
        ks.append(r / N)
        ps.append(power[mask].mean())
    return np.array(ks), np.array(ps)


class TestPowerSpectrum:
    def test_slope_matches_alpha(self):
        N = 32
        alpha = 3.0
        task = GaussianRandomField(field_size=N)
        theta = jnp.tile(jnp.array([0.0, alpha]), (2000, 1))
        sim = task.get_simulator(jax.random.PRNGKey(7))
        images = task.unflatten_data(sim(jax.random.PRNGKey(8), theta))

        ks, ps = _radial_power_spectrum(images, N)
        slope = np.polyfit(np.log(ks), np.log(ps), 1)[0]
        assert slope == pytest.approx(-alpha, abs=0.3)


class TestReferenceSampler:
    def test_shape_is_field_space(self):
        task = GaussianRandomField(field_size=16)
        samples = task._sample_reference_posterior(
            jax.random.PRNGKey(0), num_samples=64, num_observation=1
        )
        assert samples.shape == (64, 16 * 16)

    def test_observation_parameters_deterministic(self):
        task = GaussianRandomField(field_size=16)
        a = task._get_observation_parameters(1)
        b = task._get_observation_parameters(1)
        c = task._get_observation_parameters(2)
        assert a.shape == (1, 2)
        assert jnp.allclose(a, b)
        assert not jnp.allclose(a, c)

    def test_num_observation_matches_explicit_theta(self):
        task = GaussianRandomField(field_size=16)
        theta_o = task._get_observation_parameters(3)
        key = jax.random.PRNGKey(11)
        from_idx = task._sample_reference_posterior(
            key, num_samples=8, num_observation=3
        )
        from_theta = task._sample_reference_posterior(
            key, num_samples=8, observation=theta_o
        )
        assert jnp.allclose(from_idx, from_theta)

    def test_reference_spectrum_matches_theta_o(self):
        N = 32
        task = GaussianRandomField(field_size=N)
        theta_o = jnp.array([[0.0, 3.0]])
        samples = task._sample_reference_posterior(
            jax.random.PRNGKey(12), num_samples=2000, observation=theta_o
        )
        images = task.unflatten_data(samples)
        ks, ps = _radial_power_spectrum(images, N)
        slope = np.polyfit(np.log(ks), np.log(ps), 1)[0]
        assert slope == pytest.approx(-3.0, abs=0.3)


class TestRegistry:
    def test_get_task_returns_instance(self):
        from sbibm_jax import get_task

        task = get_task("gaussian_random_field")
        assert isinstance(task, GaussianRandomField)
        assert task.dim_data == 32 * 32  # default field_size=32

    def test_get_task_passes_kwargs(self):
        from sbibm_jax import get_task

        task = get_task("gaussian_random_field", field_size=16)
        assert task.dim_data == 16 * 16

    def test_available_tasks_includes_grf(self):
        from sbibm_jax import get_available_tasks

        assert "gaussian_random_field" in get_available_tasks()


@pytest.mark.slow
class TestOracleCrossCheck:
    def test_power_spectrum_matches_fyeldgenerator(self):
        FyeldGenerator = pytest.importorskip("FyeldGenerator")
        generate_field = FyeldGenerator.generate_field

        N = 32
        alpha = 3.0
        log_std = 0.0

        # --- numpy oracle ---
        rng = np.random.default_rng(0)

        def distribution(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        def power_spectrum(k):
            return np.power(k, -alpha) * np.exp(log_std) ** 2

        oracle = np.stack([
            generate_field(
                distribution, power_spectrum, (N, N),
                unit_length=1.0 / (abs(alpha) + 1e-7),
            )
            for _ in range(2000)
        ])
        ks_o, ps_o = _radial_power_spectrum(oracle, N)
        slope_o = np.polyfit(np.log(ks_o), np.log(ps_o), 1)[0]

        # --- jax port ---
        task = GaussianRandomField(field_size=N)
        theta = jnp.tile(jnp.array([log_std, alpha]), (2000, 1))
        sim = task.get_simulator(jax.random.PRNGKey(0))
        images = task.unflatten_data(sim(jax.random.PRNGKey(1), theta))
        ks_j, ps_j = _radial_power_spectrum(images, N)
        slope_j = np.polyfit(np.log(ks_j), np.log(ps_j), 1)[0]

        assert slope_j == pytest.approx(slope_o, abs=0.2)
