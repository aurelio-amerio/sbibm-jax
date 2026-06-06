"""Tests for the Toy Gravitational Lensing task."""

import jax
import jax.numpy as jnp
import pytest

from sbibm_jax.tasks.toy_lensing.task import ToyLensing


class TestPrior:
    def test_shape(self):
        task = ToyLensing(resolution=32)
        key = jax.random.PRNGKey(42)
        samples = task.get_prior(key, num_samples=50)
        assert samples.shape == (50, 2)

    def test_single_sample(self):
        task = ToyLensing(resolution=32)
        sample = task.get_prior(jax.random.PRNGKey(0), num_samples=1)
        assert sample.shape == (1, 2)

    def test_different_keys_give_different_samples(self):
        task = ToyLensing(resolution=32)
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        s1 = task.get_prior(k1, num_samples=5)
        s2 = task.get_prior(k2, num_samples=5)
        assert not jnp.allclose(s1, s2)

    def test_bounds(self):
        task = ToyLensing(resolution=32)
        key = jax.random.PRNGKey(99)
        samples = task.get_prior(key, num_samples=500)
        radius = samples[:, 0]
        width = samples[:, 1]
        assert bool(jnp.all(radius >= 0.1))
        assert bool(jnp.all(radius <= 1.1))
        assert bool(jnp.all(width >= 0.01))
        assert bool(jnp.all(width <= 0.31))

    def test_metadata(self):
        task = ToyLensing(resolution=32)
        assert task.dim_parameters == 2
        assert task.dim_data == 32 ** 2
        assert task.name == "toy_lensing"


class TestSimulator:
    def test_shape_flattened(self):
        task = ToyLensing(resolution=32)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
        theta = task.get_prior(k1, num_samples=5)
        sim = task.get_simulator(k2)
        data = sim(k3, theta)
        assert data.shape == (5, 32 * 32)

    def test_unflatten_to_image(self):
        task = ToyLensing(resolution=32)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(1), 3)
        theta = task.get_prior(k1, num_samples=4)
        sim = task.get_simulator(k2)
        data = sim(k3, theta)
        images = task.unflatten_data(data)
        assert images.shape == (4, 32, 32)

    def test_custom_resolution(self):
        task = ToyLensing(resolution=16)
        assert task.dim_data == 16 * 16
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(2), 3)
        theta = task.get_prior(k1, num_samples=3)
        sim = task.get_simulator(k2)
        data = sim(k3, theta)
        assert data.shape == (3, 16 * 16)
        images = task.unflatten_data(data)
        assert images.shape == (3, 16, 16)

    def test_fields_are_finite(self):
        task = ToyLensing(resolution=16)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(3), 3)
        theta = task.get_prior(k1, num_samples=10)
        sim = task.get_simulator(k2)
        data = sim(k3, theta)
        assert bool(jnp.all(jnp.isfinite(data)))

    def test_deterministic_same_key(self):
        task = ToyLensing(resolution=16)
        k1, k2 = jax.random.split(jax.random.PRNGKey(4))
        theta = task.get_prior(k1, num_samples=4)
        sim1 = task.get_simulator(k1)
        d1 = sim1(k2, theta)
        sim2 = task.get_simulator(k1)
        d2 = sim2(k2, theta)
        assert jnp.allclose(d1, d2)

    def test_different_keys_give_different_output(self):
        task = ToyLensing(resolution=16)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(5), 3)
        theta = task.get_prior(k1, num_samples=4)
        sim = task.get_simulator(k1)
        d1 = sim(k2, theta)
        d2 = sim(k3, theta)
        assert not jnp.allclose(d1, d2)

    def test_budget_exceeded(self):
        from sbibm_jax.tasks.simulator import SimulationBudgetExceeded

        task = ToyLensing(resolution=16)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(6), 3)
        theta = task.get_prior(k1, num_samples=20)
        sim = task.get_simulator(k2, max_calls=10)
        with pytest.raises(SimulationBudgetExceeded):
            sim(k3, theta)


class TestDistributional:
    def test_pixel_mean_approx_zero(self):
        # Each image is normalized to zero mean before noise; noise mean is 0.
        task = ToyLensing(resolution=16)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(7), 3)
        theta = task.get_prior(k1, num_samples=200)
        sim = task.get_simulator(k2)
        data = sim(k3, theta)
        images = task.unflatten_data(data)
        # Per-image spatial mean
        per_image_means = images.mean(axis=(1, 2))
        mean_of_means = float(jnp.mean(per_image_means))
        assert abs(mean_of_means) < 0.1

    def test_pixel_std_approx_1(self):
        # After normalisation mu has std=1; adding N(0, 0.3^2) gives std=sqrt(1+0.09)~1.044.
        task = ToyLensing(resolution=16)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(8), 3)
        theta = task.get_prior(k1, num_samples=200)
        sim = task.get_simulator(k2)
        data = sim(k3, theta)
        images = task.unflatten_data(data)
        # Per-image spatial std
        per_image_stds = images.std(axis=(1, 2))
        mean_std = float(jnp.mean(per_image_stds))
        expected_std = (1.0 + 0.09) ** 0.5  # approx 1.044
        assert abs(mean_std - expected_std) < 0.1


class TestObservations:
    def test_true_parameters_shape(self):
        task = ToyLensing(resolution=16)
        theta = task.get_true_parameters(1)
        assert theta.shape == (1, 2)

    def test_true_parameters_deterministic(self):
        task = ToyLensing(resolution=16)
        a = task.get_true_parameters(1)
        b = task.get_true_parameters(1)
        c = task.get_true_parameters(2)
        assert jnp.allclose(a, b)
        assert not jnp.allclose(a, c)

    def test_true_parameters_within_bounds(self):
        task = ToyLensing(resolution=16)
        for n in range(1, task.num_observations + 1):
            theta = task.get_true_parameters(n)
            assert bool(jnp.all(theta[:, 0] >= 0.1))
            assert bool(jnp.all(theta[:, 0] <= 1.1))
            assert bool(jnp.all(theta[:, 1] >= 0.01))
            assert bool(jnp.all(theta[:, 1] <= 0.31))

    def test_observation_shape(self):
        task = ToyLensing(resolution=16)
        obs = task.get_observation(1)
        assert obs.shape == (1, task.dim_data)

    def test_observation_deterministic(self):
        task = ToyLensing(resolution=16)
        o1 = task.get_observation(1)
        o2 = task.get_observation(1)
        assert jnp.allclose(o1, o2)


class TestReferenceSampler:
    def test_raises_not_implemented(self):
        task = ToyLensing(resolution=16)
        with pytest.raises(NotImplementedError):
            task._sample_reference_posterior(
                jax.random.PRNGKey(0), num_samples=10, num_observation=1
            )


class TestRegistry:
    def test_get_task_returns_instance(self):
        from sbibm_jax import get_task

        task = get_task("toy_lensing")
        assert isinstance(task, ToyLensing)
        assert task.dim_data == 1024

    def test_get_task_passes_kwargs(self):
        from sbibm_jax import get_task

        task = get_task("toy_lensing", resolution=16)
        assert task.dim_data == 256

    def test_available_tasks_includes_toy_lensing(self):
        from sbibm_jax import get_available_tasks

        assert "toy_lensing" in get_available_tasks()
