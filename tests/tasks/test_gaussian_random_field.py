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
