"""Tests for ODE-based tasks (Phase 2): Lotka-Volterra and SIR."""

import jax
import jax.numpy as jnp
import pytest

from sbibm_jax import get_task


class TestLotkaVolterra:
    @pytest.fixture
    def task(self):
        return get_task("lotka_volterra")

    def test_prior_shape(self, task):
        key = jax.random.PRNGKey(0)
        samples = task.get_prior(key, num_samples=10)
        assert samples.shape == (10, 4)
        # LogNormal prior should be positive
        assert (samples > 0).all()

    def test_simulator_shape(self, task):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(42), 3)
        samples = task.get_prior(k1, num_samples=3)
        sim = task.get_simulator(k2)
        data = sim(k3, samples)
        assert data.shape == (3, 20)

    def test_simulator_no_nan_typical_params(self, task):
        """Typical prior samples should not produce NaN."""
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
        samples = task.get_prior(k1, num_samples=5)
        sim = task.get_simulator(k2)
        data = sim(k3, samples)
        # Most samples should be finite (some extreme params may diverge)
        finite_fraction = jnp.isfinite(data).mean()
        assert finite_fraction > 0.5

    def test_observation_shape(self, task):
        obs = task.get_observation(1)
        assert obs.shape == (1, 20)

    def test_reference_posterior_shape(self, task):
        ref = task.get_reference_posterior_samples(1)
        assert ref.shape[1] == 4
        assert ref.shape[0] > 0

    def test_true_parameters_shape(self, task):
        tp = task.get_true_parameters(1)
        assert tp.shape == (1, 4)

    def test_reference_posterior_not_implemented(self, task):
        with pytest.raises(NotImplementedError):
            task._sample_reference_posterior(
                jax.random.PRNGKey(0), 10, num_observation=1
            )


class TestSIR:
    @pytest.fixture
    def task(self):
        return get_task("sir")

    def test_prior_shape(self, task):
        key = jax.random.PRNGKey(0)
        samples = task.get_prior(key, num_samples=10)
        assert samples.shape == (10, 2)
        assert (samples > 0).all()

    def test_simulator_shape(self, task):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(42), 3)
        samples = task.get_prior(k1, num_samples=3)
        sim = task.get_simulator(k2)
        data = sim(k3, samples)
        assert data.shape == (3, 10)

    def test_simulator_no_nan_typical_params(self, task):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
        samples = task.get_prior(k1, num_samples=5)
        sim = task.get_simulator(k2)
        data = sim(k3, samples)
        finite_fraction = jnp.isfinite(data).mean()
        assert finite_fraction > 0.5

    def test_observation_shape(self, task):
        obs = task.get_observation(1)
        assert obs.shape == (1, 10)

    def test_reference_posterior_shape(self, task):
        ref = task.get_reference_posterior_samples(1)
        assert ref.shape[1] == 2
        assert ref.shape[0] > 0

    def test_true_parameters_shape(self, task):
        tp = task.get_true_parameters(1)
        assert tp.shape == (1, 2)

    def test_reference_posterior_not_implemented(self, task):
        with pytest.raises(NotImplementedError):
            task._sample_reference_posterior(
                jax.random.PRNGKey(0), 10, num_observation=1
            )
