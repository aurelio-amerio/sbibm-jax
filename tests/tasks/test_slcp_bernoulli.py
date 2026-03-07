"""Tests for SLCP and Bernoulli GLM tasks."""

import jax
import jax.numpy as jnp
import pytest

from sbibm_jax import get_task


class TestSLCP:
    @pytest.fixture
    def task(self):
        return get_task("slcp")

    def test_prior_shape(self, task):
        key = jax.random.PRNGKey(0)
        samples = task.get_prior(key, num_samples=50)
        assert samples.shape == (50, 5)

    def test_simulator_shape(self, task):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
        samples = task.get_prior(k1, num_samples=20)
        sim = task.get_simulator(k2)
        data = sim(k3, samples)
        assert data.shape == (20, 8)

    def test_observation_shape(self, task):
        obs = task.get_observation(1)
        assert obs.shape == (1, 8)

    def test_reference_posterior_shape(self, task):
        ref = task.get_reference_posterior_samples(1)
        assert ref.shape[1] == 5
        assert ref.shape[0] > 0

    def test_true_parameters_shape(self, task):
        tp = task.get_true_parameters(1)
        assert tp.shape == (1, 5)

    def test_reference_posterior_not_implemented(self, task):
        with pytest.raises(NotImplementedError):
            task._sample_reference_posterior(
                jax.random.PRNGKey(0), 10, num_observation=1
            )


class TestSLCPDistractors:
    @pytest.fixture
    def task(self):
        return get_task("slcp_distractors")

    def test_prior_shape(self, task):
        key = jax.random.PRNGKey(0)
        samples = task.get_prior(key, num_samples=20)
        assert samples.shape == (20, 5)

    def test_simulator_shape(self, task):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
        samples = task.get_prior(k1, num_samples=10)
        sim = task.get_simulator(k2)
        data = sim(k3, samples)
        assert data.shape == (10, 100)

    def test_observation_shape(self, task):
        obs = task.get_observation(1)
        assert obs.shape == (1, 100)


class TestBernoulliGLM:
    @pytest.fixture
    def task(self):
        return get_task("bernoulli_glm")

    def test_prior_shape(self, task):
        key = jax.random.PRNGKey(0)
        samples = task.get_prior(key, num_samples=50)
        assert samples.shape == (50, 10)

    def test_simulator_shape(self, task):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
        samples = task.get_prior(k1, num_samples=5)
        sim = task.get_simulator(k2)
        data = sim(k3, samples)
        assert data.shape == (5, 10)

    def test_observation_shape(self, task):
        obs = task.get_observation(1)
        assert obs.shape == (1, 10)

    def test_reference_posterior_shape(self, task):
        ref = task.get_reference_posterior_samples(1)
        assert ref.shape[1] == 10
        assert ref.shape[0] > 0

    def test_reference_posterior_not_implemented(self, task):
        with pytest.raises(NotImplementedError):
            task._sample_reference_posterior(
                jax.random.PRNGKey(0), 10, num_observation=1
            )


class TestBernoulliGLMRaw:
    @pytest.fixture
    def task(self):
        return get_task("bernoulli_glm_raw")

    def test_prior_shape(self, task):
        key = jax.random.PRNGKey(0)
        samples = task.get_prior(key, num_samples=10)
        assert samples.shape == (10, 10)

    def test_simulator_shape(self, task):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
        samples = task.get_prior(k1, num_samples=5)
        sim = task.get_simulator(k2)
        data = sim(k3, samples)
        assert data.shape == (5, 100)

    def test_observation_shape(self, task):
        obs = task.get_observation(1)
        assert obs.shape == (1, 100)
