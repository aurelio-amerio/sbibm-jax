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
