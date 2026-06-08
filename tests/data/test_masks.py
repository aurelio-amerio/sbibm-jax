# tests/data/test_masks.py
"""Masks: graph transforms, condition samplers, base + edge masks."""

import jax.numpy as jnp
import numpy as np
import pytest


class TestGraph:
    def test_moralize_symmetric(self):
        from sbibm_jax.data.masks.graph import moralize
        adj = jnp.array([[0, 1], [0, 0]], dtype=jnp.bool_)
        m = moralize(adj)
        assert bool(jnp.all(m == m.T))


class TestConditionSamplers:
    def test_posterior_mask_shape(self):
        from sbibm_jax.data.masks.condition import get_condition_mask_fn
        fn = get_condition_mask_fn("posterior")
        import jax
        m = fn(jax.random.PRNGKey(0), num_samples=3, theta_dim=2, x_dim=4)
        assert np.asarray(m).shape == (3, 6)
        assert bool(jnp.all(~m[:, :2]))   # theta not conditioned
        assert bool(jnp.all(m[:, 2:]))    # x conditioned
