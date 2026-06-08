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


class TestBaseMasks:
    @pytest.mark.parametrize("name,dt,dx", [
        ("two_moons", 2, 2), ("gaussian_linear", 10, 10),
        ("gaussian_linear_uniform", 10, 10), ("gaussian_mixture", 2, 2),
        ("slcp", 5, 8),
    ])
    def test_base_mask_shape(self, name, dt, dx):
        from sbibm_jax.data.masks.base import get_base_mask_fn
        fn = get_base_mask_fn(name, dim_theta=dt, dim_x=dx)
        node_ids = jnp.arange(dt + dx)
        mask = fn(node_ids, None)
        assert np.asarray(mask).shape == (dt + dx, dt + dx)
        assert np.asarray(mask).dtype == np.bool_

    def test_unsupported_raises(self):
        from sbibm_jax.data.masks.base import get_base_mask_fn
        with pytest.raises(NotImplementedError):
            get_base_mask_fn("bernoulli_glm", dim_theta=10, dim_x=10)


class TestEdgeMasks:
    @pytest.mark.parametrize("variant", ["undirected", "directed", "none"])
    def test_edge_variants(self, variant):
        from sbibm_jax.data.masks import get_edge_mask_fn
        fn = get_edge_mask_fn("two_moons", variant, dim_theta=2, dim_x=2)
        node_ids = jnp.arange(4)
        cond = jnp.zeros(4, dtype=jnp.bool_)
        out = fn(node_ids, cond)
        if variant == "none":
            assert out is None
        else:
            assert np.asarray(out).shape == (4, 4)

    def test_unknown_variant_raises(self):
        from sbibm_jax.data.masks import get_edge_mask_fn
        with pytest.raises(NotImplementedError):
            get_edge_mask_fn("two_moons", "bogus", dim_theta=2, dim_x=2)
