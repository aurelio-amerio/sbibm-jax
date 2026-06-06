"""Tests for hf.generate: seeding, chunking, validity policies."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sbibm_jax import get_task
from sbibm_jax.hf.generate import (
    derive_task_keys,
    generate_samples,
    iter_chunks,
)


class TestSeeding:
    def test_stable_across_calls(self):
        k1 = derive_task_keys("gaussian_linear", master_seed=0)
        k2 = derive_task_keys("gaussian_linear", master_seed=0)
        for s in ("train", "validation", "test"):
            assert bool(jnp.all(k1[s] == k2[s]))

    def test_different_seeds_differ(self):
        k1 = derive_task_keys("gaussian_linear", master_seed=0)
        k2 = derive_task_keys("gaussian_linear", master_seed=1)
        assert not bool(jnp.all(k1["train"] == k2["train"]))

    def test_different_tasks_differ(self):
        k_a = derive_task_keys("gaussian_linear", master_seed=0)
        k_b = derive_task_keys("two_moons", master_seed=0)
        assert not bool(jnp.all(k_a["train"] == k_b["train"]))

    def test_splits_are_independent(self):
        keys = derive_task_keys("gaussian_linear", master_seed=0)
        assert not bool(jnp.all(keys["train"] == keys["validation"]))
        assert not bool(jnp.all(keys["train"] == keys["test"]))
        assert not bool(jnp.all(keys["validation"] == keys["test"]))


class TestChunkedGeneration:
    def test_iter_chunks_sums_to_n(self):
        task = get_task("gaussian_linear")
        key = jax.random.PRNGKey(0)
        chunks = list(iter_chunks(
            task, key, n=10, resample_invalid=False, chunk_size=4,
            dtype=np.float32, max_factor=2.0, stats={},
        ))
        thetas = np.concatenate([c[0] for c in chunks], axis=0)
        xs = np.concatenate([c[1] for c in chunks], axis=0)
        assert thetas.shape == (10, task.dim_parameters)
        assert xs.shape == (10, task.dim_data)

    def test_generate_samples_shapes_and_dtype(self):
        task = get_task("two_moons")
        key = jax.random.PRNGKey(0)
        thetas, xs, stats = generate_samples(
            task, key, n=8, chunk_size=4,
        )
        assert thetas.shape == (8, task.dim_parameters)
        assert xs.shape == (8, task.dim_data)
        assert thetas.dtype == np.float32
        assert xs.dtype == np.float32
        assert stats["rejected"] == 0
        assert stats["total_drawn"] == 8

    def test_generate_samples_reproducibility(self):
        task = get_task("gaussian_linear")
        key = jax.random.PRNGKey(123)
        t1, x1, _ = generate_samples(task, key, n=8, chunk_size=4)
        t2, x2, _ = generate_samples(task, key, n=8, chunk_size=4)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(x1, x2)

    def test_no_nan_in_clean_task(self):
        task = get_task("gaussian_linear")
        thetas, xs, _ = generate_samples(
            task, jax.random.PRNGKey(0), n=16, chunk_size=8,
        )
        assert np.isfinite(thetas).all()
        assert np.isfinite(xs).all()


class TestDefaultValidityPolicy:
    def test_finite_assert_raises_on_nan(self):
        # A tiny stub task that emits NaN rows under the default policy.
        class _NaNTask:
            name = "nan_stub"
            dim_parameters = 1
            dim_data = 1

            def get_prior(self, key, num_samples=1):
                return jnp.zeros((num_samples, 1))

            def get_simulator(self, key, max_calls=None):
                def sim(k, theta):
                    return jnp.full((theta.shape[0], 1), jnp.nan)

                sim.flatten_data = lambda x: x.reshape(-1, 1)
                return sim

        with pytest.raises(ValueError, match="non-finite"):
            generate_samples(
                _NaNTask(), jax.random.PRNGKey(0), n=4, chunk_size=4,
                resample_invalid=False,
            )
