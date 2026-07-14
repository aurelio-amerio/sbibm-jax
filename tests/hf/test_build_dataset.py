"""End-to-end build_dataset tests on tiny sizes (CPU, no network)."""

import numpy as np
import pytest
from datasets import Dataset

from sbibm_jax import get_task
from sbibm_jax.hf import build_dataset
from sbibm_jax.hf.generate import derive_task_keys, generate_samples


SMALL_OPTS = dict(train_size=8, val_size=4, test_size=2, chunk_size=4)


class TestBuildVector:
    def test_returns_bundle(self):
        bundle = build_dataset("gaussian_linear", **SMALL_OPTS)
        assert set(bundle) == {"train", "validation", "test", "reference", "stats"}
        for k in ("train", "validation", "test"):
            assert isinstance(bundle[k], Dataset)

    def test_split_sizes(self):
        bundle = build_dataset("gaussian_linear", **SMALL_OPTS)
        assert len(bundle["train"]) == 8
        assert len(bundle["validation"]) == 4
        assert len(bundle["test"]) == 2

    def test_dtype_and_finiteness(self):
        bundle = build_dataset("gaussian_linear", **SMALL_OPTS)
        sample = bundle["train"][0]
        assert len(sample["thetas"]) == 10  # gaussian_linear dim_theta
        assert len(sample["xs"]) == 10  # gaussian_linear dim_x
        arr = np.asarray(sample["xs"], dtype=np.float64)
        assert np.isfinite(arr).all()

    def test_reproducibility(self):
        b1 = build_dataset("gaussian_linear", master_seed=42, **SMALL_OPTS)
        b2 = build_dataset("gaussian_linear", master_seed=42, **SMALL_OPTS)
        np.testing.assert_array_equal(b1["train"]["thetas"], b2["train"]["thetas"])
        np.testing.assert_array_equal(b1["train"]["xs"], b2["train"]["xs"])

    def test_reference_present(self):
        bundle = build_dataset("two_moons", **SMALL_OPTS)
        assert bundle["reference"] is not None
        assert len(bundle["reference"]) == 10  # two_moons num_observations


class TestBuildImage:
    def test_grf_image_shape(self):
        bundle = build_dataset(
            "gaussian_random_field",
            **SMALL_OPTS,
            task_kwargs={"field_size": 8},
        )
        # Each row is an 8x8 image stored via Array2D.
        sample = bundle["train"][0]
        arr = np.asarray(sample["xs"], dtype=np.float32)
        assert arr.shape == (8, 8)
        assert np.isfinite(arr).all()

    def test_grf_no_reference(self):
        bundle = build_dataset(
            "gaussian_random_field",
            **SMALL_OPTS,
            task_kwargs={"field_size": 8},
        )
        assert bundle["reference"] is None


class TestBuildStats:
    def test_stats_match_materialized_train_split(self):
        # build_dataset must compute train-split stats equal to a direct pass.
        bundle = build_dataset(
            "two_moons", train_size=512, val_size=16, test_size=16,
        )
        stats = bundle["stats"]
        assert stats is not None
        # Reproduce the same train draw and compare (same master seed + key).
        task = get_task("two_moons")
        key = derive_task_keys(task.name)["train"]
        thetas, xs, _ = generate_samples(task, key, 512)
        np.testing.assert_allclose(
            np.array(stats["theta_mean"]), thetas.mean(0, keepdims=True),
            rtol=1e-4, atol=1e-4,
        )
        np.testing.assert_allclose(
            np.array(stats["x_mean"]), xs.mean(0, keepdims=True),
            rtol=1e-4, atol=1e-4,
        )
        assert np.array(stats["theta_mean"]).shape == (1, task.dim_theta)
        assert np.array(stats["x_mean"]).shape == (1, task.dim_x)


class TestImageStatsShape:
    def test_grf_x_stats_are_global_scalar(self):
        bundle = build_dataset(
            "gaussian_random_field",
            **SMALL_OPTS,
            task_kwargs={"field_size": 8},
        )
        # x native shape is (H, W); global-scalar reduction -> (1, 1, 1).
        assert np.array(bundle["stats"]["x_mean"]).shape == (1, 1, 1)


def test_hf_backend_is_applied(monkeypatch):
    from sbibm_jax.hf import build as build_mod

    class _BackendTask:
        name = "backend_probe"
        dim_theta = 2
        dim_x = 3
        num_observations = 1
        hf_backend = "special"
        backend = "default"

        def get_prior(self, key, num_samples=1):
            import jax.numpy as jnp
            return jnp.zeros((num_samples, 2))

        def get_simulator(self, key, max_calls=None):
            assert self.backend == "special"
            import jax.numpy as jnp

            def sim(k, theta):
                return jnp.zeros((theta.shape[0], 3))

            sim.flatten_data = lambda x: x.reshape(-1, 3)
            return sim

        def get_observation(self, i):
            raise FileNotFoundError

        def get_reference_posterior_samples(self, i):
            raise FileNotFoundError

        def get_true_parameters(self, i):
            raise FileNotFoundError

    monkeypatch.setattr(
        build_mod, "get_task", lambda name, **kw: _BackendTask()
    )
    bundle = build_mod.build_dataset(
        "backend_probe", train_size=4, val_size=2, test_size=2,
        chunk_size=4,
    )
    assert len(bundle["train"]) == 4
