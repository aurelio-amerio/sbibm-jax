"""End-to-end build_dataset tests on tiny sizes (CPU, no network)."""

import numpy as np
import pytest
from datasets import Dataset

from sbibm_jax.hf import build_dataset


SMALL_OPTS = dict(train_size=8, val_size=4, test_size=2, chunk_size=4)


class TestBuildVector:
    def test_returns_bundle(self):
        bundle = build_dataset("gaussian_linear", **SMALL_OPTS)
        assert set(bundle) == {"train", "validation", "test", "reference"}
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
        assert len(sample["thetas"]) == 10  # gaussian_linear dim_parameters
        assert len(sample["xs"]) == 10  # gaussian_linear dim_data
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
