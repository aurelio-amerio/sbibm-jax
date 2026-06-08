"""Tests for hf.metadata.make_metadata."""

import json
from pathlib import Path

import pytest

from sbibm_jax.hf.metadata import make_metadata, merge_metadata


class TestMakeMetadata:
    def test_returns_dict_per_task(self):
        meta = make_metadata(["gaussian_linear", "two_moons"])
        assert set(meta) == {"gaussian_linear", "two_moons"}

    def test_vector_task_schema(self):
        meta = make_metadata(["gaussian_linear"])
        m = meta["gaussian_linear"]
        assert m["dim_theta"] == 10
        assert m["dim_x"] == 10
        assert m["data_kind"] == "vector"
        assert m["data_shape"] == [10]
        assert m["splits"] == {
            "train": 1_000_000,
            "validation": 10_000,
            "test": 10_000,
        }
        assert m["has_reference"] is True
        assert m["num_observations"] == 10

    def test_image_task_schema(self):
        meta = make_metadata(["gaussian_random_field"])
        m = meta["gaussian_random_field"]
        assert m["data_kind"] == "image"
        assert m["data_shape"] == [32, 32]
        assert m["has_reference"] is False
        # Per-task hf_split_sizes override: capped at 100k train.
        assert m["splits"] == {
            "train": 100_000,
            "validation": 10_000,
            "test": 10_000,
        }

    def test_partial_override_keeps_task_cap(self):
        # gaussian_random_field caps train at 100k. Overriding only validation
        # must keep train/test at the task cap, not fall back to the 1M global
        # default — metadata must match what the dataset build would generate.
        meta = make_metadata(["gaussian_random_field"], val_size=5000)
        assert meta["gaussian_random_field"]["splits"] == {
            "train": 100_000,
            "validation": 5000,
            "test": 10_000,
        }

    def test_grf_256_image_schema(self):
        meta = make_metadata(["gaussian_random_field_256"])
        m = meta["gaussian_random_field_256"]
        assert m["data_kind"] == "image"
        assert m["data_shape"] == [256, 256]
        assert m["dim_x"] == 256 * 256
        assert m["has_reference"] is False
        assert m["splits"] == {
            "train": 100_000, "validation": 10_000, "test": 10_000,
        }

    def test_writes_json_file(self, tmp_path):
        out = tmp_path / "metadata.json"
        meta = make_metadata(["gaussian_linear"], output_path=out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded == meta


class TestMergeMetadata:
    def test_disjoint_keys_union(self):
        assert merge_metadata({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_local_overrides_shared_key(self):
        assert merge_metadata({"a": 1}, {"a": 2}) == {"a": 2}

    def test_empty_remote_returns_local(self):
        assert merge_metadata({}, {"a": 1}) == {"a": 1}

    def test_empty_local_returns_remote(self):
        assert merge_metadata({"a": 1}, {}) == {"a": 1}

    def test_does_not_mutate_inputs(self):
        remote, local = {"a": 1}, {"b": 2}
        merge_metadata(remote, local)
        assert remote == {"a": 1}
        assert local == {"b": 2}
