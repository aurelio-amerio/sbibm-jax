"""Tests for hf.metadata.make_metadata."""

import json
from pathlib import Path

import pytest

from sbibm_jax.hf.metadata import make_metadata


class TestMakeMetadata:
    def test_returns_dict_per_task(self):
        meta = make_metadata(["gaussian_linear", "two_moons"])
        assert set(meta) == {"gaussian_linear", "two_moons"}

    def test_vector_task_schema(self):
        meta = make_metadata(["gaussian_linear"])
        m = meta["gaussian_linear"]
        assert m["dim_parameters"] == 10
        assert m["dim_data"] == 10
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

    def test_writes_json_file(self, tmp_path):
        out = tmp_path / "metadata.json"
        meta = make_metadata(["gaussian_linear"], output_path=out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded == meta
