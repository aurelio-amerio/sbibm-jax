# tests/data/test_dataset.py
"""TaskDataset: metadata parsing, dims, stats, repo default, normalize."""

import json

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from sbibm_jax.hf import config


def _fake_metadata(tmp_path):
    meta = {
        "two_moons": {
            "dim_theta": 2, "dim_x": 2, "data_kind": "vector",
            "data_shape": [2], "splits": {"train": 8, "validation": 4, "test": 4},
            "has_reference": True, "num_observations": 2,
            "stats": {
                "theta_mean": [[0.0, 0.0]], "theta_std": [[1.0, 1.0]],
                "x_mean": [[0.0, 0.0]], "x_std": [[1.0, 1.0]],
                "theta_axes": [0], "x_axes": [0],
            },
        }
    }
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(meta))
    return str(p)


def _fake_main_dataset():
    rows = {"thetas": np.zeros((8, 2), np.float32), "xs": np.ones((8, 2), np.float32)}
    d = Dataset.from_dict(rows)
    return DatasetDict({"train": d, "validation": d, "test": d})


@pytest.fixture
def patched(monkeypatch, tmp_path):
    meta_path = _fake_metadata(tmp_path)
    monkeypatch.setattr(
        "sbibm_jax.data.dataset.hf_hub_download", lambda **kw: meta_path,
    )
    monkeypatch.setattr(
        "sbibm_jax.data.dataset.load_dataset",
        lambda repo, name=None, **kw: _fake_main_dataset(),
    )


class TestConstruction:
    def test_dims_and_stats_parsed(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons")
        assert ds.dim_theta == 2
        assert ds.dim_x == 2
        assert ds.data_kind == "vector"
        assert tuple(ds.data_shape) == (2,)
        assert np.array(ds.theta_mean).shape == (1, 2)

    def test_default_repo_is_test(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons")
        assert ds.repo == config.TEST_REPO

    def test_joint_sets_dim_joint(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons", kind="joint")
        assert ds.dim_joint == 4
