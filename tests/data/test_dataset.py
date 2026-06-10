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
            "x_kind": "vector", "x_shape": [2],
            "theta_kind": "vector", "theta_shape": [2],
            "splits": {"train": 8, "validation": 4, "test": 4},
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
        assert ds.x_kind == "vector"
        assert tuple(ds.x_shape) == (2,)
        assert ds.theta_kind == "vector"
        assert tuple(ds.theta_shape) == (2,)
        assert np.array(ds.theta_mean).shape == (1, 2)

    def test_default_repo_is_test(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons")
        assert ds.repo == config.TEST_REPO

    def test_joint_sets_dim_joint(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons", kind="joint")
        assert ds.dim_joint == 4


class TestLoaders:
    def test_train_loader_yields_tokenized_batches(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons", kind="conditional")
        loader = ds.get_train_loader(batch_size=4)
        theta, x = next(iter(loader))
        assert np.asarray(theta).shape == (4, 2, 1)
        assert np.asarray(x).shape == (4, 2, 1)

    def test_num_samples_subsamples_prefix(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons")
        loader = ds.get_train_loader(batch_size=2, num_samples=4)
        theta, x = next(iter(loader))
        assert np.asarray(theta).shape[0] == 2

    def test_max_workers_clamped(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons", max_workers=64)
        assert ds.max_workers == 8

    def test_prefetching_loader_iterates(self, patched):
        # The numpy collate must survive grain's mp_prefetch (worker
        # subprocesses pickle batches across the process boundary). max_workers
        # small to keep it light on the shared node.
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons", use_prefetching=True, max_workers=2)
        loader = ds.get_train_loader(batch_size=2)
        theta, x = next(iter(loader))
        assert np.asarray(theta).shape == (2, 2, 1)


def _fake_posterior():
    d = Dataset.from_dict({
        "observations": np.arange(2 * 2, dtype=np.float32).reshape(2, 2),
        "reference_samples": np.zeros((2, 10, 2), np.float32),
        "true_parameters": np.ones((2, 2), np.float32),
    })
    return DatasetDict({"reference_posterior": d})


class TestReference:
    def test_get_reference_indexes_observation(self, monkeypatch, patched):
        from sbibm_jax.data import TaskDataset

        def fake_load(repo, name=None, **kw):
            if name and name.endswith("_posterior"):
                return _fake_posterior()
            return _fake_main_dataset()

        monkeypatch.setattr("sbibm_jax.data.dataset.load_dataset", fake_load)
        ds = TaskDataset("two_moons")
        obs, samples = ds.get_reference(num_observation=2)
        assert np.asarray(obs).shape == (2,)
        assert np.asarray(samples).shape == (10, 2)
        assert np.asarray(ds.get_true_parameters(2)).shape == (2,)

    def test_get_reference_without_posterior_raises(self, monkeypatch, tmp_path):
        # has_reference False -> informative error
        import json
        meta = {"t": {"x_kind": "vector", "x_shape": [2],
                      "theta_kind": "vector", "theta_shape": [2],
                      "splits": {"train": 8, "validation": 4, "test": 4},
                      "has_reference": False, "num_observations": 1,
                      "stats": None}}
        p = tmp_path / "metadata.json"
        p.write_text(json.dumps(meta))
        monkeypatch.setattr("sbibm_jax.data.dataset.hf_hub_download", lambda **kw: str(p))
        monkeypatch.setattr("sbibm_jax.data.dataset.load_dataset",
                            lambda repo, name=None, **kw: _fake_main_dataset())
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("t")
        with pytest.raises(ValueError, match="no reference"):
            ds.get_reference(1)


class TestNormalizeMethods:
    def test_normalize_x_roundtrip(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons", normalize=True)
        x = np.ones((3, 2, 1), np.float32)
        back = ds.unnormalize_x(ds.normalize_x(x))
        np.testing.assert_allclose(np.asarray(back), x, atol=1e-5)


def _fake_ts_dataset():
    rows = {"thetas": np.zeros((8, 2), np.float32),
            "xs": np.ones((8, 5, 2), np.float32)}
    d = Dataset.from_dict(rows)
    return DatasetDict({"train": d, "validation": d, "test": d})


class TestTimeSeriesLoader:
    def _meta(self, tmp_path, x_mean, x_std):
        meta = {"gw": {
            "x_kind": "timeseries", "x_shape": [5, 2],
            "theta_kind": "vector", "theta_shape": [2],
            "splits": {"train": 8, "validation": 8, "test": 8},
            "has_reference": False, "num_observations": 1,
            "stats": {
                "theta_mean": [[0.0, 0.0]], "theta_std": [[1.0, 1.0]],
                "x_mean": x_mean, "x_std": x_std,
                "theta_axes": [0], "x_axes": [0, 1],
            },
        }}
        p = tmp_path / "metadata.json"
        p.write_text(json.dumps(meta))
        return str(p)

    def test_conditional_shapes(self, monkeypatch, tmp_path):
        mp = self._meta(tmp_path, [[[0.0, 0.0]]], [[[1.0, 1.0]]])
        monkeypatch.setattr(
            "sbibm_jax.data.dataset.hf_hub_download", lambda **kw: mp)
        monkeypatch.setattr(
            "sbibm_jax.data.dataset.load_dataset",
            lambda repo, name=None, **kw: _fake_ts_dataset())
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("gw", kind="conditional")
        assert ds.x_kind == "timeseries"
        assert ds.dim_x == 10
        assert tuple(ds.x_shape) == (5, 2)
        theta, x = next(iter(ds.get_train_loader(batch_size=4)))
        assert np.asarray(theta).shape == (4, 2, 1)
        assert np.asarray(x).shape == (4, 5, 2, 1)

    def test_normalize_broadcasts_per_channel(self, monkeypatch, tmp_path):
        # x all ones; x_mean 1, x_std 1 -> 0 after normalization.
        mp = self._meta(tmp_path, [[[1.0, 1.0]]], [[[1.0, 1.0]]])
        monkeypatch.setattr(
            "sbibm_jax.data.dataset.hf_hub_download", lambda **kw: mp)
        monkeypatch.setattr(
            "sbibm_jax.data.dataset.load_dataset",
            lambda repo, name=None, **kw: _fake_ts_dataset())
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("gw", normalize=True)
        _, x = next(iter(ds.get_train_loader(batch_size=4)))
        np.testing.assert_allclose(np.asarray(x), 0.0, atol=1e-6)
