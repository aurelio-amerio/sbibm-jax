# tests/data/test_online_dataset.py
"""OnlineTaskDataset: simulate-on-the-fly source dataset and loader."""

import json

import jax
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# _SimIterDataset / _SimIterator (no metadata needed: built from a task)
# ---------------------------------------------------------------------------

def _make_sim_ds(seed=0, batch_size=4):
    from sbibm_jax.data.dataset import _SimIterDataset
    from sbibm_jax.tasks import get_task
    task = get_task("two_moons")
    sim = task.get_simulator(jax.random.PRNGKey(0), max_calls=None)
    return _SimIterDataset(task, sim, seed, batch_size)


class TestSimIterDataset:
    def test_yields_raw_numpy_batches(self):
        batch = next(iter(_make_sim_ds()))
        assert isinstance(batch["thetas"], np.ndarray)
        assert isinstance(batch["xs"], np.ndarray)
        assert batch["thetas"].shape == (4, 2)
        assert batch["xs"].shape == (4, 2)
        assert np.isfinite(batch["thetas"]).all()
        assert np.isfinite(batch["xs"]).all()

    def test_consecutive_batches_differ(self):
        it = iter(_make_sim_ds())
        b1, b2 = next(it), next(it)
        assert not np.allclose(b1["thetas"], b2["thetas"])

    def test_same_seed_reproduces_stream(self):
        b1 = next(iter(_make_sim_ds(seed=3)))
        b2 = next(iter(_make_sim_ds(seed=3)))
        np.testing.assert_array_equal(b1["thetas"], b2["thetas"])
        np.testing.assert_array_equal(b1["xs"], b2["xs"])

    def test_different_seed_differs(self):
        b1 = next(iter(_make_sim_ds(seed=3)))
        b2 = next(iter(_make_sim_ds(seed=4)))
        assert not np.allclose(b1["thetas"], b2["thetas"])

    def test_set_slice_changes_stream(self):
        # grain calls set_slice(slice(worker_index, None, num_workers)) per
        # worker; the worker index must change the stream (else every worker
        # would replay the same data — silent duplication).
        ds0, ds1 = _make_sim_ds(), _make_sim_ds()
        ds1.set_slice(slice(1, None, 2))
        b0, b1 = next(iter(ds0)), next(iter(ds1))
        assert not np.allclose(b0["thetas"], b1["thetas"])

    def test_state_roundtrip(self):
        # get_state after batch 1, set_state on a fresh iterator -> batch 2
        # reproduced exactly (grain's checkpoint/seek protocol).
        ds = _make_sim_ds()
        it1 = iter(ds)
        next(it1)
        state = it1.get_state()
        b2 = next(it1)
        it2 = iter(_make_sim_ds())
        it2.set_state(state)
        b2_again = next(it2)
        np.testing.assert_array_equal(b2["thetas"], b2_again["thetas"])
        np.testing.assert_array_equal(b2["xs"], b2_again["xs"])


# ---------------------------------------------------------------------------
# OnlineTaskDataset (metadata faked locally; never hits the Hub)
# ---------------------------------------------------------------------------

def _fake_metadata(tmp_path):
    # two_moons shapes match the real task (dim_theta=2, dim_x=2); stats are
    # deliberately non-trivial so normalize tests detect a no-op.
    meta = {
        "two_moons": {
            "x_kind": "vector", "x_shape": [2],
            "theta_kind": "vector", "theta_shape": [2],
            "splits": {"train": 8, "validation": 4, "test": 4},
            "has_reference": True, "num_observations": 2,
            "stats": {
                "theta_mean": [[0.5, -0.5]], "theta_std": [[2.0, 2.0]],
                "x_mean": [[0.1, 0.2]], "x_std": [[3.0, 3.0]],
                "theta_axes": [0], "x_axes": [0],
            },
        },
        # File-backed task: get_simulator raises NotImplementedError.
        "gravitational_waves": {
            "x_kind": "timeseries", "x_shape": [8192, 2],
            "theta_kind": "vector", "theta_shape": [2],
            "splits": {"train": 8, "validation": 4, "test": 4},
            "has_reference": False, "num_observations": 1,
            "stats": None,
        },
    }
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(meta))
    return str(p)


@pytest.fixture
def patched_meta(monkeypatch, tmp_path):
    meta_path = _fake_metadata(tmp_path)
    monkeypatch.setattr(
        "sbibm_jax.data.dataset.hf_hub_download", lambda **kw: meta_path,
    )


class TestOnlineConstruction:
    def test_builds_with_eager_simulator(self, patched_meta):
        from sbibm_jax.data import OnlineTaskDataset
        from sbibm_jax.tasks.simulator import Simulator
        ds = OnlineTaskDataset("two_moons")
        assert ds.dim_theta == 2
        assert ds.dim_x == 2
        assert ds.task.name == "two_moons"
        assert isinstance(ds.simulator, Simulator)
        assert ds.simulator.max_calls is None

    def test_no_simulator_task_fails_at_construction(self, patched_meta):
        # hf_external tasks (gravitational_waves) have no simulator yet; the
        # eager build surfaces that immediately, not on first next().
        from sbibm_jax.data import OnlineTaskDataset
        with pytest.raises(NotImplementedError, match="simulator"):
            OnlineTaskDataset("gravitational_waves")


class TestOfflineLoadersRaise:
    @pytest.mark.parametrize("method", [
        "get_train_loader", "get_val_loader", "get_test_loader",
    ])
    def test_informative_error(self, patched_meta, method):
        from sbibm_jax.data import OnlineTaskDataset
        ds = OnlineTaskDataset("two_moons")
        with pytest.raises(NotImplementedError,
                           match="get_online_train_loader"):
            getattr(ds, method)(4)


class TestOnlineLoader:
    def test_conditional_yields_jnp_token_batches(self, patched_meta):
        from sbibm_jax.data import OnlineTaskDataset
        ds = OnlineTaskDataset("two_moons")
        theta, x = next(iter(ds.get_online_train_loader(batch_size=4)))
        assert isinstance(theta, jax.Array)
        assert isinstance(x, jax.Array)
        assert theta.shape == (4, 2, 1)
        assert x.shape == (4, 2, 1)

    def test_joint_yields_concatenated_jnp(self, patched_meta):
        from sbibm_jax.data import OnlineTaskDataset
        ds = OnlineTaskDataset("two_moons", kind="joint")
        out = next(iter(ds.get_online_train_loader(batch_size=4)))
        assert isinstance(out, jax.Array)
        assert out.shape == (4, 4, 1)

    def test_same_seed_identical_first_batch(self, patched_meta):
        from sbibm_jax.data import OnlineTaskDataset
        ds = OnlineTaskDataset("two_moons")
        t1, x1 = next(iter(ds.get_online_train_loader(batch_size=4, seed=7)))
        t2, x2 = next(iter(ds.get_online_train_loader(batch_size=4, seed=7)))
        np.testing.assert_array_equal(np.asarray(t1), np.asarray(t2))
        np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))

    def test_different_seed_differs(self, patched_meta):
        from sbibm_jax.data import OnlineTaskDataset
        ds = OnlineTaskDataset("two_moons")
        t1, _ = next(iter(ds.get_online_train_loader(batch_size=4, seed=7)))
        t2, _ = next(iter(ds.get_online_train_loader(batch_size=4, seed=8)))
        assert not np.allclose(np.asarray(t1), np.asarray(t2))

    def test_consecutive_batches_differ(self, patched_meta):
        from sbibm_jax.data import OnlineTaskDataset
        ds = OnlineTaskDataset("two_moons")
        it = iter(ds.get_online_train_loader(batch_size=4))
        t1, _ = next(it)
        t2, _ = next(it)
        assert not np.allclose(np.asarray(t1), np.asarray(t2))

    def test_normalize_matches_manual_collate(self, patched_meta):
        from sbibm_jax.data import OnlineTaskDataset
        from sbibm_jax.data.dataset import _SimIterDataset
        from sbibm_jax.data.process import make_collate_jax
        ds = OnlineTaskDataset("two_moons", normalize=True)
        theta_n, x_n = next(iter(
            ds.get_online_train_loader(batch_size=4, seed=7)))
        # Same raw draw, collated manually with the same stats.
        raw = next(iter(_SimIterDataset(ds.task, ds.simulator, 7, 4)))
        collate = make_collate_jax(kind="conditional", x_kind="vector",
                                   normalize=True, stats=ds._stats)
        theta_m, x_m = collate(raw)
        np.testing.assert_allclose(np.asarray(theta_n), np.asarray(theta_m),
                                   atol=1e-6)
        np.testing.assert_allclose(np.asarray(x_n), np.asarray(x_m),
                                   atol=1e-6)
        # And it actually normalized (stats are non-trivial in the fixture).
        raw_tok = np.asarray(raw["thetas"], np.float32)[..., None]
        assert not np.allclose(np.asarray(theta_n), raw_tok)


class TestReferenceStillWorks:
    def test_get_reference_via_posterior_config(
            self, monkeypatch, patched_meta):
        from datasets import Dataset, DatasetDict
        from sbibm_jax.data import OnlineTaskDataset

        def fake_load(repo, name=None, **kw):
            assert name == "two_moons_posterior"
            d = Dataset.from_dict({
                "observations": np.arange(4, dtype=np.float32).reshape(2, 2),
                "reference_samples": np.zeros((2, 10, 2), np.float32),
                "true_parameters": np.ones((2, 2), np.float32),
            })
            return DatasetDict({"reference_posterior": d})

        monkeypatch.setattr("sbibm_jax.data.dataset.load_dataset", fake_load)
        ds = OnlineTaskDataset("two_moons")
        obs, samples = ds.get_reference(num_observation=2)
        assert np.asarray(obs).shape == (2,)
        assert np.asarray(samples).shape == (10, 2)


class TestMultiprocessSmoke:
    def test_one_worker_end_to_end(self, patched_meta):
        # Exercises spawn + cloudpickle of the closure-based Simulator,
        # _worker_init (jax -> cpu in the worker), numpy across the pickle
        # boundary, and the main-process jnp collate. NOTE: grain skips
        # set_slice for num_workers==1 (worker_ds = ds in process_prefetch);
        # that protocol is covered by test_set_slice_changes_stream.
        from sbibm_jax.data import OnlineTaskDataset
        ds = OnlineTaskDataset("two_moons")
        loader = ds.get_online_train_loader(batch_size=2, num_workers=1)
        it = iter(loader)
        try:
            theta, x = next(it)
            assert np.asarray(theta).shape == (2, 2, 1)
            assert np.asarray(x).shape == (2, 2, 1)
            assert np.isfinite(np.asarray(theta)).all()
            assert np.isfinite(np.asarray(x)).all()
        finally:
            # grain recommends closing mp_prefetch iterators explicitly.
            if hasattr(it, "close"):
                it.close()
