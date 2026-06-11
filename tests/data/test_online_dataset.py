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
