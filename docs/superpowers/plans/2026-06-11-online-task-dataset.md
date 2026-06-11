# OnlineTaskDataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `OnlineTaskDataset.get_online_train_loader(batch_size, ...)` — an infinite grain loader that simulates fresh `(theta, x)` batches from a task's prior + simulator, with tokenization/normalization identical to the offline `TaskDataset`.

**Architecture:** A `TaskDataset` subclass in `src/sbibm_jax/data/dataset.py` that reuses the base's metadata/stats parsing (via a small `__init__` refactor), skips the HF split download, and builds the JAX task + simulator eagerly. The pipeline is a custom source `grain.IterDataset` (simulates in spawn workers under `mp_prefetch`, emitting raw numpy) followed by a main-process `.map` with a new jnp collate. Worker identity comes from grain's `set_slice` protocol (mandatory for a source `IterDataset` under `mp_prefetch`); workers are forced onto CPU via `jax.config.update` in `worker_init_fn`.

**Tech Stack:** JAX, grain 0.2.17, HuggingFace `huggingface_hub` (metadata only), pytest.

**Spec:** `docs/superpowers/specs/2026-06-11-online-task-dataset-design.md` — read it first; it documents six verified grain/JAX facts the code comments refer to.

---

## Working environment

- Branch: `online-task-dataset` (already checked out in the main repo). If executing in a worktree (user's preferred workflow), create it off the current HEAD via the `superpowers:using-git-worktrees` skill and run everything from the worktree root.
- `uv` fails inside the sandbox (read-only caches). Run tests with the main checkout's venv python plus `PYTHONPATH` pointing at the *current* checkout's `src`:

  ```bash
  VENV=/lustre/ific.uv.es/ml/ific088/github/sbibm-jax/.venv
  PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data -v
  ```

  `JAX_PLATFORMS=cpu` and `-n 2` are injected by the pytest config in `pyproject.toml`; no GPU is touched.
- Lint: `$VENV/bin/flake8 src/sbibm_jax/data tests/data`. The flake8 baseline is **never clean** (pre-existing E501 etc.); judge only violations introduced relative to HEAD.
- Never exceed 8 workers/cores for anything (shared node).

## File map

| File | Change |
|---|---|
| `src/sbibm_jax/data/dataset.py` | Refactor `TaskDataset.__init__`; add `_worker_init`, `_SimIterDataset`, `_SimIterator`, `OnlineTaskDataset` |
| `src/sbibm_jax/data/process.py` | Add `_stat_array_jax`, `make_collate_jax`; amend module docstring |
| `src/sbibm_jax/data/__init__.py` | Export `OnlineTaskDataset` |
| `tests/data/test_dataset.py` | One new test (`_stats` stored by refactor) |
| `tests/data/test_process.py` | New `make_collate_jax` test classes |
| `tests/data/test_online_dataset.py` | **New** — all online tests (source dataset, loader, errors, mp smoke) |
| `CLAUDE.md` | One paragraph documenting `OnlineTaskDataset` |

---

### Task 1: Refactor `TaskDataset.__init__` into `_init_metadata` / `_init_splits`

Pure refactor (behavior unchanged) plus one new attribute: `_init_metadata` must store the raw stats dict as `self._stats` so the subclass can rebuild the collate without re-parsing metadata.

**Files:**
- Modify: `src/sbibm_jax/data/dataset.py:22-81` (the `TaskDataset.__init__` body)
- Test: `tests/data/test_dataset.py`

- [ ] **Step 1: Write the failing test**

Append to `TestConstruction` in `tests/data/test_dataset.py`:

```python
    def test_raw_stats_dict_stored(self, patched):
        # _init_metadata keeps the raw metadata stats dict on the instance so
        # subclasses (OnlineTaskDataset) can rebuild the collate from it.
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons")
        assert ds._stats["theta_mean"] == [[0.0, 0.0]]
        assert ds._stats["x_std"] == [[1.0, 1.0]]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data/test_dataset.py::TestConstruction::test_raw_stats_dict_stored -v
```

Expected: FAIL with `AttributeError: 'TaskDataset' object has no attribute '_stats'`.

- [ ] **Step 3: Refactor `__init__`**

Replace the body of `TaskDataset.__init__` (everything from `meta_path = hf_hub_download(` through `self._posterior = None`) with three method calls, and add the three methods. The class then reads:

```python
class TaskDataset:
    def __init__(
        self,
        name,
        *,
        kind="conditional",
        repo=None,
        normalize=False,
        dtype=np.float32,
        seed=42,
        use_prefetching=True,
        max_workers=None,
    ):
        self.name = name
        self.kind = kind
        self.repo = repo if repo is not None else config.TEST_REPO
        self.normalize = normalize
        self.dtype = dtype
        self.seed = seed
        self.use_prefetching = use_prefetching
        self.max_workers = (
            None if max_workers is None else min(int(max_workers), _MAX_WORKERS_CAP)
        )

        self._init_metadata(self._load_metadata_entry())
        self._init_splits()

    def _load_metadata_entry(self):
        """Download metadata.json from the Hub and return this task's entry."""
        meta_path = hf_hub_download(
            repo_id=self.repo, filename="metadata.json", repo_type="dataset",
        )
        with open(meta_path) as f:
            return json.load(f)[self.name]

    def _init_metadata(self, entry):
        """Shapes, kinds, dims, reference info, stats, and the collate fn.

        Shared by TaskDataset and OnlineTaskDataset.
        """
        self.x_kind = entry["x_kind"]
        self.x_shape = tuple(entry["x_shape"])
        self.theta_kind = entry["theta_kind"]
        self.theta_shape = tuple(entry["theta_shape"])
        self.dim_x = int(np.prod(self.x_shape))
        self.dim_theta = int(np.prod(self.theta_shape))
        self.num_observations = int(entry["num_observations"])
        self.has_reference = bool(entry["has_reference"])
        self.dim_joint = (
            self.dim_theta + self.dim_x if self.kind == "joint" else None
        )

        stats = entry.get("stats")
        self._stats = stats
        if stats is not None:
            self.theta_mean = stats["theta_mean"]
            self.theta_std = stats["theta_std"]
            self.x_mean = stats["x_mean"]
            self.x_std = stats["x_std"]
        else:
            self.theta_mean = self.theta_std = self.x_mean = self.x_std = None

        self._collate = make_collate(
            kind=self.kind, x_kind=self.x_kind, theta_kind=self.theta_kind,
            normalize=self.normalize, stats=stats, dtype=self.dtype,
        )
        self._posterior = None  # lazily loaded in get_reference

    def _init_splits(self):
        """Offline-only: download the HF splits."""
        self.dataset = load_dataset(self.repo, self.name).with_format("numpy")
        self.df_train = self.dataset["train"]
        self.df_val = self.dataset["validation"]
        self.df_test = self.dataset["test"]
        self.max_samples = self.df_train.num_rows
```

Note the two locals `name`/`kind` from the old body become `self.name`/`self.kind`. Everything else is verbatim relocation.

- [ ] **Step 4: Run the full data test file to verify no regression + new test passes**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data/test_dataset.py -v
```

Expected: all PASS (including `test_raw_stats_dict_stored`).

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/data/dataset.py tests/data/test_dataset.py
git commit -m "refactor(data): split TaskDataset.__init__ into _init_metadata/_init_splits"
```

---

### Task 2: `make_collate_jax` in `process.py`

A jnp twin of `make_collate` for the online main-process path. The numpy `make_collate` and `_stat_array` stay untouched (spec decision: no shared-backend parameterization).

**Files:**
- Modify: `src/sbibm_jax/data/process.py`
- Test: `tests/data/test_process.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_process.py`:

```python
class TestMakeCollateJax:
    """jnp twin of make_collate (online path: collation in the main process)."""

    def test_returns_jax_arrays_with_token_shapes(self):
        import jax
        from sbibm_jax.data.process import make_collate_jax
        collate = make_collate_jax(kind="conditional", x_kind="vector")
        theta, x = collate(_batch())
        assert isinstance(theta, jax.Array)
        assert isinstance(x, jax.Array)
        assert theta.shape == (2, 3, 1)
        assert x.shape == (2, 5, 1)

    def test_joint_concats_to_single_jax_array(self):
        import jax
        from sbibm_jax.data.process import make_collate_jax
        collate = make_collate_jax(kind="joint", x_kind="vector")
        out = collate(_batch())
        assert isinstance(out, jax.Array)
        assert out.shape == (2, 3 + 5, 1)

    def test_normalize_applies_stats(self):
        from sbibm_jax.data.process import make_collate_jax
        stats = {"theta_mean": [[1.0, 1.0, 1.0]], "theta_std": [[1.0, 1.0, 1.0]],
                 "x_mean": [[1.0, 1.0, 1.0, 1.0, 1.0]],
                 "x_std": [[2.0, 2.0, 2.0, 2.0, 2.0]]}
        collate = make_collate_jax(kind="conditional", x_kind="vector",
                                   normalize=True, stats=stats)
        _, x = collate(_batch())
        # x all ones, mean 1, std 2 -> 0
        np.testing.assert_allclose(np.asarray(x), 0.0, atol=1e-6)

    def test_normalize_without_stats_raises(self):
        from sbibm_jax.data.process import make_collate_jax
        with pytest.raises(ValueError, match="requires stats"):
            make_collate_jax(kind="conditional", x_kind="vector",
                             normalize=True)

    def test_joint_raises_for_non_vector(self):
        from sbibm_jax.data.process import make_collate_jax
        with pytest.raises(ValueError, match="joint.*vector"):
            make_collate_jax(kind="joint", x_kind="image")
        with pytest.raises(ValueError, match="joint.*vector"):
            make_collate_jax(kind="joint", x_kind="vector", theta_kind="image")

    def test_unknown_kind_raises(self):
        from sbibm_jax.data.process import make_collate_jax
        with pytest.raises(ValueError, match="Unknown kind"):
            make_collate_jax(kind="bogus", x_kind="vector")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data/test_process.py::TestMakeCollateJax -v
```

Expected: FAIL/ERROR with `ImportError: cannot import name 'make_collate_jax'`.

- [ ] **Step 3: Implement `make_collate_jax`**

In `src/sbibm_jax/data/process.py`: add `import jax.numpy as jnp` below `import numpy as np`, and append at the end of the file:

```python
def _stat_array_jax(values, dtype):
    """jnp twin of _stat_array: metadata stat -> trailing-dim for tokens."""
    a = jnp.asarray(np.asarray(values), dtype=dtype)
    return a[..., None]  # (1, dim) -> (1, dim, 1); (1,1,1) -> (1,1,1,1)


def make_collate_jax(
    *, kind, x_kind, theta_kind="vector", normalize=False, stats=None,
    dtype=jnp.float32,
):
    """jnp twin of make_collate, for the online (main-process) path.

    Same semantics as make_collate, but the returned collate yields jax
    arrays. Used by OnlineTaskDataset, where collation runs in the consumer
    process after grain's pickle boundary — so jnp is safe there, unlike in
    mp_prefetch workers (see module docstring).
    """
    if kind not in ("joint", "conditional"):
        raise ValueError(f"Unknown kind {kind!r}.")
    if kind == "joint" and (x_kind != "vector" or theta_kind != "vector"):
        raise ValueError(
            f"kind='joint' is vector-only; got x_kind={x_kind!r}, "
            f"theta_kind={theta_kind!r}. Use kind='conditional'."
        )

    if normalize:
        if stats is None:
            raise ValueError("normalize=True requires stats.")
        tm = _stat_array_jax(stats["theta_mean"], dtype)
        ts = _stat_array_jax(stats["theta_std"], dtype)
        xm = _stat_array_jax(stats["x_mean"], dtype)
        xs_ = _stat_array_jax(stats["x_std"], dtype)

    def collate(batch):
        theta = jnp.asarray(batch["thetas"], dtype=dtype)[..., None]
        x = jnp.asarray(batch["xs"], dtype=dtype)[..., None]
        if normalize:
            theta = (theta - tm) / ts
            x = (x - xm) / xs_
        if kind == "conditional":
            return theta, x
        return jnp.concatenate((theta, x), axis=1)

    return collate
```

Also amend the module docstring: after the sentence ending "wrap in jnp.asarray downstream for jax.", insert:

```
make_collate_jax is the jnp twin for the online path (OnlineTaskDataset),
where collation runs in the main process after the pickle boundary.
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data/test_process.py -v
```

Expected: all PASS (old numpy tests too).

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/data/process.py tests/data/test_process.py
git commit -m "feat(data): add make_collate_jax (jnp collate for the online path)"
```

---

### Task 3: `_SimIterDataset` / `_SimIterator` source dataset + `_worker_init`

The simulate-on-the-fly grain source. Three load-bearing grain facts (verified in the spec against grain 0.2.17 — do not "simplify" them away):

1. A parentless source `IterDataset` under `mp_prefetch` **must** implement `set_slice(sl, sequential_slice=False)` (grain's `SupportsInPlaceSlicing`, runtime-checkable protocol — duck typing suffices, no extra base class). Grain calls it with `slice(worker_index, None, num_workers)`, so `sl.start` is the worker index — fold it into the key for independent per-worker streams.
2. `grain.DatasetIterator.get_state`/`set_state` are abstract AND actually called by the worker loop — implement them for real.
3. `worker_init_fn` must use `jax.config.update("jax_platforms", "cpu")`, **not** `os.environ` — jax captures `JAX_PLATFORMS` at import time and is already imported in the worker before the init fn runs; the config update works because grain calls `worker_init_fn` before unpickling the dataset (first backend touch).

**Files:**
- Modify: `src/sbibm_jax/data/dataset.py` (imports + three new module-level definitions, placed after `TaskDataset`)
- Test: `tests/data/test_online_dataset.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_online_dataset.py`:

```python
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
```

(The `import json` is used by Task 4's fixtures in this same file.)

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data/test_online_dataset.py -v
```

Expected: ERROR with `ImportError: cannot import name '_SimIterDataset'`.

- [ ] **Step 3: Implement**

In `src/sbibm_jax/data/dataset.py`, extend the imports:

```python
import jax
import jax.numpy as jnp

from sbibm_jax.tasks import get_task
from sbibm_jax.data.process import make_collate, make_collate_jax, _stat_array
```

(`import jax`/`jnp` go in the third-party block; `get_task` joins the first-party block; `make_collate_jax` is added to the existing process import.)

Append after the `TaskDataset` class:

```python
def _worker_init(worker_index, worker_count):
    """grain mp_prefetch worker init: force JAX onto CPU in the worker.

    Must be jax.config.update, not os.environ["JAX_PLATFORMS"]: jax captures
    the env var at import time, and jax is already imported here (cloudpickle
    loads this function by reference, importing this module first). The
    update is effective because grain runs worker_init_fn before unpickling
    the dataset — i.e. before anything touches a JAX backend. Spawn start
    method is guaranteed by grain itself.
    """
    del worker_index, worker_count
    jax.config.update("jax_platforms", "cpu")


class _SimIterDataset(grain.IterDataset):
    """Infinite source IterDataset drawing (theta, x) from prior + simulator.

    Implements grain's SupportsInPlaceSlicing protocol: under mp_prefetch,
    grain calls set_slice(slice(worker_index, None, num_workers)) on each
    worker's copy — required for a parentless source IterDataset, and it
    doubles as the per-worker stream id (folded into the PRNG key so workers
    produce independent streams).
    """

    def __init__(self, task, simulator, seed, batch_size):
        super().__init__()
        self._task = task
        self._simulator = simulator
        self._seed = int(seed)  # plain int; keys built lazily in the iterator
        self._batch_size = int(batch_size)
        self._worker_index = 0
        self._worker_count = 1

    def set_slice(self, sl, sequential_slice=False):
        del sequential_slice
        self._worker_index = sl.start or 0
        self._worker_count = sl.step or 1

    def __iter__(self):
        return _SimIterator(self)


class _SimIterator(grain.DatasetIterator):
    """Iterator holding the running PRNG key as checkpointable state."""

    def __init__(self, parent):
        super().__init__()
        self._p = parent
        base = jax.random.PRNGKey(parent._seed)
        self._key = jax.random.fold_in(base, parent._worker_index)

    def __next__(self):
        self._key, sub = jax.random.split(self._key)
        kt, ks = jax.random.split(sub)
        theta = self._p._task.get_prior(kt, self._p._batch_size)
        x = self._p._simulator(ks, theta)
        # Raw numpy across the pickle boundary; tokenization happens in the
        # main process (make_collate_jax).
        return {"thetas": np.asarray(theta), "xs": np.asarray(x)}

    # Abstract on DatasetIterator and genuinely called by grain's worker
    # loop (checkpoint/seek) — real implementations, not stubs. PRNGKey is a
    # raw uint32 (2,) array, so the key IS the state.
    def get_state(self):
        return {"key": np.asarray(self._key).tolist()}

    def set_state(self, state):
        self._key = jnp.asarray(state["key"], dtype=jnp.uint32)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data/test_online_dataset.py -v
```

Expected: 6 PASS. (If `iter(ds)` returns a stats-wrapping object without `get_state`, switch the test to `ds.__iter__()` — grain's `__init_subclass__` injector normally returns the iterator itself.)

Also re-run the offline tests to confirm the new imports didn't break anything:

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data -v
```

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/data/dataset.py tests/data/test_online_dataset.py
git commit -m "feat(data): grain source IterDataset simulating (theta, x) on the fly"
```

---

### Task 4: `OnlineTaskDataset` + `get_online_train_loader`

**Files:**
- Modify: `src/sbibm_jax/data/dataset.py` (new class after `_SimIterator`)
- Test: `tests/data/test_online_dataset.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_online_dataset.py`:

```python
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
    def test_get_reference_via_posterior_config(self, monkeypatch, patched_meta):
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data/test_online_dataset.py -v
```

Expected: Task 3 classes PASS; new classes ERROR with `ImportError: cannot import name 'OnlineTaskDataset'`.

- [ ] **Step 3: Implement `OnlineTaskDataset`**

Append to `src/sbibm_jax/data/dataset.py` (after `_SimIterator`):

```python
class OnlineTaskDataset(TaskDataset):
    """Simulate-on-the-fly variant of TaskDataset.

    Serves fresh (theta, x) batches from the task's prior + simulator instead
    of the pre-generated HF splits; metadata-driven shapes, stats, and
    tokenization are identical to TaskDataset (same metadata.json). The HF
    splits are never downloaded; get_reference/get_true_parameters still work
    (separate {name}_posterior config).

    Assumes the simulator always yields finite rows: tasks whose simulators
    legitimately diverge (hf_resample_invalid=True, i.e. ODE/PEtab) are not
    intended for online use, and hf_external tasks without a simulator
    (gravitational_waves) fail at construction.

    Simulator.num_simulations is only meaningful with num_workers=0: under
    mp_prefetch each worker counts on its own pickled copy.
    """

    def __init__(
        self,
        name,
        *,
        kind="conditional",
        repo=None,
        normalize=False,
        dtype=jnp.float32,
        seed=42,
    ):
        self.name = name
        self.kind = kind
        self.repo = repo if repo is not None else config.TEST_REPO
        self.normalize = normalize
        self.dtype = dtype
        self.seed = seed

        self._init_metadata(self._load_metadata_entry())
        # Replace the numpy collate set by _init_metadata: the online path
        # collates in the main process, after the pickle boundary, so jnp is
        # safe (and saves a host round-trip before the training step).
        self._collate = make_collate_jax(
            kind=kind, x_kind=self.x_kind, theta_kind=self.theta_kind,
            normalize=normalize, stats=self._stats, dtype=dtype,
        )

        self.task = get_task(name)
        # Eager build: tasks without a simulator raise NotImplementedError
        # here, at construction, instead of on the first next().
        self.simulator = self.task.get_simulator(
            jax.random.PRNGKey(self.seed), max_calls=None,
        )

    def _offline_error(self):
        return NotImplementedError(
            "OnlineTaskDataset generates batches on the fly; use "
            "get_online_train_loader."
        )

    def get_train_loader(self, batch_size, num_samples=None):
        raise self._offline_error()

    def get_val_loader(self, batch_size):
        raise self._offline_error()

    def get_test_loader(self, batch_size):
        raise self._offline_error()

    def get_online_train_loader(self, batch_size, *, seed=None, num_workers=0):
        """Infinite loader of freshly simulated, tokenized jnp batches.

        Reproducible for a fixed (seed, num_workers); changing num_workers
        changes the stream (grain stateful-transform caveat) but stays
        deterministic. num_workers=0 simulates in-process (on the default JAX
        device); num_workers>=1 simulates in CPU spawn workers, leaving the
        GPU to the training step. Pass a distinct seed for independent
        concurrent loaders.
        """
        seed = self.seed if seed is None else seed
        num_workers = min(int(num_workers), _MAX_WORKERS_CAP)
        ds = _SimIterDataset(self.task, self.simulator, seed, batch_size)
        if num_workers > 0:
            ds = ds.mp_prefetch(
                grain.MultiprocessingOptions(num_workers=num_workers),
                worker_init_fn=_worker_init,
            )
        return ds.map(self._collate)
```

- [ ] **Step 4: Export from the subpackage**

In `src/sbibm_jax/data/__init__.py`, replace the import/`__all__` lines:

```python
from sbibm_jax.data.dataset import OnlineTaskDataset, TaskDataset  # noqa: E402

__all__ = ["OnlineTaskDataset", "TaskDataset"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data/test_online_dataset.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sbibm_jax/data/dataset.py src/sbibm_jax/data/__init__.py tests/data/test_online_dataset.py
git commit -m "feat(data): OnlineTaskDataset with get_online_train_loader"
```

---

### Task 5: `mp_prefetch` smoke test (1 worker)

End-to-end through the real multiprocessing path: spawn, cloudpickle of the closure-based simulator, `set_slice`, `worker_init_fn`, numpy across the boundary, jnp collate in the main process. Kept to one worker / one batch for CI cost (spawn + jax import in the worker takes tens of seconds — same precedent as `test_prefetching_loader_iterates` in `test_dataset.py`).

**Files:**
- Test: `tests/data/test_online_dataset.py`

- [ ] **Step 1: Write the test**

Append to `tests/data/test_online_dataset.py`:

```python
class TestMultiprocessSmoke:
    def test_one_worker_end_to_end(self, patched_meta):
        # Exercises spawn + cloudpickle of the closure-based Simulator,
        # set_slice (grain calls it with slice(0, None, 1)), _worker_init
        # (jax -> cpu in the worker), numpy across the pickle boundary, and
        # the main-process jnp collate.
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
```

- [ ] **Step 2: Run it (this is the real verification — no fail-first step, it's an integration smoke test)**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data/test_online_dataset.py::TestMultiprocessSmoke -v
```

Expected: PASS (allow up to ~2 min for the spawn). If it fails inside the worker, the traceback crosses the process boundary via grain — read it from the test output and apply `superpowers:systematic-debugging` before touching code.

- [ ] **Step 3: Run the whole data suite**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/data/test_online_dataset.py
git commit -m "test(data): mp_prefetch smoke test for the online loader"
```

---

### Task 6: Docs, lint, full suite

**Files:**
- Modify: `CLAUDE.md` (the "Consumer loader" paragraph)

- [ ] **Step 1: Document in CLAUDE.md**

In the `**Consumer loader (src/sbibm_jax/data/).**` paragraph, after the sentence ending "`get_train_loader(num_samples=N)` subsamples a prefix.", insert:

```
`OnlineTaskDataset` (same module) is the simulate-on-the-fly variant: it
reads the same `metadata.json` (shapes/stats; no split download), builds the
task's prior + simulator eagerly (`hf_external` tasks fail at construction),
and serves an infinite `get_online_train_loader(batch_size, seed=,
num_workers=)` — a custom grain source `IterDataset` that draws fresh
`(theta, x)` per batch, optionally in CPU spawn workers (`mp_prefetch`;
worker identity via grain's `set_slice` protocol, CPU forced via
`jax.config.update` in `worker_init_fn`), collated to jnp tokens in the main
process via `make_collate_jax`. Offline loaders raise on it; reference
access still works. Finite-simulator tasks only (not `hf_resample_invalid`
ODE/PEtab tasks).
```

- [ ] **Step 2: Lint (new violations vs HEAD only)**

```bash
$VENV/bin/flake8 src/sbibm_jax/data tests/data
```

Compare against the same command on HEAD if unsure; fix any violation introduced by this branch (the pre-existing E501 baseline is expected noise).

- [ ] **Step 3: Run the full test suite**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest
```

Expected: everything green except the 6 known petab-only environment failures (pre-existing; unrelated to this branch).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document OnlineTaskDataset in CLAUDE.md"
```

---

## Self-review notes (already applied)

- Spec coverage: `_init_metadata`/`_init_splits` refactor + `self._stats` (Task 1), `make_collate_jax` (Task 2), source dataset with `set_slice`/state/CPU-forcing init fn (Task 3), `OnlineTaskDataset` with eager simulator, overridden offline loaders, online loader with worker cap and `num_workers=0` in-process mode (Task 4), mp smoke (Task 5), docs (Task 6). Every spec test bullet maps to a test in Tasks 1–5.
- Type consistency: `_SimIterDataset(task, simulator, seed, batch_size)` is constructed identically in Task 3 tests, Task 4 implementation, and the Task 4 normalize test. `_stats` defined in Task 1 is consumed in Task 4. `make_collate_jax` kwargs match `make_collate`'s.
- Grain facts used here (set_slice protocol, abstract state methods, `jax.config.update` over env var, spawn+cloudpickle, `num_workers=0` no-op) were verified against the installed grain 0.2.17 — see the spec's "Verified grain 0.2.17 facts" section.
