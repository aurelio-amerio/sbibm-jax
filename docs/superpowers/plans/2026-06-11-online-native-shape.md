# OnlineTaskDataset Native-Shape (Non-Vector) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the vector-only restriction on `OnlineTaskDataset` so non-vector tasks — primarily `gaussian_random_field` (image) — simulate online with tokenization/normalization identical to the offline `TaskDataset`.

**Architecture:** `_SimIterDataset` gains an optional `x_shape`; when set, the iterator reshapes the simulator's flat output to `(n, *x_shape)` (free numpy view) so raw worker batches are layout-identical to offline HF rows and the collate/stats need zero changes. `OnlineTaskDataset` passes the published metadata `x_shape` down and its construction guard shrinks to theta-only.

**Tech Stack:** JAX, grain 0.2.17, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-06-11-online-native-shape-design.md` — read it first (decision table for why metadata `x_shape`, not `unflatten_data`).

---

## Working environment

- Branch: `online-native-shape` (already created off `main` in the main repo). If executing in a worktree (user's preferred workflow), create it off this branch's HEAD via the `superpowers:using-git-worktrees` skill.
- `uv` fails inside the sandbox (read-only caches). Run tests with the main checkout's venv python plus `PYTHONPATH` pointing at the *current* checkout's `src`:

  ```bash
  VENV=/lustre/ific.uv.es/ml/ific088/github/sbibm-jax/.venv
  PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data -v
  ```

  `JAX_PLATFORMS=cpu` and `-n 2` are injected by the pytest config; no GPU is touched.
- Lint: `$VENV/bin/flake8 src/sbibm_jax/data tests/data`. The baseline is **never clean** (pre-existing E501); judge only violations introduced relative to HEAD.
- Never exceed 8 workers/cores (shared node).
- The working tree may carry PRE-EXISTING uncommitted changes (`pyproject.toml`, `uv.lock`, deleted `metadata.json`, spec docs). Never stage or commit those; `git add` only the files each task lists.

## File map

| File | Change |
|---|---|
| `src/sbibm_jax/data/dataset.py` | `_SimIterDataset(..., x_shape=None)` + iterator reshape; `OnlineTaskDataset` guard → theta-only, pass `x_shape`, docstrings |
| `tests/data/test_online_dataset.py` | source-level reshape tests; guard test repurposed; GRF loader + normalize tests; timeseries-shape test; GRF mp smoke |
| `CLAUDE.md` | update the `OnlineTaskDataset` sentence |

---

### Task 1: `x_shape` on `_SimIterDataset` (worker-side flat→native reshape)

**Files:**
- Modify: `src/sbibm_jax/data/dataset.py` (`_SimIterDataset.__init__`, `_SimIterator.__next__`, class docstring)
- Test: `tests/data/test_online_dataset.py` (`TestSimIterDataset`)

- [ ] **Step 1: Write the failing tests**

Append to `TestSimIterDataset` in `tests/data/test_online_dataset.py` (after `test_state_roundtrip`):

```python
    def test_x_shape_reshapes_to_native(self):
        # GRF's simulator emits flat (n, 1024); x_shape from metadata makes
        # the source emit native rows, layout-identical to offline HF rows.
        from sbibm_jax.data.dataset import _SimIterDataset
        from sbibm_jax.tasks import get_task
        task = get_task("gaussian_random_field")
        sim = task.get_simulator(jax.random.PRNGKey(0), max_calls=None)
        ds = _SimIterDataset(task, sim, 0, 2, x_shape=(32, 32))
        batch = next(iter(ds))
        assert batch["xs"].shape == (2, 32, 32)
        assert batch["thetas"].shape == (2, 2)
        assert isinstance(batch["xs"], np.ndarray)
        assert np.isfinite(batch["xs"]).all()

    def test_x_shape_default_stays_flat(self):
        # Without x_shape the source keeps the simulator's flat output.
        from sbibm_jax.data.dataset import _SimIterDataset
        from sbibm_jax.tasks import get_task
        task = get_task("gaussian_random_field")
        sim = task.get_simulator(jax.random.PRNGKey(0), max_calls=None)
        batch = next(iter(_SimIterDataset(task, sim, 0, 2)))
        assert batch["xs"].shape == (2, 1024)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest "tests/data/test_online_dataset.py::TestSimIterDataset" -v
```

Expected: `test_x_shape_reshapes_to_native` FAILS with `TypeError: _SimIterDataset.__init__() got an unexpected keyword argument 'x_shape'`; `test_x_shape_default_stays_flat` PASSES (current behavior); the 6 existing tests PASS.

- [ ] **Step 3: Implement**

In `src/sbibm_jax/data/dataset.py`, change `_SimIterDataset.__init__` to:

```python
    def __init__(self, task, simulator, seed, batch_size, x_shape=None):
        super().__init__()
        self._task = task
        self._simulator = simulator
        self._seed = int(seed)  # plain int; keys built lazily in the iterator
        self._batch_size = int(batch_size)
        self._x_shape = None if x_shape is None else tuple(x_shape)
        self._worker_index = 0
        self._worker_count = 1
```

Append to the `_SimIterDataset` class docstring (after the existing paragraph, separated by a blank line):

```
    With x_shape set (the published metadata x_shape), the simulator's flat
    rows are reshaped to (n, *x_shape) before crossing the pickle boundary,
    so raw batches are layout-identical to offline HF rows (a no-op for
    vector tasks, native for image/timeseries). None keeps flat output.
```

In `_SimIterator.__next__`, replace the return block (the comment plus `return` statement) with:

```python
        xs = np.asarray(x)
        if self._p._x_shape is not None:
            # Native layout matching the offline HF rows (metadata x_shape);
            # a free view — same bytes across the pickle boundary.
            xs = xs.reshape(-1, *self._p._x_shape)
        # Raw numpy across the pickle boundary; tokenization happens in the
        # main process (make_collate_jax).
        return {"thetas": np.asarray(theta), "xs": xs}
```

- [ ] **Step 4: Run the online test file to verify all pass**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data/test_online_dataset.py -v
```

Expected: all PASS (8 in `TestSimIterDataset`, rest unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/data/dataset.py tests/data/test_online_dataset.py
git commit -m "feat(data): optional flat->native x_shape reshape in _SimIterDataset"
```

---

### Task 2: theta-only guard + pass `x_shape`; GRF and timeseries loader tests

**Files:**
- Modify: `src/sbibm_jax/data/dataset.py` (`OnlineTaskDataset`)
- Test: `tests/data/test_online_dataset.py`

- [ ] **Step 1: Write the failing tests**

In `tests/data/test_online_dataset.py`:

(a) Add a `gaussian_random_field` entry to the `meta` dict inside `_fake_metadata` (after the `"gravitational_waves"` entry; stats are global-scalar like the real image pipeline, deliberately non-trivial):

```python
        # Image task: flat (n, 1024) simulator rows, native (32, 32) storage.
        "gaussian_random_field": {
            "x_kind": "image", "x_shape": [32, 32],
            "theta_kind": "vector", "theta_shape": [2],
            "splits": {"train": 8, "validation": 4, "test": 4},
            "has_reference": False, "num_observations": 10,
            "stats": {
                "theta_mean": [[0.0, 3.0]], "theta_std": [[0.3, 0.5]],
                "x_mean": [[[0.1]]], "x_std": [[[2.0]]],
                "theta_axes": [0], "x_axes": [0, 1, 2],
            },
        },
```

(b) REPLACE the whole `test_non_vector_task_fails_at_construction` method of `TestOnlineConstruction` with (guard is now theta-only; non-vector x must construct fine — that path is covered by `TestOnlineLoaderNativeShape` below):

```python
    def test_non_vector_theta_fails_at_construction(
            self, monkeypatch, patched_meta, tmp_path):
        # Theta is served as flat tokens; no task has non-vector theta and
        # the online path makes no provision for it.
        meta = {"two_moons": {
            "x_kind": "vector", "x_shape": [2],
            "theta_kind": "image", "theta_shape": [2, 1],
            "splits": {"train": 8, "validation": 4, "test": 4},
            "has_reference": True, "num_observations": 2, "stats": None,
        }}
        p = tmp_path / "theta_image_metadata.json"
        p.write_text(json.dumps(meta))
        monkeypatch.setattr(
            "sbibm_jax.data.dataset.hf_hub_download", lambda **kw: str(p),
        )
        from sbibm_jax.data import OnlineTaskDataset
        with pytest.raises(NotImplementedError, match="vector theta"):
            OnlineTaskDataset("two_moons")
```

(c) Append a new test class after `TestOnlineLoader`:

```python
class TestOnlineLoaderNativeShape:
    """Non-vector x: worker source reshapes flat -> metadata x_shape."""

    def test_image_tokens_native_shape(self, patched_meta):
        from sbibm_jax.data import OnlineTaskDataset
        ds = OnlineTaskDataset("gaussian_random_field")
        theta, x = next(iter(ds.get_online_train_loader(batch_size=4)))
        assert isinstance(x, jax.Array)
        assert theta.shape == (4, 2, 1)
        assert x.shape == (4, 32, 32, 1)
        assert np.isfinite(np.asarray(x)).all()

    def test_image_normalize_matches_manual_collate(self, patched_meta):
        from sbibm_jax.data import OnlineTaskDataset
        from sbibm_jax.data.dataset import _SimIterDataset
        from sbibm_jax.data.process import make_collate_jax
        ds = OnlineTaskDataset("gaussian_random_field", normalize=True)
        theta_n, x_n = next(iter(
            ds.get_online_train_loader(batch_size=4, seed=7)))
        assert x_n.shape == (4, 32, 32, 1)
        # Same raw draw, collated manually with the same global-scalar stats.
        raw = next(iter(_SimIterDataset(
            ds.task, ds.simulator, 7, 4, x_shape=(32, 32))))
        collate = make_collate_jax(kind="conditional", x_kind="image",
                                   normalize=True, stats=ds._stats)
        theta_m, x_m = collate(raw)
        np.testing.assert_allclose(np.asarray(theta_n), np.asarray(theta_m),
                                   atol=1e-6)
        np.testing.assert_allclose(np.asarray(x_n), np.asarray(x_m),
                                   atol=1e-6)
        # And it actually normalized (stats are non-trivial in the fixture).
        raw_tok = np.asarray(raw["xs"], np.float32)[..., None]
        assert not np.allclose(np.asarray(x_n), raw_tok)

    def test_timeseries_rank_generality(
            self, monkeypatch, patched_meta, tmp_path):
        # two_moons faked as a (2, 1) timeseries: element count matches
        # dim_x=2, proving the reshape is rank-generic without a heavy task.
        meta = {"two_moons": {
            "x_kind": "timeseries", "x_shape": [2, 1],
            "theta_kind": "vector", "theta_shape": [2],
            "splits": {"train": 8, "validation": 4, "test": 4},
            "has_reference": True, "num_observations": 2, "stats": None,
        }}
        p = tmp_path / "ts_metadata.json"
        p.write_text(json.dumps(meta))
        monkeypatch.setattr(
            "sbibm_jax.data.dataset.hf_hub_download", lambda **kw: str(p),
        )
        from sbibm_jax.data import OnlineTaskDataset
        ds = OnlineTaskDataset("two_moons")
        theta, x = next(iter(ds.get_online_train_loader(batch_size=4)))
        assert theta.shape == (4, 2, 1)
        assert x.shape == (4, 2, 1, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest "tests/data/test_online_dataset.py::TestOnlineConstruction" "tests/data/test_online_dataset.py::TestOnlineLoaderNativeShape" -v
```

Expected: `test_non_vector_theta_fails_at_construction` FAILS (current guard message says "vector-only", not "vector theta"); all three `TestOnlineLoaderNativeShape` tests FAIL with `NotImplementedError: OnlineTaskDataset is vector-only ...` raised at construction.

- [ ] **Step 3: Implement**

In `src/sbibm_jax/data/dataset.py`, inside `OnlineTaskDataset`:

(a) REPLACE the guard block at the end of `__init__` (the four comment lines starting `# The Simulator wrapper emits flat rows` plus the `if self.x_kind ...: raise ...` statement) with:

```python
        # Theta is served as flat tokens; no task has non-vector theta and
        # the online path makes no provision for it. (x is fine at any rank:
        # the source reshapes flat simulator rows to the metadata x_shape.)
        if self.theta_kind != "vector":
            raise NotImplementedError(
                f"OnlineTaskDataset requires vector theta; task {name!r} "
                f"has theta_kind={self.theta_kind!r}."
            )
```

(b) In the class docstring, REPLACE the sentence `Vector x/theta only for now: simulators emit flat rows and the online path has no flat->native reshape.` with:

```
    x is reshaped flat -> native (metadata x_shape) in the worker source, so
    image/timeseries tasks work online; theta must be vector.
```

(c) In `get_online_train_loader`, change the `_SimIterDataset` construction line to:

```python
        ds = _SimIterDataset(
            self.task, self.simulator, seed, batch_size, x_shape=self.x_shape,
        )
```

(For vector tasks `x_shape == (dim_x,)`, so the reshape is a no-op — existing vector tests must stay green unchanged.)

- [ ] **Step 4: Run the online test file, then the whole data suite**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data/test_online_dataset.py -v
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data -v
```

Expected: all PASS (no regressions in vector tests; GRF loader/normalize/timeseries tests green).

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/data/dataset.py tests/data/test_online_dataset.py
git commit -m "feat(data): OnlineTaskDataset serves non-vector tasks (native x_shape)"
```

---

### Task 3: GRF `mp_prefetch` smoke test (1 worker)

End-to-end through the real multiprocessing path for the primary non-vector use case: spawn + cloudpickle of the FFT closure simulator, CPU `worker_init`, native-shaped numpy across the pickle boundary, jnp image tokens in the main process. One worker / one batch for CI cost (same precedent as the existing `TestMultiprocessSmoke`).

**Files:**
- Test: `tests/data/test_online_dataset.py`

- [ ] **Step 1: Write the test**

Append at the very end of `tests/data/test_online_dataset.py` (after `TestMultiprocessSmoke`):

```python
class TestMultiprocessSmokeGRF:
    def test_one_worker_image_end_to_end(self, patched_meta):
        # The primary non-vector use case through the real mp path: spawn +
        # cloudpickle of the FFT closure simulator, _worker_init (jax -> cpu),
        # native-shaped numpy across the pickle boundary, jnp image tokens in
        # the main process. NOTE: grain skips set_slice for num_workers==1.
        from sbibm_jax.data import OnlineTaskDataset
        ds = OnlineTaskDataset("gaussian_random_field")
        loader = ds.get_online_train_loader(batch_size=2, num_workers=1)
        it = iter(loader)
        try:
            theta, x = next(it)
            assert np.asarray(theta).shape == (2, 2, 1)
            assert np.asarray(x).shape == (2, 32, 32, 1)
            assert np.isfinite(np.asarray(x)).all()
        finally:
            # grain recommends closing mp_prefetch iterators explicitly.
            if hasattr(it, "close"):
                it.close()
```

- [ ] **Step 2: Run it (integration smoke — no fail-first step)**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest "tests/data/test_online_dataset.py::TestMultiprocessSmokeGRF" -v
```

Expected: PASS (allow up to ~2 min for the spawn; set a generous bash timeout, e.g. 300000 ms). If it fails inside the worker, the traceback crosses the process boundary via grain — debug from the actual traceback before touching code.

- [ ] **Step 3: Run the whole data suite**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest tests/data -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/data/test_online_dataset.py
git commit -m "test(data): GRF mp_prefetch smoke test for the native-shape online path"
```

---

### Task 4: Docs, lint, full suite

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

In the `OnlineTaskDataset` sentence of the Consumer-loader paragraph, REPLACE:

```
Finite-simulator, vector-x/theta tasks only (not
`hf_resample_invalid` ODE/PEtab tasks; no flat->native reshape yet).
```

with:

```
Finite-simulator, vector-theta tasks only (not
`hf_resample_invalid` ODE/PEtab tasks); x is reshaped flat->native from the
metadata `x_shape` in the worker source, so image/timeseries tasks (e.g.
`gaussian_random_field`) work online.
```

(Keep the file's existing line-wrapping style.)

- [ ] **Step 2: Lint (new violations vs HEAD only)**

```bash
$VENV/bin/flake8 src/sbibm_jax/data tests/data
```

Fix any violation on a line introduced by this branch (pre-existing E501 baseline is expected noise).

- [ ] **Step 3: Run the full test suite**

```bash
PYTHONPATH=src HF_HOME=$TMPDIR/hfhome $VENV/bin/python -m pytest
```

Expected: all green (289 + the new tests; pypesto is installed on this node, so no petab failures). Report the exact summary line. Allow ~10 min (bash timeout 600000 ms).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: OnlineTaskDataset supports non-vector tasks via metadata x_shape"
```

---

## Self-review notes (already applied)

- Spec coverage: worker-source `x_shape` reshape (Task 1), theta-only guard + metadata `x_shape` pass-through + GRF/normalize/timeseries tests (Task 2), GRF mp smoke (Task 3), docs (Task 4). Every spec test bullet maps to a test; the spec's "repurposed guard test" is Task 2(b) plus the image-x construct-and-load coverage in `TestOnlineLoaderNativeShape`.
- Type consistency: `_SimIterDataset(task, sim, seed, batch_size, x_shape=...)` is identical in Task 1 tests, Task 2 implementation (c), and the Task 2 normalize test. The guard message "requires vector theta" matches the test regex `"vector theta"`. The fake GRF stats shapes (`x_mean [[[0.1]]]` → `_stat_array_jax` → `(1,1,1,1)`) match the real image-pipeline reduction axes `(0,1,2)`.
- Grain facts: no new grain surface is touched — `mp_prefetch`/`set_slice`/state methods are unchanged from phase 1; the reshape happens inside `__next__` before the existing numpy emission.
