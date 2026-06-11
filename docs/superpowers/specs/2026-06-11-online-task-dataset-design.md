# OnlineTaskDataset — simulate-on-the-fly training loader

**Date:** 2026-06-11 (revised 2026-06-11 after grain API verification)
**Status:** Approved (design)

## Goal

Add an online training loader to `src/sbibm_jax/data/dataset.py` that generates
fresh `(theta, x)` batches by calling the task's prior + simulator, instead of
streaming pre-generated rows from the HuggingFace dataset. The public entry
point is `OnlineTaskDataset.get_online_train_loader(batch_size, ...)`.

This serves SBI methods that train on a live simulator (e.g. sequential / online
NPE-style training) rather than a fixed corpus, while keeping the output format,
tokenization, and normalization identical to the offline `TaskDataset` loaders.

## Non-goals

- No rejection sampling / NaN handling. The online path **assumes the simulator
  always yields finite rows**. Tasks whose simulators legitimately diverge
  (ODE/PEtab, i.e. those that set `hf_resample_invalid=True`) are not intended
  for online use.
- No online validation/test loaders. Validation/test stay fixed; only training
  generates on the fly. Reference posterior access is inherited unchanged.
- No changes to the offline `TaskDataset` behavior or to the existing
  numpy `make_collate`.

## Verified grain 0.2.17 facts (drive the design below)

These were checked against the installed grain source; the design depends on
them, so re-verify on a grain upgrade:

1. `IterDataset.mp_prefetch(options, worker_init_fn, ...)` exists;
   `worker_init_fn(worker_index, worker_count)` runs in each worker **before**
   the dataset is unpickled (`process_prefetch.py`: parse flags →
   `worker_init_fn()` → `cloudpickle.loads(pickled_ds)`).
2. `num_workers=0` is a documented no-op — the pipeline runs in-process.
3. The start method is hard-coded to **spawn**
   (`mp.get_context("spawn")`, `process_prefetch.py:356`), so workers never
   inherit an initialized CUDA context. Datasets and the init fn are shipped
   with **cloudpickle**, so the closure-based `Simulator` and the task instance
   pickle fine.
4. **Sharding contract:** with `num_workers >= 1`, grain wraps each worker's
   copy of the dataset in `_LazyWorkerSliceIterDataset`, which calls
   `_set_slice_iter_dataset`. For a parentless *source* `IterDataset` this
   **requires** the `SupportsInPlaceSlicing` protocol —
   `set_slice(sl, sequential_slice=False)` — otherwise grain raises
   `ValueError("Cannot slice IterDataset source")`. Worker `i` receives
   `slice(i, None, num_workers)`, i.e. `sl.start` *is* the worker index and
   `sl.step` the worker count.
5. `grain.DatasetIterator` has **abstract** `get_state()` / `set_state(state)`,
   and the worker loop genuinely calls `set_state` (checkpoint/seek protocol) —
   they must be real implementations, not `pass`-stubs.
6. JAX captures env-var defaults (`JAX_PLATFORMS`) at **import** time, and jax
   is already imported in the worker before `worker_init_fn()` is *called*
   (cloudpickle loads the init fn by reference → imports its defining module →
   imports jax transitively via `sbibm_jax`). Therefore
   `os.environ["JAX_PLATFORMS"] = "cpu"` inside the init fn is silently
   ineffective; `jax.config.update("jax_platforms", "cpu")` works any time
   before first backend use, and per fact 1 the init fn runs before anything
   (the pickled dataset) could touch a backend.

## Architecture

### Class layout — `OnlineTaskDataset(TaskDataset)`

A new subclass in `dataset.py`. It reuses the base's metadata/stats handling,
**skips** the HF split download, and instead loads the JAX task.

Refactor `TaskDataset.__init__` to separate its two concerns so the subclass can
reuse one and skip the other:

- `_init_metadata(entry)` — shapes, kinds, dims, `num_observations`,
  `has_reference`, stats, and `self._collate`. **Shared** by both classes.
  It must store the raw stats dict as `self._stats` (today `stats` is a local
  passed straight to `make_collate`) so the subclass can rebuild the collate
  without re-parsing metadata.
- `_init_splits()` — `load_dataset(...)`, `df_train/val/test`, `max_samples`.
  **Offline-only.**

`TaskDataset.__init__` calls both (behavior unchanged).
`OnlineTaskDataset.__init__`:

1. Downloads + parses `metadata.json` and calls `_init_metadata(entry)` (so
   normalization stats and shapes match the published dataset exactly).
2. Rebuilds `self._collate` with `make_collate_jax` (same kwargs, jnp backend),
   overriding the numpy collate `_init_metadata` set.
3. Does **not** call `_init_splits()` — pure-online use never triggers a
   dataset download.
4. Sets `self.task = get_task(name)` and builds the simulator **eagerly**:
   `self.simulator = self.task.get_simulator(key, max_calls=None)`. Building
   eagerly means tasks without a simulator (`gravitational_waves`,
   `hf_external` → `NotImplementedError`) fail immediately at construction with
   a clear error, not on first `next()`.

Consequences:

- Inherited offline `get_train_loader` / `get_val_loader` / `get_test_loader`
  are overridden to raise
  `NotImplementedError("OnlineTaskDataset generates batches on the fly; use "
  "get_online_train_loader.")`.
- `get_reference` / `get_true_parameters` still work — they lazily load the
  separate `{name}_posterior` config and do not depend on the train/val/test
  splits. So online training + HF reference for evaluation is a valid combo.
- The `Simulator.num_simulations` counter is only meaningful in-process: under
  `mp_prefetch` each worker increments its own pickled copy and the
  main-process count stays 0. Harmless (`max_calls=None`), but document it in
  the `get_online_train_loader` docstring so nobody trusts the counter for
  online runs.

### Collate — `make_collate_jax` (new, in `process.py`)

Add a standalone `make_collate_jax` alongside the existing numpy `make_collate`.
Same logic and signature, but `jnp.asarray` / `jnp.concatenate`, with the stats
tokenized to trailing-dim once at construction. The existing numpy
`make_collate` and `_stat_array` are left **untouched** (no shared-backend
parameterization — a separate function is clearer and lower-risk).

```python
def make_collate_jax(*, kind, x_kind, theta_kind="vector", normalize=False,
                     stats=None, dtype=jnp.float32):
    # mirrors make_collate; conditional -> (theta_tok, x_tok),
    # joint -> jnp.concatenate, vector-only guard for joint, [..., None] tokens.
```

## Pipeline (grain, `mp_prefetch`)

The loader runs the simulator in CPU worker processes via grain `mp_prefetch`,
keeping the GPU free for the training step in the main process. Collation +
jnp conversion happen in the main process, **after** the pickle boundary.

```
_SimIterDataset (workers, CPU)  ->  mp_prefetch (numpy across pickle boundary)
                                ->  .map(make_collate_jax) (main, jnp)
```

### 1. `_SimIterDataset` — custom infinite source `IterDataset` (runs in workers)

Holds the task, the simulator, the **seed as a plain int** (not a PRNG key
device array — keys are built lazily inside the iterator, so nothing
GPU-resident crosses the pickle boundary and iterator state stays trivially
serializable), and the batch size.

**Worker identity comes from grain's slicing protocol, not from
`worker_init_fn`** (verified fact 4 — implementing `set_slice` is mandatory
anyway, and it delivers the worker index for free):

```python
class _SimIterDataset(grain.IterDataset):
    def __init__(self, task, simulator, seed, batch_size):
        super().__init__()
        self._task, self._simulator = task, simulator
        self._seed, self._batch_size = seed, batch_size
        self._worker_index, self._worker_count = 0, 1

    # grain SupportsInPlaceSlicing: mp_prefetch calls this on each worker's
    # copy with slice(worker_index, None, num_workers). Required for a source
    # IterDataset under mp_prefetch; also our per-worker stream seed.
    def set_slice(self, sl, sequential_slice=False):
        self._worker_index = sl.start or 0
        self._worker_count = sl.step or 1

    def __iter__(self):
        return _SimIterator(self)
```

With `num_workers=0`, `set_slice` is never called and the defaults
(`worker_index=0`) apply — the in-process stream equals worker 0's stream.

### 2. `_SimIterator` — `grain.DatasetIterator` with real state methods

Folds the worker index into the base key so workers produce independent
streams (without this, every worker would replay the *same* stream — a silent
duplication bug), holds the running key as state, and advances it per
`__next__` (infinite stream):

```python
class _SimIterator(grain.DatasetIterator):
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
        return {"thetas": np.asarray(theta), "xs": np.asarray(x)}

    # get_state/set_state are ABSTRACT on DatasetIterator and the grain worker
    # loop calls set_state (checkpoint/seek). Real implementations, not stubs:
    def get_state(self):
        return {"key_data": np.asarray(jax.random.key_data(self._key)).tolist()}

    def set_state(self, state):
        self._key = jnp.asarray(state["key_data"], dtype=jnp.uint32)
```

It emits **raw numpy** `{thetas, xs}` — no tokenization in the worker — because
numpy pickles cheaply across the process boundary while jax device arrays are a
footgun there.

### 3. `worker_init_fn(worker_index, worker_count)`

Runs in each worker before data flows. Its **only** job is forcing JAX onto
CPU so workers simulate on CPU and leave the GPU (and its default memory
preallocation) for training:

```python
def _worker_init(worker_index, worker_count):
    import jax
    jax.config.update("jax_platforms", "cpu")
```

- It must use `jax.config.update`, **not** `os.environ["JAX_PLATFORMS"]` —
  the env var is captured at jax import time and jax is already imported in
  the worker before the init fn is called (verified fact 6). The config update
  works because the init fn runs before the dataset (and its first jax op) is
  unpickled (verified fact 1).
- Spawn start method is guaranteed by grain itself (verified fact 3) — no
  inherited CUDA context to worry about.
- It does **not** carry the worker index; that arrives via `set_slice`
  (section 1).

### 4. `mp_prefetch`

`.mp_prefetch(grain.MultiprocessingOptions(num_workers=N), worker_init_fn=_worker_init)`.
`num_workers=0` is a valid no-op mode → simulation runs in-process (on GPU,
since `_worker_init` never runs); `worker_index` defaults to 0. `num_workers`
is clamped to `_MAX_WORKERS_CAP` (8; shared-node rule).

### 5. Main-process `.map(make_collate_jax)`

A `.map` chained after `mp_prefetch` runs in the consumer (main) process. This
is where tokenization, optional normalization, and the jnp conversion happen,
producing the final output:

- `kind="conditional"` → `(theta_tok, x_tok)` jnp arrays
- `kind="joint"` → single concatenated jnp array (vector-only guard inherited)

Output shapes/semantics match the offline loader exactly, as **jnp** instead of
numpy.

### Public method

```python
def get_online_train_loader(self, batch_size, *, seed=None, num_workers=0):
    seed = self.seed if seed is None else seed
    ...
```

- `seed` defaults to `self.seed`; pass a distinct seed for independent
  concurrent loaders.
- Worker buffering is configured via `grain.MultiprocessingOptions`
  (`per_worker_buffer_size`); the default is used unless a need arises to expose
  it.
- Reproducible for a fixed `(seed, num_workers)`. Per grain's stateful-transform
  warning, changing `num_workers` changes the stream (still deterministic).
  This is documented behavior.

## Error handling

- No simulator (`gravitational_waves`, `hf_external`): `NotImplementedError`
  surfaces at construction (eager simulator build).
- Offline loaders on `OnlineTaskDataset`: overridden to raise an informative
  `NotImplementedError` pointing at `get_online_train_loader`.
- `kind="joint"` with non-vector x/theta: existing vector-only guard, inherited
  via `make_collate_jax`.

## Testing

CPU-forced like the rest of the suite (`JAX_PLATFORMS=cpu`). Use a cheap
analytical task (e.g. `gaussian_linear` / `two_moons`). **No network:** reuse
the existing `tests/data/test_dataset.py` pattern of monkeypatching
`hf_hub_download` to a local fake `metadata.json` (online tests never need the
split download, only metadata + optional posterior — both already faked by
that fixture).

- Construction: `OnlineTaskDataset(name)` builds; simulator created eagerly.
- `get_online_train_loader(batch_size)` with **`num_workers=0`** (in-process,
  fast, deterministic) yields batches of correct shapes and **jnp** dtype, for
  both `kind="conditional"` and `kind="joint"`.
- Reproducibility: same `seed` → identical first batches; different `seed` →
  different.
- Fresh draws: consecutive batches differ.
- Normalization: `normalize=True` output equals `make_collate_jax` applied
  manually to a known prior+sim draw.
- `set_slice`: calling it with `slice(1, None, 2)` changes the stream vs. the
  default worker-0 stream (unit-level, no multiprocessing needed).
- State round-trip: `get_state` → fresh iterator → `set_state` reproduces the
  next batch.
- Offline loaders raise the informative `NotImplementedError`; `get_reference`
  still works for a task that ships a reference.
- One smoke test with `num_workers=1` to exercise the worker / `set_slice` /
  `worker_init_fn` / cloudpickle path end to end (kept minimal for CI cost).
