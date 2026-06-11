# OnlineTaskDataset — simulate-on-the-fly training loader

**Date:** 2026-06-11
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

## Architecture

### Class layout — `OnlineTaskDataset(TaskDataset)`

A new subclass in `dataset.py`. It reuses the base's metadata/stats handling,
**skips** the HF split download, and instead loads the JAX task.

Refactor `TaskDataset.__init__` to separate its two concerns so the subclass can
reuse one and skip the other:

- `_init_metadata(entry)` — shapes, kinds, dims, `num_observations`,
  `has_reference`, stats, and `self._collate`. **Shared** by both classes.
- `_init_splits()` — `load_dataset(...)`, `df_train/val/test`, `max_samples`.
  **Offline-only.**

`TaskDataset.__init__` calls both (behavior unchanged).
`OnlineTaskDataset.__init__`:

1. Downloads + parses `metadata.json` and calls `_init_metadata(entry)` (so
   normalization stats and shapes match the published dataset exactly).
2. Does **not** call `_init_splits()` — pure-online use never triggers a
   dataset download.
3. Sets `self.task = get_task(name)` and builds the simulator **eagerly**:
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

`OnlineTaskDataset` builds `self._collate` with `make_collate_jax` (overriding
the numpy collate that `_init_metadata` would otherwise set — i.e.
`_init_metadata` is given a hook/flag to pick the collate backend, or the
subclass rebuilds `self._collate` after calling it).

## Pipeline (grain, `mp_prefetch`)

The loader runs the simulator in CPU worker processes via grain `mp_prefetch`,
keeping the GPU free for the training step in the main process. Collation +
jnp conversion happen in the main process, **after** the pickle boundary.

```
_SimIterDataset (workers, CPU)  ->  mp_prefetch (numpy across pickle boundary)
                                ->  .map(make_collate_jax) (main, jnp)
```

### 1. `_SimIterDataset` — custom infinite `IterDataset` (runs in workers)

Each worker's iterator folds its **worker index** into the base key so workers
produce independent streams (without this, every worker replays the *same*
stream — a silent duplication bug):

```python
key = jax.random.fold_in(base_key, worker_index)   # worker_index via worker_init_fn
# __next__:
#   key, sub = jax.random.split(key)
#   kt, ks   = jax.random.split(sub)
#   theta    = task.get_prior(kt, batch_size)
#   x        = simulator(ks, theta)
#   return {"thetas": np.asarray(theta), "xs": np.asarray(x)}   # numpy, picklable
```

The iterator holds the running key as state and advances it per `__next__`
(infinite stream). It emits **raw numpy** `{thetas, xs}` — no tokenization in
the worker — because numpy pickles cleanly across the process boundary while
jax device arrays are a footgun there.

### 2. `worker_init_fn(worker_index, worker_count)`

Runs in each worker before data flows. It:

- Sets `os.environ["JAX_PLATFORMS"] = "cpu"` so workers simulate on CPU and
  leave the GPU for training. This requires the **spawn** start method so a
  worker does not inherit an already-initialized CUDA context from the parent;
  the implementation must confirm grain's `MultiprocessingOptions` uses (or can
  be made to use) spawn.
- Stashes `worker_index` into a module-level global that `_SimIterDataset`
  reads when constructing its per-worker key.

### 3. `mp_prefetch`

`.mp_prefetch(grain.MultiprocessingOptions(num_workers=N), worker_init_fn=_init)`.
`num_workers=0` is a valid no-op mode → simulation runs in-process (on GPU);
`worker_index` defaults to 0. `num_workers` is clamped to `_MAX_WORKERS_CAP` (8;
shared-node rule).

### 4. Main-process `.map(make_collate_jax)`

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
    base_key = jax.random.PRNGKey(self.seed if seed is None else seed)
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
analytical task (e.g. `gaussian_linear` / `two_moons`). These tasks need no HF
split download (metadata + optional posterior only).

- Construction: `OnlineTaskDataset(name)` builds; simulator created eagerly.
- `get_online_train_loader(batch_size)` with **`num_workers=0`** (in-process,
  fast, deterministic) yields batches of correct shapes and **jnp** dtype, for
  both `kind="conditional"` and `kind="joint"`.
- Reproducibility: same `seed` → identical first batches; different `seed` →
  different.
- Fresh draws: consecutive batches differ.
- Normalization: `normalize=True` output equals `make_collate_jax` applied
  manually to a known prior+sim draw.
- Offline loaders raise the informative `NotImplementedError`; `get_reference`
  still works for a task that ships a reference.
- One smoke test with `num_workers=1` to exercise the worker / `worker_init_fn`
  / pickle path (kept minimal for CI cost).
```

The grain `IterDataset` / `DatasetIterator` subclassing API and the
spawn-context behavior of `MultiprocessingOptions` are the two details to verify
first during implementation.
