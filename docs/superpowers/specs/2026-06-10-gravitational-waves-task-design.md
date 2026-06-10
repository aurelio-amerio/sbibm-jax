# Gravitational Waves task — design

**Date:** 2026-06-10
**Status:** Approved scope — file-backed (no simulator) time-series task; prior +
simulator + reference intentionally `NotImplementedError`; dataset built from
pre-generated `.pt`→`.npz` shards via a dedicated mirror script.

## What this is

A new `sbibm-jax` benchmark task, `gravitational_waves`, porting the GW task
from the GenSBI/SBI-benchmarks ecosystem:

- Consumer reference: `GenSBI-examples/src/gensbi_examples/tasks.py`
  (`GravitationalWaves(Task)`) — conditional-only, `theta` = 2 params,
  `x` = a `(8192, 2)` two-channel time series; reference posterior and true
  parameters raise `NotImplementedError`.
- Upload reference: `SBI-benchmarks-data/sbi_benchmarks/simulators/gw_dataset.py`
  — loads 10 `.pt` shards (`thetas_{0..9}`, `xs_{0..9}`), concatenates, splits,
  and pushes to HF as config `gravitational_waves` with features
  `xs: Array2D((8192, 2))`, `thetas: List(float32)`.

Unlike every existing `sbibm_jax` task, GW has **no simulator**: its data is a
fixed corpus of pre-generated `(theta, x)` rows shipped as files, not generated
on the fly from a prior+simulator. The simulator that produced these rows is not
available to us yet, so the task exposes a **mock** `get_simulator` (and
`get_prior`) that raises `NotImplementedError`. The dataset is published to the
Hub by a **dedicated mirror script** that reads the converted `.npz` shards,
mirroring `make_dataset.py` but bypassing the simulator-driven generation path.

The package and the loader stay **torch-free**: torch is needed only by a
one-time conversion script (the existing `torch` dependency group), which writes
`.npz` files that everything downstream consumes.

## Scope

In scope:
- Task class `GravitationalWaves` with HF hints (`hf_data_kind="timeseries"`,
  `hf_data_shape=(8192, 2)`, `hf_stats_axes`, `hf_external=True`), registry entry.
- Mock `get_prior` / `get_simulator` / `_sample_reference_posterior` raising
  `NotImplementedError`; no reference, no observations.
- Conversion script `scripts/convert_gw_to_npz.py` (torch group): `.pt` → `.npz`,
  self-verifying orientation/shape/dtype.
- File-backed builder `src/sbibm_jax/hf/external.py`
  (`build_gw_dataset` / `upload_gw_dataset`) reusing the existing HF primitives.
- Mirror CLI `scripts/make_gw_dataset.py`, parallel to `make_dataset.py`.
- `make_dataset.py` skipping `hf_external` tasks (so `--all` no longer trips on
  GW's mock simulator).
- Tests for the task, the builder, and the external-skip; docs note.

Out of scope / deferred:
- The actual GW simulator and prior (not yet available).
- Reference posterior samples / true parameters / observations (none exist).
- Any C2ST / metric wiring.

## Background: the original upload (`gw_dataset.py`)

- Raw `.pt` shards in `/lhome/ific/a/aamerio/data/GW/`: `thetas_{0..9}.pt`
  (`(N, 2)`), `xs_{0..9}.pt` (channels-first `(N, 2, 8192)`).
- Pooled shards 0–8 (`torch.cat`), then `np.permute_dims(xs, (0, 2, 1))` →
  `(N, 8192, 2)` (time-first). Split:
  - **train** = pooled rows `[:-512]`,
  - **validation** = pooled rows `[-512:]` (last 512),
  - **test** = shard 9 (entire), also transposed to `(N, 8192, 2)`.
- Pushed under `config_name="gravitational_waves"`, splits `train` /
  `validation` / `test`, features `xs: Array2D((8192, 2))`,
  `thetas: List(float32)`.
- GenSBI's hardcoded normalization stats imply **per-channel** `x` stats
  (`x_mean` shape `(1, 1, 2)`) and **per-feature** `theta` stats
  (`theta_mean` shape `(1, 2)`).

As of writing, the θ shards are complete (`81185` bytes each ≈ 10 000 rows ×
2 float32 → ~100 000 rows total across 10 shards); the `xs` shards are still
downloading, so the conversion script must verify shapes at run time rather than
assume them.

## How this fits the existing architecture

The `task ↔ exporter ↔ loader` contract is driven entirely by `metadata.json`.
`timeseries` is already a first-class data-kind:
- `TimeSeriesExporter` owns `features()` (`Array2D((T, C))` + `List(float32)`)
  and the flat→native reshape.
- The consumer `TaskDataset` + `make_collate(data_kind="timeseries")` already
  serve `(theta, x)` with tokenization (`[..., None]`) and optional
  normalization using metadata stats.

So GW needs **no new data-kind** and **no loader change** — only a task that
declares the timeseries shape, and a mirror script that fills the dataset and
the metadata block from files instead of a simulator. The generic
simulator-driven `build.py` path is left untouched.

## Design

### 0. Conversion script — `scripts/convert_gw_to_npz.py`

One-time, run with the torch group:
`uv run --group torch python scripts/convert_gw_to_npz.py`.

- Args: `--data-dir` (default `/lhome/ific/a/aamerio/data/GW`),
  `--out-dir` (default = `--data-dir`), `--shards` (default 10).
- Per shard `i`:
  - `thetas_i.pt` → `torch.load(..., weights_only=True)` → assert 2-D with last
    dim 2 → `.numpy().astype(float32)` → `np.savez_compressed(thetas_i.npz,
    data=theta)`, shape `(N, 2)`.
  - `xs_i.pt` → assert 3-D with exactly one axis of size `2` (channel) and one
    of size `8192` (time). **Detect orientation** and transpose to
    `(N, 8192, 2)` only if stored channel-first — correct whether torch saved
    `(N, 2, 8192)` (old script) or `(N, 8192, 2)`. Cast float32 →
    `np.savez_compressed(xs_i.npz, data=xs)`.
  - Assert `theta.shape[0] == xs.shape[0]`; print `name → shape, dtype` per shard.
- Key name `"data"` and float32 are thus **guaranteed by our own converter**,
  not assumed.
- Uses `torch.load` + `numpy` only; never imported by the package or tests.

### 1. Task class — `src/sbibm_jax/tasks/gravitational_waves/task.py`

Directory name `gravitational_waves` (must equal the HF config name the loader
loads). `GravitationalWaves(Task)`:

```python
self.dim_theta = 2
self.dim_x = 8192 * 2          # 16384 (flat), native (8192, 2)
self.hf_data_kind = "timeseries"
self.hf_data_shape = (8192, 2)
# per-channel x stats -> (1, 1, 2); per-feature theta -> (1, 2)
self.hf_stats_axes = {"theta": (0,), "x": (0, 1)}
self.hf_external = True         # file-backed; skipped by make_dataset.py
```

Base-class call: `dim_theta=2`, `dim_x=16384`, `name="gravitational_waves"`,
`name_display="Gravitational Waves"`, `num_observations=1` (nominal; no
reference), `num_posterior_samples=10000`, `path=__file__.parent`.

Mock methods (the simulator that produced the corpus is unavailable):
```python
def get_prior(self, key, num_samples=1):
    raise NotImplementedError(
        "The gravitational_waves prior is not available; the dataset is the "
        "only data source (see scripts/make_gw_dataset.py)."
    )

def get_simulator(self, key, max_calls=None):
    raise NotImplementedError(
        "The gravitational_waves simulator has not yet been implemented; "
        "use the published dataset via sbibm_jax.data.TaskDataset."
    )

def _sample_reference_posterior(self, key, num_samples,
                                num_observation=None, observation=None):
    raise NotImplementedError(
        "gravitational_waves has no reference posterior."
    )
```

No `files/` directory is shipped, so the base `get_observation` /
`get_true_parameters` / `get_reference_posterior_samples` raise
`FileNotFoundError`; `load_reference(task, exporter)` catches that and returns
`None`, giving `has_reference=False` with no extra code.

`unflatten_data` overridden to `(-1, 8192, 2)` for consistency (not used by the
mirror path, which stores native rows directly).

Registry: add an `elif task_name == "gravitational_waves"` branch to
`get_task()`; it is auto-discovered by `get_available_tasks()` (real directory).

### 2. File-backed builder — `src/sbibm_jax/hf/external.py`

New module parallel to `build.py` / `upload.py`. Houses the GW-specific reader
so it is importable and unit-testable without network, keeping the
simulator-driven `build.py` untouched.

```python
def build_gw_dataset(data_dir, *, val_size=512, num_shards=10,
                     dtype=config.DEFAULT_DTYPE, task=None) -> dict:
    """Read per-shard npz, build train/val/test Datasets + train stats.

    Returns {"train", "validation", "test", "stats",
             "sizes": {"train": Ntr, "validation": Nval, "test": Nte}}.
    """
```

- Task + exporter: `task = task or get_task("gravitational_waves")`
  (the `task=None` injection point lets tests pass a task with a reduced
  `hf_data_shape`); `exporter = get_exporter(task)` (a `TimeSeriesExporter`) —
  used only for `exporter.features()` and `data_shape`. Split sizes from the
  exporter are ignored; GW's actual sizes come from the files.
- Shape validation: each loaded `x_np`'s native shape `x_np.shape[1:]` is
  asserted equal to `tuple(exporter.data_shape)` (and `theta_np.shape[1:] ==
  (task.dim_theta,)`), so a converter/orientation mistake fails loudly at build
  time rather than producing a malformed dataset.
- Reader (`_load_shard(i)` → `(theta_np (N,2), x_np (N,8192,2))` via
  `np.load(path)["data"]`), one shard at a time to bound RAM (~one shard).
- Split policy (mirrors the original, streaming-friendly — the pool is shards
  0..num_shards-2 in order, so its last `val_size` rows are the tail of the
  last pool shard, asserting that shard has `≥ val_size` rows):
  - **train** = shards `0 .. num_shards-2`, dropping the last `val_size` rows of
    the final pool shard.
  - **validation** = last `val_size` rows of the final pool shard.
  - **test** = shard `num_shards-1` (entire).
- Each split is a `Dataset.from_generator(row_gen, features=exporter.features())`
  yielding `{"xs": x[i], "thetas": theta[i]}` per row; the generator loads each
  shard lazily so no split is ever fully resident.
- Stats: `StatsAccumulator(*resolve_stats_axes(task))` over the **train** split
  (iterating the built train Dataset, cache-independent — same approach as
  `build._compute_train_stats`), yielding `theta_mean/std` `(1,2)` and
  `x_mean/std` `(1,1,2)`.

```python
def upload_gw_dataset(repo, data_dir, **build_opts) -> dict:
    bundle = build_gw_dataset(data_dir, **build_opts)
    for split in ("train", "validation", "test"):
        bundle[split].push_to_hub(repo, config_name="gravitational_waves",
                                   split=split, private=False)
    return {"stats": bundle["stats"], "sizes": bundle["sizes"]}
```

No `_posterior` config (no reference). `push_to_hub` / `upload_file` /
`hf_hub_download` remain the only network surface (monkeypatchable in tests).

### 3. Mirror CLI — `scripts/make_gw_dataset.py`

Thin driver, structurally parallel to `make_dataset.py`:

- Args: `--data-dir` (default `/lhome/ific/a/aamerio/data/GW`), `--prod`,
  `--dry-run`, `--val-size` (default 512), `--num-shards` (default 10),
  `--metadata-path` (default `metadata.json`), `--verbose`.
- `repo = config.DEFAULT_REPO if --prod else config.TEST_REPO`; print the same
  `Target repo: … (TEST|PRODUCTION)` banner.
- Real run:
  1. `result = upload_gw_dataset(repo, data_dir, val_size=…, num_shards=…)`.
  2. `local_meta = make_metadata(["gravitational_waves"],
     train_size=Ntr, val_size=Nval, test_size=Nte,
     stats_by_task={"gravitational_waves": result["stats"]})` — passing the
     **actual** row counts so the recorded `splits` match the uploaded data.
  3. `remote = fetch_remote_metadata(repo)`;
     `merged = merge_metadata(remote, local_meta)`; write; `upload_metadata`;
     delete local file (clean state) — identical round-trip to `make_dataset.py`.
- `--dry-run`: write `make_metadata(["gravitational_waves"])` only (no data read,
  no stats → `stats: null`), matching `make_dataset.py` dry-run semantics. (Dry
  run uses the exporter's nominal split sizes since the data isn't read.)

### 4. `make_dataset.py` — skip external tasks

After resolving `task_names`, filter out any task with `hf_external=True`
(via `getattr(get_task(name), "hf_external", False)`) from **both** generation
and metadata, logging:
`Skipping gravitational_waves (external/file-backed; use
scripts/make_gw_dataset.py)`. GW's metadata block is owned solely by
`make_gw_dataset.py`; the non-destructive `merge_metadata` preserves it across
other tasks' runs. This keeps `--all` working without touching the mock
simulator.

### 5. Resulting metadata block

```json
"gravitational_waves": {
  "dim_theta": 2,
  "dim_x": 16384,
  "data_kind": "timeseries",
  "data_shape": [8192, 2],
  "splits": {"train": <Ntr>, "validation": 512, "test": <Nte>},
  "has_reference": false,
  "num_observations": 1,
  "stats": {
    "theta_mean": [[...]], "theta_std": [[...]],   // (1, 2)
    "x_mean": [[[...]]],   "x_std": [[[...]]],      // (1, 1, 2)
    "theta_axes": [0], "x_axes": [0, 1]
  }
}
```

This is exactly what `TaskDataset("gravitational_waves")` consumes: conditional
loader yields `(theta (B,2,1), x (B,8192,2,1))`; `normalize=True` broadcasts the
per-channel/per-feature stats. `joint` kind is rejected by `make_collate`
(timeseries is conditional-only) — matching GenSBI's `assert kind ==
"conditional"`.

## Testing (TDD)

New file `tests/tasks/test_gravitational_waves.py`:
1. **Metadata:** `dim_theta == 2`, `dim_x == 16384`, `name ==
   "gravitational_waves"`, `hf_data_kind == "timeseries"`,
   `hf_data_shape == (8192, 2)`, `hf_stats_axes == {"theta": (0,), "x": (0, 1)}`,
   `hf_external is True`.
2. **Mocks raise:** `get_prior`, `get_simulator`,
   `_sample_reference_posterior` each raise `NotImplementedError`.
3. **Registry:** `get_task("gravitational_waves")` returns a
   `GravitationalWaves`; `get_available_tasks()` includes it.

New file `tests/hf/test_external.py` (under the `datasets`-gated `tests/hf`):
4. **Build / split policy:** write tiny fake npz shards (small `T` to keep it
   fast, e.g. 3 shards of `(rows, 8, 2)` with a 4-row tail) to a tmp dir; run
   `build_gw_dataset(data_dir, val_size=4, num_shards=3, task=<task with
   hf_data_shape=(8, 2)>)` (the injected task makes the exporter's `Array2D`
   feature match the reduced `T`); assert split sizes follow the mirror policy
   (test = last shard, val = last 4 of pool, train = rest) and that
   `len(train)+len(val)+len(test)` equals total rows.
5. **Features / shape:** built rows have `xs` shape `(T, 2)` and `thetas` length
   2; stats shapes are `(1, 2)` (theta) and `(1, 1, 2)` (x).
6. **Upload / metadata:** monkeypatch `push_to_hub` (capture
   `config_name`/`split`), `fetch_remote_metadata` (return a sibling task block),
   `upload_file`; run the `make_gw_dataset` main with `--data-dir tmp`; assert
   three pushes under `config_name="gravitational_waves"`, the merged metadata
   keeps the sibling block and records the actual GW splits + stats.

New test in `tests/hf/test_driver.py` (or `test_metadata.py`):
7. **External skip:** `make_dataset.py` with `--all` (or an explicit list
   including `gravitational_waves`) does not call `upload_dataset` for GW and
   omits it from the built metadata; a non-external sibling is still processed.

The test arrays use a reduced `T` (the builder takes `data_shape` from the task,
so tests either patch `hf_data_shape` or assert against the row's actual shape)
to keep the suite fast and CPU-only (per `JAX_PLATFORMS=cpu`).

## Files

- **New:**
  - `scripts/convert_gw_to_npz.py`
  - `scripts/make_gw_dataset.py`
  - `src/sbibm_jax/hf/external.py`
  - `src/sbibm_jax/tasks/gravitational_waves/__init__.py` (empty),
    `src/sbibm_jax/tasks/gravitational_waves/task.py`
  - `tests/tasks/test_gravitational_waves.py`
  - `tests/hf/test_external.py`
- **Modify:**
  - `src/sbibm_jax/tasks/__init__.py` (add `gravitational_waves` branch)
  - `src/sbibm_jax/hf/__init__.py` (export `build_gw_dataset` / `upload_gw_dataset`)
  - `scripts/make_dataset.py` (skip `hf_external` tasks)
  - `CLAUDE.md` (note the external/file-backed path + GW task)
  - possibly `tests/hf/test_driver.py` (external-skip test)

## Open items (non-blocking)
- If/when the real GW simulator + prior become available, `get_prior` /
  `get_simulator` can be implemented and GW could move onto the generic
  generation path (drop `hf_external`); the dataset schema would be unchanged.
- `hf/external.py` is GW-specific for now but structured so a second file-backed
  task (e.g. a future GenSBI-style `lensing`) could reuse the split/stream/stats
  scaffold.
