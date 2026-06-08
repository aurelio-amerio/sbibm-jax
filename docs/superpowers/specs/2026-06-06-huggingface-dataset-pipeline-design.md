# HuggingFace dataset pipeline (`sbibm_jax.hf`) — design

**Date:** 2026-06-06
**Status:** Approved scope — general generate→build→upload pipeline driven by the
JAX `Task` API, with data-kind exporter subclasses.

## What this is

A port of the `SBI-benchmarks-data` dataset-creation pipeline into `sbibm-jax`.
The original repo (`/lhome/ific/a/aamerio/data/github/SBI-benchmarks-data`) used
the **PyTorch** `sbibm` package + `torch` to draw `(θ, x)` and per-observation
reference data, build HuggingFace `datasets`, and `push_to_hub`. This rewrite
reproduces the same published dataset but generates everything from the native
JAX `Task` API in this repo — **no torch, no original `sbibm`**.

The result is a new importable subpackage `sbibm_jax.hf` plus a thin driver
script. It is a **general, abstract pipeline** that works for every current and
future task, with subclassing to handle different data structures (flat vectors
vs. images / time-series, the latter stored via HuggingFace `Array2D`).

## Background: what the original does

Source: `SBI-benchmarks-data/sbi_benchmarks/{sbi_tasks.py, hf_hub.py}`.

- `get_task_data(name, n)`: `task = sbibm.get_task(name)`; draws
  `θ = prior(n)`, `x = simulator(θ)` (→ `.numpy()`); for observations `1..10`
  reads `get_observation`, `get_reference_posterior_samples`,
  `get_true_parameters`. `lensing` is special-cased to its own **JAX**
  `LensingSimulator` (no reference posteriors).
- `make_dataset(name)`: splits into **train = 1,000,000 / val = 10,000 /
  test = 10,000** (one stream, sliced), casts to `float32`, builds three HF
  `Dataset`s via `Dataset.from_generator` plus a reference-posterior `Dataset`
  from a dict (`reference_samples`, `observations`, `true_parameters`).
- `upload_dataset(repo, name)`: pushes train/val/test under
  `config_name=name`, and the reference posterior under
  `config_name=f"{name}_posterior"`, to repo `aurelio-amerio/SBI-benchmarks`.
- `lensing`'s `xs` used an explicit `Features({"xs": Array2D((32,32)), ...})`;
  all flat tasks used `features=None` (auto-inferred 1-D lists). Data-kind
  selection was an ad-hoc `if task_name == "lensing"` branch.

### How NaN / failed sims are handled today (investigated)

- **Task/simulator layer — NaN is propagated, never dropped.** The Beer task
  emits a full NaN row on AMICI failure
  (`tasks/beer_molbiosystems/task.py:155`); ODE tasks do the same for divergent
  parameters (documented convention).
- **Reference pipeline (`diffusion-experiments/case_study2`) — NaN rows are
  kept in the saved data and filtered only at evaluation time** (e.g.
  `helper_pypesto.py:453, 468–469` count `count_nan_data` and drop NaN rows
  before computing metrics). There is **no** drop-at-generation or
  resample-to-N logic anywhere, and **nothing imputes/extrapolates** values.

The published dataset must contain **no NaN rows** and do **no imputation**, so
this pipeline *adds* a generation-time validity step the reference code lacks
(see §5).

## Target task API (what we build on)

`Task` (`src/sbibm_jax/tasks/task.py`) provides, all JAX/numpyro:

- `get_prior(key, num_samples) -> (n, dim_parameters)`
- `get_simulator(key, max_calls=None) -> Simulator`; `sim(key, θ) -> (n, dim_data)`
  (flattened via `flatten_data`); `max_calls=None` disables budget enforcement.
- `get_observation(i)`, `get_reference_posterior_samples(i)`,
  `get_true_parameters(i)` — CSV loaders (1-indexed). 8 of 10 tasks ship these
  files; `gaussian_random_field` and `beer_molbiosystems` do **not**.
- Metadata: `dim_parameters`, `dim_data`, `num_observations`, `name`.

## Design decisions (resolved)

1. **Scope:** general pipeline over all registry tasks (current + future),
   abstract base + data-kind exporter subclasses.
2. **Specialization:** by **data kind** — `VectorExporter` (default),
   `ImageExporter` (`Array2D`), `TimeSeriesExporter`.
3. **Shape source (hybrid):** exporter defaults to `VectorExporter` from
   `task.dim_data`; only structured tasks declare a small export hint **on the
   Task**. Flat analytical tasks are untouched.
4. **Reference block is optional per task:** read existing CSVs; skip when
   absent.
5. **Packaging:** new subpackage `sbibm_jax.hf` + optional extra `[hf]`; torch
   dropped; thin `scripts/make_dataset.py` driver.
6. **Split sizes:** global default `1e6 / 1e4 / 1e4` + per-task override.
7. **Seeding:** master seed → stable per-task `fold_in` → independent per-split
   keys.
8. **`metadata.json`:** auto-generated from task attributes.
9. **HF layout:** same repo `aurelio-amerio/SBI-benchmarks`; configs `<task>`
   (train/validation/test) + `<task>_posterior`; dtype `float32`.
10. **Validity policy (per task):** default takes N raw draws with a finite
    assert; ODE/PEtab tasks opt into drop + rejection-resample to exact N.

## Module layout

```
src/sbibm_jax/hf/
  __init__.py     # public API: build_dataset, upload_dataset, make_metadata, get_exporter
  exporter.py     # DatasetExporter base + VectorExporter, ImageExporter, TimeSeriesExporter
  registry.py     # data_kind -> ExporterClass ; get_exporter(task, **overrides)
  generate.py     # chunked JAX generation of (theta, x): seeding + validity policy
  reference.py    # load optional reference block from CSV via Task loaders
  metadata.py     # auto-generate metadata.json from task attributes
  upload.py       # push_to_hub / upload_file helpers (mockable)
  config.py       # defaults: split sizes, repo name, dtype, chunk size
scripts/make_dataset.py   # thin driver (replaces the old make_dataset.py)
```

## Components

### 1. The exporter (core abstraction)

`DatasetExporter(task, *, train_size, val_size, test_size, dtype=np.float32,
chunk_size, max_factor)`. Responsibilities: build the `<task>` dataset
(train/val/test) and the optional `<task>_posterior` dataset; own the HF
`Features` schema and flat-`x` reshaping.

Override points (data-kind subclasses change only these):

- `data_kind: str` (class attribute)
- `x_feature() -> datasets.Feature` — the HF feature for a single `x` row.
- `shape_x(x_flat) -> array` — reshape a flat `(batch, dim_data)` block to native
  storage shape.

Concrete subclasses:

| Class | `data_kind` | `x_feature()` | `shape_x` |
|-------|-------------|---------------|-----------|
| `VectorExporter` (default) | `"vector"` | `List(Value("float32"))` | identity (flat) |
| `ImageExporter` | `"image"` | `Array2D(data_shape, "float32")` | `(batch, H, W)` |
| `TimeSeriesExporter` | `"timeseries"` | `Array2D((T, C), "float32")` | `(batch, T, C)` |

`thetas` is always `List(Value("float32"))` (parameter vectors).

Build flow (per split): call `generate_samples` (§3) → reshape `x` via
`shape_x` → stream rows into `Dataset.from_generator` with `features()`.
`ImageExporter` is the direct target for the GRF (`32×32`) and in-flight
`toy_lensing` (`32×32`) tasks.

### 2. Binding tasks → exporters (hybrid)

`registry.py` maps `data_kind -> ExporterClass`. `get_exporter(task,
**overrides)` reads optional class attributes via `getattr` with safe defaults:

- `hf_data_kind` → `"vector"`
- `hf_data_shape` → `(task.dim_data,)`
- `hf_resample_invalid` → `False`
- `hf_split_sizes` → global default (optional override)

Per-task changes are minimal and isolated:

- Flat analytical tasks: **no change**.
- `GaussianRandomField`: set `hf_data_kind = "image"`,
  `hf_data_shape = (field_size, field_size)` (in `__init__`, since `field_size`
  is instance state).
- ODE (`lotka_volterra`, `sir`) and `beer_molbiosystems`:
  `hf_resample_invalid = True`.

### 3. Generation + seeding + validity

`generate_samples(task, key, n, *, resample_invalid, chunk_size, dtype,
max_factor) -> (thetas, xs_flat, stats)`:

- **Seeding:** a single master seed (a pipeline parameter, default constant in
  `config.py`, overridable per run) → per-task key via stable
  `jax.random.fold_in(master, zlib.crc32(name.encode()))` (stable across runs,
  unlike salted `hash()`) → `jax.random.split` into independent train/val/test
  keys. Independent per-split keys make the splits effectively disjoint (for
  continuous priors, collisions have negligible probability).
- **Chunked:** each chunk derives `(theta_key, sim_key)`; draws
  `task.get_prior(theta_key, chunk)` and runs the simulator with
  `max_calls=None`; chunks are streamed to `Dataset.from_generator` to bound
  memory (critical for GRF `1024`-dim @ `1e6` ≈ 4 GB if materialised).
- **Default policy (`resample_invalid=False`):** assert all-finite; if a
  non-resample task ever emits NaN, raise with task name + offending count
  (loud failure, never a silent NaN ship). Take exactly N.
- **Resample policy (`resample_invalid=True`):** drop non-finite `(θ, x)` rows,
  keep drawing chunks until N valid rows are produced; cap total draws at
  `max_factor × N` and raise if exceeded; record the rejection rate in `stats`
  and `log` it. This is pure **rejection sampling** — no imputation — so kept
  rows are genuine i.i.d. prior draws; the effective prior becomes
  "prior ∩ {sim succeeds}".

### 4. Reference block (optional, per task)

`load_reference(task, exporter) -> Optional[datasets.Dataset]`: for `i` in
`1..num_observations` read `get_observation` / `get_reference_posterior_samples`
/ `get_true_parameters`. If the files are absent (GRF, Beer) return `None` →
the `<task>_posterior` config is **skipped** (not an error). Observations are
reshaped via `exporter.shape_x`; `reference_samples` and `true_parameters` stay
flat parameter vectors. Schema matches the original: `reference_samples`,
`observations`, `true_parameters`.

### 5. Metadata + upload

- `make_metadata(tasks) -> dict` and writes `metadata.json`, **auto-generated**
  per task: `dim_parameters`, `dim_data`, `data_kind`, `data_shape`, split
  sizes, `has_reference`, `num_observations`. Replaces the hand-maintained file.
- `upload.py`: `upload_metadata(path, repo)` (`upload_file`) and
  `upload_dataset(repo, task_name, ...)` which pushes each split with the right
  `config_name`/`split` and the reference config when present. The HF calls are
  isolated here so tests can monkeypatch them (no network).

### 6. Public API + driver

`sbibm_jax.hf.__init__` re-exports `build_dataset(task_name, **opts)`,
`upload_dataset(repo, task_name, **opts)`, `make_metadata(...)`,
`get_exporter(task, **overrides)`. `scripts/make_dataset.py` is a thin driver
(loop over `get_available_tasks()` or an explicit list; build metadata; upload),
replacing the original `make_dataset.py`.

## Dependencies / torch removal

Add an optional extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
hf = ["datasets", "huggingface_hub"]
```

No torch anywhere in the pipeline. Importing `sbibm_jax.hf` without the extra
raises an informative `ImportError` pointing at `pip install sbibm-jax[hf]`
(mirrors the existing `pypesto` extra pattern). The legacy `torch` dependency
group stays solely for the one-off `scripts/convert_torch_to_npz.py`.

## Error handling

- Missing `datasets`/`huggingface_hub` → informative `ImportError` naming the
  `[hf]` extra.
- Unexpected NaN in a non-resample task → raise with task name + NaN count.
- Resample cap (`max_factor × N`) exceeded → raise with achieved/required counts
  and the rejection rate.
- Missing reference files → skip the `_posterior` config (not an error).
- Unknown `data_kind` in the registry → raise listing the known kinds.

## Testing

CPU-only, tiny sizes, no network (consistent with the existing suite). The
`[hf]` extra (`datasets`) must be available in the test environment; tests
skip with a clear reason if it is not importable.

- **Build correctness** on cheap real tasks (`gaussian_linear`, `two_moons`) at
  tiny sizes (e.g. 8/4/2): split row counts, `float32` dtype, **no NaN**,
  correct `Features`, and reproducibility (same seed → identical data).
- **Exporter selection:** `VectorExporter` for flat tasks; `ImageExporter`
  (`Array2D` feature, correct `(H, W)`) for `gaussian_random_field`.
- **Reference block:** present (10 rows) for `two_moons`; `None` for
  `gaussian_random_field`.
- **Validity policy:** a fake NaN-emitting task → resample yields exactly N
  finite rows and a sane rejection rate; default policy raises.
- **Metadata:** `make_metadata` emits correct dims / kinds / shapes /
  `has_reference`.
- **Upload:** monkeypatch `push_to_hub` / `upload_file`; assert the
  `config_name` / `split` arguments per call. No real HF calls; mark
  network/slow as appropriate.

## Deferred / out of scope

- Actually publishing/regenerating the live HF dataset (this spec covers the
  code + tests, not a production upload run).
- Sharded/resumable generation beyond in-process chunked streaming.
- Reference posteriors for tasks that currently lack them (GRF, Beer,
  toy_lensing) — their `_posterior` configs simply stay absent until those files
  exist.
- A full CLI beyond the thin `scripts/make_dataset.py` driver.
