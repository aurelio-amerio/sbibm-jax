# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`sbibm-jax` is a JAX/NumPyro rewrite of the Simulation-Based Inference Benchmark
(sbibm). It provides a set of benchmark *tasks* — each defining a prior, a
simulator, reference observations, and reference posterior samples — for
evaluating SBI methods. The original sbibm was PyTorch/Pyro based; this port
replaces those with JAX, `numpyro.distributions`, and `diffrax` (for ODE tasks).
Any task can also be streamed out as a HuggingFace dataset via the optional
`sbibm_jax.hf` subpackage.

## Commands

This project uses `uv`. Python is pinned to 3.12.

```bash
uv sync --all-groups            # install all dependency groups (dev work)
uv run pytest                   # run the test suite (CPU-forced, 2 workers via xdist)
uv run pytest tests/tasks/test_analytical.py            # run one file
uv run pytest tests/tasks/test_ode.py::TestLotkaVolterra # run one class
uv run pytest -k two_moons                              # run by keyword
uv run pytest -m "not slow"     # skip slow-marked tests
uv run flake8 src tests         # lint
```

Test configuration lives in `pyproject.toml` (`[tool.pytest.ini_options]`):
`JAX_PLATFORMS=cpu` is injected via `pytest-env` so tests never touch the GPU,
and `-n 2` (pytest-xdist) runs them in parallel. Markers: `slow`, `experimental`.

The default JAX install is the CUDA 12 build (`jax[cuda12]`). PyTorch is only
needed for the one-time data-conversion script and is pinned to the CPU index;
pull it in with the `torch` group: `uv run --group torch python scripts/convert_torch_to_npz.py`.

The `beer_molbiosystems` PEtab task needs the optional `pypesto` extra
(`pypesto`, `petab`, `amici`, `benchmark-models-petab`, `joblib`, `scipy`):
`uv sync --extra pypesto`. Installing it triggers a one-time AMICI compile of
the Beer model (needs a C/C++ compiler, SWIG, and BLAS). The task constructs
without the extra (for registry discovery) but raises an informative error when
the prior/simulator/reference-posterior methods are called without it. Its
helper code is a verbatim port of `diffusion-experiments/case_study2`.

The `sbibm_jax.hf` HuggingFace export pipeline needs the optional `[hf]` extra
(`datasets`, `huggingface_hub`): `uv sync --extra hf` (it is also in the `hf`
dependency group, so `uv sync --all-groups` pulls it in). Build/upload datasets
through the thin driver `scripts/make_dataset.py`:

```bash
# Write metadata.json only, no HF push (custom split sizes):
uv run python scripts/make_dataset.py --tasks two_moons --train-size 1000 --dry-run
uv run python scripts/make_dataset.py --all            # every task -> TEST repo
uv run python scripts/make_dataset.py --all --prod     # every task -> PRODUCTION repo
```

Uploads target the **test** repo (`config.TEST_REPO`) by default; pass `--prod`
to target production (`config.DEFAULT_REPO`). Each run prints a `Target repo: …
(TEST|PRODUCTION)` banner. Subset uploads are non-destructive: the remote
`metadata.json` is fetched and merged so untouched tasks are preserved, and the
local `metadata.json` is deleted after a successful real upload.

Pass `--chunk-size N` to shrink the per-chunk generation batch if a GPU OOMs
on large image tasks (e.g. `gaussian_random_field_256`).

## Architecture

**Task abstraction.** `src/sbibm_jax/tasks/task.py` defines the abstract `Task`
base class. Every benchmark is a subclass living in
`src/sbibm_jax/tasks/<name>/task.py`. A task carries dimensionality/budget
metadata and implements three abstract methods:
- `get_prior(key, num_samples)` → parameter samples `(num_samples, dim_parameters)`
- `get_simulator(key, max_calls)` → a `Simulator` instance
- `_sample_reference_posterior(...)` → reference posterior (closed-form where possible)

Priors are `numpyro.distributions` objects stored on `self.prior_dist`. The base
class also provides CSV loaders for observations, true parameters, and reference
posterior samples.

**Simulator wrapper.** `src/sbibm_jax/tasks/simulator.py` wraps each task's raw
simulator function `(key, parameters) -> data`. It enforces a simulation budget
(`max_calls`, raising `SimulationBudgetExceeded`), counts calls, normalizes input
shapes, and flattens output via the task's `flatten_data`. Tasks define the
simulator as a closure inside `get_simulator` and return
`Simulator(task=self, simulator=fn, max_calls=...)`.

**Registry.** `src/sbibm_jax/tasks/__init__.py` maps task-name strings to classes
in `get_task()`, with lazy per-branch imports. Some names are aliases/variants of
the same class passing different kwargs (e.g. `slcp_distractors` → `SLCP(distractors=True)`,
`bernoulli_glm_raw` → `BernoulliGLM(summary="raw")`, `gaussian_nonlinear` → `SLCP`,
`gaussian_random_field_256` → `GaussianRandomField(field_size=256)`).
`get_available_tasks()` discovers task directories on disk and appends these
extra variant names. The top-level `sbibm_jax` package re-exports `get_task` and
`get_available_tasks`.

**Task data files.** Each task directory has a `files/` subtree:
`files/num_observation_<N>/{observation.csv, true_parameters.csv,
reference_posterior_samples.csv.bz2}`. These are read by the base-class loaders
via `sbibm_jax/utils/io.py` (pandas → numpy → JAX, default dtype `float32`,
`atleast_2d`). Some tasks also have task-specific data (design matrices, GMM
params) stored as `.npz`, converted from the original PyTorch `.pt`/`.torch`
files by `scripts/convert_torch_to_npz.py`.

**ODE tasks** (Lotka-Volterra, SIR) use `diffrax` for integration — vector field
function + `diffeqsolve` with `Tsit5`/`PIDController`, `jax.vmap`ed over the
parameter batch. They may produce NaNs for divergent parameters; simulators
propagate NaN rows rather than failing.

**HuggingFace export (`src/sbibm_jax/hf/`).** Optional subpackage, gated by the
`[hf]` extra with an import-guard that mirrors the `pypesto` pattern (informative
ImportError pointing at `pip install sbibm-jax[hf]`). The key architectural fact
is the **task ↔ exporter contract**: a task drives the export purely through a
few optional `hf_*` attributes read via `getattr`-with-defaults in
`registry.get_exporter`, so *any* task exports as a flat parameter-vector dataset
with zero task-side changes; declaring `hf_x_kind` switches it to an image or
time-series storage shape. One full build is:
`build_dataset(task_name)` → `get_exporter` (dispatches `hf_x_kind` →
`VectorExporter` / `ImageExporter` / `TimeSeriesExporter`, which own the HF
`Features` schema and the flat-to-native reshape) → `derive_task_keys` (stable
per-task PRNG keys via a `zlib.crc32` fold-in, *not* Python's salted `hash()`) →
chunked streaming generation fed into `Dataset.from_generator` for the
train/validation/test splits, plus an optional reference block built from the
task's reference-posterior CSVs (skipped, returning `None`, when those files are
absent). The non-finite-row behavior of the ODE/PEtab simulators above drives the
validity policy: the default raises loudly on any NaN/Inf, and tasks whose
simulators legitimately diverge set `hf_resample_invalid=True` to switch to
rejection sampling (drop bad rows, redraw to exactly `n`, capped at
`max_factor * n`). Defaults (split sizes, chunk size, target repo, master seed)
live in `hf/config.py`; `scripts/make_dataset.py` is the only CLI entry point.
Normalization stats (mean/std of `theta` and `x`) are accumulated over the train
split during generation (float64, streamed) and written into each task's
`metadata.json` block; the per-task reduction axes default to per-feature and are
overridden via the task's `hf_stats_axes` (image tasks use a global scalar).
Stats are absent (`null`) under `--dry-run`.

**Consumer loader (`src/sbibm_jax/data/`).** Optional subpackage gated by the
`[loader]` extra (`grain`, `datasets`, `huggingface_hub`), with the same
import-guard pattern as `hf` (informative ImportError → `pip install
sbibm-jax[loader]`). `from sbibm_jax.data import TaskDataset` loads an
SBI-benchmarks task straight from the Hub. It is driven *entirely* by the
published `metadata.json` (x_kind/x_shape, theta_kind/theta_shape, splits, stats) — no
per-task code. The default repo is the **TEST** repo (`config.TEST_REPO`); pass
`repo=config.DEFAULT_REPO` for production. `kind="conditional"` serves
`(theta, x)`; `kind="joint"` concatenates them along the feature axis
(vector-only). Both reproduce GenSBI's tokenization (each scalar feature → a
length-1 token via a trailing `[..., None]`); `normalize=True` applies the
gen-time stats from `metadata.json`. `get_train_loader` / `get_val_loader` /
`get_test_loader` return `grain` pipelines (shuffle→repeat→batch→tokenizing
collate, optional multiprocess `mp_prefetch`); `max_workers` is clamped to ≤8
(shared-node rule). `get_train_loader(num_samples=N)` subsamples a prefix.
`normalize_theta`/`normalize_x` (+ `unnormalize_*`) expose the stats directly;
`get_reference`/`get_true_parameters` read the separate `{task}_posterior`
config (raising when the task ships no reference). Graph/causal masks are
**opt-in** via `sbibm_jax.data.masks` (`get_base_mask_fn`, `get_edge_mask_fn`,
`get_condition_mask_fn`) — the core loader never imports it, and base/edge masks
cover only the 5 analytical base tasks (`two_moons`, `gaussian_linear`,
`gaussian_linear_uniform`, `gaussian_mixture`, `slcp`).

### Conventions

- All array ops use `jax.numpy`; randomness is explicit PRNG keys split with
  `jax.random.split`. Functions threading keys take `key` as the first argument.
- Task names equal their directory name (set via `Path(__file__).parent.name`),
  and `name_display` carries the human-readable label.
- Tasks are grouped into "phases" (analytical, ODE, …) reflected in test files,
  not in the package layout.
- HuggingFace export is opt-in-by-attribute: a task only sets `hf_*` attributes
  (`hf_x_kind`, `hf_x_shape`, `hf_resample_invalid`, `hf_split_sizes`) when
  it needs to deviate from the flat-vector default. The image tasks
  `gaussian_random_field` and `toy_lensing` declare `hf_x_kind="image"` plus a
  2-D `hf_x_shape`; ODE/PEtab tasks set `hf_resample_invalid=True`. The
  expensive tasks `toy_lensing`, `gaussian_random_field`, and `beer_molbiosystems`
  also set `hf_split_sizes` to cap `train` at `100_000` (vs. the global
  `1_000_000` default); validation/test stay at `10_000`. The
  `gaussian_random_field_256` alias is a 256×256 high-resolution variant of
  `gaussian_random_field` (same `hf_split_sizes` cap of `100_000` train); each
  256×256 float32 row is ~256 KiB, so its train split is ~25 GiB on disk —
  generated incrementally via the chunked `Dataset.from_generator` streaming,
  never held whole in RAM. There is no per-task
  budget ladder — each dataset is generated once at its largest useful size and
  consumers subsample smaller budgets by indexing the dataset prefix (valid
  because every `(θ, x)` row is an independent draw).

The `gravitational_waves` task is **file-backed**: it has no simulator yet
(`get_prior`/`get_simulator` raise `NotImplementedError`) and sets
`hf_external=True`, so `make_dataset.py` skips it. Its dataset is a fixed corpus
of pre-generated `(theta, x)` rows (`x` is a `(8192, 2)` two-channel time
series, `hf_x_kind="timeseries"`). A one-off `scripts/convert_gw_to_npz.py`
(torch group) converts the raw `.pt` shards to `.npz`; the torch-free
`scripts/make_gw_dataset.py` then reads those shards, builds the
train/validation/test splits (mirroring the original `gw_dataset.py`: shard 9 =
test, last 512 pooled rows = validation, the rest = train), accumulates stats,
pushes under `config_name="gravitational_waves"`, and merges its block into
`metadata.json`. There is no reference posterior. The simulator will be added in
a future rework, at which point GW can move onto the generic generation path.

## Note on `diffusion-experiments/`

The `diffusion-experiments/` directory is untracked reference/research code
(case studies, separate from the `sbibm_jax` package). It is not part of the
installable package or the test suite.
