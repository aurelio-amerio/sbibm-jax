# HuggingFace dataset loading (`sbibm_jax.data`) — design

**Date:** 2026-06-08
**Status:** Approved scope — move the consumer-side HF dataset loading from
`GenSBI-examples` into `sbibm-jax`, making `sbibm-jax` the single source of truth
for the SBI-benchmarks datasets (build + serve + metadata).

## What this is

`sbibm-jax` already **produces** the SBI-benchmarks HuggingFace datasets (the
`sbibm_jax.hf` generate→build→upload pipeline). The consuming half lives in a
separate project, `GenSBI-examples` (`src/gensbi_examples/tasks.py`): a `Task`
class that loads those datasets from the Hub, wraps the splits in
[`grain`](https://github.com/google/grain) dataloaders, exposes task
dimensionality, ships precomputed normalization statistics, and provides
graph/causal **masks** for the analytical base tasks.

This design **moves that consumer half into `sbibm-jax`** as a new optional
subpackage `sbibm_jax.data`, so a single library both builds and serves the
datasets. As part of the move we also fix a provenance weakness: normalization
statistics — currently hand-computed and shipped as `.npz` files in
`GenSBI-examples` — are instead **computed at generation time** by the producer
pipeline and published in `metadata.json` alongside the data, so they can never
drift from the dataset they describe.

`GenSBI-examples` is **left untouched** by this effort. A later, separate
migration will delete its `tasks.py` / `stats/` / `mask.py` / `graph.py` and
re-point it at `sbibm_jax.data` (with clean-name call-site updates). That
migration is out of scope here; see §8.

## Background: what `GenSBI-examples` does today

Source: `GenSBI-examples/src/gensbi_examples/{tasks.py, graph.py, mask.py}` and
the shipped `stats/stats_<task>.npz` files. Key facts established by inspection:

- **Repo:** loads from `aurelio-amerio/SBI-benchmarks` (production), via
  `hf_hub_download(..., "metadata.json")` for dims, then
  `load_dataset(repo, task_name).with_format("numpy")` for `train`/`validation`/
  `test`, and `load_dataset(repo, f"{task_name}_posterior")` for reference
  posteriors.
- **Framing (generative-model naming):** `obs` = parameters θ (stored column
  `thetas`), `cond` = data x (stored column `xs`). Exposes `dim_obs`/`dim_cond`/
  `dim_joint`, `obs_mean`/`obs_std`/`cond_mean`/`cond_std`,
  `normalize_obs`/`normalize_cond`.
- **Processing baked into the dataloader** via `process_fn`:
  - `process_joint`: `concat(thetas[...,None], xs[...,None], axis=1)` →
    `(batch, dim_θ+dim_x, 1)` — every scalar feature becomes a length-1 **token**
    so a graph transformer can index each θ_i / x_j as a node.
  - `process_conditional`: returns `(thetas[...,None], xs[...,None])`.
  - Normalized variants subtract/divide by the per-token `(1, dim, 1)` stats.
- **`grain` loaders:** `get_train_dataset(batch_size, nsamples)`,
  `get_val_dataset`, `get_test_dataset` build
  `grain.MapDataset.source(split).shuffle(seed).repeat().to_iter_dataset()
  .batch(batch_size).map(process_fn)` with optional `mp_prefetch`.
- **Reference:** `get_reference(num_observation)` → `(observation, samples)`;
  `get_true_parameters(num_observation)`.
- **Stats are per-task in shape**, confirming reduction is task-specific:
  - analytical tasks: per-feature `(1, dim, 1)` (reduce over batch only),
    shipped as `.npz`.
  - `GravitationalWaves`: x stats per-channel `(1, 1, 2)`, θ per-feature
    `(1, 2)`, hardcoded in **bfloat16**, applied in a bespoke `split_data`.
  - `GravitationalLensing`: x stats global-scalar `(1, 1, 1)`, θ per-feature
    `(1, 2)`, hardcoded in bfloat16, bespoke `split_data`.
- **Masks** (graph/causal): `get_base_mask_fn()` returns a boolean adjacency
  matrix of shape `(dim_θ+dim_x, dim_θ+dim_x)` plus a `base_mask_fn(node_ids,
  node_meta_data)` closure that sub-indexes it. `get_edge_mask_fn(name)` derives
  `faithfull` / `min_faithfull` / `undirected` (moralized) / `directed` / `none`
  variants via `graph.py` (`faithfull_mask`, `min_faithfull_mask`, `moralize`).
  Implemented for `two_moons`, `gaussian_linear`, `gaussian_linear_uniform`,
  `gaussian_mixture`, `slcp`; `bernoulli_glm` raises `NotImplementedError`;
  GW/lensing have none (conditional-only).

## What the producer already provides (we build on this)

- `metadata.json` per task: `dim_parameters`, `dim_data`, `data_kind`
  (`vector`/`image`/`timeseries`), `data_shape`, `splits`, `has_reference`,
  `num_observations` (`sbibm_jax/hf/metadata.py`).
- Upload layout (`sbibm_jax/hf/upload.py`, **confirmed**): main config
  `task_name` with splits `train`/`validation`/`test`; reference, when present,
  pushed under config `f"{task_name}_posterior"`, split `reference_posterior`,
  with columns `reference_samples`, `observations`, `true_parameters` —
  deliberately matching the original SBI-benchmarks schema
  (`sbibm_jax/hf/reference.py`).
- Exporters (`sbibm_jax/hf/exporter.py`) own the flat↔native reshape:
  `VectorExporter` / `ImageExporter` / `TimeSeriesExporter`, selected from the
  task's `hf_data_kind` / `hf_data_shape` via `get_exporter`.
- Default target repo is `config.TEST_REPO` (`aurelio-amerio/SBI-benchmarks-test`);
  `--prod` switches to `config.DEFAULT_REPO`.

## Design decisions (locked via brainstorming)

1. **Full move**, `sbibm-jax` becomes the single source of truth.
   `GenSBI-examples` is not modified in this effort.
2. **Clean names only** in the new API — no `obs`/`cond` compatibility aliases.
   `GenSBI-examples` updates its call sites when it migrates.
3. **Stats computed at generation time** → published in `metadata.json`
   (option 1), with a **per-task reduction-axis spec**.
4. **Masks: full port** (base masks + edge transforms), living **entirely in a
   dedicated optional submodule** `sbibm_jax.data.masks`; the core loader and the
   producer `Task` stay mask-free.
5. **Core loader keeps GenSBI's tokenization** (`[...,None]`): joint/conditional
   processing reproduces `process_joint`/`process_conditional` verbatim,
   trailing channel dim included.
6. **Default repo = `config.TEST_REPO`** (`SBI-benchmarks-test`), overridable to
   production, consistent with `make_dataset`.

## §1 — New subpackage `sbibm_jax.data`

A consumer-side sibling to `sbibm_jax.hf`. `hf/` *builds & uploads* (producer);
`data/` *loads & serves* (consumer). They share only the `metadata.json`
contract and the HF repo.

- Gated by a **new `[loader]` optional extra**: `grain`, `datasets`,
  `huggingface_hub`. An import-guard mirrors the `[hf]`/pypesto pattern — the
  subpackage imports without the extra (for discovery) but raises an informative
  `ImportError` pointing at `pip install sbibm-jax[loader]` when used.
- `grain` is a heavy, consumer-only dependency; keeping the loader in its own
  subpackage with its own extra keeps the producer/build path light. (Rejected
  alternative: folding the loader into `sbibm_jax.hf`.)

Files (proposed):

```
src/sbibm_jax/data/
  __init__.py        # public API: TaskDataset; import-guard
  dataset.py         # TaskDataset class
  process.py         # joint/conditional collate (tokenization + normalization)
  masks/
    __init__.py      # get_base_mask_fn, get_edge_mask_fn (per-task dispatch)
    base.py          # per-task base adjacency builders (parameterized by dims)
    graph.py         # ported faithfull_mask / min_faithfull_mask / moralize
```

## §2 — `TaskDataset` (public entry)

Named to avoid clashing with the producer's `get_task` / `get_available_tasks`
(already re-exported from the top-level `sbibm_jax` package).

```python
from sbibm_jax.data import TaskDataset

ds = TaskDataset("two_moons", kind="conditional")   # repo defaults to TEST
loader = ds.get_train_loader(batch_size=256)
theta, x = next(iter(loader))                        # tokenized: (B, dim, 1)
```

Construction:

```python
TaskDataset(
    name: str,
    *,
    kind: str = "conditional",      # "joint" | "conditional"
    repo: str | None = None,        # None -> config.TEST_REPO
    normalize: bool = False,        # apply published stats in the collate
    dtype = jnp.float32,            # e.g. jnp.bfloat16 for GW/lensing
    seed: int = 42,
    use_prefetching: bool = True,
    max_workers: int | None = None, # capped at <= 8 (shared node)
)
```

Attributes:

- **Dims:** `dim_parameters`, `dim_data`, `data_kind`, `data_shape`,
  `dim_joint` (set only when `kind="joint"`).
- **Stats** (read from `metadata.json`, native-shaped): `theta_mean`,
  `theta_std`, `x_mean`, `x_std`. `None` when the task ships no stats.

Methods:

- **Loaders (grain):** `get_train_loader(batch_size, num_samples=None)`,
  `get_val_loader(batch_size)`, `get_test_loader(batch_size)`. `num_samples`
  subsamples the train prefix (valid because every `(θ, x)` row is an
  independent draw; matches the producer's "subsample by prefix" convention).
  `max_workers` is clamped to ≤ 8.
- **Normalization:** `normalize_theta(t)`, `unnormalize_theta(t)`,
  `normalize_x(x)`, `unnormalize_x(x)` (broadcast the published stats).
- **Reference:** `get_reference(num_observation=1)` → `(observation, samples)`;
  `get_true_parameters(num_observation=1)`. Both load the `f"{name}_posterior"`
  config / `reference_posterior` split and index `num_observation - 1` into
  `observations` / `reference_samples` / `true_parameters`. Tasks without a
  `_posterior` config raise an informative error (as GW/lensing do today).

## §3 — One generic processing path (no per-task overrides)

The collate (`process.py`) is **fully generic**, driven by `metadata.json`:

1. read native-shaped arrays from the split (`xs` already stored at
   `data_shape` for image/timeseries; flat for vector),
2. optionally normalize with the published stats (when `normalize=True`),
3. cast to the `dtype` parameter,
4. **tokenize** (`[...,None]`) and, for `kind="joint"`, concatenate the θ/x
   tokens along the feature axis — reproducing GenSBI's
   `process_joint`/`process_conditional` exactly.

This generic path reproduces **both** the analytical `process_*` **and** the
GW/lensing bespoke `split_data` — verified: `split_data` is only
*reshape-to-native + bfloat16 + normalize*, all of which the generic path covers
once `dtype` is a loader parameter and stats live in metadata. Therefore **no
task needs a custom `process_fn`.** If a future task genuinely needs bespoke
shaping, an opt-in hook can be added then (YAGNI for now).

**Joint mode is vector-only.** Concatenating θ-tokens with x-tokens along one
axis is only meaningful for flat-vector data; for `data_kind in {image,
timeseries}` `kind="joint"` raises an informative error. This matches GenSBI,
where the image/GW tasks are conditional-only. Conditional mode works for all
data kinds.

Output shapes (tokenized, per decision 5):

- `conditional` → `(theta: (B, dim_parameters, 1), x: (B, *data_shape, 1))`
- `joint` (vector only) → `(B, dim_parameters + dim_data, 1)`

## §4 — Stats computed at generation time → `metadata.json`

Producer-side change in `sbibm_jax.hf`:

- During generation of the **train** split, accumulate running mean/std over the
  **native-shaped** x (via `exporter.shape_x`) and over θ, using a numerically
  stable accumulator (Welford / sum + sum-of-squares in **float64**) so the
  result is exact over ~10^6 rows without holding them in RAM. This piggybacks
  on the existing chunked streaming in `hf/generate.py` / `hf/build.py`.
- **Per-task reduction axes** via a new opt-in attribute on the task, in the
  same `hf_*` idiom as `hf_data_kind` etc.:

  ```python
  # default (per-feature: reduce the batch axis only)
  hf_stats_axes = {"theta": (0,), "x": (0,)}
  # GravitationalLensing — global scalar over the whole image:
  hf_stats_axes = {"theta": (0,), "x": (0, 1, 2)}
  # GravitationalWaves — per-channel (keep the channel axis):
  hf_stats_axes = {"theta": (0,), "x": (0, 1)}
  ```

  Axes refer to the **native batch shape** `(batch, *data_shape)`. `keepdims`
  preserves broadcastability against native data.
- Write the reduced `theta_mean`/`theta_std`/`x_mean`/`x_std` (and the
  `stats_axes` used, for transparency) into the per-task `metadata.json` block.
  These are small (per-feature / per-channel / scalar — never per-pixel, which
  is exactly what the axis spec prevents), so JSON storage is appropriate. They
  are merged non-destructively like the rest of `metadata.json`.
- The loader reads stats straight from the already-fetched `metadata.json` — no
  extra downloads, always in sync with the published data.

This replaces the hand-shipped `GenSBI-examples/stats/*.npz` files (which stay in
that repo until its later migration).

## §5 — Masks: dedicated optional submodule `sbibm_jax.data.masks`

Full port of the mask feature, **self-contained** — neither the core
`TaskDataset` nor the producer `Task` imports it. Consumers building graph/causal
transformers opt in explicitly:

```python
from sbibm_jax.data.masks import get_base_mask_fn, get_edge_mask_fn
base_fn = get_base_mask_fn("slcp", dim_parameters=5, dim_data=8)
edge_fn = get_edge_mask_fn("slcp", "undirected", dim_parameters=5, dim_data=8)
```

- **Base masks** (`base.py`): per-task adjacency builders, **parameterized by
  the task's dims at call time** (`dim_parameters`, `dim_data`) rather than
  hardcoded, so they cannot silently desync from the data. Ported for the 5
  analytical base tasks; unsupported tasks raise `NotImplementedError` (as
  today).
- **Edge transforms** (`graph.py`): direct port of `faithfull_mask`,
  `min_faithfull_mask`, `moralize`, exposed through `get_edge_mask_fn(name, ...)`
  with the named variants `faithfull` / `min_faithfull` / `undirected` /
  `directed` / `none`. Generic — applied to any base mask.

Rationale: the edge-mask machinery targets a narrow family of graph transformers
that enforce conditional-independence structure via an attention mask; the large
majority of SBI consumers (flow matching, diffusion, NPE/NRE, plain
transformers) only want θ–x pairs. Masks are therefore an opt-in bonus, kept out
of the core API's weight.

## §6 — Repo / safety default

`TaskDataset(repo=None)` resolves to `config.TEST_REPO`
(`aurelio-amerio/SBI-benchmarks-test`), mirroring the producer's default so test
runs never touch production. Production is reached by passing
`repo=config.DEFAULT_REPO` (or the literal repo id).

## §7 — Reference / posterior loading (confirmed)

The producer stores reference posteriors as a **separate** `f"{task}_posterior"`
config (split `reference_posterior`; columns `reference_samples`,
`observations`, `true_parameters`), precisely because not every task has them.
`TaskDataset.get_reference` / `get_true_parameters` mirror this: load that
config, index by `num_observation - 1`. No reconciliation needed.

## §8 — Out of scope (later, separate effort)

Migrating `GenSBI-examples` onto `sbibm_jax.data`: delete its `tasks.py` /
`stats/` / `mask.py` / `graph.py`, depend on `sbibm-jax[loader]`, and update call
sites from the `obs`/`cond` names to the clean names. Tracked separately.

## Testing

- **Construction & metadata:** build a small local/mocked `metadata.json` (+ tiny
  local HF dataset); assert `dim_parameters`/`dim_data`/`data_kind`/`data_shape`
  and stats parse correctly; default `repo` resolves to `TEST_REPO`.
- **Processing:** joint/conditional output shapes including the `[...,None]`
  token channel; `kind="joint"` raises for `image`/`timeseries`; `dtype` honored
  (incl. bfloat16); normalize/unnormalize round-trip.
- **Gen-time stats:** for each reduction mode (per-feature, per-channel,
  global-scalar) compare the streamed accumulator against `np.mean/np.std` on a
  small fully-materialized sample; assert published shapes match the documented
  native-axis reductions.
- **Masks:** base adjacency shapes `(dim_θ+dim_x, dim_θ+dim_x)` for the 5
  supported tasks; `node_ids` sub-indexing; each edge-transform variant; an
  unsupported task raises `NotImplementedError`.
- **Reference:** `get_reference` / `get_true_parameters` indexing against a
  mocked `_posterior` config; a task without `_posterior` raises informatively.
- **grain smoke test:** iterate a few batches of a small split on CPU with
  `max_workers <= 8`.
