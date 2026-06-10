# Gravitational Waves task + metadata schema symmetry — design

**Date:** 2026-06-10
**Status:** Approved scope. Three coherent pieces: (A) a breaking
`metadata.json` schema refactor making `x`/`theta` symmetric, (B) the
file-backed `gravitational_waves` task + standalone upload/conversion scripts
(no tests, minimal surface), (C) loader test coverage for the timeseries path
and the new schema fields.

## Overview

We are adding a Gravitational Waves benchmark to `sbibm-jax`, ported from the
GenSBI/SBI-benchmarks ecosystem. GW is unlike every existing task: it has **no
simulator** (the generator is not available yet) and its data is a fixed corpus
of pre-generated `(theta, x)` rows shipped as files. So the task exposes a
**mock** simulator/prior that raise `NotImplementedError`, and the dataset is
published by a **dedicated standalone script** that reads converted `.npz`
shards — bypassing the simulator-driven `make_dataset.py` path. GW will be
reworked once the real simulator lands, so its upload path is intentionally
minimal (a script, no package module, no tests).

Investigating the consumer side surfaced that our metadata-driven loader
(`sbibm_jax.data.TaskDataset`) **already** loads GW-shaped data correctly via
the generic `timeseries` path — the only missing thing was test coverage. It
also surfaced a schema asymmetry worth fixing now, while the dataset repo is
still the **test** repo (pre-freeze): `x` has a rich `data_shape` but `theta`
only had a scalar `dim_theta`. We make the schema symmetric (`x_*` / `theta_*`),
which also future-proofs non-vector `theta` (e.g. a future image-θ task).

Pieces:
- **A — schema refactor (generic, all tasks):** rename/restructure
  `metadata.json` to `{x_kind, x_shape, theta_kind, theta_shape, …}`; drop
  `dim_x`/`dim_theta` (derived from shapes). Re-upload all tasks to the TEST
  repo (also refreshes the stale metadata already pending there).
- **B — GW task + scripts:** mock task class, `.pt`→`.npz` converter, standalone
  uploader. No tests for the scripts.
- **C — loader tests:** lock in the `timeseries` conditional + normalize path
  and the new schema parsing in `TaskDataset`.

## Background

- **GenSBI consumer** (`GenSBI-examples/.../tasks.py`, `GravitationalWaves`):
  conditional-only; `theta` = 2 params (`dim_obs=2, ch_obs=1`), `x` = a
  `(8192, 2)` two-channel series (`dim_cond=8192, ch_cond=2`); `split_data`
  normalizes (bfloat16 stats) and reshapes to `theta (B,2,1)`, `x (B,8192,2,1)`;
  `get_reference`/`get_true_parameters` raise `NotImplementedError`. The
  `dim_cond`/`ch_cond`/`dim_obs`/`ch_obs` names are GenSBI's; reconciling
  gensbi-examples to our `x_shape`/`theta_shape` is **out of scope** here.
- **Original upload** (`SBI-benchmarks-data/.../gw_dataset.py`): 10 `.pt` shards
  `thetas_{0..9}` `(N,2)`, `xs_{0..9}` channels-first `(N,2,8192)`. Pool shards
  0–8, transpose `xs` to `(N,8192,2)`; **train** = pool `[:-512]`,
  **validation** = pool `[-512:]`, **test** = shard 9 (whole). Pushed under
  `config_name="gravitational_waves"`, features `xs: Array2D((8192,2))`,
  `thetas: List(float32)`. GenSBI's hardcoded stats imply per-channel `x`
  (`(1,1,2)`) and per-feature `theta` (`(1,2)`).
- As of writing: θ shards complete (~10 000 rows each → ~100 000 total);
  `xs` shards still downloading — so the converter must verify shapes at run
  time, not assume them.

### Why GW already loads through `TaskDataset`

`make_collate(kind="conditional", …)` does `theta[...,None]`, `x[...,None]`;
with `x` stored `(B,8192,2)` that yields `(B,8192,2,1)`, and `theta (B,2)` →
`(B,2,1)` — identical to GenSBI's `split_data`. Normalization broadcasts:
`theta_mean (1,2)`→`(1,2,1)`, `x_mean (1,1,2)`→`(1,1,2,1)`. dtype is
configurable (`jnp.bfloat16` supported). So **no loader code change** is needed
for GW beyond the schema refactor; the gap was only test coverage.

---

## A. Metadata schema refactor (generic)

### New per-task block

```json
"<task>": {
  "x_kind":      "vector" | "image" | "timeseries",
  "x_shape":     [ ... ],
  "theta_kind":  "vector" | "image" | "timeseries",
  "theta_shape": [ ... ],
  "splits":      {"train": int, "validation": int, "test": int},
  "has_reference": bool,
  "num_observations": int,
  "stats": { ... } | null
}
```

- **Renamed:** `data_kind` → `x_kind`, `data_shape` → `x_shape`.
- **Added:** `theta_kind`, `theta_shape`.
- **Removed:** `dim_x`, `dim_theta` — derived as `prod(x_shape)` /
  `prod(theta_shape)`.
- Today every task has `theta_kind="vector"`, `theta_shape=[dim_theta]` (all θ
  are flat vectors); GW is `x_kind="timeseries"`, `x_shape=[8192,2]`,
  `theta_kind="vector"`, `theta_shape=[2]`. A vector task (two_moons) is
  `x_kind="vector"`, `x_shape=[2]`, `theta_kind="vector"`, `theta_shape=[2]`.

Breaking change is acceptable: the live datasets are on the **test** repo and
get re-uploaded (which also clears the stale TEST metadata already noted as
pending). No backward-compat fallback in the loader — the new keys are required.

Channels are **derived** from shape (`shape[-1]` when rank ≥ 2, else 1); no
separate channel field (single source of truth, can't drift).

### Internal renames (end-to-end consistency)

So `exporter.<attr>` matches the metadata keys:

1. **`src/sbibm_jax/hf/exporter.py`**
   - `DatasetExporter`: class attr `data_kind` → `x_kind`; add `theta_kind`
     (default `"vector"`). Instance `self.data_shape` → `self.x_shape`; add
     `self.theta_shape` (default `(task.dim_theta,)`). Constructor params
     `data_shape` → `x_shape`, plus `theta_shape=None`/`theta_kind="vector"`.
   - `VectorExporter` / `ImageExporter` / `TimeSeriesExporter`: `data_kind` →
     `x_kind`; `shape_x` bodies and the `data_shape` ctor param → `x_shape`;
     error messages updated. `theta_feature()` stays `List(float32)` (all θ are
     vectors today; a future image-θ would add a θ-storage dispatch — out of
     scope).
2. **`src/sbibm_jax/hf/registry.py`**
   - `DATA_KIND_REGISTRY` → `X_KIND_REGISTRY`.
   - `get_exporter`: read `hf_x_kind` (default `"vector"`), `hf_x_shape`
     (default `(task.dim_x,)`), `hf_theta_kind` (default `"vector"`),
     `hf_theta_shape` (default `(task.dim_theta,)`); pass `x_shape`/`theta_shape`
     /`theta_kind` to the exporter. Docstring updated.
3. **`src/sbibm_jax/hf/metadata.py`**
   - Emit `x_kind`, `x_shape`, `theta_kind`, `theta_shape`; drop `dim_x`,
     `dim_theta`. Read from `exporter.x_kind` / `.x_shape` / `.theta_kind` /
     `.theta_shape`.
4. **Tasks setting the x-hint:** `gaussian_random_field/task.py`,
   `toy_lensing/task.py`: `hf_data_kind` → `hf_x_kind`,
   `hf_data_shape` → `hf_x_shape`. (They don't set θ-hints → defaults apply.)
   ODE/PEtab tasks set only `hf_resample_invalid` — unaffected.
5. **`src/sbibm_jax/data/dataset.py` (`TaskDataset`)**
   - Parse `self.x_kind`, `self.x_shape`, `self.theta_kind`, `self.theta_shape`
     (required keys). Derive `self.dim_x = int(prod(x_shape))`,
     `self.dim_theta = int(prod(theta_shape))` (kept so masks /
     `dim_joint = dim_theta + dim_x` are unaffected — `get_base_mask_fn` takes
     `dim_theta`/`dim_x` as explicit args from the consumer). Pass
     `x_kind=self.x_kind, theta_kind=self.theta_kind` to `make_collate`.
6. **`src/sbibm_jax/data/process.py` (`make_collate`)**
   - Param `data_kind` → `x_kind`; add `theta_kind="vector"`. Joint guard
     becomes: `kind=="joint"` requires `x_kind=="vector"` **and**
     `theta_kind=="vector"` (joint concatenates flat tokens). Tokenization
     unchanged.

`build.py` and `reference.py` call `exporter.shape_x(...)` and `task.dim_x`
(the **Task** class keeps `dim_x`/`dim_theta`; only the *metadata* drops them) —
no change beyond the method-body rename inside `exporter.py`.

### `data_kind` keep rationale

`x_kind` (and `theta_kind`) are retained even though shape rank hints at kind:
`image` and `timeseries` are both rank-2 `x_shape`, so shape alone can't
disambiguate them (different semantics, same `Array2D` storage). Vector vs
non-vector also gates `joint`.

---

## B. GW task + scripts (minimal surface, no tests)

### B0. Converter — `scripts/convert_gw_to_npz.py` (torch group, one-time)

`uv run --group torch python scripts/convert_gw_to_npz.py`. Args `--data-dir`
(default `/lhome/ific/a/aamerio/data/GW`), `--out-dir` (default = data-dir),
`--num-shards` (default 10). Per shard:
- `thetas_i.pt` → `torch.load(weights_only=True)` → assert 2-D, last dim 2 →
  float32 → `np.savez_compressed(thetas_i.npz, data=…)`, `(N,2)`.
- `xs_i.pt` → assert 3-D with one axis == 2 (channel) and one == 8192 (time);
  **detect orientation** and transpose to `(N,8192,2)` only if channel-first
  (correct whether torch stored `(N,2,8192)` or `(N,8192,2)`) → float32 →
  `np.savez_compressed(xs_i.npz, data=…)`.
- Assert θ/xs row counts match; print `name → shape, dtype`.

This **guarantees** the downstream layout (key `"data"`, `(N,8192,2)` / `(N,2)`,
float32) rather than assuming it. Imports torch only; never imported by the
package or tests.

### B1. Task — `src/sbibm_jax/tasks/gravitational_waves/task.py`

Directory `gravitational_waves` (== HF config name the loader reads).
`GravitationalWaves(Task)`:
- Base: `dim_theta=2`, `dim_x=16384`, `name="gravitational_waves"`,
  `name_display="Gravitational Waves"`, `num_observations=1` (nominal; no
  reference), `num_posterior_samples=10000`, `path=__file__.parent`.
- HF hints: `hf_x_kind="timeseries"`, `hf_x_shape=(8192,2)`,
  `hf_stats_axes={"theta": (0,), "x": (0, 1)}` (→ θ `(1,2)`, x `(1,1,2)`),
  `hf_external=True`. (θ defaults: `theta_kind="vector"`, `theta_shape=(2,)`.)
- Mocks raise `NotImplementedError`: `get_prior` ("prior not available; the
  dataset is the only source"), `get_simulator` ("simulator not yet
  implemented; use the published dataset via sbibm_jax.data.TaskDataset"),
  `_sample_reference_posterior`.
- Ships **no** `files/` → base `get_observation` etc. raise `FileNotFoundError`
  → `load_reference` returns `None` → `has_reference=False` automatically.
- `unflatten_data` → `(-1, 8192, 2)` for consistency.
- Registry: add `elif task_name == "gravitational_waves"` to `get_task()`;
  auto-discovered by `get_available_tasks()`.

### B2. Uploader — `scripts/make_gw_dataset.py` (standalone)

Self-contained (approach B); reuses package helpers it can import but adds no
new package module and no tests. Mirrors `make_dataset.py`'s CLI/flow:
- Args: `--data-dir` (default `/lhome/ific/a/aamerio/data/GW`), `--prod`,
  `--dry-run`, `--val-size` (default 512), `--num-shards` (default 10),
  `--metadata-path` (default `metadata.json`), `--verbose`.
- `repo = config.DEFAULT_REPO if --prod else config.TEST_REPO`; print
  `Target repo: … (TEST|PRODUCTION)`.
- Read per-shard npz (`np.load(p)["data"]`), one shard at a time (RAM ≈ one
  shard). Split (mirror original, streaming-friendly — pool is shards
  0..n-2 in order, so its last `val_size` rows are the tail of the last pool
  shard; assert that shard has ≥ `val_size`):
  - **train** = shards `0..n-2` minus last `val_size` rows of the final pool
    shard; **validation** = those last `val_size` rows; **test** = shard `n-1`.
- Build each split via `Dataset.from_generator(row_gen, features=…)` with
  features `{"xs": Array2D((8192,2), "float32"), "thetas": List(float32)}`
  (built inline, or via `get_exporter(get_task("gravitational_waves"))
  .features()`); generator loads shards lazily so no split is fully resident.
- Validate each loaded `x` native shape == `(8192,2)` and `theta` == `(2,)` so a
  converter mistake fails loudly.
- Compute train stats with `StatsAccumulator(*resolve_stats_axes(task))` over the
  built train split → θ `(1,2)`, x `(1,1,2)`.
- Push three splits under `config_name="gravitational_waves"` (no `_posterior`).
- Metadata round-trip exactly like `make_dataset.py`:
  `make_metadata(["gravitational_waves"], train_size=Ntr, val_size=Nval,
  test_size=Nte, stats_by_task={"gravitational_waves": stats})` (actual counts so
  recorded `splits` match) → `fetch_remote_metadata` → `merge_metadata` →
  write → `upload_metadata` → delete local file.
- `--dry-run`: write `make_metadata(["gravitational_waves"])` only (no read, no
  stats → `stats: null`).

### B3. `make_dataset.py` — skip external tasks

Filter out `getattr(get_task(name), "hf_external", False)` tasks from **both**
generation and metadata, logging `Skipping gravitational_waves
(external/file-backed; use scripts/make_gw_dataset.py)`. Keeps `--all` working
without hitting GW's mock simulator; GW's metadata block is owned by
`make_gw_dataset.py` and preserved by the non-destructive `merge_metadata`.

---

## C. Loader test coverage

In `tests/data/test_dataset.py` (the `[loader]`-gated suite):
1. **New-schema parsing:** fake metadata uses `x_kind/x_shape/theta_kind/
   theta_shape` (no `dim_x`/`dim_theta`); assert `TaskDataset` exposes
   `x_kind`, `x_shape`, `theta_kind`, `theta_shape` and **derives**
   `dim_x == prod(x_shape)`, `dim_theta == prod(theta_shape)`; `dim_joint`
   (joint) still correct.
2. **Timeseries conditional path:** fake a GW-like task (`x_kind="timeseries"`,
   `x_shape=[T,2]` with small `T`, stats θ `(1,2)`/x `(1,1,2)`); assert the
   conditional loader yields `theta (B,2,1)`, `x (B,T,2,1)`; with
   `normalize=True` the per-channel/per-feature stats broadcast and a
   normalize→unnormalize round-trip holds.
3. **Joint guard:** `kind="joint"` on a non-vector `x_kind` raises (vector-only).

`tests/data/test_process.py`: update `make_collate` call sites (`data_kind` →
`x_kind`, add `theta_kind`), add a timeseries-conditional shape case and the
joint-guard case.

Schema-refactor test updates (mechanical): `tests/hf/test_exporter.py`,
`tests/hf/test_registry.py`, `tests/hf/test_metadata.py`,
`tests/tasks/test_gaussian_random_field.py` (assert `hf_x_kind`/`hf_x_shape`),
and any `tests/hf/test_reference.py` / `test_build_dataset.py` /
`test_generate.py` references to the old metadata keys. `tests/data/test_masks.py`
passes dims as ints → unaffected.

GW task test — `tests/tasks/test_gravitational_waves.py`:
- dims/name/hints (`hf_x_kind=="timeseries"`, `hf_x_shape==(8192,2)`,
  `hf_stats_axes`, `hf_external is True`);
- `get_prior`/`get_simulator`/`_sample_reference_posterior` raise
  `NotImplementedError`;
- registry: `get_task("gravitational_waves")` instance; in
  `get_available_tasks()`.

(The upload/conversion **scripts** get no tests, per scope.)

## Files

**New:**
- `scripts/convert_gw_to_npz.py`
- `scripts/make_gw_dataset.py`
- `src/sbibm_jax/tasks/gravitational_waves/__init__.py` (empty), `…/task.py`
- `tests/tasks/test_gravitational_waves.py`

**Modify (schema refactor):**
- `src/sbibm_jax/hf/exporter.py`, `…/registry.py`, `…/metadata.py`
- `src/sbibm_jax/tasks/gaussian_random_field/task.py`,
  `src/sbibm_jax/tasks/toy_lensing/task.py`
- `src/sbibm_jax/data/dataset.py`, `src/sbibm_jax/data/process.py`
- `tests/hf/test_exporter.py`, `…/test_registry.py`, `…/test_metadata.py`,
  `tests/tasks/test_gaussian_random_field.py`, and old-key refs in
  `tests/hf/test_reference.py` / `test_build_dataset.py` / `test_generate.py`
- `tests/data/test_dataset.py`, `tests/data/test_process.py`

**Modify (GW):**
- `src/sbibm_jax/tasks/__init__.py` (register `gravitational_waves`)
- `scripts/make_dataset.py` (skip `hf_external`)

**Docs:** `CLAUDE.md` — update the schema description (`x_*`/`theta_*`, dropped
`dim_*`), note the external/file-backed GW path and its scripts.

## Out of scope / deferred
- The real GW simulator + prior (when available, implement `get_prior`/
  `get_simulator`, drop `hf_external`; schema unchanged).
- Non-vector `theta` storage dispatch (theta_feature stays `List(float32)`);
  `theta_kind`/`theta_shape` are recorded now for future use.
- Reconciling gensbi-examples' `dim_cond`/`ch_cond`/`dim_obs`/`ch_obs` names to
  our `x_shape`/`theta_shape` (gensbi-examples side, separate work).
- Tests for the GW upload/conversion scripts.
- Reference posterior / true parameters / observations for GW (none exist).
