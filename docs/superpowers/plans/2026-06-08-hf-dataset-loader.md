# HF Dataset Loader (`sbibm_jax.data`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the consumer-side HuggingFace dataset loading from `GenSBI-examples` into a new optional `sbibm_jax.data` subpackage, so `sbibm-jax` both builds and serves the SBI-benchmarks datasets — with `grain` dataloaders, gen-time normalization stats published in `metadata.json`, and a self-contained masks submodule.

**Architecture:** Four sequential phases, **each its own PR**: (0) rename `dim_parameters`/`dim_data` → `dim_theta`/`dim_x` package-wide; (1) compute normalization stats during generation and store them in `metadata.json`; (2) the `sbibm_jax.data.TaskDataset` core loader (grain, joint/conditional with GenSBI tokenization); (3) the `sbibm_jax.data.masks` opt-in submodule (base masks + edge transforms). The loader is driven entirely by the published `metadata.json` contract, with no per-task code.

**Tech Stack:** Python 3.12, JAX/numpyro, `datasets` + `huggingface_hub` (`[hf]` extra), `grain` (new `[loader]` extra), `numpy`, pytest + pytest-xdist (CPU-forced, `-n 2`).

**Source spec:** `docs/superpowers/specs/2026-06-08-hf-dataset-loader-design.md`

**Conventions for every task:** `uv run pytest <path> -v` runs tests (CPU-forced via `pytest-env`). Lint with `uv run flake8 <paths>`; judge by *new* violations vs HEAD (bare flake8 is never clean — pre-existing E501). Never run jobs with more than 8 cores. Commit messages end with the `Co-Authored-By` trailer used in this repo.

---

## File Structure

```
src/sbibm_jax/
  tasks/task.py            # MODIFY (P0): dim_parameters->dim_theta, dim_data->dim_x
  tasks/**/task.py         # MODIFY (P0): same rename; (P1) hf_stats_axes on image tasks
  hf/metadata.py           # MODIFY (P0 keys, P1 stats injection)
  hf/exporter.py           # MODIFY (P0 rename)
  hf/registry.py           # MODIFY (P0 rename)
  hf/build.py              # MODIFY (P1): accumulate train-split stats -> bundle["stats"]
  hf/upload.py             # MODIFY (P1): upload_dataset returns stats
  hf/stats.py              # CREATE (P1): StatsAccumulator + axis resolution
  data/                    # CREATE (P2/P3): the loader subpackage
    __init__.py            #   import guard + public API (TaskDataset)
    dataset.py             #   TaskDataset class
    process.py             #   joint/conditional collate (tokenize + normalize)
    masks/
      __init__.py          #   get_base_mask_fn, get_edge_mask_fn, get_condition_mask_fn
      graph.py             #   verbatim port of GenSBI graph.py
      condition.py         #   verbatim port of GenSBI mask.py samplers
      base.py              #   per-task base adjacency builders (param by dims)
scripts/make_dataset.py    # MODIFY (P1): inject stats post-build, upload metadata after builds
pyproject.toml             # MODIFY (P2): add [loader] extra + dependency-group
tests/
  hf/test_stats.py         # CREATE (P1)
  hf/test_metadata.py      # MODIFY (P1)
  hf/test_build_dataset.py # MODIFY (P1)
  data/                    # CREATE (P2/P3)
    conftest.py            #   importorskip grain
    test_import_guard.py
    test_process.py
    test_dataset.py
    test_masks.py
```

---

# PHASE 0 — Nomenclature rename (`dim_parameters`→`dim_theta`, `dim_data`→`dim_x`)

Mechanical, package-wide. The "test" is the existing suite staying green plus a grep proving no stragglers. Do this as its own PR before anything else so later phases build on the final names.

### Task 0.1: Rename dim attributes package-wide

**Files:**
- Modify: every `*.py` under `src/sbibm_jax/`, `tests/`, `scripts/` that contains `dim_parameters` or `dim_data`
- Key spots: `src/sbibm_jax/tasks/task.py:18-19,43-44`, `src/sbibm_jax/hf/metadata.py:29-30,48-49`, `src/sbibm_jax/hf/exporter.py:49`, `src/sbibm_jax/hf/registry.py:35,49`

- [ ] **Step 1: Record the baseline (suite green before the rename)**

Run: `uv run pytest -q -n 2`
Expected: PASS (note the passed count; the rename must not change it).

- [ ] **Step 2: Confirm the two tokens never appear as substrings of unrelated identifiers**

Run: `grep -rnE 'dim_data|dim_parameters' src tests scripts | grep -vE '\bdim_data\b|\bdim_parameters\b' || echo "CLEAN: only whole-word matches"`
Expected: `CLEAN: only whole-word matches` (proves `dim_data` is never part of e.g. `data_shape`; the rename is safe).

- [ ] **Step 3: Apply the rename across code (NOT docs)**

```bash
grep -rlE 'dim_parameters|dim_data' src tests scripts \
  | xargs sed -i 's/dim_parameters/dim_theta/g; s/dim_data/dim_x/g'
```

This renames the `Task` constructor params/attributes, every task's `super().__init__(dim_x=..., dim_theta=...)`, the `hf` pipeline references, **and** the `metadata.json` string keys in `metadata.py` (`"dim_parameters"`→`"dim_theta"`, `"dim_data"`→`"dim_x"`) plus its schema docstring. Docs under `docs/` are intentionally left untouched (the spec documents the rename).

- [ ] **Step 4: Verify zero stragglers remain in code**

Run: `grep -rnE 'dim_parameters|dim_data' src tests scripts || echo "NONE LEFT"`
Expected: `NONE LEFT`.

- [ ] **Step 5: Run the full suite — must be identical green**

Run: `uv run pytest -q -n 2`
Expected: PASS, same count as Step 1.

- [ ] **Step 6: Lint the touched files (no new violations)**

Run: `uv run flake8 src/sbibm_jax/tasks/task.py src/sbibm_jax/hf/metadata.py src/sbibm_jax/hf/exporter.py src/sbibm_jax/hf/registry.py`
Expected: no *new* violations vs HEAD (the rename shortens names, so line lengths only shrink).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: normalize dim_parameters->dim_theta, dim_data->dim_x package-wide

sbibm-jax convention is theta (parameters) / x (simulator output); the stored
columns are already thetas/xs. Rename the dimension attributes to match across
Task, all tasks, the hf pipeline, metadata.json keys, and tests. Mechanical;
suite unchanged. metadata.json on the test repo refreshes on the next
make_dataset run (stored data columns are untouched).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

> **Ops note (not a code step):** the test repo's existing `metadata.json` still has the old keys until a `make_dataset` run re-uploads it. The loader (Phase 2) only reads from a repo whose `metadata.json` was produced by this renamed pipeline, so refresh the test repo's metadata before loading from it live.

---

# PHASE 1 — Gen-time normalization stats → `metadata.json`

Compute exact mean/std over the **train** split during generation (float64, streamed), reduced per a per-task axis spec, and publish them in `metadata.json`. Its own PR.

### Task 1.1: `StatsAccumulator` + axis resolution (`hf/stats.py`)

**Files:**
- Create: `src/sbibm_jax/hf/stats.py`
- Test: `tests/hf/test_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hf/test_stats.py
"""Tests for hf.stats: streaming reduction matches numpy; axis resolution."""

import numpy as np
import pytest

from sbibm_jax.hf.stats import StatsAccumulator, resolve_stats_axes


class TestStatsAccumulator:
    def test_per_feature_matches_numpy(self):
        rng = np.random.default_rng(0)
        theta = rng.normal(size=(1000, 3)).astype(np.float32)
        x = rng.normal(size=(1000, 5)).astype(np.float32)
        acc = StatsAccumulator(theta_axes=(0,), x_axes=(0,))
        for i in range(0, 1000, 256):                       # stream in chunks
            acc.update(theta[i:i + 256], x[i:i + 256])
        res = acc.result()
        np.testing.assert_allclose(
            np.array(res["theta_mean"]), theta.mean(0, keepdims=True),
            rtol=1e-5, atol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(res["x_std"]), x.std(0, keepdims=True),
            rtol=1e-5, atol=1e-4,
        )
        assert np.array(res["theta_mean"]).shape == (1, 3)
        assert np.array(res["x_mean"]).shape == (1, 5)

    def test_global_scalar_image_reduction(self):
        rng = np.random.default_rng(1)
        theta = rng.normal(size=(64, 2)).astype(np.float32)
        x = rng.normal(size=(64, 8, 8)).astype(np.float32)   # native image
        acc = StatsAccumulator(theta_axes=(0,), x_axes=(0, 1, 2))
        acc.update(theta, x)
        res = acc.result()
        assert np.array(res["x_mean"]).shape == (1, 1, 1)
        np.testing.assert_allclose(
            np.array(res["x_mean"]).item(), x.mean(), rtol=1e-5, atol=1e-5,
        )


class TestResolveStatsAxes:
    def test_default_is_reduce_batch_only(self):
        class T:  # no hf_stats_axes
            pass
        theta_axes, x_axes = resolve_stats_axes(T())
        assert theta_axes == (0,)
        assert x_axes == (0,)

    def test_task_override_wins(self):
        class T:
            hf_stats_axes = {"theta": (0,), "x": (0, 1, 2)}
        theta_axes, x_axes = resolve_stats_axes(T())
        assert x_axes == (0, 1, 2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/hf/test_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbibm_jax.hf.stats'`.

- [ ] **Step 3: Implement `hf/stats.py`**

```python
# src/sbibm_jax/hf/stats.py
"""Streaming normalization statistics over native-shaped (theta, x).

Stats are accumulated in float64 (sum + sum-of-squares) so the mean/std are
exact over ~1e6 rows without holding them in RAM. Reduction axes refer to the
native BATCH-INCLUSIVE shape (axis 0 = batch); output keeps reduced dims so the
result broadcasts against a single native row. The published shapes are e.g.
(1, dim_theta) for per-feature theta, (1, 1, 1) for a global-scalar image x,
(1, 1, C) for a per-channel time-series x.
"""

from typing import Tuple

import numpy as np


class _Reducer:
    def __init__(self, axes: Tuple[int, ...]):
        self.axes = tuple(axes)
        self._sum = None
        self._sumsq = None
        self._count = 0

    def update(self, arr: np.ndarray) -> None:
        a = np.asarray(arr, dtype=np.float64)
        s = a.sum(axis=self.axes, keepdims=True)
        ss = (a * a).sum(axis=self.axes, keepdims=True)
        n = 1
        for ax in self.axes:
            n *= a.shape[ax]
        if self._sum is None:
            self._sum, self._sumsq = s, ss
        else:
            self._sum += s
            self._sumsq += ss
        self._count += n

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        mean = self._sum / self._count
        var = np.maximum(self._sumsq / self._count - mean * mean, 0.0)
        std = np.sqrt(var)
        return mean.astype(np.float32), std.astype(np.float32)


class StatsAccumulator:
    """Accumulate mean/std for theta and x over their reduction axes."""

    def __init__(self, theta_axes, x_axes):
        self.theta_axes = tuple(theta_axes)
        self.x_axes = tuple(x_axes)
        self._t = _Reducer(self.theta_axes)
        self._x = _Reducer(self.x_axes)

    def update(self, theta_native: np.ndarray, x_native: np.ndarray) -> None:
        self._t.update(theta_native)
        self._x.update(x_native)

    def result(self) -> dict:
        tm, ts = self._t.finalize()
        xm, xs = self._x.finalize()
        return {
            "theta_mean": tm.tolist(),
            "theta_std": ts.tolist(),
            "x_mean": xm.tolist(),
            "x_std": xs.tolist(),
            "theta_axes": list(self.theta_axes),
            "x_axes": list(self.x_axes),
        }


def resolve_stats_axes(task) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return (theta_axes, x_axes) from task.hf_stats_axes, default reduce-batch.

    Default reduces only the batch axis (per-feature stats). Tasks whose x is an
    image / time-series set hf_stats_axes to avoid per-pixel stats, e.g.
    {"theta": (0,), "x": (0, 1, 2)} for a global-scalar image.
    """
    spec = getattr(task, "hf_stats_axes", None)
    if spec is None:
        return (0,), (0,)
    return tuple(spec["theta"]), tuple(spec["x"])
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/hf/test_stats.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/hf/stats.py tests/hf/test_stats.py
git commit -m "feat(hf): StatsAccumulator + per-task stats-axis resolution

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 1.2: Accumulate train-split stats in `build_dataset`

**Files:**
- Modify: `src/sbibm_jax/hf/build.py:14-31` (`_build_split`), `:34-67` (`build_dataset`)
- Test: `tests/hf/test_build_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hf/test_build_dataset.py  (add to the existing file)
import numpy as np

from sbibm_jax import get_task
from sbibm_jax.hf.build import build_dataset
from sbibm_jax.hf.generate import derive_task_keys, generate_samples


class TestBuildStats:
    def test_stats_match_materialized_train_split(self):
        # build_dataset must compute train-split stats equal to a direct pass.
        bundle = build_dataset(
            "two_moons", train_size=512, val_size=16, test_size=16,
        )
        stats = bundle["stats"]
        assert stats is not None
        # Reproduce the same train draw and compare (same master seed + key).
        task = get_task("two_moons")
        key = derive_task_keys(task.name)["train"]
        thetas, xs, _ = generate_samples(task, key, 512)
        np.testing.assert_allclose(
            np.array(stats["theta_mean"]), thetas.mean(0, keepdims=True),
            rtol=1e-4, atol=1e-4,
        )
        np.testing.assert_allclose(
            np.array(stats["x_mean"]), xs.mean(0, keepdims=True),
            rtol=1e-4, atol=1e-4,
        )
        assert np.array(stats["theta_mean"]).shape == (1, task.dim_theta)
        assert np.array(stats["x_mean"]).shape == (1, task.dim_x)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/hf/test_build_dataset.py::TestBuildStats -v`
Expected: FAIL with `KeyError: 'stats'`.

- [ ] **Step 3: Compute train-split stats from the *materialized* train Dataset**

> **Critical gotcha (why we do NOT accumulate inside the generator):**
> `Dataset.from_generator` fingerprints the generator + closure and, on a
> **cache hit** (same task + seed + size → same fingerprint), loads the Arrow
> file *without ever calling the generator*. Accumulating stats as a generator
> side effect would then leave the accumulator empty on every re-run, and
> `finalize()` would divide by zero. Instead, compute stats by iterating the
> **already-built** train `Dataset` — iterating a materialized dataset always
> reads rows (cache-independent), and `x` is already stored native-shaped, so
> no reshape is needed. Bonus: no re-simulation.

Leave `_build_split` exactly as it is today (no `accumulator` param). Add a
`_compute_train_stats` helper, an `import numpy as np`, the
`StatsAccumulator`/`resolve_stats_axes` import, and a `"stats"` key in
`build_dataset`:

```python
# src/sbibm_jax/hf/build.py  (add near the top)
import numpy as np
from sbibm_jax.hf.stats import StatsAccumulator, resolve_stats_axes


def _compute_train_stats(task, exporter, train_dataset) -> dict:
    """Accumulate normalization stats by iterating the built train split.

    Cache-independent (unlike accumulating inside Dataset.from_generator's
    generator, which is skipped on a cache hit). x is stored native-shaped.
    """
    theta_axes, x_axes = resolve_stats_axes(task)
    acc = StatsAccumulator(theta_axes, x_axes)
    for batch in train_dataset.with_format("numpy").iter(
        batch_size=exporter.chunk_size
    ):
        acc.update(np.asarray(batch["thetas"]), np.asarray(batch["xs"]))
    return acc.result()
```

Then change the `build_dataset` return so the train split is built once and
reused for stats:

```python
    keys = derive_task_keys(task.name, master_seed=master_seed)
    train = _build_split(exporter, keys["train"], exporter.train_size)
    return {
        "train": train,
        "validation": _build_split(exporter, keys["validation"], exporter.val_size),
        "test": _build_split(exporter, keys["test"], exporter.test_size),
        "reference": load_reference(task, exporter),
        "stats": _compute_train_stats(task, exporter, train),
    }
```

(Also update the `build_dataset` docstring to mention the `"stats"` key.)

- [ ] **Step 4: Run to verify it passes — TWICE (cache-hit regression guard)**

Run it once, then immediately again **without** clearing `~/.cache/huggingface`:
`uv run pytest tests/hf/test_build_dataset.py::TestBuildStats -v && uv run pytest tests/hf/test_build_dataset.py::TestBuildStats -v`
Expected: PASS both times. The second run exercises the `Dataset.from_generator`
cache hit; stats must still be correct (this is the exact failure mode the
materialized-iteration approach fixes — if stats were accumulated inside the
generator, the second run would error in `finalize()`).

- [ ] **Step 5: Run the rest of the build tests (no regressions)**

Run: `uv run pytest tests/hf/test_build_dataset.py -v`
Expected: PASS (existing tests still green; a `"stats"` key is now present in bundles).

- [ ] **Step 6: Commit**

```bash
git add src/sbibm_jax/hf/build.py tests/hf/test_build_dataset.py
git commit -m "feat(hf): compute train-split normalization stats during build

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 1.3: Per-task `hf_stats_axes` on image tasks

Image x (e.g. `toy_lensing`, `gaussian_random_field`) would otherwise get per-pixel stats. Set them to a global scalar over the spatial axes.

**Files:**
- Modify: `src/sbibm_jax/tasks/toy_lensing/task.py` (near the other `hf_*` attrs, ~line 38), `src/sbibm_jax/tasks/gaussian_random_field/task.py` (same)
- Test: `tests/hf/test_build_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hf/test_build_dataset.py  (add)
class TestImageStatsShape:
    def test_toy_lensing_x_stats_are_global_scalar(self):
        bundle = build_dataset(
            "toy_lensing", train_size=32, val_size=8, test_size=8,
        )
        import numpy as np
        # x native shape is (H, W); global-scalar reduction -> (1, 1, 1).
        assert np.array(bundle["stats"]["x_mean"]).shape == (1, 1, 1)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/hf/test_build_dataset.py::TestImageStatsShape -v`
Expected: FAIL — default axes `(0,)` give x_mean shape `(1, H, W)`, not `(1, 1, 1)`.

- [ ] **Step 3: Add `hf_stats_axes` to the two image tasks**

In `src/sbibm_jax/tasks/toy_lensing/task.py`, right after `self.hf_data_shape = (resolution, resolution)`:

```python
        # Normalize the whole image with a single global scalar (avoid
        # per-pixel stats). Axes refer to the native batch shape (B, H, W).
        self.hf_stats_axes = {"theta": (0,), "x": (0, 1, 2)}
```

In `src/sbibm_jax/tasks/gaussian_random_field/task.py`, right after its `self.hf_data_shape = (field_size, field_size)`:

```python
        self.hf_stats_axes = {"theta": (0,), "x": (0, 1, 2)}
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/hf/test_build_dataset.py::TestImageStatsShape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/tasks/toy_lensing/task.py src/sbibm_jax/tasks/gaussian_random_field/task.py tests/hf/test_build_dataset.py
git commit -m "feat(tasks): global-scalar stats axes for image tasks

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 1.4: Inject stats into `metadata.json`; reorder the driver

`make_metadata` cannot compute stats (no generation), so the driver collects them from each build and folds them into the metadata before the (now post-build) metadata upload. `upload_dataset` returns its bundle's stats.

**Files:**
- Modify: `src/sbibm_jax/hf/metadata.py:12-64` (add `stats_by_task` param), `src/sbibm_jax/hf/upload.py:51-74` (return stats), `scripts/make_dataset.py:121-142` (reorder)
- Test: `tests/hf/test_metadata.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hf/test_metadata.py  (add)
from sbibm_jax.hf.metadata import make_metadata


class TestMetadataStats:
    def test_stats_injected_when_provided(self):
        stats_by_task = {
            "two_moons": {
                "theta_mean": [[0.0, 0.0]], "theta_std": [[1.0, 1.0]],
                "x_mean": [[0.0, 0.0]], "x_std": [[1.0, 1.0]],
                "theta_axes": [0], "x_axes": [0],
            }
        }
        meta = make_metadata(["two_moons"], stats_by_task=stats_by_task)
        assert meta["two_moons"]["stats"] == stats_by_task["two_moons"]

    def test_stats_absent_when_not_provided(self):
        meta = make_metadata(["two_moons"])
        assert meta["two_moons"]["stats"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/hf/test_metadata.py::TestMetadataStats -v`
Expected: FAIL with `TypeError: make_metadata() got an unexpected keyword argument 'stats_by_task'`.

- [ ] **Step 3: Add `stats_by_task` to `make_metadata`**

In `src/sbibm_jax/hf/metadata.py`, add the parameter and the per-entry `"stats"` field:

```python
def make_metadata(
    task_names: Iterable[str],
    *,
    output_path: Optional[Path] = None,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    stats_by_task: Optional[dict] = None,
) -> dict:
```

In the per-task dict (after `"num_observations": ...`), add:

```python
            "stats": (stats_by_task or {}).get(name),
```

(Update the schema docstring to list `stats: dict | None`.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/hf/test_metadata.py::TestMetadataStats -v`
Expected: PASS.

- [ ] **Step 5: Make `upload_dataset` return stats**

In `src/sbibm_jax/hf/upload.py`, change the signature/return:

```python
def upload_dataset(repo_name: str, task_name: str, **build_opts) -> dict:
    """Build the dataset for `task_name`, push each split, and return its stats."""
    bundle = build_dataset(task_name, **build_opts)
    bundle["train"].push_to_hub(
        repo_name, config_name=task_name, split="train", private=False,
    )
    bundle["validation"].push_to_hub(
        repo_name, config_name=task_name, split="validation", private=False,
    )
    bundle["test"].push_to_hub(
        repo_name, config_name=task_name, split="test", private=False,
    )
    if bundle.get("reference") is not None:
        bundle["reference"].push_to_hub(
            repo_name,
            config_name=f"{task_name}_posterior",
            split="reference_posterior",
            private=False,
        )
    return bundle["stats"]
```

- [ ] **Step 6: Reorder the driver to upload metadata (with stats) after builds**

In `scripts/make_dataset.py`, replace the block from `local_meta = make_metadata(...)` through `metadata_path.unlink(...)` (lines ~121-142) with:

```python
    metadata_path = Path(args.metadata_path)
    # Dry-run: metadata only, no generation, so no stats.
    local_meta = make_metadata(
        task_names,
        output_path=metadata_path,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    print(f"Wrote {metadata_path}")

    if args.dry_run:
        print("Dry run — skipping HF uploads.")
        return

    # Build + upload each task's data, collecting train-split stats.
    stats_by_task = {}
    for name in task_names:
        print(f"Uploading dataset for task: {name}")
        stats_by_task[name] = upload_dataset(repo, name, **build_opts)

    # Rebuild metadata WITH stats, merge non-destructively, upload once.
    local_meta = make_metadata(
        task_names,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        stats_by_task=stats_by_task,
    )
    remote_meta = fetch_remote_metadata(repo)
    merged_meta = merge_metadata(remote_meta, local_meta)
    metadata_path.write_text(json.dumps(merged_meta, indent=4))
    upload_metadata(str(metadata_path), repo)
    metadata_path.unlink(missing_ok=True)
    print(f"Removed local {metadata_path} (clean state).")
```

- [ ] **Step 7: Run the driver + upload tests with mocked network**

Run: `uv run pytest tests/hf/test_driver.py tests/hf/test_upload.py tests/hf/test_metadata.py -v`
Expected: PASS. If `tests/hf/test_driver.py` asserts the old ordering (metadata uploaded before data), update those assertions to the new order: data builds first, metadata (with stats) uploaded once at the end. Show the updated assertion inline when you touch it.

- [ ] **Step 8: Commit**

```bash
git add src/sbibm_jax/hf/metadata.py src/sbibm_jax/hf/upload.py scripts/make_dataset.py tests/hf/test_metadata.py tests/hf/test_driver.py tests/hf/test_upload.py
git commit -m "feat(hf): publish gen-time stats in metadata.json (driver reorder)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 1.5: Update CLAUDE.md for the new metadata stats

**Files:**
- Modify: `CLAUDE.md` (the HF export section describing the pipeline)

- [ ] **Step 1: Add a sentence to the HF export paragraph**

Append to the HF export description in `CLAUDE.md`:

> Normalization stats (mean/std of `theta` and `x`) are accumulated over the train split during generation (float64, streamed) and written into each task's `metadata.json` block; the per-task reduction axes default to per-feature and are overridden via the task's `hf_stats_axes` (image tasks use a global scalar). Stats are absent (`null`) under `--dry-run`.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: note gen-time stats in metadata.json

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

# PHASE 2 — Core loader `sbibm_jax.data`

The consumer subpackage. Its own PR. Default repo is `config.TEST_REPO`. Output reproduces GenSBI's tokenized joint/conditional processing.

### Task 2.1: `[loader]` extra + subpackage skeleton + import guard

**Files:**
- Modify: `pyproject.toml:17-31` (optional-deps), `:33-63` (dependency-groups)
- Create: `src/sbibm_jax/data/__init__.py`, `tests/data/__init__.py`, `tests/data/conftest.py`, `tests/data/test_import_guard.py`

- [x] **Step 1: Add the `[loader]` extra and dependency-group**

In `pyproject.toml`, under `[project.optional-dependencies]` add:

```toml
loader = [
    "datasets>=2.20.0",
    "huggingface_hub>=0.24.0",
    "grain>=0.2.15",
]
```

Under `[dependency-groups]` add a matching group and include it in `dev`:

```toml
loader = [
    "datasets>=2.20.0",
    "huggingface_hub>=0.24.0",
    "grain>=0.2.15",
]
```

and add `{include-group = "loader"}` to the `dev` group's list.

- [ ] **Step 2: Sync the new group**

Run: `uv sync --all-groups`
Expected: resolves and installs `grain`.

- [ ] **Step 3: Write the import-guard test**

```python
# tests/data/conftest.py
"""Skip the whole data test subdir if grain/datasets aren't importable."""
import pytest
pytest.importorskip("grain", reason="The [loader] extra is not installed.")
pytest.importorskip("datasets", reason="The [loader] extra is not installed.")
```

```python
# tests/data/test_import_guard.py
"""The data subpackage exposes TaskDataset when the extra is present."""

def test_taskdataset_importable():
    from sbibm_jax.data import TaskDataset
    assert TaskDataset is not None
```

```python
# tests/data/__init__.py   (empty)
```

- [ ] **Step 4: Run to verify it fails**

Run: `uv run pytest tests/data/test_import_guard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbibm_jax.data'`.

- [ ] **Step 5: Create the guarded `__init__.py`**

```python
# src/sbibm_jax/data/__init__.py
"""Consumer-side loading of the SBI-benchmarks HuggingFace datasets.

Requires the optional `[loader]` extra (`grain`, `datasets`, `huggingface_hub`).
Importing without it raises an informative ImportError, mirroring the `[hf]`
pattern. The loader serves theta/x pairs via grain, exposes task dims and
gen-time normalization stats from metadata.json, and (opt-in) graph masks via
`sbibm_jax.data.masks`.
"""

try:
    import grain  # noqa: F401
    import datasets  # noqa: F401
    import huggingface_hub  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The sbibm_jax.data subpackage requires the optional `[loader]` extra. "
        "Install it with `uv sync --extra loader` or `pip install sbibm-jax[loader]`."
    ) from e

from sbibm_jax.data.dataset import TaskDataset  # noqa: E402

__all__ = ["TaskDataset"]
```

(`dataset.py` is created in Task 2.3; this import resolves once that file exists. Implement 2.2 and 2.3 before re-running this test, or temporarily stub `dataset.py` with `class TaskDataset: ...` — but the plan order below creates `process.py` then `dataset.py` next, so just proceed.)

- [ ] **Step 6: Commit (after 2.3 lands the real `dataset.py`)**

Defer the commit of `__init__.py` until `dataset.py` exists (Task 2.3 Step 6) so the import resolves. Stage `pyproject.toml` + tests now:

```bash
git add pyproject.toml uv.lock tests/data/__init__.py tests/data/conftest.py tests/data/test_import_guard.py
git commit -m "build(loader): add [loader] extra (grain) + data test scaffold

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 2.2: Tokenizing joint/conditional collate (`process.py`)

Reproduces GenSBI's `process_joint`/`process_conditional` (+ normalized variants): tokenize each feature with a trailing `[...,None]`, normalize post-tokenization with trailing-dim stats, cast dtype.

**Files:**
- Create: `src/sbibm_jax/data/process.py`
- Test: `tests/data/test_process.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_process.py
"""Collate: tokenization, joint concat, normalization, dtype, joint guard."""

import numpy as np
import pytest

from sbibm_jax.data.process import make_collate


def _batch():
    return {
        "thetas": np.arange(2 * 3, dtype=np.float32).reshape(2, 3),  # (B=2, 3)
        "xs": np.ones((2, 5), dtype=np.float32),                     # (B=2, 5) vector
    }


class TestConditional:
    def test_tokenizes_to_trailing_channel(self):
        collate = make_collate(kind="conditional", data_kind="vector")
        theta, x = collate(_batch())
        assert theta.shape == (2, 3, 1)
        assert x.shape == (2, 5, 1)

    def test_normalize_applies_stats(self):
        stats = {"theta_mean": [[1.0, 1.0, 1.0]], "theta_std": [[1.0, 1.0, 1.0]],
                 "x_mean": [[1.0, 1.0, 1.0, 1.0, 1.0]],
                 "x_std": [[2.0, 2.0, 2.0, 2.0, 2.0]]}
        collate = make_collate(kind="conditional", data_kind="vector",
                               normalize=True, stats=stats)
        theta, x = collate(_batch())
        # x all ones, mean 1, std 2 -> 0
        np.testing.assert_allclose(np.asarray(x), 0.0, atol=1e-6)


class TestJoint:
    def test_joint_concats_along_feature_axis(self):
        collate = make_collate(kind="joint", data_kind="vector")
        out = collate(_batch())
        assert out.shape == (2, 3 + 5, 1)

    def test_joint_raises_for_image(self):
        with pytest.raises(ValueError, match="joint.*vector"):
            make_collate(kind="joint", data_kind="image")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/data/test_process.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbibm_jax.data.process'`.

- [ ] **Step 3: Implement `process.py`**

```python
# src/sbibm_jax/data/process.py
"""Joint/conditional collate reproducing GenSBI's tokenized processing.

Each scalar feature becomes a length-1 token via a trailing [..., None] so a
graph transformer can index it as a node. Normalization (optional) is applied
post-tokenization with trailing-dim stats, matching GenSBI's process_*_norm.

NumPy throughout (not jnp): grain `.map(collate)` runs in worker subprocesses
under `mp_prefetch`, and numpy arrays pickle cleanly across the process
boundary where jax device arrays are a footgun. GenSBI's hot path likewise uses
np.concatenate. The loader therefore yields numpy arrays; wrap in jnp.asarray
downstream for jax. `dtype` defaults to np.float32 (jnp.float32 IS np.float32,
so a jnp dtype passed from TaskDataset works too).
"""

import numpy as np


def _stat_array(values, dtype):
    """metadata stat (native-reduced, e.g. (1, dim)) -> trailing-dim for tokens."""
    a = np.asarray(np.array(values), dtype=dtype)
    return a[..., None]  # (1, dim) -> (1, dim, 1); (1,1,1) -> (1,1,1,1)


def make_collate(*, kind, data_kind, normalize=False, stats=None, dtype=np.float32):
    """Return a collate fn mapping a {'thetas','xs'} batch to model-ready arrays.

    kind="conditional" -> (theta, x); kind="joint" -> concat([theta, x], axis=1).
    Joint is vector-only. `stats` is the metadata stats dict (native-reduced).
    """
    if kind == "joint" and data_kind != "vector":
        raise ValueError(
            f"kind='joint' is vector-only; task data_kind={data_kind!r} supports "
            "kind='conditional' only."
        )
    if kind not in ("joint", "conditional"):
        raise ValueError(f"Unknown kind {kind!r}.")

    if normalize:
        if stats is None:
            raise ValueError("normalize=True requires stats.")
        tm = _stat_array(stats["theta_mean"], dtype)
        ts = _stat_array(stats["theta_std"], dtype)
        xm = _stat_array(stats["x_mean"], dtype)
        xs_ = _stat_array(stats["x_std"], dtype)

    def collate(batch):
        theta = np.asarray(batch["thetas"], dtype=dtype)[..., None]
        x = np.asarray(batch["xs"], dtype=dtype)[..., None]
        if normalize:
            theta = (theta - tm) / ts
            x = (x - xm) / xs_
        if kind == "conditional":
            return theta, x
        return np.concatenate((theta, x), axis=1)

    return collate
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/data/test_process.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/data/process.py tests/data/test_process.py
git commit -m "feat(data): tokenizing joint/conditional collate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 2.3: `TaskDataset` — metadata, dims, stats, repo default

**Files:**
- Create: `src/sbibm_jax/data/dataset.py`
- Test: `tests/data/test_dataset.py`

- [ ] **Step 1: Write the failing test (metadata-driven construction, mocked network)**

```python
# tests/data/test_dataset.py
"""TaskDataset: metadata parsing, dims, stats, repo default, normalize."""

import json

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from sbibm_jax.hf import config


def _fake_metadata(tmp_path):
    meta = {
        "two_moons": {
            "dim_theta": 2, "dim_x": 2, "data_kind": "vector",
            "data_shape": [2], "splits": {"train": 8, "validation": 4, "test": 4},
            "has_reference": True, "num_observations": 2,
            "stats": {
                "theta_mean": [[0.0, 0.0]], "theta_std": [[1.0, 1.0]],
                "x_mean": [[0.0, 0.0]], "x_std": [[1.0, 1.0]],
                "theta_axes": [0], "x_axes": [0],
            },
        }
    }
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(meta))
    return str(p)


def _fake_main_dataset():
    rows = {"thetas": np.zeros((8, 2), np.float32), "xs": np.ones((8, 2), np.float32)}
    d = Dataset.from_dict(rows)
    return DatasetDict({"train": d, "validation": d, "test": d})


@pytest.fixture
def patched(monkeypatch, tmp_path):
    meta_path = _fake_metadata(tmp_path)
    monkeypatch.setattr(
        "sbibm_jax.data.dataset.hf_hub_download", lambda **kw: meta_path,
    )
    monkeypatch.setattr(
        "sbibm_jax.data.dataset.load_dataset",
        lambda repo, name=None, **kw: _fake_main_dataset(),
    )


class TestConstruction:
    def test_dims_and_stats_parsed(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons")
        assert ds.dim_theta == 2
        assert ds.dim_x == 2
        assert ds.data_kind == "vector"
        assert tuple(ds.data_shape) == (2,)
        assert np.array(ds.theta_mean).shape == (1, 2)

    def test_default_repo_is_test(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons")
        assert ds.repo == config.TEST_REPO

    def test_joint_sets_dim_joint(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons", kind="joint")
        assert ds.dim_joint == 4
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/data/test_dataset.py::TestConstruction -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbibm_jax.data.dataset'`.

- [ ] **Step 3: Implement `dataset.py`**

```python
# src/sbibm_jax/data/dataset.py
"""TaskDataset: load an SBI-benchmarks task from the Hub and serve theta/x.

Driven entirely by the published metadata.json (dims, data_kind/shape, splits,
stats) — no per-task code. Default repo is the TEST repo (config.TEST_REPO);
pass repo=config.DEFAULT_REPO for production.
"""

import json

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from sbibm_jax.hf import config
from sbibm_jax.data.process import make_collate

_MAX_WORKERS_CAP = 8  # shared node; never exceed (see CLAUDE.md / memory).


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

        meta_path = hf_hub_download(
            repo_id=self.repo, filename="metadata.json", repo_type="dataset",
        )
        with open(meta_path) as f:
            entry = json.load(f)[name]

        self.dim_theta = int(entry["dim_theta"])
        self.dim_x = int(entry["dim_x"])
        self.data_kind = entry["data_kind"]
        self.data_shape = tuple(entry["data_shape"])
        self.num_observations = int(entry["num_observations"])
        self.has_reference = bool(entry["has_reference"])
        self.dim_joint = self.dim_theta + self.dim_x if kind == "joint" else None

        stats = entry.get("stats")
        if stats is not None:
            self.theta_mean = stats["theta_mean"]
            self.theta_std = stats["theta_std"]
            self.x_mean = stats["x_mean"]
            self.x_std = stats["x_std"]
        else:
            self.theta_mean = self.theta_std = self.x_mean = self.x_std = None

        self._collate = make_collate(
            kind=kind, data_kind=self.data_kind,
            normalize=normalize, stats=stats, dtype=dtype,
        )

        self.dataset = load_dataset(self.repo, name).with_format("numpy")
        self.df_train = self.dataset["train"]
        self.df_val = self.dataset["validation"]
        self.df_test = self.dataset["test"]
        self.max_samples = self.df_train.num_rows
        self._posterior = None  # lazily loaded in get_reference
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/data/test_dataset.py::TestConstruction -v`
Expected: PASS.

- [ ] **Step 5: Verify the import-guard test now passes too**

Run: `uv run pytest tests/data/test_import_guard.py -v`
Expected: PASS (`TaskDataset` now importable).

- [ ] **Step 6: Commit (includes the `__init__.py` from Task 2.1)**

```bash
git add src/sbibm_jax/data/__init__.py src/sbibm_jax/data/dataset.py tests/data/test_dataset.py
git commit -m "feat(data): TaskDataset metadata-driven construction

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 2.4: grain loaders (`get_train_loader` / `get_val_loader` / `get_test_loader`)

**Files:**
- Modify: `src/sbibm_jax/data/dataset.py` (add methods)
- Test: `tests/data/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_dataset.py  (add)
class TestLoaders:
    def test_train_loader_yields_tokenized_batches(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons", kind="conditional")
        loader = ds.get_train_loader(batch_size=4)
        theta, x = next(iter(loader))
        assert np.asarray(theta).shape == (4, 2, 1)
        assert np.asarray(x).shape == (4, 2, 1)

    def test_num_samples_subsamples_prefix(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons")
        loader = ds.get_train_loader(batch_size=2, num_samples=4)
        theta, x = next(iter(loader))
        assert np.asarray(theta).shape[0] == 2

    def test_max_workers_clamped(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons", max_workers=64)
        assert ds.max_workers == 8

    def test_prefetching_loader_iterates(self, patched):
        # The numpy collate must survive grain's mp_prefetch (worker
        # subprocesses pickle batches across the process boundary). max_workers
        # small to keep it light on the shared node.
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons", use_prefetching=True, max_workers=2)
        loader = ds.get_train_loader(batch_size=2)
        theta, x = next(iter(loader))
        assert np.asarray(theta).shape == (2, 2, 1)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/data/test_dataset.py::TestLoaders -v`
Expected: FAIL with `AttributeError: 'TaskDataset' object has no attribute 'get_train_loader'`.

- [ ] **Step 3: Add the loader methods to `TaskDataset`**

```python
# src/sbibm_jax/data/dataset.py  (add imports + methods)
import grain


class TaskDataset:
    # ... existing __init__ ...

    def _loader(self, split, batch_size, num_samples=None):
        if num_samples is not None:
            if num_samples > split.num_rows:
                raise ValueError(
                    f"num_samples={num_samples} exceeds split size {split.num_rows}."
                )
            split = split.select(range(int(num_samples)))
        pipe = (
            grain.MapDataset.source(split)
            .shuffle(self.seed)
            .repeat()
            .to_iter_dataset()
            .batch(batch_size)
            .map(self._collate)
        )
        if self.use_prefetching and self.max_workers:
            cfg = grain.experimental.pick_performance_config(
                ds=pipe, ram_budget_mb=1024, max_workers=self.max_workers,
                max_buffer_size=None,
            )
            pipe = pipe.mp_prefetch(cfg.multiprocessing_options)
        return pipe

    def get_train_loader(self, batch_size, num_samples=None):
        return self._loader(self.df_train, batch_size, num_samples)

    def get_val_loader(self, batch_size):
        return self._loader(self.df_val, batch_size)

    def get_test_loader(self, batch_size):
        return self._loader(self.df_test, batch_size)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/data/test_dataset.py::TestLoaders -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/data/dataset.py tests/data/test_dataset.py
git commit -m "feat(data): grain train/val/test loaders (workers clamped to 8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 2.5: Reference / true-parameter access + normalize methods

**Files:**
- Modify: `src/sbibm_jax/data/dataset.py`
- Test: `tests/data/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_dataset.py  (add)
from datasets import Dataset, DatasetDict


def _fake_posterior():
    d = Dataset.from_dict({
        "observations": np.arange(2 * 2, dtype=np.float32).reshape(2, 2),
        "reference_samples": np.zeros((2, 10, 2), np.float32),
        "true_parameters": np.ones((2, 2), np.float32),
    })
    return DatasetDict({"reference_posterior": d})


class TestReference:
    def test_get_reference_indexes_observation(self, monkeypatch, patched):
        from sbibm_jax.data import TaskDataset

        def fake_load(repo, name=None, **kw):
            if name and name.endswith("_posterior"):
                return _fake_posterior()
            return _fake_main_dataset()

        monkeypatch.setattr("sbibm_jax.data.dataset.load_dataset", fake_load)
        ds = TaskDataset("two_moons")
        obs, samples = ds.get_reference(num_observation=2)
        assert np.asarray(obs).shape == (2,)
        assert np.asarray(samples).shape == (10, 2)
        assert np.asarray(ds.get_true_parameters(2)).shape == (2,)

    def test_get_reference_without_posterior_raises(self, monkeypatch, tmp_path):
        # has_reference False -> informative error
        import json
        meta = {"t": {"dim_theta": 2, "dim_x": 2, "data_kind": "vector",
                      "data_shape": [2], "splits": {"train": 8, "validation": 4, "test": 4},
                      "has_reference": False, "num_observations": 1, "stats": None}}
        p = tmp_path / "metadata.json"
        p.write_text(json.dumps(meta))
        monkeypatch.setattr("sbibm_jax.data.dataset.hf_hub_download", lambda **kw: str(p))
        monkeypatch.setattr("sbibm_jax.data.dataset.load_dataset",
                            lambda repo, name=None, **kw: _fake_main_dataset())
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("t")
        with pytest.raises(ValueError, match="no reference"):
            ds.get_reference(1)


class TestNormalizeMethods:
    def test_normalize_x_roundtrip(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons", normalize=True)
        x = np.ones((3, 2, 1), np.float32)
        back = ds.unnormalize_x(ds.normalize_x(x))
        np.testing.assert_allclose(np.asarray(back), x, atol=1e-5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/data/test_dataset.py::TestReference tests/data/test_dataset.py::TestNormalizeMethods -v`
Expected: FAIL (`get_reference` / `normalize_x` not defined).

- [ ] **Step 3: Add reference + normalize methods**

```python
# src/sbibm_jax/data/dataset.py  (add to TaskDataset)
from sbibm_jax.data.process import _stat_array


class TaskDataset:
    # ...

    def _ensure_posterior(self):
        if not self.has_reference:
            raise ValueError(
                f"Task {self.name!r} has no reference posterior "
                f"({self.name}_posterior config absent)."
            )
        if self._posterior is None:
            self._posterior = load_dataset(
                self.repo, f"{self.name}_posterior",
            ).with_format("numpy")["reference_posterior"]
        return self._posterior

    def _check_obs(self, num_observation):
        if not 1 <= num_observation <= self.num_observations:
            raise ValueError(
                f"num_observation must be in [1, {self.num_observations}]."
            )

    def get_reference(self, num_observation=1):
        self._check_obs(num_observation)
        post = self._ensure_posterior()
        i = num_observation - 1
        return post["observations"][i], post["reference_samples"][i]

    def get_true_parameters(self, num_observation=1):
        self._check_obs(num_observation)
        return self._ensure_posterior()["true_parameters"][num_observation - 1]

    def _norm(self, arr, mean, std):
        m = _stat_array(mean, self.dtype)
        s = _stat_array(std, self.dtype)
        return (np.asarray(arr, dtype=self.dtype) - m) / s

    def _unnorm(self, arr, mean, std):
        m = _stat_array(mean, self.dtype)
        s = _stat_array(std, self.dtype)
        return np.asarray(arr, dtype=self.dtype) * s + m

    def normalize_theta(self, theta):
        return self._norm(theta, self.theta_mean, self.theta_std)

    def unnormalize_theta(self, theta):
        return self._unnorm(theta, self.theta_mean, self.theta_std)

    def normalize_x(self, x):
        return self._norm(x, self.x_mean, self.x_std)

    def unnormalize_x(self, x):
        return self._unnorm(x, self.x_mean, self.x_std)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/data/test_dataset.py -v`
Expected: PASS (all dataset tests).

- [ ] **Step 5: Lint + commit**

Run: `uv run flake8 src/sbibm_jax/data/`
Expected: no new violations.

```bash
git add src/sbibm_jax/data/dataset.py tests/data/test_dataset.py
git commit -m "feat(data): reference access + normalize_theta/x methods

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

# PHASE 3 — Masks submodule `sbibm_jax.data.masks`

Opt-in graph/causal masks for the 5 analytical base tasks. Self-contained; the core loader never imports it. Its own PR.

### Task 3.1: Verbatim port of `graph.py` and condition-mask samplers

**Files:**
- Create: `src/sbibm_jax/data/masks/__init__.py`, `src/sbibm_jax/data/masks/graph.py`, `src/sbibm_jax/data/masks/condition.py`
- Source (copy verbatim): `/lhome/ific/a/aamerio/data/github/GenSBI-examples/src/gensbi_examples/graph.py` and `.../mask.py`
- Test: `tests/data/test_masks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_masks.py
"""Masks: graph transforms, condition samplers, base + edge masks."""

import jax.numpy as jnp
import numpy as np
import pytest


class TestGraph:
    def test_moralize_symmetric(self):
        from sbibm_jax.data.masks.graph import moralize
        adj = jnp.array([[0, 1], [0, 0]], dtype=jnp.bool_)
        m = moralize(adj)
        assert bool(jnp.all(m == m.T))


class TestConditionSamplers:
    def test_posterior_mask_shape(self):
        from sbibm_jax.data.masks.condition import get_condition_mask_fn
        fn = get_condition_mask_fn("posterior")
        import jax
        m = fn(jax.random.PRNGKey(0), num_samples=3, theta_dim=2, x_dim=4)
        assert np.asarray(m).shape == (3, 6)
        assert bool(jnp.all(~m[:, :2]))   # theta not conditioned
        assert bool(jnp.all(m[:, 2:]))    # x conditioned
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/data/test_masks.py::TestGraph tests/data/test_masks.py::TestConditionSamplers -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbibm_jax.data.masks'`.

- [ ] **Step 3: Copy the two source files verbatim**

```bash
mkdir -p src/sbibm_jax/data/masks
cp /lhome/ific/a/aamerio/data/github/GenSBI-examples/src/gensbi_examples/graph.py \
   src/sbibm_jax/data/masks/graph.py
cp /lhome/ific/a/aamerio/data/github/GenSBI-examples/src/gensbi_examples/mask.py \
   src/sbibm_jax/data/masks/condition.py
```

Create a temporary package init so the imports resolve (the full public API lands in Task 3.4):

```python
# src/sbibm_jax/data/masks/__init__.py
from sbibm_jax.data.masks.condition import get_condition_mask_fn  # noqa: F401
from sbibm_jax.data.masks import graph  # noqa: F401
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/data/test_masks.py::TestGraph tests/data/test_masks.py::TestConditionSamplers -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/data/masks/ tests/data/test_masks.py
git commit -m "feat(masks): port graph.py + condition-mask samplers from GenSBI

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 3.2: Per-task base adjacency builders (`base.py`)

Parameterized by `dim_theta`/`dim_x` at call time (no hardcoded dims). Ported verbatim from GenSBI's `get_base_mask_fn` bodies for the 5 supported tasks.

**Files:**
- Create: `src/sbibm_jax/data/masks/base.py`
- Test: `tests/data/test_masks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_masks.py  (add)
class TestBaseMasks:
    @pytest.mark.parametrize("name,dt,dx", [
        ("two_moons", 2, 2), ("gaussian_linear", 10, 10),
        ("gaussian_linear_uniform", 10, 10), ("gaussian_mixture", 2, 2),
        ("slcp", 5, 8),
    ])
    def test_base_mask_shape(self, name, dt, dx):
        from sbibm_jax.data.masks.base import get_base_mask_fn
        fn = get_base_mask_fn(name, dim_theta=dt, dim_x=dx)
        node_ids = jnp.arange(dt + dx)
        mask = fn(node_ids, None)
        assert np.asarray(mask).shape == (dt + dx, dt + dx)
        assert np.asarray(mask).dtype == np.bool_

    def test_unsupported_raises(self):
        from sbibm_jax.data.masks.base import get_base_mask_fn
        with pytest.raises(NotImplementedError):
            get_base_mask_fn("bernoulli_glm", dim_theta=10, dim_x=10)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/data/test_masks.py::TestBaseMasks -v`
Expected: FAIL (`base` module / `get_base_mask_fn` missing).

- [ ] **Step 3: Implement `base.py`**

Port each task's adjacency from GenSBI `tasks.py` (`TwoMoons`/`GaussianLinear`/`GaussianLinearUniform`/`GaussianMixture`/`SLCP` `.get_base_mask_fn`), replacing `self.dim_obs`/`self.dim_cond` with the `dim_theta`/`dim_x` arguments:

```python
# src/sbibm_jax/data/masks/base.py
"""Per-task base adjacency masks, parameterized by (dim_theta, dim_x).

Each builder returns a `base_mask_fn(node_ids, node_meta_data)` closure that
sub-indexes a boolean (dim_theta+dim_x)^2 adjacency. Ported from GenSBI's
get_base_mask_fn bodies (self.dim_obs->dim_theta, self.dim_cond->dim_x).
"""

import jax
import jax.numpy as jnp


def _require_equal_dims(name, dim_theta, dim_x):
    # These two block layouts (ones((dim_x, dim_theta)) beside a
    # (dim_theta, dim_x) block; eye(dim_x) in the off-diagonal) only assemble
    # when dim_theta == dim_x — true for all four tasks that use them. Guard so
    # an unequal-dim caller gets a clear error instead of a cryptic jnp.block
    # shape failure.
    if dim_theta != dim_x:
        raise NotImplementedError(
            f"base mask for {name!r} assumes dim_theta == dim_x; "
            f"got {dim_theta} != {dim_x}."
        )


def _two_moons_like(dim_theta, dim_x):
    # two_moons / gaussian_mixture: x depends on all theta (lower-tri block).
    _require_equal_dims("two_moons/gaussian_mixture", dim_theta, dim_x)
    thetas_mask = jnp.eye(dim_theta, dtype=jnp.bool_)
    x_mask = jnp.tril(jnp.ones((dim_theta, dim_x), dtype=jnp.bool_))
    return jnp.block([
        [thetas_mask, jnp.zeros((dim_theta, dim_x))],
        [jnp.ones((dim_x, dim_theta)), x_mask],
    ]).astype(jnp.bool_)


def _gaussian_linear_like(dim_theta, dim_x):
    _require_equal_dims("gaussian_linear", dim_theta, dim_x)
    thetas_mask = jnp.eye(dim_theta, dtype=jnp.bool_)
    x_i_mask = jnp.eye(dim_x, dtype=jnp.bool_)
    return jnp.block([
        [thetas_mask, jnp.zeros((dim_theta, dim_x))],
        [jnp.eye(dim_x), x_i_mask],
    ]).astype(jnp.bool_)


def _slcp(dim_theta, dim_x):
    thetas_mask = jnp.eye(dim_theta, dtype=jnp.bool_)
    x_i_dim = dim_x // 4
    x_i_mask = jax.scipy.linalg.block_diag(
        *tuple([jnp.tril(jnp.ones((x_i_dim, x_i_dim), dtype=jnp.bool_))] * 4)
    )
    return jnp.block([
        [thetas_mask, jnp.zeros((dim_theta, dim_x))],
        [jnp.ones((dim_x, dim_theta)), x_i_mask],
    ]).astype(jnp.bool_)


_BUILDERS = {
    "two_moons": _two_moons_like,
    "gaussian_mixture": _two_moons_like,
    "gaussian_linear": _gaussian_linear_like,
    "gaussian_linear_uniform": _gaussian_linear_like,
    "slcp": _slcp,
}


def get_base_mask_fn(name, *, dim_theta, dim_x):
    """Return base_mask_fn(node_ids, node_meta_data) for `name`."""
    builder = _BUILDERS.get(name)
    if builder is None:
        raise NotImplementedError(
            f"Task {name!r} has no base mask "
            f"(supported: {sorted(_BUILDERS)})."
        )
    base_mask = builder(dim_theta, dim_x)

    def base_mask_fn(node_ids, node_meta_data):
        return base_mask[node_ids, :][:, node_ids]

    return base_mask_fn
```

> Note: `gaussian_nonlinear` is a registry alias of `SLCP`; if mask support for it is wanted, map it to `_slcp` too. Left out per YAGNI until a consumer asks.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/data/test_masks.py::TestBaseMasks -v`
Expected: PASS (6 cases).

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/data/masks/base.py tests/data/test_masks.py
git commit -m "feat(masks): per-task base adjacency builders (param by dims)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 3.3: `get_edge_mask_fn` + public masks API

**Files:**
- Modify: `src/sbibm_jax/data/masks/__init__.py` (full API), create `src/sbibm_jax/data/masks/edge.py`
- Test: `tests/data/test_masks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_masks.py  (add)
class TestEdgeMasks:
    @pytest.mark.parametrize("variant", ["undirected", "directed", "none"])
    def test_edge_variants(self, variant):
        from sbibm_jax.data.masks import get_edge_mask_fn
        fn = get_edge_mask_fn("two_moons", variant, dim_theta=2, dim_x=2)
        node_ids = jnp.arange(4)
        cond = jnp.zeros(4, dtype=jnp.bool_)
        out = fn(node_ids, cond)
        if variant == "none":
            assert out is None
        else:
            assert np.asarray(out).shape == (4, 4)

    def test_unknown_variant_raises(self):
        from sbibm_jax.data.masks import get_edge_mask_fn
        with pytest.raises(NotImplementedError):
            get_edge_mask_fn("two_moons", "bogus", dim_theta=2, dim_x=2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/data/test_masks.py::TestEdgeMasks -v`
Expected: FAIL (`get_edge_mask_fn` not exported).

- [ ] **Step 3: Implement `edge.py` and finalize `__init__.py`**

Port GenSBI's `get_edge_mask_fn` dispatch (`tasks.py:312-350`), parameterized by dims:

```python
# src/sbibm_jax/data/masks/edge.py
"""Edge-mask transforms applied to a task's base mask (ported from GenSBI)."""

from sbibm_jax.data.masks.base import get_base_mask_fn
from sbibm_jax.data.masks.graph import (
    faithfull_mask,
    min_faithfull_mask,
    moralize,
)


def get_edge_mask_fn(name, variant="undirected", *, dim_theta, dim_x):
    base_mask_fn = get_base_mask_fn(name, dim_theta=dim_theta, dim_x=dim_x)
    v = variant.lower()

    if v == "faithfull":
        def fn(node_id, condition_mask, meta_data=None):
            return faithfull_mask(base_mask_fn(node_id, meta_data), condition_mask)
        return fn
    if v == "min_faithfull":
        def fn(node_id, condition_mask, meta_data=None):
            return min_faithfull_mask(base_mask_fn(node_id, meta_data), condition_mask)
        return fn
    if v == "undirected":
        def fn(node_id, condition_mask, meta_data=None):
            return moralize(base_mask_fn(node_id, meta_data))
        return fn
    if v == "directed":
        def fn(node_id, condition_mask, meta_data=None):
            return base_mask_fn(node_id, meta_data)
        return fn
    if v == "none":
        return lambda node_id, condition_mask, *a, **k: None
    raise NotImplementedError(f"Unknown edge-mask variant {variant!r}.")
```

```python
# src/sbibm_jax/data/masks/__init__.py
"""Opt-in graph/causal masks for the analytical base tasks.

Not imported by the core loader. Build base masks with get_base_mask_fn and
edge-transformed masks with get_edge_mask_fn; sample conditioning masks with
get_condition_mask_fn.
"""

from sbibm_jax.data.masks.base import get_base_mask_fn
from sbibm_jax.data.masks.condition import get_condition_mask_fn
from sbibm_jax.data.masks.edge import get_edge_mask_fn

__all__ = ["get_base_mask_fn", "get_edge_mask_fn", "get_condition_mask_fn"]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/data/test_masks.py -v`
Expected: PASS (all mask tests).

- [ ] **Step 5: Lint + commit**

Run: `uv run flake8 src/sbibm_jax/data/masks/`
Expected: no new violations (the ported `graph.py` may carry pre-existing style noise — judge by new violations only; do not reformat the verbatim port beyond what's needed to import cleanly).

```bash
git add src/sbibm_jax/data/masks/ tests/data/test_masks.py
git commit -m "feat(masks): get_edge_mask_fn + public masks API

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 3.4: Document the loader in CLAUDE.md + README

**Files:**
- Modify: `CLAUDE.md` (add a `sbibm_jax.data` paragraph next to the `sbibm_jax.hf` one)

- [ ] **Step 1: Add a paragraph to CLAUDE.md**

Document: the `[loader]` extra; `from sbibm_jax.data import TaskDataset`; default repo is TEST; `kind="conditional"`/`"joint"` (joint vector-only) with GenSBI tokenization; loaders `get_{train,val,test}_loader`; stats read from `metadata.json`; reference via the `{task}_posterior` config; masks are opt-in via `sbibm_jax.data.masks` and cover only the 5 analytical base tasks.

- [ ] **Step 2: Run the whole suite once more**

Run: `uv run pytest -q -n 2`
Expected: PASS (full suite green, including `tests/hf` and `tests/data`).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document sbibm_jax.data loader subpackage

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §0 rename → Phase 0 (Task 0.1). ✓
- §1 subpackage `[loader]` extra + skeleton → Task 2.1. ✓
- §2 `TaskDataset` API (dims, stats, loaders, reference, normalize, repo default, max_workers≤8) → Tasks 2.3–2.5. ✓
- §3 generic collate, tokenization, joint-vector-only → Task 2.2. ✓
- §4 gen-time stats + `hf_stats_axes` + metadata + driver → Tasks 1.1–1.4. ✓
- §5 masks (graph port, base, edge, condition samplers) → Tasks 3.1–3.3. ✓
- §6 default repo TEST → Task 2.3 (`test_default_repo_is_test`). ✓
- §7 reference `{task}_posterior` → Task 2.5. ✓
- Testing section (construction, processing, stats modes, masks, reference, grain smoke) → covered across the test files. ✓

**Placeholder scan:** No "TBD"/"implement later"; verbatim ports name the exact source files (`graph.py`, `mask.py`) — fully specified content, not a placeholder.

**Type/name consistency:** `dim_theta`/`dim_x` used everywhere post-Phase 0; `make_collate(kind, data_kind, normalize, stats, dtype)` signature matches its callers in `dataset.py`; `_stat_array` defined in `process.py` and reused in `dataset.py`; `get_base_mask_fn(name, *, dim_theta, dim_x)` and `get_edge_mask_fn(name, variant, *, dim_theta, dim_x)` signatures match their tests; `build_dataset` now returns `"stats"`, consumed by `upload_dataset` and the driver.

**Known cross-phase dependency:** Phase 2/3 import `from sbibm_jax.hf import config` (for `TEST_REPO`/`DEFAULT_REPO`); `sbibm_jax.hf` requires the `[hf]` extra, which the `[loader]` extra's deps (`datasets`, `huggingface_hub`) already satisfy. If importing `config` is undesirable, copy the two repo constants into `sbibm_jax/data` — but reuse is preferred (DRY).
