# Gravitational Waves task + x/theta metadata symmetry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a file-backed `gravitational_waves` benchmark (mock simulator,
dataset published from pre-generated `.npz` shards) and make the dataset
`metadata.json` schema symmetric between `x` and `theta`.

**Architecture:** Three pieces. (A) A breaking metadata schema refactor:
`{x_kind, x_shape, theta_kind, theta_shape, …}`, dropping derived
`dim_x`/`dim_theta`. (B) A mock GW `Task` plus two standalone scripts (torch→npz
converter, npz→Hub uploader) — no tests, minimal surface, since GW is reworked
once a simulator exists. (C) Loader test coverage for the already-working
`timeseries` conditional path and the new schema fields.

**Tech Stack:** Python 3.12, JAX/NumPyro, `datasets`/`huggingface_hub`, `grain`,
`pytest` (`uv run pytest`), `flake8`. PyTorch only in the one-off converter
(`torch` dependency group).

**Conventions for the executor:**
- Run tests with `uv run pytest <path> -v`; lint with `uv run flake8 src tests`.
  Judge lint by *new* violations vs HEAD (bare flake8 is never clean — default
  line length 79, pre-existing E501).
- Every commit message ends with the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- The `[hf]` and `[loader]` extras must be installed (`uv sync --all-groups`);
  `tests/hf` and `tests/data` self-skip otherwise.

---

## File structure

**Schema refactor (piece A):**
- `src/sbibm_jax/hf/exporter.py` — rename `data_kind`→`x_kind`,
  `data_shape`→`x_shape`; add `theta_kind`/`theta_shape`.
- `src/sbibm_jax/hf/registry.py` — `DATA_KIND_REGISTRY`→`X_KIND_REGISTRY`; read
  `hf_x_kind`/`hf_x_shape`/`hf_theta_kind`/`hf_theta_shape`.
- `src/sbibm_jax/hf/metadata.py` — emit new keys; drop `dim_x`/`dim_theta`.
- `src/sbibm_jax/tasks/{toy_lensing,gaussian_random_field}/task.py` —
  `hf_data_*`→`hf_x_*`.
- `src/sbibm_jax/data/process.py` — `make_collate` `data_kind`→`x_kind`,
  add `theta_kind`, joint guard.
- `src/sbibm_jax/data/dataset.py` — parse new schema; derive `dim_x`/`dim_theta`.

**GW task + scripts (piece B):**
- `src/sbibm_jax/tasks/gravitational_waves/{__init__.py,task.py}` — mock task.
- `src/sbibm_jax/tasks/__init__.py` — registry branch.
- `scripts/convert_gw_to_npz.py` — torch→npz converter (no test).
- `scripts/make_gw_dataset.py` — npz→Hub uploader (no test).
- `scripts/make_dataset.py` — skip `hf_external` tasks.

**Tests (piece C + refactor updates):**
- `tests/hf/{test_exporter,test_registry,test_metadata}.py`,
  `tests/tasks/test_gaussian_random_field.py` (refactor updates).
- `tests/data/{test_process,test_dataset}.py` (refactor + timeseries coverage).
- `tests/tasks/test_gravitational_waves.py` (new).
- `tests/hf/test_driver.py` (external-skip test).

**Docs:** `CLAUDE.md`.

---

## Task 1: HF export-layer schema (exporter + registry + metadata + task hints)

Coordinated rename — these files are coupled (registry constructs the exporter,
metadata reads it), so they change together and the suite returns green at the
end of the task.

**Files:**
- Modify: `src/sbibm_jax/hf/exporter.py`, `src/sbibm_jax/hf/registry.py`,
  `src/sbibm_jax/hf/metadata.py`,
  `src/sbibm_jax/tasks/toy_lensing/task.py`,
  `src/sbibm_jax/tasks/gaussian_random_field/task.py`
- Test: `tests/hf/test_exporter.py`, `tests/hf/test_registry.py`,
  `tests/hf/test_metadata.py`, `tests/tasks/test_gaussian_random_field.py`

- [ ] **Step 1: Update `tests/hf/test_exporter.py`** (full new content)

```python
"""Tests for DatasetExporter and its x-kind subclasses."""

import numpy as np
import pytest
from datasets import Array2D, Features, List, Value

from sbibm_jax import get_task
from sbibm_jax.hf.exporter import (
    DatasetExporter,
    ImageExporter,
    TimeSeriesExporter,
    VectorExporter,
)


class TestVectorExporter:
    def test_x_kind(self):
        task = get_task("gaussian_linear")
        exp = VectorExporter(task, train_size=4, val_size=2, test_size=2)
        assert exp.x_kind == "vector"

    def test_theta_defaults(self):
        task = get_task("gaussian_linear")
        exp = VectorExporter(task, train_size=4, val_size=2, test_size=2)
        assert exp.theta_kind == "vector"
        assert tuple(exp.theta_shape) == (task.dim_theta,)
        assert tuple(exp.x_shape) == (task.dim_x,)

    def test_features_schema(self):
        task = get_task("gaussian_linear")
        exp = VectorExporter(task, train_size=4, val_size=2, test_size=2)
        feats = exp.features()
        assert isinstance(feats, Features)
        assert isinstance(feats["xs"], List)
        assert isinstance(feats["xs"].feature, Value)
        assert feats["xs"].feature.dtype == "float32"
        assert isinstance(feats["thetas"], List)
        assert feats["thetas"].feature.dtype == "float32"

    def test_shape_x_identity(self):
        task = get_task("gaussian_linear")
        exp = VectorExporter(task, train_size=4, val_size=2, test_size=2)
        flat = np.zeros((3, task.dim_x), dtype=np.float32)
        out = exp.shape_x(flat)
        assert out.shape == (3, task.dim_x)
        assert out.dtype == np.float32

    def test_base_class_is_abstract(self):
        task = get_task("gaussian_linear")
        with pytest.raises(TypeError):
            DatasetExporter(task, train_size=1, val_size=1, test_size=1)


class TestImageExporter:
    def test_x_kind(self):
        task = get_task("gaussian_random_field", field_size=8)
        exp = ImageExporter(
            task, x_shape=(8, 8), train_size=4, val_size=2, test_size=2,
        )
        assert exp.x_kind == "image"

    def test_features_schema(self):
        task = get_task("gaussian_random_field", field_size=8)
        exp = ImageExporter(
            task, x_shape=(8, 8), train_size=4, val_size=2, test_size=2,
        )
        feats = exp.features()
        assert isinstance(feats["xs"], Array2D)
        assert feats["xs"].shape == (8, 8)
        assert feats["xs"].dtype == "float32"

    def test_shape_x_reshapes_to_image(self):
        task = get_task("gaussian_random_field", field_size=8)
        exp = ImageExporter(
            task, x_shape=(8, 8), train_size=4, val_size=2, test_size=2,
        )
        flat = np.zeros((5, 8 * 8), dtype=np.float32)
        out = exp.shape_x(flat)
        assert out.shape == (5, 8, 8)
        assert out.dtype == np.float32

    def test_rejects_non_2d_shape(self):
        task = get_task("gaussian_random_field", field_size=8)
        with pytest.raises(ValueError, match="2-D x_shape"):
            ImageExporter(
                task, x_shape=(8, 8, 3), train_size=4, val_size=2, test_size=2,
            )


class TestTimeSeriesExporter:
    def test_x_kind(self):
        task = get_task("gaussian_linear")  # any task; x_shape is what counts
        exp = TimeSeriesExporter(
            task, x_shape=(5, 2), train_size=4, val_size=2, test_size=2,
        )
        assert exp.x_kind == "timeseries"

    def test_features_schema(self):
        task = get_task("gaussian_linear")
        exp = TimeSeriesExporter(
            task, x_shape=(5, 2), train_size=4, val_size=2, test_size=2,
        )
        feats = exp.features()
        assert isinstance(feats["xs"], Array2D)
        assert feats["xs"].shape == (5, 2)
        assert feats["xs"].dtype == "float32"

    def test_shape_x_reshapes_to_tc(self):
        task = get_task("gaussian_linear")
        exp = TimeSeriesExporter(
            task, x_shape=(5, 2), train_size=4, val_size=2, test_size=2,
        )
        flat = np.zeros((7, 5 * 2), dtype=np.float32)
        out = exp.shape_x(flat)
        assert out.shape == (7, 5, 2)
        assert out.dtype == np.float32

    def test_rejects_non_2d_shape(self):
        task = get_task("gaussian_linear")
        with pytest.raises(ValueError, match="2-D x_shape"):
            TimeSeriesExporter(
                task, x_shape=(5,), train_size=4, val_size=2, test_size=2,
            )
```

- [ ] **Step 2: Update `tests/hf/test_registry.py`** (full new content)

```python
"""Tests for hf.registry.get_exporter."""

import pytest

from sbibm_jax import get_task
from sbibm_jax.hf import config
from sbibm_jax.hf.exporter import ImageExporter, VectorExporter
from sbibm_jax.hf.registry import X_KIND_REGISTRY, get_exporter


class TestRegistry:
    def test_known_kinds(self):
        assert set(X_KIND_REGISTRY) == {"vector", "image", "timeseries"}

    def test_default_is_vector(self):
        task = get_task("gaussian_linear")
        exp = get_exporter(task)
        assert isinstance(exp, VectorExporter)
        assert exp.train_size == config.DEFAULT_SPLIT_SIZES["train"]
        assert exp.val_size == config.DEFAULT_SPLIT_SIZES["validation"]
        assert exp.test_size == config.DEFAULT_SPLIT_SIZES["test"]

    def test_split_size_overrides(self):
        task = get_task("gaussian_linear")
        exp = get_exporter(task, train_size=10, val_size=2, test_size=2)
        assert exp.train_size == 10
        assert exp.val_size == 2
        assert exp.test_size == 2

    def test_hf_x_kind_hint_selects_image(self):
        task = get_task("gaussian_linear")
        task.hf_x_kind = "image"
        task.hf_x_shape = (4, 4)
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert isinstance(exp, ImageExporter)
        assert exp.x_shape == (4, 4)

    def test_resample_invalid_hint_propagates(self):
        task = get_task("gaussian_linear")
        task.hf_resample_invalid = True
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert exp.resample_invalid is True

    def test_unknown_x_kind_raises(self):
        task = get_task("gaussian_linear")
        task.hf_x_kind = "tensor4d"
        with pytest.raises(ValueError, match="Unknown x_kind"):
            get_exporter(task)


class TestRegistryRealTasks:
    def test_grf_selects_image_exporter(self):
        task = get_task("gaussian_random_field", field_size=8)
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert isinstance(exp, ImageExporter)
        assert exp.x_shape == (8, 8)

    def test_grf_default_field_size_32(self):
        task = get_task("gaussian_random_field")
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert isinstance(exp, ImageExporter)
        assert exp.x_shape == (32, 32)

    def test_toy_lensing_selects_image_exporter(self):
        task = get_task("toy_lensing", resolution=8)
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert isinstance(exp, ImageExporter)
        assert exp.x_shape == (8, 8)

    def test_toy_lensing_default_resolution_32(self):
        task = get_task("toy_lensing")
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert isinstance(exp, ImageExporter)
        assert exp.x_shape == (32, 32)

    def test_grf_256_selects_image_exporter(self):
        task = get_task("gaussian_random_field_256")
        exp = get_exporter(task)
        assert isinstance(exp, ImageExporter)
        assert exp.x_shape == (256, 256)
        assert exp.train_size == 100_000
        assert exp.val_size == 10_000
        assert exp.test_size == 10_000


class TestResampleHints:
    @pytest.mark.parametrize(
        "name",
        ["lotka_volterra", "sir", "beer_molbiosystems"],
    )
    def test_resample_invalid_set(self, name):
        try:
            task = get_task(name)
        except ImportError as e:
            pytest.skip(f"task {name} requires an extra: {e}")
        assert getattr(task, "hf_resample_invalid", False) is True
```

- [ ] **Step 3: Update the schema-asserting methods in `tests/hf/test_metadata.py`**

Replace `test_vector_task_schema`, `test_image_task_schema`, and
`test_grf_256_image_schema` with:

```python
    def test_vector_task_schema(self):
        meta = make_metadata(["gaussian_linear"])
        m = meta["gaussian_linear"]
        assert m["x_kind"] == "vector"
        assert m["x_shape"] == [10]
        assert m["theta_kind"] == "vector"
        assert m["theta_shape"] == [10]
        assert "dim_x" not in m
        assert "dim_theta" not in m
        assert m["splits"] == {
            "train": 1_000_000,
            "validation": 10_000,
            "test": 10_000,
        }
        assert m["has_reference"] is True
        assert m["num_observations"] == 10

    def test_image_task_schema(self):
        meta = make_metadata(["gaussian_random_field"])
        m = meta["gaussian_random_field"]
        assert m["x_kind"] == "image"
        assert m["x_shape"] == [32, 32]
        assert m["theta_kind"] == "vector"
        assert m["theta_shape"] == [2]
        assert m["has_reference"] is False
        assert m["splits"] == {
            "train": 100_000,
            "validation": 10_000,
            "test": 10_000,
        }

    def test_grf_256_image_schema(self):
        meta = make_metadata(["gaussian_random_field_256"])
        m = meta["gaussian_random_field_256"]
        assert m["x_kind"] == "image"
        assert m["x_shape"] == [256, 256]
        assert m["theta_shape"] == [2]
        assert "dim_x" not in m
        assert m["has_reference"] is False
        assert m["splits"] == {
            "train": 100_000, "validation": 10_000, "test": 10_000,
        }
```

(Leave `test_partial_override_keeps_task_cap`, `test_writes_json_file`,
`TestMergeMetadata`, and `TestMetadataStats` unchanged.)

- [ ] **Step 4: Update the GRF hint assertions in `tests/tasks/test_gaussian_random_field.py:272-273`**

```python
        assert task.hf_x_kind == "image"
        assert task.hf_x_shape == (256, 256)
```

- [ ] **Step 5: Run the updated tests to verify they FAIL**

Run: `uv run pytest tests/hf/test_exporter.py tests/hf/test_registry.py tests/hf/test_metadata.py "tests/tasks/test_gaussian_random_field.py::TestHighResVariant::test_registry_alias" -v`
Expected: FAIL (e.g. `AttributeError: 'VectorExporter' object has no attribute 'x_kind'`, `ImportError: cannot import name 'X_KIND_REGISTRY'`, and the GRF test's `assert task.hf_x_kind == "image"` failing).

- [ ] **Step 6: Rewrite `src/sbibm_jax/hf/exporter.py`** (full new content)

```python
"""Dataset exporter base + x-kind subclasses.

The exporter owns the HF Features schema and the shape transformation from the
flat (batch, dim_x) block emitted by sbibm_jax.hf.generate to the
native-storage shape (vector / image / time-series). Concrete subclasses
override three things: `x_kind`, `x_feature()`, and `shape_x(x_flat)`.

theta is stored as a flat float32 vector for every current task; `theta_kind`
/ `theta_shape` are recorded (for metadata symmetry with x and to future-proof
a non-vector theta), but theta_feature() stays List(float32) for now.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from datasets import Array2D, Features, List, Value

from sbibm_jax.hf import config
from sbibm_jax.tasks.task import Task


class DatasetExporter(ABC):
    """Abstract base. Subclasses set `x_kind` and override `x_feature`/`shape_x`."""

    x_kind: str = ""
    theta_kind: str = "vector"

    def __init__(
        self,
        task: Task,
        *,
        train_size: int,
        val_size: int,
        test_size: int,
        dtype=config.DEFAULT_DTYPE,
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        max_factor: float = config.DEFAULT_MAX_FACTOR,
        x_shape: Optional[Tuple[int, ...]] = None,
        theta_shape: Optional[Tuple[int, ...]] = None,
        theta_kind: Optional[str] = None,
        resample_invalid: bool = False,
    ):
        if type(self) is DatasetExporter:
            raise TypeError(
                "DatasetExporter is abstract; instantiate VectorExporter / "
                "ImageExporter / TimeSeriesExporter."
            )
        self.task = task
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.max_factor = max_factor
        self.x_shape = x_shape if x_shape is not None else (task.dim_x,)
        self.theta_shape = (
            theta_shape if theta_shape is not None else (task.dim_theta,)
        )
        if theta_kind is not None:
            self.theta_kind = theta_kind
        self.resample_invalid = resample_invalid

    @abstractmethod
    def x_feature(self):
        """The HF feature for a single `x` row."""

    @abstractmethod
    def shape_x(self, x_flat: np.ndarray) -> np.ndarray:
        """Reshape a flat `(batch, dim_x)` block to native storage shape."""

    def theta_feature(self):
        return List(Value("float32"))

    def features(self) -> Features:
        return Features({"xs": self.x_feature(), "thetas": self.theta_feature()})


class VectorExporter(DatasetExporter):
    """Flat parameter-vector storage. The default for analytical tasks."""

    x_kind = "vector"

    def x_feature(self):
        return List(Value("float32"))

    def shape_x(self, x_flat: np.ndarray) -> np.ndarray:
        return np.asarray(x_flat, dtype=self.dtype)


class ImageExporter(DatasetExporter):
    """`Array2D`-backed image storage (e.g. Gaussian Random Field 32x32)."""

    x_kind = "image"

    def __init__(self, task: Task, *, x_shape: Tuple[int, int], **kwargs):
        if x_shape is None or len(x_shape) != 2:
            raise ValueError(
                f"ImageExporter requires a 2-D x_shape (H, W); got {x_shape!r}."
            )
        super().__init__(task, x_shape=x_shape, **kwargs)

    def x_feature(self):
        return Array2D(shape=self.x_shape, dtype="float32")

    def shape_x(self, x_flat: np.ndarray) -> np.ndarray:
        h, w = self.x_shape
        return np.asarray(x_flat, dtype=self.dtype).reshape(-1, h, w)


class TimeSeriesExporter(DatasetExporter):
    """`Array2D`-backed time-series storage (T, C)."""

    x_kind = "timeseries"

    def __init__(self, task: Task, *, x_shape: Tuple[int, int], **kwargs):
        if x_shape is None or len(x_shape) != 2:
            raise ValueError(
                f"TimeSeriesExporter requires a 2-D x_shape (T, C); got {x_shape!r}."
            )
        super().__init__(task, x_shape=x_shape, **kwargs)

    def x_feature(self):
        return Array2D(shape=self.x_shape, dtype="float32")

    def shape_x(self, x_flat: np.ndarray) -> np.ndarray:
        t, c = self.x_shape
        return np.asarray(x_flat, dtype=self.dtype).reshape(-1, t, c)
```

- [ ] **Step 7: Rewrite `src/sbibm_jax/hf/registry.py`** (full new content)

```python
"""x-kind registry: hf_x_kind hint -> exporter class."""

from typing import Type

from sbibm_jax.hf import config
from sbibm_jax.hf.exporter import (
    DatasetExporter,
    ImageExporter,
    TimeSeriesExporter,
    VectorExporter,
)
from sbibm_jax.tasks.task import Task

X_KIND_REGISTRY: dict[str, Type[DatasetExporter]] = {
    "vector": VectorExporter,
    "image": ImageExporter,
    "timeseries": TimeSeriesExporter,
}


def get_exporter(
    task: Task,
    *,
    train_size: int | None = None,
    val_size: int | None = None,
    test_size: int | None = None,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    max_factor: float = config.DEFAULT_MAX_FACTOR,
    dtype=config.DEFAULT_DTYPE,
) -> DatasetExporter:
    """Build the exporter for `task`, honouring its hf_* hint attributes.

    Hint attributes (all optional, read via getattr with safe defaults):
        hf_x_kind:           "vector" | "image" | "timeseries" (default "vector")
        hf_x_shape:          tuple[int, ...]  (default (task.dim_x,))
        hf_theta_kind:       "vector"         (default "vector")
        hf_theta_shape:      tuple[int, ...]  (default (task.dim_theta,))
        hf_resample_invalid: bool             (default False)
        hf_split_sizes:      dict             (default config.DEFAULT_SPLIT_SIZES)

    Explicit train/val/test_size keyword arguments override the task hint and
    the global default. Unknown x_kind raises ValueError.
    """
    x_kind = getattr(task, "hf_x_kind", "vector")
    if x_kind not in X_KIND_REGISTRY:
        raise ValueError(
            f"Unknown x_kind {x_kind!r} for task {task.name!r}; "
            f"known kinds: {sorted(X_KIND_REGISTRY)}."
        )

    x_shape = getattr(task, "hf_x_shape", (task.dim_x,))
    theta_kind = getattr(task, "hf_theta_kind", "vector")
    theta_shape = getattr(task, "hf_theta_shape", (task.dim_theta,))
    resample_invalid = getattr(task, "hf_resample_invalid", False)
    task_split_sizes = getattr(task, "hf_split_sizes", config.DEFAULT_SPLIT_SIZES)

    ts = train_size if train_size is not None else task_split_sizes["train"]
    vs = val_size if val_size is not None else task_split_sizes["validation"]
    es = test_size if test_size is not None else task_split_sizes["test"]

    cls = X_KIND_REGISTRY[x_kind]
    kwargs = dict(
        train_size=ts,
        val_size=vs,
        test_size=es,
        chunk_size=chunk_size,
        max_factor=max_factor,
        dtype=dtype,
        theta_kind=theta_kind,
        theta_shape=tuple(theta_shape),
        resample_invalid=resample_invalid,
    )
    if cls is VectorExporter:
        return cls(task, **kwargs)
    return cls(task, x_shape=tuple(x_shape), **kwargs)
```

- [ ] **Step 8: Update the metadata block in `src/sbibm_jax/hf/metadata.py`**

Replace the `Schema per task:` docstring block and the `meta[name] = {…}` dict
in `make_metadata`. New docstring schema lines:

```python
    Schema per task:
        x_kind:         "vector" | "image" | "timeseries"
        x_shape:        list[int]
        theta_kind:     "vector" | "image" | "timeseries"
        theta_shape:    list[int]
        splits:         dict[str, int]
        has_reference:  bool
        num_observations: int
        stats:          dict | None
```

New dict (replaces the current `meta[name] = {…}`):

```python
        meta[name] = {
            "x_kind": exporter.x_kind,
            "x_shape": list(exporter.x_shape),
            "theta_kind": exporter.theta_kind,
            "theta_shape": list(exporter.theta_shape),
            "splits": {
                "train": exporter.train_size,
                "validation": exporter.val_size,
                "test": exporter.test_size,
            },
            "has_reference": load_reference(task, exporter) is not None,
            "num_observations": int(task.num_observations),
            "stats": (stats_by_task or {}).get(name),
        }
```

- [ ] **Step 9: Rename the x-hints in the two image tasks**

In `src/sbibm_jax/tasks/toy_lensing/task.py`:
```python
        self.hf_x_kind = "image"
        self.hf_x_shape = (resolution, resolution)
```
In `src/sbibm_jax/tasks/gaussian_random_field/task.py`:
```python
        self.hf_x_kind = "image"
        self.hf_x_shape = (field_size, field_size)
```
(Leave their `hf_stats_axes` and `hf_split_sizes` lines unchanged.)

- [ ] **Step 10: Run the full hf + task suite to verify PASS**

Run: `uv run pytest tests/hf tests/tasks -q`
Expected: PASS (the `petab`-only tests may skip without the `pypesto` extra —
that is fine, treat skips as pass).

- [ ] **Step 11: Lint**

Run: `uv run flake8 src/sbibm_jax/hf src/sbibm_jax/tasks/toy_lensing src/sbibm_jax/tasks/gaussian_random_field`
Expected: no new violations vs HEAD.

- [ ] **Step 12: Commit**

```bash
git add src/sbibm_jax/hf/exporter.py src/sbibm_jax/hf/registry.py \
  src/sbibm_jax/hf/metadata.py src/sbibm_jax/tasks/toy_lensing/task.py \
  src/sbibm_jax/tasks/gaussian_random_field/task.py \
  tests/hf/test_exporter.py tests/hf/test_registry.py \
  tests/hf/test_metadata.py tests/tasks/test_gaussian_random_field.py
git commit -m "refactor(hf): symmetric x/theta metadata schema (x_kind/x_shape/theta_*)"
```

---

## Task 2: Loader schema + collate (process.py + dataset.py)

`dataset.py` calls `make_collate`, so the two change together to stay green.

**Files:**
- Modify: `src/sbibm_jax/data/process.py`, `src/sbibm_jax/data/dataset.py`
- Test: `tests/data/test_process.py`, `tests/data/test_dataset.py`

- [ ] **Step 1: Update `tests/data/test_process.py`** (full new content)

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
        collate = make_collate(kind="conditional", x_kind="vector")
        theta, x = collate(_batch())
        assert theta.shape == (2, 3, 1)
        assert x.shape == (2, 5, 1)

    def test_normalize_applies_stats(self):
        stats = {"theta_mean": [[1.0, 1.0, 1.0]], "theta_std": [[1.0, 1.0, 1.0]],
                 "x_mean": [[1.0, 1.0, 1.0, 1.0, 1.0]],
                 "x_std": [[2.0, 2.0, 2.0, 2.0, 2.0]]}
        collate = make_collate(kind="conditional", x_kind="vector",
                               normalize=True, stats=stats)
        theta, x = collate(_batch())
        # x all ones, mean 1, std 2 -> 0
        np.testing.assert_allclose(np.asarray(x), 0.0, atol=1e-6)


class TestJoint:
    def test_joint_concats_along_feature_axis(self):
        collate = make_collate(kind="joint", x_kind="vector")
        out = collate(_batch())
        assert out.shape == (2, 3 + 5, 1)

    def test_joint_raises_for_image_x(self):
        with pytest.raises(ValueError, match="joint.*vector"):
            make_collate(kind="joint", x_kind="image")
```

- [ ] **Step 2: Update `tests/data/test_dataset.py` for the new schema**

Replace `_fake_metadata` with:

```python
def _fake_metadata(tmp_path):
    meta = {
        "two_moons": {
            "x_kind": "vector", "x_shape": [2],
            "theta_kind": "vector", "theta_shape": [2],
            "splits": {"train": 8, "validation": 4, "test": 4},
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
```

Replace `test_dims_and_stats_parsed` with:

```python
    def test_dims_and_stats_parsed(self, patched):
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("two_moons")
        assert ds.dim_theta == 2
        assert ds.dim_x == 2
        assert ds.x_kind == "vector"
        assert tuple(ds.x_shape) == (2,)
        assert ds.theta_kind == "vector"
        assert tuple(ds.theta_shape) == (2,)
        assert np.array(ds.theta_mean).shape == (1, 2)
```

In `test_get_reference_without_posterior_raises`, replace the inline `meta`
dict with:

```python
        meta = {"t": {"x_kind": "vector", "x_shape": [2],
                      "theta_kind": "vector", "theta_shape": [2],
                      "splits": {"train": 8, "validation": 4, "test": 4},
                      "has_reference": False, "num_observations": 1,
                      "stats": None}}
```

- [ ] **Step 3: Run the updated tests to verify they FAIL**

Run: `uv run pytest tests/data/test_process.py tests/data/test_dataset.py -v`
Expected: FAIL (`make_collate() got an unexpected keyword argument 'x_kind'`;
`KeyError: 'data_kind'` or `AttributeError: 'TaskDataset' object has no attribute 'x_kind'`).

- [ ] **Step 4: Rewrite `src/sbibm_jax/data/process.py`** (full new content)

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


def make_collate(
    *, kind, x_kind, theta_kind="vector", normalize=False, stats=None,
    dtype=np.float32,
):
    """Return a collate fn mapping a {'thetas','xs'} batch to model-ready arrays.

    kind="conditional" -> (theta, x); kind="joint" -> concat([theta, x], axis=1).
    Joint is vector-only (both x and theta). `stats` is the metadata stats dict
    (native-reduced).
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

- [ ] **Step 5: Edit `src/sbibm_jax/data/dataset.py` — parse the new schema**

Replace this exact block:

```python
        self.dim_theta = int(entry["dim_theta"])
        self.dim_x = int(entry["dim_x"])
        self.data_kind = entry["data_kind"]
        self.data_shape = tuple(entry["data_shape"])
        self.num_observations = int(entry["num_observations"])
        self.has_reference = bool(entry["has_reference"])
        self.dim_joint = self.dim_theta + self.dim_x if kind == "joint" else None
```

with:

```python
        self.x_kind = entry["x_kind"]
        self.x_shape = tuple(entry["x_shape"])
        self.theta_kind = entry["theta_kind"]
        self.theta_shape = tuple(entry["theta_shape"])
        self.dim_x = int(np.prod(self.x_shape))
        self.dim_theta = int(np.prod(self.theta_shape))
        self.num_observations = int(entry["num_observations"])
        self.has_reference = bool(entry["has_reference"])
        self.dim_joint = self.dim_theta + self.dim_x if kind == "joint" else None
```

- [ ] **Step 6: Edit `src/sbibm_jax/data/dataset.py` — the `make_collate` call**

Replace this exact block:

```python
        self._collate = make_collate(
            kind=kind, data_kind=self.data_kind,
            normalize=normalize, stats=stats, dtype=dtype,
        )
```

with:

```python
        self._collate = make_collate(
            kind=kind, x_kind=self.x_kind, theta_kind=self.theta_kind,
            normalize=normalize, stats=stats, dtype=dtype,
        )
```

- [ ] **Step 7: Run the updated tests to verify PASS**

Run: `uv run pytest tests/data/test_process.py tests/data/test_dataset.py -v`
Expected: PASS.

- [ ] **Step 8: Lint + commit**

```bash
uv run flake8 src/sbibm_jax/data
git add src/sbibm_jax/data/process.py src/sbibm_jax/data/dataset.py \
  tests/data/test_process.py tests/data/test_dataset.py
git commit -m "refactor(data): TaskDataset/make_collate read x/theta schema, derive dims"
```

---

## Task 3: Loader timeseries coverage

Lock in the (already-generic) `timeseries` conditional path and the symmetric
joint guard. New tests only — no source change.

**Files:**
- Test: `tests/data/test_process.py`, `tests/data/test_dataset.py`

- [ ] **Step 1: Add a timeseries class to `tests/data/test_process.py`**

Append:

```python
class TestTimeSeriesConditional:
    def _ts_batch(self):
        xs = np.empty((2, 4, 2), dtype=np.float32)
        xs[..., 0] = 1.0
        xs[..., 1] = 3.0
        return {"thetas": np.ones((2, 2), dtype=np.float32), "xs": xs}

    def test_tokenizes_with_channel(self):
        collate = make_collate(kind="conditional", x_kind="timeseries")
        theta, x = collate(self._ts_batch())
        assert theta.shape == (2, 2, 1)
        assert x.shape == (2, 4, 2, 1)

    def test_normalize_per_channel(self):
        stats = {"theta_mean": [[1.0, 1.0]], "theta_std": [[1.0, 1.0]],
                 "x_mean": [[[1.0, 3.0]]], "x_std": [[[1.0, 2.0]]]}
        collate = make_collate(kind="conditional", x_kind="timeseries",
                               normalize=True, stats=stats)
        _, x = collate(self._ts_batch())
        # channel 0: (1-1)/1=0 ; channel 1: (3-3)/2=0
        np.testing.assert_allclose(np.asarray(x), 0.0, atol=1e-6)

    def test_joint_raises_for_timeseries(self):
        with pytest.raises(ValueError, match="joint.*vector"):
            make_collate(kind="joint", x_kind="timeseries")

    def test_joint_raises_for_non_vector_theta(self):
        with pytest.raises(ValueError, match="joint.*vector"):
            make_collate(kind="joint", x_kind="vector", theta_kind="image")
```

- [ ] **Step 2: Add a timeseries loader class to `tests/data/test_dataset.py`**

Append (uses the module-level `Dataset`/`DatasetDict`/`json`/`np` imports):

```python
def _fake_ts_dataset():
    rows = {"thetas": np.zeros((8, 2), np.float32),
            "xs": np.ones((8, 5, 2), np.float32)}
    d = Dataset.from_dict(rows)
    return DatasetDict({"train": d, "validation": d, "test": d})


class TestTimeSeriesLoader:
    def _meta(self, tmp_path, x_mean, x_std):
        meta = {"gw": {
            "x_kind": "timeseries", "x_shape": [5, 2],
            "theta_kind": "vector", "theta_shape": [2],
            "splits": {"train": 8, "validation": 8, "test": 8},
            "has_reference": False, "num_observations": 1,
            "stats": {
                "theta_mean": [[0.0, 0.0]], "theta_std": [[1.0, 1.0]],
                "x_mean": x_mean, "x_std": x_std,
                "theta_axes": [0], "x_axes": [0, 1],
            },
        }}
        p = tmp_path / "metadata.json"
        p.write_text(json.dumps(meta))
        return str(p)

    def test_conditional_shapes(self, monkeypatch, tmp_path):
        mp = self._meta(tmp_path, [[[0.0, 0.0]]], [[[1.0, 1.0]]])
        monkeypatch.setattr(
            "sbibm_jax.data.dataset.hf_hub_download", lambda **kw: mp)
        monkeypatch.setattr(
            "sbibm_jax.data.dataset.load_dataset",
            lambda repo, name=None, **kw: _fake_ts_dataset())
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("gw", kind="conditional")
        assert ds.x_kind == "timeseries"
        assert ds.dim_x == 10
        assert tuple(ds.x_shape) == (5, 2)
        theta, x = next(iter(ds.get_train_loader(batch_size=4)))
        assert np.asarray(theta).shape == (4, 2, 1)
        assert np.asarray(x).shape == (4, 5, 2, 1)

    def test_normalize_broadcasts_per_channel(self, monkeypatch, tmp_path):
        # x all ones; x_mean 1, x_std 1 -> 0 after normalization.
        mp = self._meta(tmp_path, [[[1.0, 1.0]]], [[[1.0, 1.0]]])
        monkeypatch.setattr(
            "sbibm_jax.data.dataset.hf_hub_download", lambda **kw: mp)
        monkeypatch.setattr(
            "sbibm_jax.data.dataset.load_dataset",
            lambda repo, name=None, **kw: _fake_ts_dataset())
        from sbibm_jax.data import TaskDataset
        ds = TaskDataset("gw", normalize=True)
        _, x = next(iter(ds.get_train_loader(batch_size=4)))
        np.testing.assert_allclose(np.asarray(x), 0.0, atol=1e-6)
```

- [ ] **Step 3: Run the new tests to verify PASS**

Run: `uv run pytest "tests/data/test_process.py::TestTimeSeriesConditional" "tests/data/test_dataset.py::TestTimeSeriesLoader" -v`
Expected: PASS (the generic collate/loader already handle timeseries).

- [ ] **Step 4: Commit**

```bash
git add tests/data/test_process.py tests/data/test_dataset.py
git commit -m "test(data): cover timeseries conditional loading + joint guard"
```

---

## Task 4: GravitationalWaves task class

**Files:**
- Create: `src/sbibm_jax/tasks/gravitational_waves/__init__.py`,
  `src/sbibm_jax/tasks/gravitational_waves/task.py`
- Modify: `src/sbibm_jax/tasks/__init__.py`
- Test: `tests/tasks/test_gravitational_waves.py`

- [ ] **Step 1: Write the failing test `tests/tasks/test_gravitational_waves.py`**

```python
"""Tests for the file-backed Gravitational Waves task."""

import jax
import pytest

from sbibm_jax.tasks.gravitational_waves.task import GravitationalWaves


class TestMetadata:
    def test_dims_and_name(self):
        task = GravitationalWaves()
        assert task.dim_theta == 2
        assert task.dim_x == 8192 * 2
        assert task.name == "gravitational_waves"
        assert task.name_display == "Gravitational Waves"

    def test_hf_hints(self):
        task = GravitationalWaves()
        assert task.hf_x_kind == "timeseries"
        assert task.hf_x_shape == (8192, 2)
        assert task.hf_stats_axes == {"theta": (0,), "x": (0, 1)}
        assert task.hf_external is True


class TestMocksRaise:
    def test_get_prior_raises(self):
        task = GravitationalWaves()
        with pytest.raises(NotImplementedError):
            task.get_prior(jax.random.PRNGKey(0), num_samples=1)

    def test_get_simulator_raises(self):
        task = GravitationalWaves()
        with pytest.raises(NotImplementedError):
            task.get_simulator(jax.random.PRNGKey(0))

    def test_reference_posterior_raises(self):
        task = GravitationalWaves()
        with pytest.raises(NotImplementedError):
            task._sample_reference_posterior(
                jax.random.PRNGKey(0), num_samples=10, num_observation=1,
            )


class TestRegistry:
    def test_get_task_returns_instance(self):
        from sbibm_jax import get_task
        task = get_task("gravitational_waves")
        assert isinstance(task, GravitationalWaves)

    def test_available_tasks_includes_gw(self):
        from sbibm_jax import get_available_tasks
        assert "gravitational_waves" in get_available_tasks()
```

- [ ] **Step 2: Run it to verify it FAILS**

Run: `uv run pytest tests/tasks/test_gravitational_waves.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named
'sbibm_jax.tasks.gravitational_waves'`).

- [ ] **Step 3: Create `src/sbibm_jax/tasks/gravitational_waves/__init__.py`**

```python
```
(Empty file.)

- [ ] **Step 4: Create `src/sbibm_jax/tasks/gravitational_waves/task.py`**

```python
"""Gravitational Waves task: file-backed time-series inference (no simulator).

Unlike the other tasks, gravitational_waves has no simulator yet: its data is a
fixed corpus of pre-generated (theta, x) rows published by
scripts/make_gw_dataset.py. get_prior / get_simulator / the reference sampler
raise NotImplementedError; consume the dataset via
sbibm_jax.data.TaskDataset("gravitational_waves").

theta = 2 parameters; x = a (8192, 2) two-channel strain time series.
"""

from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp

from sbibm_jax.tasks.task import Task

_MSG = (
    "The gravitational_waves {what} is not available yet. This is a "
    "file-backed task: load the published dataset via "
    "sbibm_jax.data.TaskDataset('gravitational_waves'). The simulator/prior "
    "will be added in a future rework."
)


class GravitationalWaves(Task):
    def __init__(self):
        super().__init__(
            dim_theta=2,
            dim_x=8192 * 2,
            name=Path(__file__).parent.name,
            name_display="Gravitational Waves",
            num_observations=1,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            path=Path(__file__).parent.absolute(),
        )
        # HF export hints: stored as a (T, C) time series via TimeSeriesExporter.
        self.hf_x_kind = "timeseries"
        self.hf_x_shape = (8192, 2)
        # per-feature theta -> (1, 2); per-channel x -> (1, 1, 2).
        self.hf_stats_axes = {"theta": (0,), "x": (0, 1)}
        # File-backed: skipped by make_dataset.py (the simulator is a mock);
        # uploaded by scripts/make_gw_dataset.py.
        self.hf_external = True

    def get_prior(
        self, key: jax.random.PRNGKey, num_samples: int = 1
    ) -> jnp.ndarray:
        raise NotImplementedError(_MSG.format(what="prior"))

    def get_simulator(self, key: jax.random.PRNGKey, max_calls: Optional[int] = None):
        raise NotImplementedError(_MSG.format(what="simulator"))

    def get_labels_parameters(self) -> List[str]:
        return ["theta_1", "theta_2"]

    def unflatten_data(self, data: jnp.ndarray) -> jnp.ndarray:
        return data.reshape(-1, 8192, 2)

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        raise NotImplementedError(_MSG.format(what="reference posterior"))
```

- [ ] **Step 5: Register in `src/sbibm_jax/tasks/__init__.py`**

Add this branch inside `get_task()` (e.g. after the `toy_lensing` branch, before
the final `else`):

```python
    elif task_name == "gravitational_waves":
        from sbibm_jax.tasks.gravitational_waves.task import GravitationalWaves
        return GravitationalWaves(*args, **kwargs)
```

- [ ] **Step 6: Run the test to verify PASS**

Run: `uv run pytest tests/tasks/test_gravitational_waves.py -v`
Expected: PASS.

- [ ] **Step 7: Lint + commit**

```bash
uv run flake8 src/sbibm_jax/tasks/gravitational_waves src/sbibm_jax/tasks/__init__.py
git add src/sbibm_jax/tasks/gravitational_waves tests/tasks/test_gravitational_waves.py \
  src/sbibm_jax/tasks/__init__.py
git commit -m "feat(tasks): add file-backed gravitational_waves task (mock simulator)"
```

---

## Task 5: `make_dataset.py` skips external tasks

**Files:**
- Modify: `scripts/make_dataset.py`
- Test: `tests/hf/test_driver.py`

- [ ] **Step 1: Write the failing test in `tests/hf/test_driver.py`**

Append:

```python
def test_external_task_skipped(monkeypatch, tmp_path):
    mod = _load_driver()
    out = tmp_path / "metadata.json"

    monkeypatch.setattr(mod, "fetch_remote_metadata", lambda repo: {})

    captured = {}

    def fake_upload_metadata(path, repo):
        captured["meta"] = json.loads(open(path).read())

    uploaded = []

    monkeypatch.setattr(mod, "upload_metadata", fake_upload_metadata)
    monkeypatch.setattr(
        mod, "upload_dataset", lambda repo, name, **o: uploaded.append(name))

    mod.main(["--tasks", "gaussian_linear", "gravitational_waves",
              "--metadata-path", str(out)])

    # gravitational_waves is hf_external -> not generated, not in metadata.
    assert uploaded == ["gaussian_linear"]
    assert "gravitational_waves" not in captured["meta"]
    assert "gaussian_linear" in captured["meta"]
```

- [ ] **Step 2: Run it to verify it FAILS**

Run: `uv run pytest tests/hf/test_driver.py::test_external_task_skipped -v`
Expected: FAIL (`gravitational_waves` appears in `uploaded` / in the metadata,
or `upload_dataset` is called for it).

- [ ] **Step 3: Add the import in `scripts/make_dataset.py`**

Change:
```python
from sbibm_jax import get_available_tasks
```
to:
```python
from sbibm_jax import get_available_tasks, get_task
```

- [ ] **Step 4: Add the skip filter in `scripts/make_dataset.py:main`**

Immediately after the `if args.all: … elif args.tasks: … else: … sys.exit(2)`
block that resolves `task_names` (and before the `repo = …` banner line), insert:

```python
    # File-backed tasks (hf_external=True) are published by their own scripts
    # (e.g. scripts/make_gw_dataset.py), not this simulator-driven path. Skip
    # them so --all does not invoke a mock simulator.
    kept = []
    for name in task_names:
        if getattr(get_task(name), "hf_external", False):
            logging.info(
                "Skipping %s (external/file-backed; use its dedicated "
                "upload script).", name,
            )
        else:
            kept.append(name)
    task_names = kept
```

- [ ] **Step 5: Run the test to verify PASS**

Run: `uv run pytest tests/hf/test_driver.py -v`
Expected: PASS (all driver tests, including the new one).

- [ ] **Step 6: Lint + commit**

```bash
uv run flake8 scripts/make_dataset.py
git add scripts/make_dataset.py tests/hf/test_driver.py
git commit -m "feat(make_dataset): skip hf_external tasks (file-backed uploads)"
```

---

## Task 6: torch→npz converter script (no test)

A one-off, torch-only converter. No automated test (per scope); it is verified
by running it on the real shards.

**Files:**
- Create: `scripts/convert_gw_to_npz.py`

- [ ] **Step 1: Create `scripts/convert_gw_to_npz.py`**

```python
"""One-time conversion of the Gravitational Waves .pt shards to .npz.

The package and the upload script (scripts/make_gw_dataset.py) stay torch-free;
this converter is the only torch consumer. Run it with the torch group:

    uv run --group torch python scripts/convert_gw_to_npz.py \
        --data-dir /lhome/ific/a/aamerio/data/GW

For each shard i it writes (alongside the .pt files by default):
    thetas_i.npz  with key "data", shape (N, 2),       float32
    xs_i.npz      with key "data", shape (N, 8192, 2),  float32

xs is stored time-first (N, 8192, 2). The raw .pt may be channel-first
(N, 2, 8192) (as in the original gw_dataset.py) or already time-first; the
orientation is detected and transposed only when needed.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

T_LEN = 8192
N_CH = 2


def _to_time_first(xs: np.ndarray) -> np.ndarray:
    if xs.ndim != 3:
        raise ValueError(f"xs must be 3-D (N, *, *); got shape {xs.shape}.")
    _, a, b = xs.shape
    if (a, b) == (T_LEN, N_CH):
        return xs
    if (a, b) == (N_CH, T_LEN):
        return np.transpose(xs, (0, 2, 1))
    raise ValueError(
        f"xs shape {xs.shape} is neither (N, {T_LEN}, {N_CH}) nor "
        f"(N, {N_CH}, {T_LEN}); cannot determine orientation."
    )


def convert_shard(data_dir: Path, out_dir: Path, i: int) -> None:
    theta = torch.load(
        data_dir / f"thetas_{i}.pt", map_location="cpu", weights_only=True,
    ).numpy()
    if theta.ndim != 2 or theta.shape[1] != 2:
        raise ValueError(f"thetas_{i} must be (N, 2); got {theta.shape}.")
    theta = theta.astype(np.float32)

    xs = torch.load(
        data_dir / f"xs_{i}.pt", map_location="cpu", weights_only=True,
    ).numpy()
    xs = _to_time_first(xs).astype(np.float32)

    if theta.shape[0] != xs.shape[0]:
        raise ValueError(
            f"shard {i}: theta rows {theta.shape[0]} != xs rows {xs.shape[0]}."
        )

    np.savez_compressed(out_dir / f"thetas_{i}.npz", data=theta)
    np.savez_compressed(out_dir / f"xs_{i}.npz", data=xs)
    print(
        f"  shard {i}: thetas {theta.shape} {theta.dtype}, "
        f"xs {xs.shape} {xs.dtype}"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/lhome/ific/a/aamerio/data/GW")
    p.add_argument("--out-dir", default=None, help="Defaults to --data-dir.")
    p.add_argument("--num-shards", type=int, default=10)
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {args.num_shards} GW shards: {data_dir} -> {out_dir}")
    for i in range(args.num_shards):
        convert_shard(data_dir, out_dir, i)
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint**

Run: `uv run flake8 scripts/convert_gw_to_npz.py`
Expected: no new violations. (Do not run the script — the `.pt` files are still
downloading; it is executed manually later.)

- [ ] **Step 3: Commit**

```bash
git add scripts/convert_gw_to_npz.py
git commit -m "feat(scripts): add GW .pt->.npz converter (torch group, one-off)"
```

---

## Task 7: npz→Hub uploader script (no test)

Standalone uploader mirroring `make_dataset.py`'s flow, reusing the package's HF
helpers. No automated test (per scope); verified by `--dry-run` and a real run.

**Files:**
- Create: `scripts/make_gw_dataset.py`

- [ ] **Step 1: Create `scripts/make_gw_dataset.py`**

```python
"""Build and upload the file-backed Gravitational Waves dataset.

GW has no simulator, so it is NOT built by scripts/make_dataset.py (that path
runs a prior + simulator). This script reads the pre-generated .npz shards
(produced by scripts/convert_gw_to_npz.py) and pushes them to the Hub with a
metadata.json block compatible with sbibm_jax.data.TaskDataset.

Uploads target the TEST repo by default; pass --prod for production.

    uv run python scripts/make_gw_dataset.py --data-dir /lhome/ific/a/aamerio/data/GW
    uv run python scripts/make_gw_dataset.py --dry-run    # metadata.json only
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from datasets import Dataset

from sbibm_jax import get_task
from sbibm_jax.hf import (
    config,
    fetch_remote_metadata,
    make_metadata,
    merge_metadata,
    upload_metadata,
)
from sbibm_jax.hf.registry import get_exporter
from sbibm_jax.hf.stats import StatsAccumulator, resolve_stats_axes

TASK_NAME = "gravitational_waves"
log = logging.getLogger("make_gw_dataset")


def _load_shard(data_dir: Path, i: int):
    theta = np.asarray(np.load(data_dir / f"thetas_{i}.npz")["data"], np.float32)
    xs = np.asarray(np.load(data_dir / f"xs_{i}.npz")["data"], np.float32)
    if theta.shape[0] != xs.shape[0]:
        raise ValueError(f"shard {i}: theta/xs row mismatch.")
    return theta, xs


def _validate(theta, xs, x_shape, dim_theta):
    if tuple(xs.shape[1:]) != tuple(x_shape):
        raise ValueError(f"xs native shape {xs.shape[1:]} != {tuple(x_shape)}.")
    if theta.shape[1] != dim_theta:
        raise ValueError(f"theta dim {theta.shape[1]} != {dim_theta}.")


def _rows(theta, xs):
    for i in range(theta.shape[0]):
        yield {"xs": xs[i], "thetas": theta[i]}


def build_splits(data_dir, *, val_size, num_shards, exporter, dim_theta):
    """Mirror-original split policy, streaming one shard at a time.

    train = shards 0..n-2 minus the last val_size pool rows;
    validation = the last val_size pool rows (tail of shard n-2);
    test = shard n-1 (whole).
    """
    data_dir = Path(data_dir)
    x_shape = exporter.x_shape
    features = exporter.features()
    last_pool = num_shards - 2  # last shard contributing to the train pool

    # Validation = tail of the last pool shard.
    th_lp, xs_lp = _load_shard(data_dir, last_pool)
    _validate(th_lp, xs_lp, x_shape, dim_theta)
    if xs_lp.shape[0] < val_size:
        raise ValueError(
            f"last pool shard {last_pool} has {xs_lp.shape[0]} rows < "
            f"val_size={val_size}."
        )
    val_theta, val_xs = th_lp[-val_size:], xs_lp[-val_size:]

    def train_gen():
        for i in range(0, num_shards - 1):
            theta, xs = _load_shard(data_dir, i)
            _validate(theta, xs, x_shape, dim_theta)
            if i == last_pool:
                theta, xs = theta[:-val_size], xs[:-val_size]
            yield from _rows(theta, xs)

    def val_gen():
        yield from _rows(val_theta, val_xs)

    def test_gen():
        theta, xs = _load_shard(data_dir, num_shards - 1)
        _validate(theta, xs, x_shape, dim_theta)
        yield from _rows(theta, xs)

    train = Dataset.from_generator(train_gen, features=features)
    val = Dataset.from_generator(val_gen, features=features)
    test = Dataset.from_generator(test_gen, features=features)
    sizes = {"train": len(train), "validation": len(val), "test": len(test)}
    return {"train": train, "validation": val, "test": test}, sizes


def compute_stats(task, train):
    theta_axes, x_axes = resolve_stats_axes(task)
    acc = StatsAccumulator(theta_axes, x_axes)
    for batch in train.with_format("numpy").iter(batch_size=128):
        acc.update(np.asarray(batch["thetas"]), np.asarray(batch["xs"]))
    return acc.result()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/lhome/ific/a/aamerio/data/GW")
    p.add_argument("--num-shards", type=int, default=10)
    p.add_argument("--val-size", type=int, default=512)
    p.add_argument("--prod", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--metadata-path", default="metadata.json")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    repo = config.DEFAULT_REPO if args.prod else config.TEST_REPO
    label = "PRODUCTION" if args.prod else "TEST"
    print(f"Target repo: {repo}  ({label})")

    metadata_path = Path(args.metadata_path)

    if args.dry_run:
        make_metadata([TASK_NAME], output_path=metadata_path)
        print(f"Wrote {metadata_path} (dry run — no upload, no stats).")
        return

    task = get_task(TASK_NAME)
    exporter = get_exporter(task)
    datasets, sizes = build_splits(
        args.data_dir, val_size=args.val_size, num_shards=args.num_shards,
        exporter=exporter, dim_theta=task.dim_theta,
    )
    print(f"Split sizes: {sizes}")

    stats = compute_stats(task, datasets["train"])

    for split in ("train", "validation", "test"):
        datasets[split].push_to_hub(
            repo, config_name=TASK_NAME, split=split, private=False,
        )

    local_meta = make_metadata(
        [TASK_NAME],
        train_size=sizes["train"],
        val_size=sizes["validation"],
        test_size=sizes["test"],
        stats_by_task={TASK_NAME: stats},
    )
    remote_meta = fetch_remote_metadata(repo)
    merged = merge_metadata(remote_meta, local_meta)
    metadata_path.write_text(json.dumps(merged, indent=4))
    upload_metadata(str(metadata_path), repo)
    metadata_path.unlink(missing_ok=True)
    print(f"Uploaded metadata and removed local {metadata_path}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-check `--dry-run` (writes metadata only, no network)**

Run: `uv run python scripts/make_gw_dataset.py --dry-run --metadata-path /tmp/gw_meta.json`
Expected: prints `Target repo: …-test  (TEST)`; `/tmp/gw_meta.json` contains a
`gravitational_waves` block with `"x_kind": "timeseries"`, `"x_shape": [8192, 2]`,
`"theta_shape": [2]`, `"stats": null`.

Verify: `uv run python -c "import json;m=json.load(open('/tmp/gw_meta.json'))['gravitational_waves'];print(m['x_kind'],m['x_shape'],m['theta_shape'])"`
Expected: `timeseries [8192, 2] [2]`

- [ ] **Step 3: Lint + commit**

```bash
uv run flake8 scripts/make_gw_dataset.py
git add scripts/make_gw_dataset.py
git commit -m "feat(scripts): add standalone GW dataset uploader (npz -> Hub)"
```

---

## Task 8: Docs (CLAUDE.md)

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the metadata-schema mentions in `CLAUDE.md`**

In the `sbibm_jax.data` loader paragraph, change
`(dims, data_kind/data_shape, splits, stats)` to
`(x_kind/x_shape, theta_kind/theta_shape, splits, stats)`.

In the HF-export paragraph, change `declaring hf_data_kind switches it to an
image or time-series storage shape` to `declaring hf_x_kind switches it to an
image or time-series storage shape`.

In the conventions bullet, change `The image tasks gaussian_random_field and
toy_lensing declare hf_data_kind="image" plus a 2-D hf_data_shape` to
`… declare hf_x_kind="image" plus a 2-D hf_x_shape`.

- [ ] **Step 2: Add a GW / external-task note to `CLAUDE.md`**

Append this paragraph at the end of the HF-export section (after the
`gaussian_random_field_256` paragraph):

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: x/theta metadata schema + file-backed gravitational_waves"
```

---

## Task 9: Full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest tests -q`
Expected: PASS (petab-only tests may skip without the `pypesto` extra — fine).

- [ ] **Step 2: Lint the touched trees**

Run: `uv run flake8 src tests scripts`
Expected: no *new* violations relative to HEAD (pre-existing E501 etc. are
baseline; compare against `git stash && uv run flake8 … ; git stash pop` if unsure).

- [ ] **Step 3: Confirm the GW dry-run end-to-end once more**

Run: `uv run python scripts/make_gw_dataset.py --dry-run --metadata-path /tmp/gw_meta2.json && uv run python -c "import json;print(json.load(open('/tmp/gw_meta2.json'))['gravitational_waves'])"`
Expected: a GW block with `x_kind=timeseries`, `x_shape=[8192,2]`,
`theta_shape=[2]`, `has_reference=false`, `stats=null`.

---

## Notes on the manual (post-merge) data upload

These are **not** plan tasks (no code), but the dataset only materializes when
run by hand once the `.pt` files finish downloading:

1. `uv run --group torch python scripts/convert_gw_to_npz.py --data-dir /lhome/ific/a/aamerio/data/GW`
2. `uv run python scripts/make_gw_dataset.py --data-dir /lhome/ific/a/aamerio/data/GW`
   (TEST repo; add `--prod` for production once verified).
3. Verify: `TaskDataset("gravitational_waves").get_train_loader(8)` yields
   `(theta (8,2,1), x (8,8192,2,1))`.
