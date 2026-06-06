# HuggingFace Dataset Pipeline (`sbibm_jax.hf`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the `SBI-benchmarks-data` dataset-creation pipeline into the `sbibm-jax` repo as a new importable subpackage `sbibm_jax.hf`, driven entirely by the JAX `Task` API — no torch, no original `sbibm`.

**Architecture:** A general abstract pipeline with one `DatasetExporter` base class and three data-kind subclasses (`VectorExporter`, `ImageExporter`, `TimeSeriesExporter`). Per-task specialization is done via tiny class-attribute hints (`hf_data_kind`, `hf_data_shape`, `hf_resample_invalid`, `hf_split_sizes`) read by a `registry.get_exporter(task)` helper. Generation (`generate.py`) is chunked JAX with stable per-task seeding and a configurable validity policy (default: finite-assert; opt-in: rejection-resample to exact N). HF-network calls (`upload.py`) are isolated for monkey-patching. Public API + a thin `scripts/make_dataset.py` driver replace the old `SBI-benchmarks-data/make_dataset.py`.

**Tech Stack:** JAX (`jax`, `jax.numpy`, `jax.random.fold_in`), `numpy`, `datasets` (Features / Array2D / Value / List / Dataset.from_generator), `huggingface_hub` (`upload_file`, `push_to_hub`), `pytest` (CPU-forced via `pyproject.toml`, `-n 2` via xdist), `pytest-mock`, Python `zlib.crc32` (for stable hashing of task names).

**Reference spec:** `docs/superpowers/specs/2026-06-06-huggingface-dataset-pipeline-design.md`

**Reference (original) code:** `/lhome/ific/a/aamerio/data/github/SBI-benchmarks-data/sbi_benchmarks/{sbi_tasks.py, hf_hub.py}` and `make_dataset.py`.

---

## File Structure

**Create (in `sbibm-jax`):**

- `src/sbibm_jax/hf/__init__.py` — package marker + public API re-exports + `[hf]` extra import guard.
- `src/sbibm_jax/hf/config.py` — defaults: split sizes (`1_000_000 / 10_000 / 10_000`), `DEFAULT_REPO = "aurelio-amerio/SBI-benchmarks"`, `DEFAULT_DTYPE = numpy.float32`, `DEFAULT_CHUNK_SIZE = 4096`, `DEFAULT_MAX_FACTOR = 10.0`, `DEFAULT_MASTER_SEED = 0`.
- `src/sbibm_jax/hf/exporter.py` — `DatasetExporter` base + `VectorExporter` / `ImageExporter` / `TimeSeriesExporter`.
- `src/sbibm_jax/hf/registry.py` — `DATA_KIND_REGISTRY` + `get_exporter(task, **overrides)`.
- `src/sbibm_jax/hf/generate.py` — `iter_chunks(...)` (streaming) + `generate_samples(...)` (materializing wrapper) + `derive_task_keys(...)` (stable seeding).
- `src/sbibm_jax/hf/reference.py` — `load_reference(task, exporter) -> Optional[datasets.Dataset]`.
- `src/sbibm_jax/hf/metadata.py` — `make_metadata(task_names, *, output_path=None, **opts) -> dict`.
- `src/sbibm_jax/hf/upload.py` — `upload_metadata(path, repo)` + `upload_dataset(repo, task_name, **opts)`.
- `src/sbibm_jax/hf/build.py` — `build_dataset(task_name, **opts)` orchestration (used by `__init__.py` re-export).
- `scripts/make_dataset.py` — thin driver script.
- `tests/hf/__init__.py` — empty package marker.
- `tests/hf/conftest.py` — `pytest.importorskip("datasets")` at module scope (skips the whole subdir if the `[hf]` extra isn't installed).
- `tests/hf/test_exporter.py` — exporter base + 3 subclasses.
- `tests/hf/test_registry.py` — `get_exporter` dispatch + hint reading.
- `tests/hf/test_generate.py` — seeding, chunking, validity policies (default + resample).
- `tests/hf/test_reference.py` — reference block presence + absence.
- `tests/hf/test_metadata.py` — `make_metadata` schema.
- `tests/hf/test_upload.py` — monkeypatched `push_to_hub` / `upload_file`.
- `tests/hf/test_build_dataset.py` — end-to-end integration on `gaussian_linear` (vector) + `gaussian_random_field` (image), tiny sizes.

**Modify:**

- `pyproject.toml` — add the `[hf]` optional extra.
- `src/sbibm_jax/tasks/gaussian_random_field/task.py` — set `self.hf_data_kind = "image"` and `self.hf_data_shape = (field_size, field_size)` in `__init__`.
- `src/sbibm_jax/tasks/lotka_volterra/task.py` — set `self.hf_resample_invalid = True` in `__init__`.
- `src/sbibm_jax/tasks/sir/task.py` — set `self.hf_resample_invalid = True` in `__init__`.
- `src/sbibm_jax/tasks/beer_molbiosystems/task.py` — set `self.hf_resample_invalid = True` in `__init__`.

All other tasks (`gaussian_linear`, `gaussian_linear_uniform`, `gaussian_mixture`, `two_moons`, `slcp`, `bernoulli_glm`) are **not touched** — they default to `VectorExporter` and the finite-assert policy.

---

## Task 1: Add `[hf]` extra and package scaffold

**Files:**
- Modify: `pyproject.toml`
- Create: `src/sbibm_jax/hf/__init__.py`
- Create: `src/sbibm_jax/hf/config.py`
- Create: `tests/hf/__init__.py`
- Create: `tests/hf/conftest.py`
- Create: `tests/hf/test_import_guard.py`

- [ ] **Step 1: Add the `[hf]` optional extra**

Edit `pyproject.toml`. In the `[project.optional-dependencies]` table (which currently only holds `pypesto`), add an `hf` entry. Show the resulting block:

```toml
[project.optional-dependencies]
hf = [
    "datasets>=2.20.0",
    "huggingface_hub>=0.24.0",
]
pypesto = [
    "pypesto[amici]>=0.6.0",
    "petab",
    "benchmark-models-petab",
    "joblib",
    "scipy",
    # petab's linter (np.issubdtype on column dtypes) breaks on pandas 3.0's
    # default StringDtype; pin to the pandas 2.x line.
    "pandas<3.0",
]
```

Then add `hf` to the `dev` dependency group so tests can run it. The existing `dev` group is:

```toml
dev = [
    {include-group = "lint"},
    {include-group = "test"},
    {include-group = "torch"},
    {include-group = "notebooks"},
]
```

Add a new `hf` group at the same level and include it in `dev`:

```toml
hf = [
    "datasets>=2.20.0",
    "huggingface_hub>=0.24.0",
]
dev = [
    {include-group = "lint"},
    {include-group = "test"},
    {include-group = "torch"},
    {include-group = "notebooks"},
    {include-group = "hf"},
]
```

- [ ] **Step 2: Sync dependencies**

Run: `uv sync --all-groups`
Expected: clean install of `datasets` and `huggingface_hub` in the dev env (no errors).

- [ ] **Step 3: Write the failing import-guard test**

Create `tests/hf/__init__.py` as an empty file. Create `tests/hf/conftest.py`:

```python
"""Skip the whole hf test subdir if `datasets` isn't importable."""

import pytest

pytest.importorskip(
    "datasets",
    reason="The [hf] extra is not installed (uv sync --extra hf).",
)
```

Create `tests/hf/test_import_guard.py`:

```python
"""Smoke tests for the sbibm_jax.hf package boundary."""


def test_package_imports():
    import sbibm_jax.hf

    assert hasattr(sbibm_jax.hf, "__all__")


def test_config_defaults():
    from sbibm_jax.hf import config

    assert config.DEFAULT_REPO == "aurelio-amerio/SBI-benchmarks"
    assert config.DEFAULT_SPLIT_SIZES == {
        "train": 1_000_000,
        "validation": 10_000,
        "test": 10_000,
    }
    assert config.DEFAULT_CHUNK_SIZE == 4096
    assert config.DEFAULT_MAX_FACTOR == 10.0
    assert config.DEFAULT_MASTER_SEED == 0
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_import_guard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbibm_jax.hf'`.

- [ ] **Step 5: Create the `config.py` defaults**

Create `src/sbibm_jax/hf/config.py`:

```python
"""Configuration defaults for the sbibm_jax.hf pipeline.

Centralised so the public API (build_dataset / upload_dataset / make_metadata),
the thin driver in scripts/make_dataset.py, and tests share a single source of
truth.
"""

import numpy as np

DEFAULT_REPO: str = "aurelio-amerio/SBI-benchmarks"

DEFAULT_SPLIT_SIZES: dict = {
    "train": 1_000_000,
    "validation": 10_000,
    "test": 10_000,
}

DEFAULT_DTYPE = np.float32

DEFAULT_CHUNK_SIZE: int = 4096

DEFAULT_MAX_FACTOR: float = 10.0

DEFAULT_MASTER_SEED: int = 0
```

- [ ] **Step 6: Create the package `__init__.py` with import guard**

Create `src/sbibm_jax/hf/__init__.py`:

```python
"""HuggingFace dataset pipeline for sbibm_jax.

Requires the optional `[hf]` extra (`datasets`, `huggingface_hub`). Importing
this subpackage without the extra raises an informative ImportError that points
at `pip install sbibm-jax[hf]`, mirroring the existing `pypesto` extra pattern.

Public API (re-exported below): build_dataset, upload_dataset, make_metadata,
get_exporter.
"""

try:
    import datasets  # noqa: F401
    import huggingface_hub  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The sbibm_jax.hf subpackage requires the optional `[hf]` extra. "
        "Install it with `uv sync --extra hf` or `pip install sbibm-jax[hf]`."
    ) from e

from sbibm_jax.hf import config  # noqa: E402

__all__ = ["config"]
```

(Public API names `build_dataset`, `upload_dataset`, `make_metadata`, `get_exporter` will be added to `__all__` and re-exported as later tasks land them.)

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_import_guard.py -v`
Expected: 2 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/sbibm_jax/hf/__init__.py src/sbibm_jax/hf/config.py \
        tests/hf/__init__.py tests/hf/conftest.py tests/hf/test_import_guard.py
git commit -m "feat(hf): add sbibm_jax.hf package scaffold with [hf] extra"
```

---

## Task 2: `DatasetExporter` base + `VectorExporter`

**Files:**
- Create: `src/sbibm_jax/hf/exporter.py`
- Test: `tests/hf/test_exporter.py`

Background: the base class owns the HF `Features` schema and reshapes the
flat-`x` block emitted by `generate.py`. Subclasses override three small things:
`data_kind` (string), `x_feature()` (per-row HF feature), and `shape_x(x_flat)`
(reshape `(batch, dim_data)` → native storage shape).

> **Implementer note (`datasets.List`):** the code below imports
> `List` from `datasets` (matching the original `SBI-benchmarks-data` code).
> `List` is the modern variable-length feature in `datasets` ≥ 3.0; the `[hf]`
> extra pins `datasets>=2.20.0`, which `uv` resolves to a 3.x build that has it.
> If a pinned environment ever lacks `datasets.List` (`ImportError`), substitute
> `Sequence(Value("float32"))` for `List(Value("float32"))` everywhere in this
> file — the on-disk Arrow type (`list<float32>`) and every assertion on
> `.feature.dtype` are identical, so the tests pass unchanged. Do not mix the
> two: pick `List` (preferred) or `Sequence` consistently.

- [ ] **Step 1: Write the failing test**

Create `tests/hf/test_exporter.py`:

```python
"""Tests for DatasetExporter and its data-kind subclasses."""

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
    def test_data_kind(self):
        task = get_task("gaussian_linear")
        exp = VectorExporter(task, train_size=4, val_size=2, test_size=2)
        assert exp.data_kind == "vector"

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
        flat = np.zeros((3, task.dim_data), dtype=np.float32)
        out = exp.shape_x(flat)
        assert out.shape == (3, task.dim_data)
        assert out.dtype == np.float32

    def test_base_class_is_abstract(self):
        task = get_task("gaussian_linear")
        with pytest.raises(TypeError):
            DatasetExporter(task, train_size=1, val_size=1, test_size=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_exporter.py::TestVectorExporter -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbibm_jax.hf.exporter'`.

- [ ] **Step 3: Implement `DatasetExporter` base + `VectorExporter`**

Create `src/sbibm_jax/hf/exporter.py`:

```python
"""Dataset exporter base + data-kind subclasses.

The exporter owns the HF Features schema and the shape transformation from the
flat (batch, dim_data) block emitted by sbibm_jax.hf.generate to the
native-storage shape (vector / image / time-series). Concrete subclasses
override three things: `data_kind`, `x_feature()`, and `shape_x(x_flat)`.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from datasets import Array2D, Features, List, Value

from sbibm_jax.hf import config
from sbibm_jax.tasks.task import Task


class DatasetExporter(ABC):
    """Abstract base. Subclasses set `data_kind` and override `x_feature`/`shape_x`."""

    data_kind: str = ""

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
        data_shape: Optional[Tuple[int, ...]] = None,
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
        self.data_shape = data_shape if data_shape is not None else (task.dim_data,)
        self.resample_invalid = resample_invalid

    @abstractmethod
    def x_feature(self):
        """The HF feature for a single `x` row."""

    @abstractmethod
    def shape_x(self, x_flat: np.ndarray) -> np.ndarray:
        """Reshape a flat `(batch, dim_data)` block to native storage shape."""

    def theta_feature(self):
        return List(Value("float32"))

    def features(self) -> Features:
        return Features({"xs": self.x_feature(), "thetas": self.theta_feature()})


class VectorExporter(DatasetExporter):
    """Flat parameter-vector storage. The default for analytical tasks."""

    data_kind = "vector"

    def x_feature(self):
        return List(Value("float32"))

    def shape_x(self, x_flat: np.ndarray) -> np.ndarray:
        return np.asarray(x_flat, dtype=self.dtype)


class ImageExporter(DatasetExporter):
    """`Array2D`-backed image storage (e.g. Gaussian Random Field 32x32)."""

    data_kind = "image"

    def __init__(self, task: Task, *, data_shape: Tuple[int, int], **kwargs):
        if data_shape is None or len(data_shape) != 2:
            raise ValueError(
                f"ImageExporter requires a 2-D data_shape (H, W); got {data_shape!r}."
            )
        super().__init__(task, data_shape=data_shape, **kwargs)

    def x_feature(self):
        return Array2D(shape=self.data_shape, dtype="float32")

    def shape_x(self, x_flat: np.ndarray) -> np.ndarray:
        h, w = self.data_shape
        return np.asarray(x_flat, dtype=self.dtype).reshape(-1, h, w)


class TimeSeriesExporter(DatasetExporter):
    """`Array2D`-backed time-series storage (T, C)."""

    data_kind = "timeseries"

    def __init__(self, task: Task, *, data_shape: Tuple[int, int], **kwargs):
        if data_shape is None or len(data_shape) != 2:
            raise ValueError(
                f"TimeSeriesExporter requires a 2-D data_shape (T, C); got {data_shape!r}."
            )
        super().__init__(task, data_shape=data_shape, **kwargs)

    def x_feature(self):
        return Array2D(shape=self.data_shape, dtype="float32")

    def shape_x(self, x_flat: np.ndarray) -> np.ndarray:
        t, c = self.data_shape
        return np.asarray(x_flat, dtype=self.dtype).reshape(-1, t, c)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_exporter.py::TestVectorExporter -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/hf/exporter.py tests/hf/test_exporter.py
git commit -m "feat(hf): add DatasetExporter base and VectorExporter"
```

---

## Task 3: `ImageExporter`

**Files:**
- Modify: `tests/hf/test_exporter.py` (add `TestImageExporter`)

The class itself was already written in Task 2 to keep `exporter.py` cohesive
(one file holding the type hierarchy). This task adds the tests that exercise
it.

- [ ] **Step 1: Write the failing test**

Append to `tests/hf/test_exporter.py`:

```python
class TestImageExporter:
    def test_data_kind(self):
        task = get_task("gaussian_random_field", field_size=8)
        exp = ImageExporter(
            task, data_shape=(8, 8), train_size=4, val_size=2, test_size=2,
        )
        assert exp.data_kind == "image"

    def test_features_schema(self):
        task = get_task("gaussian_random_field", field_size=8)
        exp = ImageExporter(
            task, data_shape=(8, 8), train_size=4, val_size=2, test_size=2,
        )
        feats = exp.features()
        assert isinstance(feats["xs"], Array2D)
        assert feats["xs"].shape == (8, 8)
        assert feats["xs"].dtype == "float32"

    def test_shape_x_reshapes_to_image(self):
        task = get_task("gaussian_random_field", field_size=8)
        exp = ImageExporter(
            task, data_shape=(8, 8), train_size=4, val_size=2, test_size=2,
        )
        flat = np.zeros((5, 8 * 8), dtype=np.float32)
        out = exp.shape_x(flat)
        assert out.shape == (5, 8, 8)
        assert out.dtype == np.float32

    def test_rejects_non_2d_shape(self):
        task = get_task("gaussian_random_field", field_size=8)
        with pytest.raises(ValueError, match="2-D data_shape"):
            ImageExporter(
                task,
                data_shape=(8, 8, 3),
                train_size=4,
                val_size=2,
                test_size=2,
            )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_exporter.py::TestImageExporter -v`
Expected: 4 tests PASS (the class was already implemented in Task 2).

- [ ] **Step 3: Commit**

```bash
git add tests/hf/test_exporter.py
git commit -m "test(hf): cover ImageExporter feature schema and reshape"
```

---

## Task 4: `TimeSeriesExporter`

**Files:**
- Modify: `tests/hf/test_exporter.py` (add `TestTimeSeriesExporter`)

- [ ] **Step 1: Write the failing test**

Append to `tests/hf/test_exporter.py`:

```python
class TestTimeSeriesExporter:
    def test_data_kind(self):
        task = get_task("gaussian_linear")  # any task; data_shape is what counts
        exp = TimeSeriesExporter(
            task, data_shape=(5, 2), train_size=4, val_size=2, test_size=2,
        )
        assert exp.data_kind == "timeseries"

    def test_features_schema(self):
        task = get_task("gaussian_linear")
        exp = TimeSeriesExporter(
            task, data_shape=(5, 2), train_size=4, val_size=2, test_size=2,
        )
        feats = exp.features()
        assert isinstance(feats["xs"], Array2D)
        assert feats["xs"].shape == (5, 2)
        assert feats["xs"].dtype == "float32"

    def test_shape_x_reshapes_to_tc(self):
        task = get_task("gaussian_linear")
        exp = TimeSeriesExporter(
            task, data_shape=(5, 2), train_size=4, val_size=2, test_size=2,
        )
        flat = np.zeros((7, 5 * 2), dtype=np.float32)
        out = exp.shape_x(flat)
        assert out.shape == (7, 5, 2)
        assert out.dtype == np.float32

    def test_rejects_non_2d_shape(self):
        task = get_task("gaussian_linear")
        with pytest.raises(ValueError, match="2-D data_shape"):
            TimeSeriesExporter(
                task,
                data_shape=(5,),
                train_size=4,
                val_size=2,
                test_size=2,
            )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_exporter.py::TestTimeSeriesExporter -v`
Expected: 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/hf/test_exporter.py
git commit -m "test(hf): cover TimeSeriesExporter feature schema and reshape"
```

---

## Task 5: Registry + `get_exporter` with hint reading

**Files:**
- Create: `src/sbibm_jax/hf/registry.py`
- Test: `tests/hf/test_registry.py`

Binds Tasks → Exporter classes via the `hf_data_kind` class/instance hint
(default `"vector"`). Reads `hf_data_shape`, `hf_resample_invalid`,
`hf_split_sizes` via `getattr` with safe defaults so flat analytical tasks need
zero changes.

- [ ] **Step 1: Write the failing test**

Create `tests/hf/test_registry.py`:

```python
"""Tests for hf.registry.get_exporter."""

import pytest

from sbibm_jax import get_task
from sbibm_jax.hf import config
from sbibm_jax.hf.exporter import ImageExporter, VectorExporter
from sbibm_jax.hf.registry import DATA_KIND_REGISTRY, get_exporter


class TestRegistry:
    def test_known_kinds(self):
        assert set(DATA_KIND_REGISTRY) == {"vector", "image", "timeseries"}

    def test_default_is_vector(self):
        task = get_task("gaussian_linear")
        exp = get_exporter(task)
        assert isinstance(exp, VectorExporter)
        assert exp.train_size == config.DEFAULT_SPLIT_SIZES["train"]
        assert exp.val_size == config.DEFAULT_SPLIT_SIZES["validation"]
        assert exp.test_size == config.DEFAULT_SPLIT_SIZES["test"]

    def test_split_size_overrides(self):
        task = get_task("gaussian_linear")
        exp = get_exporter(
            task, train_size=10, val_size=2, test_size=2,
        )
        assert exp.train_size == 10
        assert exp.val_size == 2
        assert exp.test_size == 2

    def test_hf_data_kind_hint_selects_image(self):
        # GRF gets its hint via Task 6; here we simulate the hint manually.
        task = get_task("gaussian_linear")
        task.hf_data_kind = "image"
        task.hf_data_shape = (4, 4)
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert isinstance(exp, ImageExporter)
        assert exp.data_shape == (4, 4)

    def test_resample_invalid_hint_propagates(self):
        task = get_task("gaussian_linear")
        task.hf_resample_invalid = True
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert exp.resample_invalid is True

    def test_unknown_data_kind_raises(self):
        task = get_task("gaussian_linear")
        task.hf_data_kind = "tensor4d"
        with pytest.raises(ValueError, match="Unknown data_kind"):
            get_exporter(task)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbibm_jax.hf.registry'`.

- [ ] **Step 3: Implement `registry.py`**

Create `src/sbibm_jax/hf/registry.py`:

```python
"""Data-kind registry: hf_data_kind hint -> exporter class."""

from typing import Type

from sbibm_jax.hf import config
from sbibm_jax.hf.exporter import (
    DatasetExporter,
    ImageExporter,
    TimeSeriesExporter,
    VectorExporter,
)
from sbibm_jax.tasks.task import Task

DATA_KIND_REGISTRY: dict[str, Type[DatasetExporter]] = {
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
        hf_data_kind:        "vector" | "image" | "timeseries" (default "vector")
        hf_data_shape:       tuple[int, ...]  (default (task.dim_data,))
        hf_resample_invalid: bool             (default False)
        hf_split_sizes:      dict             (default config.DEFAULT_SPLIT_SIZES)

    Explicit train/val/test_size keyword arguments override the task hint and
    the global default. Unknown data_kind raises ValueError.
    """
    data_kind = getattr(task, "hf_data_kind", "vector")
    if data_kind not in DATA_KIND_REGISTRY:
        raise ValueError(
            f"Unknown data_kind {data_kind!r} for task {task.name!r}; "
            f"known kinds: {sorted(DATA_KIND_REGISTRY)}."
        )

    data_shape = getattr(task, "hf_data_shape", (task.dim_data,))
    resample_invalid = getattr(task, "hf_resample_invalid", False)
    task_split_sizes = getattr(task, "hf_split_sizes", config.DEFAULT_SPLIT_SIZES)

    ts = train_size if train_size is not None else task_split_sizes["train"]
    vs = val_size if val_size is not None else task_split_sizes["validation"]
    es = test_size if test_size is not None else task_split_sizes["test"]

    cls = DATA_KIND_REGISTRY[data_kind]
    kwargs = dict(
        train_size=ts,
        val_size=vs,
        test_size=es,
        chunk_size=chunk_size,
        max_factor=max_factor,
        dtype=dtype,
        resample_invalid=resample_invalid,
    )
    if cls is VectorExporter:
        return cls(task, **kwargs)
    return cls(task, data_shape=tuple(data_shape), **kwargs)
```

- [ ] **Step 4: Re-export `get_exporter` from the package**

Edit `src/sbibm_jax/hf/__init__.py`. The current end of the file is:

```python
from sbibm_jax.hf import config  # noqa: E402

__all__ = ["config"]
```

Replace with:

```python
from sbibm_jax.hf import config  # noqa: E402
from sbibm_jax.hf.registry import get_exporter  # noqa: E402

__all__ = ["config", "get_exporter"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_registry.py -v`
Expected: 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sbibm_jax/hf/registry.py src/sbibm_jax/hf/__init__.py tests/hf/test_registry.py
git commit -m "feat(hf): add data-kind registry and get_exporter dispatch"
```

---

## Task 6: Hint attributes on the Gaussian Random Field task

**Files:**
- Modify: `src/sbibm_jax/tasks/gaussian_random_field/task.py`
- Test: `tests/hf/test_registry.py` (add a real-task selection test)

The GRF field is `(field_size, field_size)`; it is the in-tree image example.
`field_size` is instance state, so the hint must be set in `__init__` (after
the parent `super().__init__`).

- [ ] **Step 1: Write the failing test**

Append to `tests/hf/test_registry.py`:

```python
class TestRegistryRealTasks:
    def test_grf_selects_image_exporter(self):
        task = get_task("gaussian_random_field", field_size=8)
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert isinstance(exp, ImageExporter)
        assert exp.data_shape == (8, 8)

    def test_grf_default_field_size_32(self):
        task = get_task("gaussian_random_field")
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert isinstance(exp, ImageExporter)
        assert exp.data_shape == (32, 32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_registry.py::TestRegistryRealTasks -v`
Expected: FAIL — `assert isinstance(exp, ImageExporter)` will fail because `hf_data_kind` defaults to `"vector"` until we set it.

- [ ] **Step 3: Add hints to GRF `__init__`**

Edit `src/sbibm_jax/tasks/gaussian_random_field/task.py`. The current `__init__` body (after `field_size` is stored) is:

```python
        self.field_size = field_size
        super().__init__(
            dim_parameters=2,
            dim_data=field_size * field_size,
            name=Path(__file__).parent.name,
            name_display="Gaussian Random Field",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
        )

        self.prior_dist = dist.Independent(
```

Insert the hint assignments between `super().__init__(...)` and the
`prior_dist` assignment. The result is:

```python
        self.field_size = field_size
        super().__init__(
            dim_parameters=2,
            dim_data=field_size * field_size,
            name=Path(__file__).parent.name,
            name_display="Gaussian Random Field",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
        )

        # HF export hints: stored as (H, W) images via ImageExporter.
        self.hf_data_kind = "image"
        self.hf_data_shape = (field_size, field_size)

        self.prior_dist = dist.Independent(
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_registry.py::TestRegistryRealTasks -v`
Expected: 2 tests PASS.

Also run the existing GRF task tests to confirm no regressions:
Run: `uv run pytest tests/tasks/test_gaussian_random_field.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/tasks/gaussian_random_field/task.py tests/hf/test_registry.py
git commit -m "feat(grf): declare hf_data_kind=image and hf_data_shape hint"
```

---

## Task 7: Generation — seeding + chunking (default validity policy)

**Files:**
- Create: `src/sbibm_jax/hf/generate.py`
- Test: `tests/hf/test_generate.py`

This task implements the streaming generator (`iter_chunks`), a materializing
wrapper (`generate_samples`) for tests, the stable per-task key derivation
(`derive_task_keys`), and the default validity policy (finite-assert). The
resample policy is added in Task 8.

- [ ] **Step 1: Write the failing test**

Create `tests/hf/test_generate.py`:

```python
"""Tests for hf.generate: seeding, chunking, validity policies."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sbibm_jax import get_task
from sbibm_jax.hf.generate import (
    derive_task_keys,
    generate_samples,
    iter_chunks,
)


class TestSeeding:
    def test_stable_across_calls(self):
        k1 = derive_task_keys("gaussian_linear", master_seed=0)
        k2 = derive_task_keys("gaussian_linear", master_seed=0)
        for s in ("train", "validation", "test"):
            assert bool(jnp.all(k1[s] == k2[s]))

    def test_different_seeds_differ(self):
        k1 = derive_task_keys("gaussian_linear", master_seed=0)
        k2 = derive_task_keys("gaussian_linear", master_seed=1)
        assert not bool(jnp.all(k1["train"] == k2["train"]))

    def test_different_tasks_differ(self):
        k_a = derive_task_keys("gaussian_linear", master_seed=0)
        k_b = derive_task_keys("two_moons", master_seed=0)
        assert not bool(jnp.all(k_a["train"] == k_b["train"]))

    def test_splits_are_independent(self):
        keys = derive_task_keys("gaussian_linear", master_seed=0)
        assert not bool(jnp.all(keys["train"] == keys["validation"]))
        assert not bool(jnp.all(keys["train"] == keys["test"]))
        assert not bool(jnp.all(keys["validation"] == keys["test"]))


class TestChunkedGeneration:
    def test_iter_chunks_sums_to_n(self):
        task = get_task("gaussian_linear")
        key = jax.random.PRNGKey(0)
        chunks = list(iter_chunks(
            task, key, n=10, resample_invalid=False, chunk_size=4,
            dtype=np.float32, max_factor=2.0, stats={},
        ))
        thetas = np.concatenate([c[0] for c in chunks], axis=0)
        xs = np.concatenate([c[1] for c in chunks], axis=0)
        assert thetas.shape == (10, task.dim_parameters)
        assert xs.shape == (10, task.dim_data)

    def test_generate_samples_shapes_and_dtype(self):
        task = get_task("two_moons")
        key = jax.random.PRNGKey(0)
        thetas, xs, stats = generate_samples(
            task, key, n=8, chunk_size=4,
        )
        assert thetas.shape == (8, task.dim_parameters)
        assert xs.shape == (8, task.dim_data)
        assert thetas.dtype == np.float32
        assert xs.dtype == np.float32
        assert stats["rejected"] == 0
        assert stats["total_drawn"] == 8

    def test_generate_samples_reproducibility(self):
        task = get_task("gaussian_linear")
        key = jax.random.PRNGKey(123)
        t1, x1, _ = generate_samples(task, key, n=8, chunk_size=4)
        t2, x2, _ = generate_samples(task, key, n=8, chunk_size=4)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(x1, x2)

    def test_no_nan_in_clean_task(self):
        task = get_task("gaussian_linear")
        thetas, xs, _ = generate_samples(
            task, jax.random.PRNGKey(0), n=16, chunk_size=8,
        )
        assert np.isfinite(thetas).all()
        assert np.isfinite(xs).all()


class TestDefaultValidityPolicy:
    def test_finite_assert_raises_on_nan(self):
        # A tiny stub task that emits NaN rows under the default policy.
        class _NaNTask:
            name = "nan_stub"
            dim_parameters = 1
            dim_data = 1

            def get_prior(self, key, num_samples=1):
                return jnp.zeros((num_samples, 1))

            def get_simulator(self, key, max_calls=None):
                def sim(k, theta):
                    return jnp.full((theta.shape[0], 1), jnp.nan)

                sim.flatten_data = lambda x: x.reshape(-1, 1)
                return sim

        with pytest.raises(ValueError, match="non-finite"):
            generate_samples(
                _NaNTask(), jax.random.PRNGKey(0), n=4, chunk_size=4,
                resample_invalid=False,
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbibm_jax.hf.generate'`.

- [ ] **Step 3: Implement `generate.py`**

Create `src/sbibm_jax/hf/generate.py`:

```python
"""Chunked JAX generation of (theta, x) with stable per-task seeding.

Public surface:
    derive_task_keys(name, *, master_seed) -> dict[str, jax.PRNGKey]
        Stable per-task fold-in + split into independent train/val/test keys.

    iter_chunks(task, key, n, *, resample_invalid, chunk_size, dtype,
                max_factor, stats) -> Iterator[(theta_chunk, x_chunk)]
        Streams valid chunks of (theta, x_flat) totalling exactly n rows.
        Stats are written into the provided dict.

    generate_samples(task, key, n, **kwargs) -> (thetas, xs_flat, stats)
        Materializing convenience wrapper around iter_chunks (for tests).

Validity policy:
    resample_invalid=False (default): assert all rows are finite; raise loudly
        if any NaN/Inf appears. Used by analytical tasks.
    resample_invalid=True: drop non-finite (theta, x) rows and keep drawing
        until exactly n valid rows are produced; cap total raw draws at
        max_factor * n; raise if cap is exceeded. Used by ODE / PEtab tasks.
        This is pure rejection sampling — no imputation — so kept rows remain
        i.i.d. draws from the (prior ∩ {sim succeeds}) measure.
"""

import logging
import math
import zlib
from typing import Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sbibm_jax.hf import config

log = logging.getLogger(__name__)


def derive_task_keys(
    name: str,
    *,
    master_seed: int = config.DEFAULT_MASTER_SEED,
) -> dict:
    """Return stable, independent train/val/test PRNG keys for `name`.

    Stability across runs is guaranteed by zlib.crc32 (deterministic), unlike
    Python's salted hash().
    """
    master = jax.random.PRNGKey(master_seed)
    per_task = jax.random.fold_in(master, int(zlib.crc32(name.encode())))
    k_train, k_val, k_test = jax.random.split(per_task, 3)
    return {"train": k_train, "validation": k_val, "test": k_test}


def _draw_chunk(task, theta_key, sim_key, chunk_size, dtype):
    thetas = task.get_prior(theta_key, num_samples=chunk_size)
    sim = task.get_simulator(sim_key, max_calls=None)
    xs = sim(sim_key, thetas)
    thetas_np = np.asarray(thetas, dtype=dtype)
    xs_np = np.asarray(xs, dtype=dtype)
    return thetas_np, xs_np


def iter_chunks(
    task,
    key: jax.Array,
    n: int,
    *,
    resample_invalid: bool,
    chunk_size: int,
    dtype,
    max_factor: float,
    stats: dict,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Stream (theta_chunk, x_flat_chunk) totalling exactly `n` valid rows.

    Writes summary counters into `stats` once exhausted.
    """
    yielded = 0
    drawn = 0
    rejected = 0
    chunk_idx = 0
    cap = int(math.ceil(max_factor * n)) if resample_invalid else n

    while yielded < n:
        chunk_idx += 1
        key, sub = jax.random.split(key)
        theta_key, sim_key = jax.random.split(sub)
        size = min(chunk_size, max(n - yielded, chunk_size))

        thetas_np, xs_np = _draw_chunk(task, theta_key, sim_key, size, dtype)
        drawn += size

        finite_mask = (
            np.isfinite(thetas_np).all(axis=1) & np.isfinite(xs_np).all(axis=1)
        )
        bad = int((~finite_mask).sum())

        if not resample_invalid:
            if bad:
                raise ValueError(
                    f"Task {task.name!r} produced {bad} non-finite row(s) under "
                    "the default validity policy. Set hf_resample_invalid=True "
                    "on the task to drop them via rejection sampling."
                )
            take = min(n - yielded, thetas_np.shape[0])
            yield thetas_np[:take], xs_np[:take]
            yielded += take
        else:
            rejected += bad
            valid_t = thetas_np[finite_mask]
            valid_x = xs_np[finite_mask]
            if valid_t.shape[0] == 0:
                if drawn > cap:
                    raise ValueError(
                        f"Task {task.name!r}: rejection-sampling cap exceeded "
                        f"({drawn}/{cap} draws for n={n}); "
                        f"rejection_rate={rejected / max(drawn, 1):.3f}."
                    )
                continue
            take = min(n - yielded, valid_t.shape[0])
            yield valid_t[:take], valid_x[:take]
            yielded += take
            if drawn > cap:
                raise ValueError(
                    f"Task {task.name!r}: rejection-sampling cap exceeded "
                    f"({drawn}/{cap} draws for n={n}); "
                    f"rejection_rate={rejected / max(drawn, 1):.3f}."
                )

    stats["total_drawn"] = drawn
    stats["rejected"] = rejected
    stats["rejection_rate"] = rejected / max(drawn, 1)
    if resample_invalid and rejected:
        log.info(
            "[hf.generate] task=%s n=%d rejected=%d rate=%.4f",
            task.name, n, rejected, stats["rejection_rate"],
        )


def generate_samples(
    task,
    key: jax.Array,
    n: int,
    *,
    resample_invalid: bool = False,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    dtype=config.DEFAULT_DTYPE,
    max_factor: float = config.DEFAULT_MAX_FACTOR,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Materializing wrapper around iter_chunks.

    Returns concatenated arrays and the populated stats dict. Used by tests and
    for tasks where n is small enough to fit in memory; the streaming pipeline
    (build.py) uses iter_chunks directly to bound RAM at the chunk size.
    """
    stats: dict = {}
    chunks = list(iter_chunks(
        task, key, n,
        resample_invalid=resample_invalid,
        chunk_size=chunk_size,
        dtype=dtype,
        max_factor=max_factor,
        stats=stats,
    ))
    thetas = np.concatenate([c[0] for c in chunks], axis=0)
    xs = np.concatenate([c[1] for c in chunks], axis=0)
    return thetas, xs, stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_generate.py -v`
Expected: all PASS (TestSeeding 4, TestChunkedGeneration 4, TestDefaultValidityPolicy 1 = 9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/hf/generate.py tests/hf/test_generate.py
git commit -m "feat(hf): chunked generation with stable per-task seeding"
```

---

## Task 8: Generation — resample policy + max_factor cap

**Files:**
- Modify: `tests/hf/test_generate.py` (add `TestResamplePolicy`)

The implementation already supports `resample_invalid=True` (added in Task 7),
so this task is purely about test coverage of the policy.

- [ ] **Step 1: Write the failing test**

Append to `tests/hf/test_generate.py`:

```python
class TestResamplePolicy:
    @staticmethod
    def _make_partial_nan_task(nan_fraction: float):
        """Stub task whose simulator returns NaN on a fixed fraction of rows."""

        class _PartialNaN:
            name = "partial_nan_stub"
            dim_parameters = 1
            dim_data = 1

            def get_prior(self, key, num_samples=1):
                return jax.random.uniform(key, (num_samples, 1))

            def get_simulator(self, key, max_calls=None):
                def sim(k, theta):
                    n = theta.shape[0]
                    u = jax.random.uniform(k, (n,))
                    out = jnp.where(
                        u < nan_fraction,
                        jnp.full((n,), jnp.nan),
                        theta[:, 0],
                    ).reshape(n, 1)
                    return out

                sim.flatten_data = lambda x: x.reshape(-1, 1)
                return sim

        return _PartialNaN()

    def test_resample_returns_exactly_n_finite_rows(self):
        task = self._make_partial_nan_task(nan_fraction=0.3)
        thetas, xs, stats = generate_samples(
            task, jax.random.PRNGKey(0), n=20,
            resample_invalid=True, chunk_size=8, max_factor=10.0,
        )
        assert thetas.shape == (20, 1)
        assert xs.shape == (20, 1)
        assert np.isfinite(thetas).all()
        assert np.isfinite(xs).all()
        assert stats["rejected"] > 0
        assert 0.0 < stats["rejection_rate"] < 1.0

    def test_resample_cap_raises(self):
        # nan_fraction=1.0 means every row is NaN; cap is hit immediately.
        task = self._make_partial_nan_task(nan_fraction=1.0)
        with pytest.raises(ValueError, match="cap exceeded"):
            generate_samples(
                task, jax.random.PRNGKey(0), n=10,
                resample_invalid=True, chunk_size=4, max_factor=2.0,
            )

    def test_default_policy_raises_with_count(self):
        task = self._make_partial_nan_task(nan_fraction=0.5)
        with pytest.raises(ValueError, match="non-finite"):
            generate_samples(
                task, jax.random.PRNGKey(0), n=8, chunk_size=8,
                resample_invalid=False,
            )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_generate.py::TestResamplePolicy -v`
Expected: 3 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/hf/test_generate.py
git commit -m "test(hf): cover rejection-resample policy and max_factor cap"
```

---

## Task 9: `hf_resample_invalid` hints on ODE + PEtab tasks

**Files:**
- Modify: `src/sbibm_jax/tasks/lotka_volterra/task.py`
- Modify: `src/sbibm_jax/tasks/sir/task.py`
- Modify: `src/sbibm_jax/tasks/beer_molbiosystems/task.py`
- Test: `tests/hf/test_registry.py` (add `TestResampleHints`)

- [ ] **Step 1: Write the failing test**

Append to `tests/hf/test_registry.py`:

```python
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

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_registry.py::TestResampleHints -v`
Expected: FAIL — `getattr(task, "hf_resample_invalid", False) is True` returns `False`.

- [ ] **Step 3: Set the hint on Lotka-Volterra**

Edit `src/sbibm_jax/tasks/lotka_volterra/task.py`. After the `super().__init__(...)` block in `__init__` (which currently ends around line 60 of the existing file, before the prior-params assignment), add:

```python
        # ODE divergences emit NaN rows; rejection-resample at HF export time.
        self.hf_resample_invalid = True
```

- [ ] **Step 4: Set the hint on SIR**

Edit `src/sbibm_jax/tasks/sir/task.py`. Locate the `super().__init__(...)` in
`__init__` and add the same two-line block immediately after it:

```python
        # ODE divergences emit NaN rows; rejection-resample at HF export time.
        self.hf_resample_invalid = True
```

- [ ] **Step 5: Set the hint on Beer**

Edit `src/sbibm_jax/tasks/beer_molbiosystems/task.py`. Locate the
`super().__init__(...)` in `__init__` and add immediately after it:

```python
        # AMICI failures emit full NaN rows; rejection-resample at HF export time.
        self.hf_resample_invalid = True
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_registry.py::TestResampleHints -v`
Expected: 3 tests PASS (or 2 PASS + 1 SKIP if `pypesto` extra isn't installed).

Also confirm no task-level regressions:
Run: `uv run pytest tests/tasks/test_ode.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/sbibm_jax/tasks/lotka_volterra/task.py \
        src/sbibm_jax/tasks/sir/task.py \
        src/sbibm_jax/tasks/beer_molbiosystems/task.py \
        tests/hf/test_registry.py
git commit -m "feat(tasks): mark ODE and PEtab tasks for HF rejection resample"
```

---

## Task 10: Reference block loader

**Files:**
- Create: `src/sbibm_jax/hf/reference.py`
- Test: `tests/hf/test_reference.py`

Reads each task's `files/num_observation_<i>/{observation, reference_posterior_samples,
true_parameters}.csv*` via the existing `Task` loaders. Builds a `datasets.Dataset`
with fields `reference_samples`, `observations`, `true_parameters` (matches the
original `SBI-benchmarks-data` schema). Returns `None` when the files are
absent (GRF, Beer, future toy_lensing) — caller skips the `_posterior` config.

- [ ] **Step 1: Write the failing test**

Create `tests/hf/test_reference.py`:

```python
"""Tests for hf.reference.load_reference."""

import numpy as np
import pytest
from datasets import Dataset

from sbibm_jax import get_task
from sbibm_jax.hf.reference import load_reference
from sbibm_jax.hf.registry import get_exporter


class TestLoadReference:
    def test_two_moons_present(self):
        task = get_task("two_moons")
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        ref = load_reference(task, exp)
        assert isinstance(ref, Dataset)
        assert len(ref) == task.num_observations  # 10
        cols = set(ref.column_names)
        assert cols == {"reference_samples", "observations", "true_parameters"}

    def test_two_moons_shapes(self):
        task = get_task("two_moons")
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        ref = load_reference(task, exp)
        # observations: each row is (1, dim_data) → flat list of dim_data floats.
        row = ref[0]
        assert len(row["observations"]) == task.dim_data
        assert len(row["true_parameters"]) == task.dim_parameters
        # reference_samples is a (num_ref_posterior_samples, dim_parameters) block
        # → list-of-lists. Each inner list has length dim_parameters.
        rs = row["reference_samples"]
        assert len(rs) == task.num_reference_posterior_samples
        assert len(rs[0]) == task.dim_parameters

    def test_grf_absent_returns_none(self):
        task = get_task("gaussian_random_field", field_size=8)
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert load_reference(task, exp) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_reference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbibm_jax.hf.reference'`.

- [ ] **Step 3: Implement `reference.py`**

Create `src/sbibm_jax/hf/reference.py`:

```python
"""Optional per-task reference block loader.

For tasks that ship reference posterior CSVs under
files/num_observation_<i>/, builds a HuggingFace Dataset matching the original
SBI-benchmarks-data schema (reference_samples, observations, true_parameters).
Returns None when the files are absent so the caller can skip the _posterior
config without erroring.
"""

from typing import Optional

import numpy as np
from datasets import Dataset

from sbibm_jax.hf.exporter import DatasetExporter
from sbibm_jax.tasks.task import Task


def load_reference(task: Task, exporter: DatasetExporter) -> Optional[Dataset]:
    """Load the reference block for `task`, reshaping observations via `exporter`.

    Returns None if any required CSV is missing (e.g. for GRF, Beer).
    """
    observations = []
    reference_samples = []
    true_parameters = []

    for i in range(1, task.num_observations + 1):
        try:
            obs = np.asarray(task.get_observation(i), dtype=np.float32)
            ref = np.asarray(
                task.get_reference_posterior_samples(i), dtype=np.float32,
            )
            true_p = np.asarray(task.get_true_parameters(i), dtype=np.float32)
        except FileNotFoundError:
            return None

        # Reshape the observation `x` from (1, dim_data) into the exporter's
        # native storage shape (drop the leading sample axis after reshape).
        obs_flat = obs.reshape(1, task.dim_data)
        obs_shaped = exporter.shape_x(obs_flat)[0]

        observations.append(obs_shaped)
        reference_samples.append(ref)
        true_parameters.append(true_p.reshape(-1))

    return Dataset.from_dict({
        "reference_samples": np.stack(reference_samples).astype(np.float32),
        "observations": np.stack(observations).astype(np.float32),
        "true_parameters": np.stack(true_parameters).astype(np.float32),
    })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_reference.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/hf/reference.py tests/hf/test_reference.py
git commit -m "feat(hf): load optional per-task reference block (skipped when absent)"
```

---

## Task 11: `make_metadata` auto-generation

**Files:**
- Create: `src/sbibm_jax/hf/metadata.py`
- Test: `tests/hf/test_metadata.py`

Replaces the hand-maintained `metadata.json` from `SBI-benchmarks-data`. Built
from `task` attributes (`dim_parameters`, `dim_data`, etc.) plus
exporter-derived fields (`data_kind`, `data_shape`) and reference-block
presence (probed via `load_reference`).

- [ ] **Step 1: Write the failing test**

Create `tests/hf/test_metadata.py`:

```python
"""Tests for hf.metadata.make_metadata."""

import json
from pathlib import Path

import pytest

from sbibm_jax.hf.metadata import make_metadata


class TestMakeMetadata:
    def test_returns_dict_per_task(self):
        meta = make_metadata(["gaussian_linear", "two_moons"])
        assert set(meta) == {"gaussian_linear", "two_moons"}

    def test_vector_task_schema(self):
        meta = make_metadata(["gaussian_linear"])
        m = meta["gaussian_linear"]
        assert m["dim_parameters"] == 10
        assert m["dim_data"] == 10
        assert m["data_kind"] == "vector"
        assert m["data_shape"] == [10]
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
        assert m["data_kind"] == "image"
        assert m["data_shape"] == [32, 32]
        assert m["has_reference"] is False

    def test_writes_json_file(self, tmp_path):
        out = tmp_path / "metadata.json"
        meta = make_metadata(["gaussian_linear"], output_path=out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded == meta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_metadata.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbibm_jax.hf.metadata'`.

- [ ] **Step 3: Implement `metadata.py`**

Create `src/sbibm_jax/hf/metadata.py`:

```python
"""Auto-generate metadata.json from Task attributes."""

import json
from pathlib import Path
from typing import Iterable, Optional

from sbibm_jax import get_task
from sbibm_jax.hf import config
from sbibm_jax.hf.reference import load_reference
from sbibm_jax.hf.registry import get_exporter


def make_metadata(
    task_names: Iterable[str],
    *,
    output_path: Optional[Path] = None,
    split_sizes: Optional[dict] = None,
) -> dict:
    """Build a metadata dict (and optionally write metadata.json).

    Schema per task:
        dim_parameters: int
        dim_data:       int
        data_kind:      "vector" | "image" | "timeseries"
        data_shape:     list[int]
        splits:         dict[str, int]
        has_reference:  bool
        num_observations: int
    """
    if split_sizes is None:
        split_sizes = dict(config.DEFAULT_SPLIT_SIZES)

    meta: dict = {}
    for name in task_names:
        task = get_task(name)
        exporter = get_exporter(
            task,
            train_size=split_sizes["train"],
            val_size=split_sizes["validation"],
            test_size=split_sizes["test"],
        )
        meta[name] = {
            "dim_parameters": int(task.dim_parameters),
            "dim_data": int(task.dim_data),
            "data_kind": exporter.data_kind,
            "data_shape": list(exporter.data_shape),
            "splits": dict(split_sizes),
            "has_reference": load_reference(task, exporter) is not None,
            "num_observations": int(task.num_observations),
        }

    if output_path is not None:
        Path(output_path).write_text(json.dumps(meta, indent=4))

    return meta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_metadata.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/hf/metadata.py tests/hf/test_metadata.py
git commit -m "feat(hf): auto-generate metadata.json from task attributes"
```

---

## Task 12: Upload helpers (monkey-patchable)

**Files:**
- Create: `src/sbibm_jax/hf/upload.py`
- Test: `tests/hf/test_upload.py`

The two HF-network calls live behind a single module so tests can monkeypatch
them without any real network use. `upload_dataset` uses `build_dataset` (Task 13)
under the hood — we declare it now as a forward import and the integration is
fully exercised in Task 13.

- [ ] **Step 1: Write the failing test**

Create `tests/hf/test_upload.py`:

```python
"""Tests for hf.upload — monkeypatched, no real HF calls."""

import pytest

import sbibm_jax.hf.upload as upload_mod
from sbibm_jax.hf.upload import upload_dataset, upload_metadata


class _FakeDataset:
    def __init__(self, name):
        self.name = name
        self.push_calls = []

    def push_to_hub(self, repo_name, **kwargs):
        self.push_calls.append((repo_name, kwargs))


class TestUploadMetadata:
    def test_calls_upload_file(self, monkeypatch, tmp_path):
        path = tmp_path / "metadata.json"
        path.write_text("{}")

        calls = []

        def fake_upload_file(**kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(upload_mod, "upload_file", fake_upload_file)
        upload_metadata(str(path), "user/repo")

        assert len(calls) == 1
        kw = calls[0]
        assert kw["path_or_fileobj"] == str(path)
        assert kw["path_in_repo"] == "metadata.json"
        assert kw["repo_id"] == "user/repo"
        assert kw["repo_type"] == "dataset"


class TestUploadDataset:
    def test_pushes_each_split_with_right_config(self, monkeypatch):
        train = _FakeDataset("train")
        val = _FakeDataset("val")
        test = _FakeDataset("test")
        ref = _FakeDataset("ref")

        def fake_build(task_name, **opts):
            return {
                "train": train,
                "validation": val,
                "test": test,
                "reference": ref,
            }

        monkeypatch.setattr(upload_mod, "build_dataset", fake_build)
        upload_dataset("user/repo", "two_moons")

        assert train.push_calls == [
            ("user/repo", {
                "config_name": "two_moons",
                "split": "train",
                "private": False,
            }),
        ]
        assert val.push_calls == [
            ("user/repo", {
                "config_name": "two_moons",
                "split": "validation",
                "private": False,
            }),
        ]
        assert test.push_calls == [
            ("user/repo", {
                "config_name": "two_moons",
                "split": "test",
                "private": False,
            }),
        ]
        assert ref.push_calls == [
            ("user/repo", {
                "config_name": "two_moons_posterior",
                "split": "reference_posterior",
                "private": False,
            }),
        ]

    def test_skips_reference_when_absent(self, monkeypatch):
        train = _FakeDataset("train")
        val = _FakeDataset("val")
        test = _FakeDataset("test")

        def fake_build(task_name, **opts):
            return {
                "train": train,
                "validation": val,
                "test": test,
                "reference": None,
            }

        monkeypatch.setattr(upload_mod, "build_dataset", fake_build)
        upload_dataset("user/repo", "gaussian_random_field")

        assert train.push_calls[0][1]["config_name"] == "gaussian_random_field"
        assert val.push_calls[0][1]["split"] == "validation"
        # No assertion needed for "ref" — it does not exist (would have raised).
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_upload.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbibm_jax.hf.upload'`.

- [ ] **Step 3: Implement `upload.py`**

Create `src/sbibm_jax/hf/upload.py`:

```python
"""HuggingFace upload helpers — isolated so tests can monkeypatch.

`upload_file` and `push_to_hub` are the only network surface. Both are imported
at module scope so monkeypatching `sbibm_jax.hf.upload.upload_file` (and the
shadowing of `build_dataset` here) is sufficient for any test.
"""

from huggingface_hub import upload_file

from sbibm_jax.hf.build import build_dataset


def upload_metadata(file_path: str, repo_name: str) -> None:
    """Push a metadata.json file to the dataset repo."""
    upload_file(
        path_or_fileobj=file_path,
        path_in_repo="metadata.json",
        repo_id=repo_name,
        repo_type="dataset",
    )


def upload_dataset(repo_name: str, task_name: str, **build_opts) -> None:
    """Build the dataset for `task_name` and push each split to `repo_name`.

    The dataset is pushed under config_name=task_name with splits train /
    validation / test. If the task ships a reference block, it is pushed under
    config_name=f"{task_name}_posterior" with split "reference_posterior".
    """
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
```

- [ ] **Step 4: Create a placeholder `build.py` so the import resolves**

`upload.py` imports `build_dataset` from `sbibm_jax.hf.build`, which is fleshed
out in Task 13. Create a temporary stub so the tests in this task pass without
network calls (the tests monkeypatch `build_dataset` anyway):

Create `src/sbibm_jax/hf/build.py`:

```python
"""Top-level orchestration: build_dataset(task_name, **opts).

NOTE: This is a stub. The real implementation is added in Task 13. Tests in
Task 12 monkeypatch this symbol via sbibm_jax.hf.upload.build_dataset, so the
stub never actually runs.
"""

from typing import Any


def build_dataset(task_name: str, **opts: Any):
    raise NotImplementedError(
        "build_dataset is a stub until Task 13 implements the orchestration."
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_upload.py -v`
Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sbibm_jax/hf/upload.py src/sbibm_jax/hf/build.py tests/hf/test_upload.py
git commit -m "feat(hf): upload helpers with mockable HF surface"
```

---

## Task 13: `build_dataset` orchestration + end-to-end integration

**Files:**
- Modify: `src/sbibm_jax/hf/build.py` (replace stub)
- Modify: `src/sbibm_jax/hf/__init__.py` (re-export `build_dataset`, `upload_dataset`, `make_metadata`)
- Test: `tests/hf/test_build_dataset.py`

Wires Task 5–11 together. Streams chunks through `Dataset.from_generator` so
memory stays bounded.

- [ ] **Step 1: Write the failing test**

Create `tests/hf/test_build_dataset.py`:

```python
"""End-to-end build_dataset tests on tiny sizes (CPU, no network)."""

import numpy as np
import pytest
from datasets import Dataset

from sbibm_jax.hf import build_dataset


SMALL_OPTS = dict(train_size=8, val_size=4, test_size=2, chunk_size=4)


class TestBuildVector:
    def test_returns_bundle(self):
        bundle = build_dataset("gaussian_linear", **SMALL_OPTS)
        assert set(bundle) == {"train", "validation", "test", "reference"}
        for k in ("train", "validation", "test"):
            assert isinstance(bundle[k], Dataset)

    def test_split_sizes(self):
        bundle = build_dataset("gaussian_linear", **SMALL_OPTS)
        assert len(bundle["train"]) == 8
        assert len(bundle["validation"]) == 4
        assert len(bundle["test"]) == 2

    def test_dtype_and_finiteness(self):
        bundle = build_dataset("gaussian_linear", **SMALL_OPTS)
        sample = bundle["train"][0]
        assert len(sample["thetas"]) == 10  # gaussian_linear dim_parameters
        assert len(sample["xs"]) == 10  # gaussian_linear dim_data
        arr = np.asarray(sample["xs"], dtype=np.float64)
        assert np.isfinite(arr).all()

    def test_reproducibility(self):
        b1 = build_dataset("gaussian_linear", master_seed=42, **SMALL_OPTS)
        b2 = build_dataset("gaussian_linear", master_seed=42, **SMALL_OPTS)
        np.testing.assert_array_equal(b1["train"]["thetas"], b2["train"]["thetas"])
        np.testing.assert_array_equal(b1["train"]["xs"], b2["train"]["xs"])

    def test_reference_present(self):
        bundle = build_dataset("two_moons", **SMALL_OPTS)
        assert bundle["reference"] is not None
        assert len(bundle["reference"]) == 10  # two_moons num_observations


class TestBuildImage:
    def test_grf_image_shape(self):
        bundle = build_dataset(
            "gaussian_random_field",
            **SMALL_OPTS,
            task_kwargs={"field_size": 8},
        )
        # Each row is an 8x8 image stored via Array2D.
        sample = bundle["train"][0]
        arr = np.asarray(sample["xs"], dtype=np.float32)
        assert arr.shape == (8, 8)
        assert np.isfinite(arr).all()

    def test_grf_no_reference(self):
        bundle = build_dataset(
            "gaussian_random_field",
            **SMALL_OPTS,
            task_kwargs={"field_size": 8},
        )
        assert bundle["reference"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_build_dataset.py -v`
Expected: FAIL — current `build_dataset` raises `NotImplementedError`.

- [ ] **Step 3: Replace the `build.py` stub with the real orchestration**

Overwrite `src/sbibm_jax/hf/build.py`:

```python
"""build_dataset(task_name, **opts): the end-to-end pipeline entry point."""

from typing import Any, Optional

from datasets import Dataset

from sbibm_jax import get_task
from sbibm_jax.hf import config
from sbibm_jax.hf.generate import derive_task_keys, iter_chunks
from sbibm_jax.hf.reference import load_reference
from sbibm_jax.hf.registry import get_exporter


def _build_split(exporter, key, n: int) -> Dataset:
    """Stream (theta, x) rows of one split through Dataset.from_generator."""
    stats: dict = {}

    def row_generator():
        for theta_chunk, x_flat_chunk in iter_chunks(
            exporter.task, key, n,
            resample_invalid=exporter.resample_invalid,
            chunk_size=exporter.chunk_size,
            dtype=exporter.dtype,
            max_factor=exporter.max_factor,
            stats=stats,
        ):
            x_chunk = exporter.shape_x(x_flat_chunk)
            for i in range(theta_chunk.shape[0]):
                yield {"xs": x_chunk[i], "thetas": theta_chunk[i]}

    return Dataset.from_generator(row_generator, features=exporter.features())


def build_dataset(
    task_name: str,
    *,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    max_factor: float = config.DEFAULT_MAX_FACTOR,
    dtype=config.DEFAULT_DTYPE,
    master_seed: int = config.DEFAULT_MASTER_SEED,
    task_kwargs: Optional[dict] = None,
) -> dict:
    """Build the train / validation / test (+ optional reference) bundle.

    Returns a dict with keys "train", "validation", "test", "reference"
    (the last may be None if the task ships no reference CSVs).
    """
    task = get_task(task_name, **(task_kwargs or {}))
    exporter = get_exporter(
        task,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        chunk_size=chunk_size,
        max_factor=max_factor,
        dtype=dtype,
    )
    keys = derive_task_keys(task.name, master_seed=master_seed)
    return {
        "train": _build_split(exporter, keys["train"], exporter.train_size),
        "validation": _build_split(exporter, keys["validation"], exporter.val_size),
        "test": _build_split(exporter, keys["test"], exporter.test_size),
        "reference": load_reference(task, exporter),
    }
```

- [ ] **Step 4: Re-export public API**

Edit `src/sbibm_jax/hf/__init__.py`. Replace the current trailing block:

```python
from sbibm_jax.hf import config  # noqa: E402
from sbibm_jax.hf.registry import get_exporter  # noqa: E402

__all__ = ["config", "get_exporter"]
```

with:

```python
from sbibm_jax.hf import config  # noqa: E402
from sbibm_jax.hf.build import build_dataset  # noqa: E402
from sbibm_jax.hf.metadata import make_metadata  # noqa: E402
from sbibm_jax.hf.registry import get_exporter  # noqa: E402
from sbibm_jax.hf.upload import upload_dataset, upload_metadata  # noqa: E402

__all__ = [
    "build_dataset",
    "config",
    "get_exporter",
    "make_metadata",
    "upload_dataset",
    "upload_metadata",
]
```

- [ ] **Step 5: Run all hf tests**

Run: `uv run pytest tests/hf/ -v`
Expected: all existing PLUS the new build_dataset tests PASS (7 new tests).

- [ ] **Step 6: Commit**

```bash
git add src/sbibm_jax/hf/build.py src/sbibm_jax/hf/__init__.py tests/hf/test_build_dataset.py
git commit -m "feat(hf): wire build_dataset orchestration + integration tests"
```

---

## Task 14: Driver script `scripts/make_dataset.py`

**Files:**
- Create: `scripts/make_dataset.py`
- Test: `tests/hf/test_driver.py`

Replaces the old `SBI-benchmarks-data/make_dataset.py`. Loops over an explicit
task list (or all available tasks via `--all`), builds metadata, optionally
uploads. Supports a `--dry-run` flag so a smoke test can exercise the CLI
without touching HF.

- [ ] **Step 1: Write the failing test**

Create `tests/hf/test_driver.py`:

```python
"""Smoke tests for scripts/make_dataset.py (no real HF calls)."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "scripts" / "make_dataset.py"


def test_driver_help_runs():
    result = subprocess.run(
        [sys.executable, str(DRIVER), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "usage" in result.stdout.lower()


def test_dry_run_writes_metadata(tmp_path):
    out = tmp_path / "metadata.json"
    result = subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--tasks",
            "gaussian_linear",
            "--metadata-path",
            str(out),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.exists()
    assert "gaussian_linear" in out.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_driver.py -v`
Expected: FAIL — `scripts/make_dataset.py` does not exist.

- [ ] **Step 3: Implement the driver**

Create `scripts/make_dataset.py`:

```python
"""Build (and optionally upload) HuggingFace datasets for sbibm_jax tasks.

Replaces SBI-benchmarks-data/make_dataset.py. Usage:

    # Default repo, all available tasks, real upload:
    uv run python scripts/make_dataset.py --all

    # Explicit task list, dry-run (writes metadata.json, no HF push):
    uv run python scripts/make_dataset.py --tasks gaussian_linear two_moons --dry-run

    # Custom split sizes:
    uv run python scripts/make_dataset.py --tasks two_moons \
        --train-size 1000 --val-size 100 --test-size 100
"""

import argparse
import logging
import sys
from pathlib import Path

from sbibm_jax import get_available_tasks
from sbibm_jax.hf import (
    config,
    make_metadata,
    upload_dataset,
    upload_metadata,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tasks",
        nargs="+",
        help="Explicit task names. Use --all for every registered task.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Process every task returned by get_available_tasks().",
    )
    p.add_argument(
        "--repo",
        default=config.DEFAULT_REPO,
        help=f"HuggingFace dataset repo (default: {config.DEFAULT_REPO}).",
    )
    p.add_argument(
        "--metadata-path",
        default="metadata.json",
        help="Where to write metadata.json (default: ./metadata.json).",
    )
    p.add_argument("--train-size", type=int, default=None)
    p.add_argument("--val-size", type=int, default=None)
    p.add_argument("--test-size", type=int, default=None)
    p.add_argument("--master-seed", type=int, default=config.DEFAULT_MASTER_SEED)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build metadata.json but skip all HF uploads.",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.all:
        task_names = get_available_tasks()
    elif args.tasks:
        task_names = args.tasks
    else:
        print("ERROR: pass --tasks NAME [NAME ...] or --all", file=sys.stderr)
        sys.exit(2)

    build_opts = {}
    if args.train_size is not None:
        build_opts["train_size"] = args.train_size
    if args.val_size is not None:
        build_opts["val_size"] = args.val_size
    if args.test_size is not None:
        build_opts["test_size"] = args.test_size
    build_opts["master_seed"] = args.master_seed

    metadata_path = Path(args.metadata_path)
    split_sizes = None
    if any(k in build_opts for k in ("train_size", "val_size", "test_size")):
        split_sizes = {
            "train": build_opts.get(
                "train_size", config.DEFAULT_SPLIT_SIZES["train"]),
            "validation": build_opts.get(
                "val_size", config.DEFAULT_SPLIT_SIZES["validation"]),
            "test": build_opts.get(
                "test_size", config.DEFAULT_SPLIT_SIZES["test"]),
        }
    make_metadata(task_names, output_path=metadata_path, split_sizes=split_sizes)
    print(f"Wrote {metadata_path}")

    if args.dry_run:
        print("Dry run — skipping HF uploads.")
        return

    upload_metadata(str(metadata_path), args.repo)
    for name in task_names:
        print(f"Uploading dataset for task: {name}")
        upload_dataset(args.repo, name, **build_opts)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_driver.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Sanity-run the dry-run path end-to-end**

Run: `uv run python scripts/make_dataset.py --tasks gaussian_linear --train-size 8 --val-size 4 --test-size 2 --metadata-path /tmp/sbibm_jax_test_meta.json --dry-run`
Expected: writes `/tmp/sbibm_jax_test_meta.json` and prints "Dry run — skipping HF uploads." (no exception).

- [ ] **Step 6: Run the full hf test subdir as a regression check**

Run: `uv run pytest tests/hf/ -v`
Expected: all PASS.

- [ ] **Step 7: Run the whole test suite for regressions**

Run: `uv run pytest -m "not slow"`
Expected: all PASS (the new tests are CPU-only and tiny).

- [ ] **Step 8: Commit**

```bash
git add scripts/make_dataset.py tests/hf/test_driver.py
git commit -m "feat(hf): thin driver script replacing the old make_dataset.py"
```

---

## Done

After all 14 tasks land:

- `sbibm_jax.hf` is a self-contained, no-torch, no-original-sbibm subpackage
  that builds the same published HuggingFace dataset entirely from the JAX
  Task API.
- New tasks added to `sbibm_jax/tasks/*` automatically work in the pipeline:
  flat analytical tasks need zero changes; structured tasks set tiny
  `hf_data_kind` / `hf_data_shape` / `hf_resample_invalid` hints on
  themselves.
- All HF-network calls are isolated in `sbibm_jax/hf/upload.py` and can be
  monkeypatched for tests.
- The pipeline is covered by ~30+ CPU-only tests under `tests/hf/`, none of
  which touch the network.
