# Gaussian Random Field 256×256 Variant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `gaussian_random_field_256`, a 256×256 high-resolution variant of the Gaussian Random Field task, registered as an alias and exportable to HuggingFace with train/validation/test split sizes 100k/10k/10k.

**Architecture:** SLCP-style variant — no new task directory. `GaussianRandomField` gains optional `name`/`name_display` constructor args (default `None` → unchanged behavior); a registry alias constructs it with `field_size=256` and the distinct name. The HF export already dispatches on `hf_data_kind="image"` + `hf_data_shape`, and the chunked `Dataset.from_generator` streaming bounds RAM, so no export-side changes are needed beyond exposing a `--chunk-size` knob on the driver for the larger per-chunk GPU footprint.

**Tech Stack:** JAX, NumPyro, HuggingFace `datasets`, pytest. Lint: flake8 (default 79-char limit — keep all new lines ≤79).

---

## File Structure

- **Modify** `src/sbibm_jax/tasks/gaussian_random_field/task.py` — add optional `name`/`name_display` ctor args.
- **Modify** `src/sbibm_jax/tasks/__init__.py` — add `gaussian_random_field_256` registry branch + `tasks_extra` entry.
- **Modify** `scripts/make_dataset.py` — add `--chunk-size` flag, thread into `build_opts`.
- **Modify** `tests/tasks/test_gaussian_random_field.py` — variant + default-name + smoke-gen tests.
- **Modify** `tests/hf/test_registry.py` — variant exporter test.
- **Modify** `tests/hf/test_metadata.py` — variant metadata-schema test.
- **Modify** `tests/hf/test_driver.py` — `--chunk-size` forwarding test.
- **Modify** `CLAUDE.md` — document the variant and the `--chunk-size` flag.

---

### Task 1: Parametrize `GaussianRandomField` name

**Files:**
- Modify: `src/sbibm_jax/tasks/gaussian_random_field/task.py:21-37`
- Test: `tests/tasks/test_gaussian_random_field.py`

- [ ] **Step 1: Write the failing tests**

Append a new class at the end of `tests/tasks/test_gaussian_random_field.py` (before the `if __name__` block if present, otherwise at end):

```python
class TestNameOverride:
    def test_default_name_unchanged(self):
        t = GaussianRandomField(field_size=16)
        assert t.name == "gaussian_random_field"
        assert t.name_display == "Gaussian Random Field"

    def test_explicit_name_override(self):
        t = GaussianRandomField(
            field_size=256,
            name="gaussian_random_field_256",
            name_display="Gaussian Random Field (256x256)",
        )
        assert t.name == "gaussian_random_field_256"
        assert t.name_display == "Gaussian Random Field (256x256)"
        assert t.field_size == 256
        assert t.dim_data == 256 * 256
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tasks/test_gaussian_random_field.py::TestNameOverride -v`
Expected: FAIL — `test_explicit_name_override` errors with `TypeError: __init__() got an unexpected keyword argument 'name'`.

- [ ] **Step 3: Add the optional ctor args**

In `src/sbibm_jax/tasks/gaussian_random_field/task.py`, change the signature and the two `super().__init__` lines. Replace:

```python
    def __init__(self, field_size: int = 32):
```

with:

```python
    def __init__(
        self,
        field_size: int = 32,
        name: Optional[str] = None,
        name_display: Optional[str] = None,
    ):
```

Then inside that `super().__init__(...)` call, replace:

```python
            name=Path(__file__).parent.name,
            name_display="Gaussian Random Field",
```

with:

```python
            name=name or Path(__file__).parent.name,
            name_display=name_display or "Gaussian Random Field",
```

(`Optional` is already imported via `from typing import Optional`.)

Also update the docstring `Args:` section to mention the two new optional args, e.g. add under `field_size`:

```python
            name: Optional task name override (defaults to the directory
                name). Used by the high-resolution registry alias.
            name_display: Optional human-readable label override.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tasks/test_gaussian_random_field.py -v`
Expected: PASS — all GRF tests, including the new class and the existing `test_metadata` (`name == "gaussian_random_field"` for `field_size=16`).

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/tasks/gaussian_random_field/task.py \
        tests/tasks/test_gaussian_random_field.py
git commit -m "feat(grf): optional name/name_display ctor args"
```

---

### Task 2: Register the `gaussian_random_field_256` alias

**Files:**
- Modify: `src/sbibm_jax/tasks/__init__.py:65-69` (registry branch) and `:90` (`tasks_extra`)
- Test: `tests/tasks/test_gaussian_random_field.py`, `tests/hf/test_registry.py`, `tests/hf/test_metadata.py`

- [ ] **Step 1: Write the failing tests**

In `tests/tasks/test_gaussian_random_field.py`, append:

```python
class TestHighResVariant:
    def test_registry_alias(self):
        from sbibm_jax import get_task
        task = get_task("gaussian_random_field_256")
        assert task.name == "gaussian_random_field_256"
        assert task.name_display == "Gaussian Random Field (256x256)"
        assert task.field_size == 256
        assert task.dim_data == 256 * 256
        assert task.hf_data_kind == "image"
        assert task.hf_data_shape == (256, 256)
        assert task.hf_split_sizes == {
            "train": 100_000, "validation": 10_000, "test": 10_000,
        }

    def test_in_available_tasks(self):
        from sbibm_jax import get_available_tasks
        assert "gaussian_random_field_256" in get_available_tasks()

    def test_smoke_generation(self):
        from sbibm_jax import get_task
        task = get_task("gaussian_random_field_256")
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
        theta = task.get_prior(k1, num_samples=2)
        sim = task.get_simulator(k2)
        data = sim(k3, theta)
        assert data.shape == (2, 256 * 256)
        assert jnp.all(jnp.isfinite(data))
        assert task.unflatten_data(data).shape == (2, 256, 256)
```

In `tests/hf/test_registry.py`, inside class `TestRegistryRealTasks`, append a method:

```python
    def test_grf_256_selects_image_exporter(self):
        task = get_task("gaussian_random_field_256")
        exp = get_exporter(task)
        assert isinstance(exp, ImageExporter)
        assert exp.data_shape == (256, 256)
        assert exp.train_size == 100_000
        assert exp.val_size == 10_000
        assert exp.test_size == 10_000
```

In `tests/hf/test_metadata.py`, inside class `TestMakeMetadata`, append a method:

```python
    def test_grf_256_image_schema(self):
        meta = make_metadata(["gaussian_random_field_256"])
        m = meta["gaussian_random_field_256"]
        assert m["data_kind"] == "image"
        assert m["data_shape"] == [256, 256]
        assert m["dim_data"] == 256 * 256
        assert m["has_reference"] is False
        assert m["splits"] == {
            "train": 100_000, "validation": 10_000, "test": 10_000,
        }
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest \
  tests/tasks/test_gaussian_random_field.py::TestHighResVariant \
  tests/hf/test_registry.py::TestRegistryRealTasks::test_grf_256_selects_image_exporter \
  tests/hf/test_metadata.py::TestMakeMetadata::test_grf_256_image_schema -v
```
Expected: FAIL — `get_task("gaussian_random_field_256")` raises `NotImplementedError: Task 'gaussian_random_field_256' not found.`

- [ ] **Step 3: Add the registry branch and `tasks_extra` entry**

In `src/sbibm_jax/tasks/__init__.py`, immediately after the existing `gaussian_random_field` branch (the one returning `GaussianRandomField(*args, **kwargs)`), add:

```python
    elif task_name == "gaussian_random_field_256":
        from sbibm_jax.tasks.gaussian_random_field.task import (
            GaussianRandomField,
        )
        return GaussianRandomField(
            *args,
            field_size=256,
            name="gaussian_random_field_256",
            name_display="Gaussian Random Field (256x256)",
            **kwargs,
        )
```

Then in `get_available_tasks`, change:

```python
    tasks_extra = ["slcp_distractors", "bernoulli_glm_raw"]
```

to:

```python
    tasks_extra = [
        "slcp_distractors",
        "bernoulli_glm_raw",
        "gaussian_random_field_256",
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest \
  tests/tasks/test_gaussian_random_field.py \
  tests/hf/test_registry.py tests/hf/test_metadata.py -v
```
Expected: PASS — all tests, including the three new ones.

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/tasks/__init__.py \
        tests/tasks/test_gaussian_random_field.py \
        tests/hf/test_registry.py tests/hf/test_metadata.py
git commit -m "feat(grf): register gaussian_random_field_256 alias"
```

---

### Task 3: Add `--chunk-size` flag to the driver

**Files:**
- Modify: `scripts/make_dataset.py` (argparse block + `build_opts` assembly)
- Test: `tests/hf/test_driver.py`

- [ ] **Step 1: Write the failing test**

In `tests/hf/test_driver.py`, append (it reuses the existing `_load_driver` helper):

```python
def test_chunk_size_forwarded_to_upload(monkeypatch, tmp_path):
    mod = _load_driver()
    out = tmp_path / "metadata.json"

    monkeypatch.setattr(mod, "fetch_remote_metadata", lambda repo: {})
    monkeypatch.setattr(mod, "upload_metadata", lambda path, repo: None)

    upload_opts = {}

    def fake_upload_dataset(repo, name, **opts):
        upload_opts[name] = opts

    monkeypatch.setattr(mod, "upload_dataset", fake_upload_dataset)

    mod.main(["--tasks", "gaussian_linear", "--chunk-size", "256",
              "--metadata-path", str(out)])

    assert upload_opts["gaussian_linear"]["chunk_size"] == 256


def test_chunk_size_absent_by_default(monkeypatch, tmp_path):
    mod = _load_driver()
    out = tmp_path / "metadata.json"

    monkeypatch.setattr(mod, "fetch_remote_metadata", lambda repo: {})
    monkeypatch.setattr(mod, "upload_metadata", lambda path, repo: None)

    upload_opts = {}

    def fake_upload_dataset(repo, name, **opts):
        upload_opts[name] = opts

    monkeypatch.setattr(mod, "upload_dataset", fake_upload_dataset)

    mod.main(["--tasks", "gaussian_linear", "--metadata-path", str(out)])

    assert "chunk_size" not in upload_opts["gaussian_linear"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/hf/test_driver.py::test_chunk_size_forwarded_to_upload -v`
Expected: FAIL — argparse errors with `unrecognized arguments: --chunk-size 256` (SystemExit).

- [ ] **Step 3: Add the flag and thread it through**

In `scripts/make_dataset.py`, in `parse_args`, after the `--test-size` argument add:

```python
    p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Per-chunk generation batch size (rows). Lower it if a GPU "
            "OOMs on large image tasks; defaults to config.DEFAULT_CHUNK_SIZE."
        ),
    )
```

In `main`, in the `build_opts` assembly block (alongside the size overrides, before `build_opts["master_seed"] = args.master_seed`), add:

```python
    if args.chunk_size is not None:
        build_opts["chunk_size"] = args.chunk_size
```

(Confirm: `build_opts` is forwarded only via `upload_dataset(repo, name, **build_opts)` → `build_dataset(task_name, **build_opts)`, which already accepts `chunk_size`. `make_metadata` does not receive it and does not need it — chunk size does not affect recorded metadata.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/hf/test_driver.py -v`
Expected: PASS — all driver tests, including the two new ones.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_dataset.py tests/hf/test_driver.py
git commit -m "feat(hf): add --chunk-size flag to make_dataset driver"
```

---

### Task 4: Documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Document the variant in the registry/aliases description**

In `CLAUDE.md`, in the **Registry** paragraph, extend the alias examples. Find:

```
the same class passing different kwargs (e.g. `slcp_distractors` → `SLCP(distractors=True)`,
`bernoulli_glm_raw` → `BernoulliGLM(summary="raw")`, `gaussian_nonlinear` → `SLCP`).
```

and append the new alias to that list, e.g.:

```
the same class passing different kwargs (e.g. `slcp_distractors` → `SLCP(distractors=True)`,
`bernoulli_glm_raw` → `BernoulliGLM(summary="raw")`, `gaussian_nonlinear` → `SLCP`,
`gaussian_random_field_256` → `GaussianRandomField(field_size=256)`).
```

- [ ] **Step 2: Document the variant in the image-tasks convention note**

In the **Conventions** section, find the sentence beginning "The image tasks `gaussian_random_field` and `toy_lensing` declare `hf_data_kind="image"`…" and add a sentence noting the high-res variant, e.g. after that sentence:

```
The `gaussian_random_field_256` alias is a 256×256 high-resolution variant of
`gaussian_random_field` (same `hf_split_sizes` cap of 100k train); each 256×256
float32 row is ~256 KiB, so its train split is ~25 GiB on disk — generated
incrementally via the chunked `Dataset.from_generator` streaming, never held
whole in RAM.
```

- [ ] **Step 3: Document the `--chunk-size` flag**

In the **Commands** section, near the `make_dataset.py` examples, add a line noting the flag, e.g.:

```
Pass `--chunk-size N` to shrink the per-chunk generation batch if a GPU OOMs on
large image tasks (e.g. `gaussian_random_field_256`).
```

- [ ] **Step 4: Verify the full suite and lint are clean**

Run: `uv run pytest -m "not slow"`
Expected: PASS (no regressions).

Run: `uv run flake8 src tests scripts`
Expected: no NEW violations introduced by this change (baseline flake8 has pre-existing E501s elsewhere; verify none of the touched files gained a violation — compare against `git stash`/`HEAD` if unsure).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document gaussian_random_field_256 + --chunk-size"
```

---

## Self-Review Notes

- **Spec coverage:** variant task (Tasks 1–2), 100k/10k/10k splits (inherited via GRF `hf_split_sizes`, asserted in Task 2), `--chunk-size` knob (Task 3), docs (Task 4). All design points covered.
- **No new directory / no split-size change / no `hf_chunk_size` attribute** — confirmed out of scope per the design.
- **Name-collision safety:** the existing `test_metadata` asserts `GaussianRandomField(field_size=16).name == "gaussian_random_field"`; Task 1's `name or <dir name>` default preserves it (Task 1 Step 1 also asserts this directly).
- **CI-cost safety:** no test triggers a real full build; the smoke test generates only n=2 rows. `--all`/`get_available_tasks` is only membership-asserted, not exact-set.
