# Test-repo default, `--prod` guardrail, and non-destructive metadata — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scripts/make_dataset.py` upload to the test HF repo by default (requiring `--prod` for production) and make `metadata.json` uploads non-destructive so a subset run preserves other tasks' entries.

**Architecture:** Two additive, well-bounded pieces. (1) The driver resolves the target repo from a new `--prod` flag (test by default), prints a TEST/PRODUCTION banner, and drops the old `--repo` flag. (2) Metadata becomes non-destructive via two new single-purpose functions — `fetch_remote_metadata(repo)` (network, in `upload.py`) and a pure `merge_metadata(remote, local)` (in `metadata.py`) — orchestrated by the driver: build subset → fetch remote → merge → write merged → upload → delete local file. `upload_metadata` stays a pure uploader, so its existing test is unaffected.

**Tech Stack:** Python 3.12, `huggingface_hub` 1.18.0 (`hf_hub_download`, `upload_file`, `EntryNotFoundError`/`RepositoryNotFoundError` from `huggingface_hub.utils`), `pytest`, `uv`.

**Spec:** `docs/superpowers/specs/2026-06-06-make-dataset-test-repo-design.md`

---

## File Structure

- `src/sbibm_jax/hf/config.py` — add `TEST_REPO` constant.
- `src/sbibm_jax/hf/metadata.py` — add pure `merge_metadata(remote, local)`.
- `src/sbibm_jax/hf/upload.py` — add `fetch_remote_metadata(repo_name)`; `upload_metadata` unchanged.
- `src/sbibm_jax/hf/__init__.py` — re-export the two new functions.
- `scripts/make_dataset.py` — drop `--repo`, add `--prod`, banner, merge orchestration, post-upload delete; update docstring.
- `CLAUDE.md` — update the `make_dataset.py` usage block.
- Tests: `tests/hf/test_import_guard.py`, `tests/hf/test_metadata.py`, `tests/hf/test_upload.py`, `tests/hf/test_driver.py`.

## Pre-existing caveat (do NOT fix here)

`tests/hf/test_import_guard.py::test_config_defaults` asserts `config.DEFAULT_MASTER_SEED == 0`, but the working tree has an **uncommitted** change setting it to `42`. That single assertion may fail independently of this work. It is out of scope — do not change `DEFAULT_MASTER_SEED` or that assertion. When verifying, run the specific test classes/files added below rather than relying on a fully-green `test_import_guard.py`.

---

## Task 1: Add `TEST_REPO` to config

**Files:**
- Modify: `src/sbibm_jax/hf/config.py:5`
- Test: `tests/hf/test_import_guard.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/hf/test_import_guard.py` (new function, independent of the seed assertion):

```python
def test_test_repo_constant():
    from sbibm_jax.hf import config

    assert config.TEST_REPO == "aurelio-amerio/SBI-benchmarks-test"
    # production constant is unchanged
    assert config.DEFAULT_REPO == "aurelio-amerio/SBI-benchmarks"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_import_guard.py::test_test_repo_constant -v`
Expected: FAIL — `AttributeError: module 'sbibm_jax.hf.config' has no attribute 'TEST_REPO'`.

- [ ] **Step 3: Add the constant**

In `src/sbibm_jax/hf/config.py`, change the `DEFAULT_REPO` line (line 5) to add a clarifying comment and the new constant:

```python
# Production dataset repo. NOTE: this is NOT the CLI default — make_dataset.py
# targets TEST_REPO unless --prod is passed.
DEFAULT_REPO: str = "aurelio-amerio/SBI-benchmarks"

# Default target for make_dataset.py (safe; use --prod to hit DEFAULT_REPO).
TEST_REPO: str = "aurelio-amerio/SBI-benchmarks-test"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_import_guard.py::test_test_repo_constant -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/hf/config.py tests/hf/test_import_guard.py
git commit -m "feat(hf): add TEST_REPO config constant"
```

---

## Task 2: Pure `merge_metadata`

**Files:**
- Modify: `src/sbibm_jax/hf/metadata.py` (append after `make_metadata`, ~line 55)
- Test: `tests/hf/test_metadata.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/hf/test_metadata.py`:

```python
from sbibm_jax.hf.metadata import merge_metadata


class TestMergeMetadata:
    def test_disjoint_keys_union(self):
        assert merge_metadata({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_local_overrides_shared_key(self):
        assert merge_metadata({"a": 1}, {"a": 2}) == {"a": 2}

    def test_empty_remote_returns_local(self):
        assert merge_metadata({}, {"a": 1}) == {"a": 1}

    def test_empty_local_returns_remote(self):
        assert merge_metadata({"a": 1}, {}) == {"a": 1}

    def test_does_not_mutate_inputs(self):
        remote, local = {"a": 1}, {"b": 2}
        merge_metadata(remote, local)
        assert remote == {"a": 1}
        assert local == {"b": 2}
```

(The existing `from sbibm_jax.hf.metadata import make_metadata` import stays; this adds a second import line.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_metadata.py::TestMergeMetadata -v`
Expected: FAIL — `ImportError: cannot import name 'merge_metadata'`.

- [ ] **Step 3: Add the function**

Append to `src/sbibm_jax/hf/metadata.py`:

```python
def merge_metadata(remote: dict, local: dict) -> dict:
    """Merge freshly-built ``local`` entries over ``remote``.

    Tasks present in ``local`` overwrite their own entries; every other task
    already documented in ``remote`` is preserved. Pure — no I/O, no mutation
    of the inputs.
    """
    return {**remote, **local}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_metadata.py::TestMergeMetadata -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/hf/metadata.py tests/hf/test_metadata.py
git commit -m "feat(hf): add pure merge_metadata helper"
```

---

## Task 3: `fetch_remote_metadata` in `upload.py`

**Files:**
- Modify: `src/sbibm_jax/hf/upload.py:8` (imports) and append a function
- Test: `tests/hf/test_upload.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/hf/test_upload.py` (top of file already has `import pytest` and `import sbibm_jax.hf.upload as upload_mod`):

```python
import json

from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from sbibm_jax.hf.upload import fetch_remote_metadata


class TestFetchRemoteMetadata:
    def test_returns_parsed_dict(self, monkeypatch, tmp_path):
        f = tmp_path / "metadata.json"
        f.write_text(json.dumps({"two_moons": {"dim": 2}}))
        monkeypatch.setattr(
            upload_mod, "hf_hub_download", lambda **kw: str(f))
        assert fetch_remote_metadata("user/repo") == {"two_moons": {"dim": 2}}

    def test_entry_not_found_returns_empty(self, monkeypatch):
        def boom(**kw):
            raise EntryNotFoundError.__new__(EntryNotFoundError)
        monkeypatch.setattr(upload_mod, "hf_hub_download", boom)
        assert fetch_remote_metadata("user/repo") == {}

    def test_repo_not_found_returns_empty(self, monkeypatch):
        def boom(**kw):
            raise RepositoryNotFoundError.__new__(RepositoryNotFoundError)
        monkeypatch.setattr(upload_mod, "hf_hub_download", boom)
        assert fetch_remote_metadata("user/repo") == {}

    def test_other_error_propagates(self, monkeypatch):
        def boom(**kw):
            raise ValueError("transient network error")
        monkeypatch.setattr(upload_mod, "hf_hub_download", boom)
        with pytest.raises(ValueError):
            fetch_remote_metadata("user/repo")
```

Note: `EntryNotFoundError`/`RepositoryNotFoundError` subclass `HfHubHTTPError` and need a `response` kwarg to construct normally; `cls.__new__(cls)` makes a raisable, catchable instance without it.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_upload.py::TestFetchRemoteMetadata -v`
Expected: FAIL — `ImportError: cannot import name 'fetch_remote_metadata'`.

- [ ] **Step 3: Implement**

In `src/sbibm_jax/hf/upload.py`, replace the import block (lines 8-10) with:

```python
import json

from huggingface_hub import hf_hub_download, upload_file
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from sbibm_jax.hf.build import build_dataset
```

Then append this function (after `upload_metadata`, before `upload_dataset` is fine too — keep it grouped with `upload_metadata`):

```python
def fetch_remote_metadata(repo_name: str) -> dict:
    """Return the repo's existing metadata.json as a dict, or {} if absent.

    Downloads with force_download to avoid a stale local cache. A missing file
    (EntryNotFoundError) or a non-existent repo (RepositoryNotFoundError) is
    treated as "no remote metadata" and returns {}. Any other error (auth,
    HTTP/connection) propagates — a transient failure must never be silently
    treated as an empty remote, which would drop sibling task entries on merge.
    """
    try:
        local_path = hf_hub_download(
            repo_id=repo_name,
            filename="metadata.json",
            repo_type="dataset",
            force_download=True,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return {}
    with open(local_path) as f:
        return json.load(f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_upload.py::TestFetchRemoteMetadata -v`
Expected: PASS (4 tests). Also confirm the pre-existing `TestUploadMetadata` still passes:
Run: `uv run pytest tests/hf/test_upload.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/hf/upload.py tests/hf/test_upload.py
git commit -m "feat(hf): add fetch_remote_metadata for non-destructive merge"
```

---

## Task 4: Re-export new helpers from the package

**Files:**
- Modify: `src/sbibm_jax/hf/__init__.py:20-33`
- Test: `tests/hf/test_import_guard.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/hf/test_import_guard.py`:

```python
def test_new_helpers_reexported():
    import sbibm_jax.hf as hf

    assert hasattr(hf, "merge_metadata")
    assert hasattr(hf, "fetch_remote_metadata")
    assert "merge_metadata" in hf.__all__
    assert "fetch_remote_metadata" in hf.__all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_import_guard.py::test_new_helpers_reexported -v`
Expected: FAIL — `AssertionError` (attributes/`__all__` entries missing).

- [ ] **Step 3: Implement**

In `src/sbibm_jax/hf/__init__.py`, update the two import lines and `__all__`:

```python
from sbibm_jax.hf.metadata import make_metadata, merge_metadata  # noqa: E402
from sbibm_jax.hf.registry import get_exporter  # noqa: E402
from sbibm_jax.hf.upload import (  # noqa: E402
    fetch_remote_metadata,
    upload_dataset,
    upload_metadata,
)
```

```python
__all__ = [
    "build_dataset",
    "config",
    "fetch_remote_metadata",
    "get_exporter",
    "make_metadata",
    "merge_metadata",
    "upload_dataset",
    "upload_metadata",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/hf/test_import_guard.py::test_new_helpers_reexported tests/hf/test_import_guard.py::test_package_imports -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/hf/__init__.py tests/hf/test_import_guard.py
git commit -m "feat(hf): re-export merge_metadata and fetch_remote_metadata"
```

---

## Task 5: Driver — `--prod` flag, target banner, drop `--repo`

**Files:**
- Modify: `scripts/make_dataset.py` (docstring, `parse_args`, `main`)
- Test: `tests/hf/test_driver.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/hf/test_driver.py`:

```python
def test_dry_run_banner_defaults_to_test(tmp_path):
    out = tmp_path / "metadata.json"
    result = subprocess.run(
        [sys.executable, str(DRIVER), "--tasks", "gaussian_linear",
         "--metadata-path", str(out), "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "(TEST)" in result.stdout
    assert "aurelio-amerio/SBI-benchmarks-test" in result.stdout


def test_dry_run_banner_prod(tmp_path):
    out = tmp_path / "metadata.json"
    result = subprocess.run(
        [sys.executable, str(DRIVER), "--tasks", "gaussian_linear",
         "--metadata-path", str(out), "--prod", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "(PRODUCTION)" in result.stdout
    # the production repo id, without the -test suffix, must appear
    assert "aurelio-amerio/SBI-benchmarks\n" in result.stdout \
        or "aurelio-amerio/SBI-benchmarks " in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_driver.py::test_dry_run_banner_defaults_to_test -v`
Expected: FAIL — no `(TEST)` in stdout (banner not implemented).

- [ ] **Step 3: Implement the CLI + banner**

In `scripts/make_dataset.py`:

(a) Update the module docstring usage block to:

```python
"""Build (and optionally upload) HuggingFace datasets for sbibm_jax tasks.

Uploads target the TEST repo by default; pass --prod for production.

    # Test repo (default), all available tasks, real upload:
    uv run python scripts/make_dataset.py --all

    # Production repo:
    uv run python scripts/make_dataset.py --all --prod

    # Explicit task list, dry-run (writes metadata.json, no HF push):
    uv run python scripts/make_dataset.py --tasks gaussian_linear two_moons --dry-run

    # Custom split sizes:
    uv run python scripts/make_dataset.py --tasks two_moons \
        --train-size 1000 --val-size 100 --test-size 100
"""
```

(b) In `parse_args`, **delete** the `--repo` argument block (lines 42-46) and **add** in its place:

```python
    p.add_argument(
        "--prod",
        action="store_true",
        help=(
            "Upload to the PRODUCTION repo (config.DEFAULT_REPO). Without it, "
            "uploads target the test repo (config.TEST_REPO)."
        ),
    )
```

Also update the `--metadata-path` help to mention deletion:

```python
    p.add_argument(
        "--metadata-path",
        default="metadata.json",
        help=(
            "Where to write metadata.json (default: ./metadata.json). Deleted "
            "after a successful real upload; kept on --dry-run."
        ),
    )
```

(c) In `main`, after the `task_names` resolution block and before `build_opts`, add target resolution + banner:

```python
    repo = config.DEFAULT_REPO if args.prod else config.TEST_REPO
    label = "PRODUCTION" if args.prod else "TEST"
    print(f"Target repo: {repo}  ({label})")
```

(d) Replace the two `args.repo` uses in the upload section with `repo` (full orchestration is finalized in Task 6; for this task just make it use `repo`):

```python
    upload_metadata(str(metadata_path), repo)
    for name in task_names:
        print(f"Uploading dataset for task: {name}")
        upload_dataset(repo, name, **build_opts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/hf/test_driver.py -v`
Expected: all PASS (help, dry-run metadata, both banner tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/make_dataset.py tests/hf/test_driver.py
git commit -m "feat(hf): default make_dataset to test repo; add --prod guardrail + banner"
```

---

## Task 6: Driver — non-destructive merge orchestration + local-file delete

**Files:**
- Modify: `scripts/make_dataset.py` (imports + `main` upload path)
- Test: `tests/hf/test_driver.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/hf/test_driver.py` (add `import importlib.util` and `import json` at the top of the file):

```python
def _load_driver():
    spec = importlib.util.spec_from_file_location("make_dataset_mod", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_real_upload_merges_remote_and_deletes_local(monkeypatch, tmp_path):
    mod = _load_driver()
    out = tmp_path / "metadata.json"

    # Remote already documents a *different* task; merge must preserve it.
    monkeypatch.setattr(
        mod, "fetch_remote_metadata", lambda repo: {"two_moons": {"x": 1}})

    captured = {}

    def fake_upload_metadata(path, repo):
        captured["content"] = json.loads(open(path).read())
        captured["repo"] = repo

    monkeypatch.setattr(mod, "upload_metadata", fake_upload_metadata)
    monkeypatch.setattr(mod, "upload_dataset", lambda repo, name, **o: None)

    mod.main(["--tasks", "gaussian_linear", "--metadata-path", str(out)])

    # merged: remote task preserved + selected task added
    assert "two_moons" in captured["content"]
    assert "gaussian_linear" in captured["content"]
    # default target is the test repo
    assert captured["repo"] == "aurelio-amerio/SBI-benchmarks-test"
    # local artifact deleted -> clean state
    assert not out.exists()


def test_dry_run_keeps_local_and_skips_network(monkeypatch, tmp_path):
    mod = _load_driver()
    out = tmp_path / "metadata.json"

    def boom(repo):
        raise AssertionError("network must not be touched on --dry-run")

    monkeypatch.setattr(mod, "fetch_remote_metadata", boom)
    mod.main(["--tasks", "gaussian_linear", "--metadata-path", str(out),
              "--dry-run"])
    assert out.exists()  # kept for inspection
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/hf/test_driver.py::test_real_upload_merges_remote_and_deletes_local -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'fetch_remote_metadata'` (driver doesn't import it yet) or the merge/delete behavior is absent.

- [ ] **Step 3: Implement the orchestration**

In `scripts/make_dataset.py`:

(a) Add `import json` to the top imports, and extend the package import to pull in the two helpers:

```python
from sbibm_jax.hf import (
    config,
    fetch_remote_metadata,
    make_metadata,
    merge_metadata,
    upload_dataset,
    upload_metadata,
)
```

(b) Capture the built dict from `make_metadata`:

```python
    local_meta = make_metadata(
        task_names, output_path=metadata_path, split_sizes=split_sizes)
    print(f"Wrote {metadata_path}")
```

(c) Keep the dry-run early return as-is (file kept):

```python
    if args.dry_run:
        print("Dry run — skipping HF uploads.")
        return
```

(d) Replace the upload section with merge → write → upload → delete:

```python
    remote_meta = fetch_remote_metadata(repo)
    merged_meta = merge_metadata(remote_meta, local_meta)
    metadata_path.write_text(json.dumps(merged_meta, indent=4))
    upload_metadata(str(metadata_path), repo)
    for name in task_names:
        print(f"Uploading dataset for task: {name}")
        upload_dataset(repo, name, **build_opts)
    metadata_path.unlink(missing_ok=True)
    print(f"Removed local {metadata_path} (clean state).")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/hf/test_driver.py -v`
Expected: all PASS (help, dry-run metadata, both banner tests, merge+delete, dry-run-keeps-local).

- [ ] **Step 5: Commit**

```bash
git add scripts/make_dataset.py tests/hf/test_driver.py
git commit -m "feat(hf): merge metadata against remote and clean up local file on upload"
```

---

## Task 7: Docs — update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (the `make_dataset.py` usage block, ~lines 48-53)

- [ ] **Step 1: Update the usage block**

Replace the fenced `make_dataset.py` example block in `CLAUDE.md` with:

````markdown
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
````

- [ ] **Step 2: Sanity check**

Run: `git diff CLAUDE.md` and confirm only the intended block changed.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document test-repo default + --prod + non-destructive metadata"
```

---

## Task 8: Full verification

- [ ] **Step 1: Lint**

Run: `uv run flake8 src tests scripts`
Expected: no output (clean). Fix any new findings in the files you touched.

- [ ] **Step 2: Run the hf test suite (scoped)**

Run: `uv run pytest tests/hf -v`
Expected: all PASS **except** possibly `test_import_guard.py::test_config_defaults` if the uncommitted `DEFAULT_MASTER_SEED = 42` change is present (pre-existing, out of scope). Every test added/modified by this plan must pass.

- [ ] **Step 3: Manual card-preservation verification on the TEST repo**

This is the one path the metadata merge does not cover: `push_to_hub` makes the
`datasets` library read-modify-write the repo `README.md` YAML (`configs:` /
`dataset_info:` listing every config). Confirm a subset push does not drop other
configs. Requires HF auth (`huggingface-cli login`).

```bash
# 1. Upload one small task to the test repo:
uv run python scripts/make_dataset.py --tasks gaussian_linear \
    --train-size 64 --val-size 16 --test-size 16

# 2. Upload a different small task to the same repo:
uv run python scripts/make_dataset.py --tasks two_moons \
    --train-size 64 --val-size 16 --test-size 16
```

Then verify (e.g. in `uv run python`):

```python
from huggingface_hub import hf_hub_download
import json
repo = "aurelio-amerio/SBI-benchmarks-test"
meta = json.load(open(hf_hub_download(
    repo, "metadata.json", repo_type="dataset", force_download=True)))
assert {"gaussian_linear", "two_moons"} <= set(meta), meta.keys()   # metadata merged

from datasets import load_dataset
load_dataset(repo, "gaussian_linear", split="test")  # first task still resolves
load_dataset(repo, "two_moons", split="test")        # second task resolves
```

Also open the test repo's `README.md` on the Hub and confirm **both** configs are
listed. If a config was dropped, the `datasets` card writer does not merge on this
version — record it as a follow-up (card-merge step / alternate upload), out of
scope for this change.

- [ ] **Step 4: Final commit (if any verification fixups were needed)**

```bash
git add -A
git commit -m "test(hf): verification fixups for make_dataset test-repo feature"
```

---

## Self-review notes

- **Spec coverage:** Part 1 (test default / `--prod` / banner / drop `--repo`) → Tasks 1, 5. Part 2 (fetch + merge + orchestration + delete) → Tasks 2, 3, 6. Re-export → Task 4. Docs → Task 7. Card-preservation verification → Task 8 Step 3.
- **Type/name consistency:** `merge_metadata(remote, local)`, `fetch_remote_metadata(repo_name)`, `repo`/`local_meta`/`remote_meta`/`merged_meta` names are used identically across tasks.
- **`upload_metadata` unchanged**, so the existing `tests/hf/test_upload.py::TestUploadMetadata::test_calls_upload_file` keeps passing without edits.
