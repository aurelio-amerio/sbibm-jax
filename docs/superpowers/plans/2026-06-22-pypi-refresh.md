# PyPI Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `sbibm-jax` publishable to PyPI and approachable for newcomers by adding GitHub Actions (publish + CI), full PyPI metadata, and a README with dataset-usage instructions.

**Architecture:** Three independent deliverables — two GitHub Actions workflows copied/adapted from the sibling GenSBI repos, `pyproject.toml` metadata, and a new `README.md`. No package code changes. Each task ends with a concrete validation command.

**Tech Stack:** `uv` + `uv_build`, GitHub Actions (OIDC trusted publishing), flake8, pytest/pytest-cov/pytest-xdist, genbadge, JAX/NumPyro.

## Global Constraints

- Python is pinned to **3.12** (all workflow `uv python install` steps use `3.12`).
- Repo: `github.com/aurelio-amerio/sbibm-jax`; author **Aurelio Amerio** `dev@gensbi.com`; license **MIT**.
- `/lhome/ific/a/aamerio/data/github/sbibm-jax` is a symlink to the primary lustre checkout — edit either path, it's one file.
- Build backend is `uv_build`; do **not** change `name`, `version` (`0.1.1`), `requires-python`, dependencies, optional-dependencies, or dependency-groups.
- Production dataset repo: `aurelio-amerio/SBI-benchmarks`; in-code loader default: `aurelio-amerio/SBI-benchmarks-test`.
- Tests are CPU-forced by `pytest-env` and run under `-n 2` (both in `pyproject.toml`); the suite needs no network (Hub is mocked) and no `pypesto` (petab tests `skipif`).
- `uv`/`uv build`/`uv sync` may fail under the command sandbox (read-only caches); if a `uv` step errors with a permission/cache error, re-run it with the sandbox disabled. Pure-Python validation steps use `.venv/bin/python` and work in-sandbox.
- flake8 baseline on `src tests` is 283 pre-existing violations; this plan adds **no** Python, so that count must stay 283 (judge by *new* violations).

---

### Task 1: PyPI publish workflow

**Files:**
- Create: `.github/workflows/python-publish.yml`

**Interfaces:**
- Consumes: nothing.
- Produces: a workflow that, on a published GitHub Release, builds and uploads to `https://pypi.org/p/sbibm-jax` via OIDC trusted publishing (no API token). A PyPI Trusted Publisher and a `pypi` GitHub environment must be configured manually by the maintainer — out of scope for this task.

- [ ] **Step 1: Create the workflow file**

Create `.github/workflows/python-publish.yml` with exactly:

```yaml
name: Publish Python 🐍 distribution 📦 to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    name: Build and publish 📦 to PyPI
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/sbibm-jax
    permissions:
      id-token: write
      contents: read
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v7
      - name: Set up Python
        run: uv python install 3.12
      - name: Build
        run: uv build
      - name: Publish distribution 📦 to PyPI
        run: uv publish
```

- [ ] **Step 2: Validate the YAML parses**

Run: `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/python-publish.yml')); print('publish yaml OK')"`
Expected: `publish yaml OK`

- [ ] **Step 3: Confirm the PyPI URL and Python version**

Run: `grep -E "pypi.org/p/sbibm-jax|uv python install 3.12" .github/workflows/python-publish.yml`
Expected: both lines present (URL points at `sbibm-jax`, Python is 3.12).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/python-publish.yml
git commit -m "ci: add PyPI publish workflow (OIDC trusted publishing)"
```

---

### Task 2: Build/CI workflow

**Files:**
- Create: `.github/workflows/python-app.yml`

**Interfaces:**
- Consumes: dependency groups `lint`, `test`, `hf`, `loader`; pytest markers `slow`, `experimental`; package `sbibm_jax` (for `--cov`).
- Produces: a `Build` workflow with `lint`, `test`, and `publish-badges` jobs. `publish-badges` commits `img/badges/coverage.svg` and `img/badges/tests.svg` to `main` (referenced by the README in Task 4).

- [ ] **Step 1: Create the workflow file**

Create `.github/workflows/python-app.yml` with exactly:

```yaml
# Installs dependencies, lints, runs the test suite, and publishes
# coverage/test badges. Adapted from GenSBI (Sphinx docs jobs dropped;
# badge commit uses the default GITHUB_TOKEN instead of a GitHub App).
name: Build

on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["main"]

permissions:
  contents: read

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v7
      - name: Set up Python 3.12
        run: uv python install 3.12
      - name: Install dependencies
        run: uv sync --group lint
      - name: Lint with flake8
        run: |
          # stop the build if there are Python syntax errors or undefined names
          uv run flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
          # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
          uv run flake8 src tests --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v7
      - name: Set up Python 3.12
        run: uv python install 3.12
      - name: Install dependencies
        run: uv sync --group test --group hf --group loader
      - name: Run fast tests
        run: |
          uv run pytest -m "not slow and not experimental" --junitxml=reports/junit/junit-fast.xml --cov=sbibm_jax --cov-report=term-missing
      - name: Run slow tests
        run: |
          uv run pytest -m "slow" --junitxml=reports/junit/junit-slow.xml --cov=sbibm_jax --cov-append --cov-report=term-missing
      - name: Run experimental tests
        run: |
          uv run pytest -m "experimental" --junitxml=reports/junit/junit-experimental.xml --cov=sbibm_jax --cov-append --cov-report=xml:reports/coverage/coverage.xml --cov-report=term-missing
      - name: Merge Test Reports
        run: |
          uv run python -m junitparser merge reports/junit/junit-fast.xml reports/junit/junit-slow.xml reports/junit/junit-experimental.xml reports/junit/junit.xml
      - name: Generate Badges
        run: |
          uv run genbadge coverage --input-file reports/coverage/coverage.xml --output-file reports/badges/coverage.svg
          uv run genbadge tests --input-file reports/junit/junit.xml --output-file reports/badges/tests.svg
      - name: Upload Test Reports
        uses: actions/upload-artifact@v4
        with:
          name: test-reports
          path: reports/

  publish-badges:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: write
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Download Test Reports
        uses: actions/download-artifact@v4
        with:
          name: test-reports
          path: reports
      - name: Prepare Badges
        run: |
          mkdir -p img/badges
          cp -r reports/badges/*.svg img/badges/
      - name: Commit and push badges
        uses: stefanzweifel/git-auto-commit-action@v6
        with:
          commit_message: "Update badges [skip ci]"
          branch: main
          file_pattern: img/badges/*.svg
```

- [ ] **Step 2: Validate the YAML parses**

Run: `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/python-app.yml')); print('build yaml OK')"`
Expected: `build yaml OK`

- [ ] **Step 3: Verify the referenced groups and markers exist**

Run: `.venv/bin/python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); g=d['dependency-groups']; assert all(k in g for k in ('lint','test','hf','loader')), 'missing group'; m=' '.join(d['tool']['pytest']['ini_options']['markers']); assert 'slow' in m and 'experimental' in m; print('groups+markers OK')"`
Expected: `groups+markers OK`

- [ ] **Step 4: Sanity-check the marker passes are non-empty (no pytest exit-5)**

Run: `JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python -m pytest -m "slow" --co -q 2>/dev/null | tail -3`
Expected: at least one collected item (the GRF slow test is not pypesto-gated). The `experimental` pass collects the petab tests and skips them — collection is non-empty either way, so neither pass triggers pytest's "no tests collected" exit code 5.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/python-app.yml
git commit -m "ci: add Build workflow (lint, test, coverage badges)"
```

---

### Task 3: Full PyPI metadata in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml` (the `[project]` table; add `[project.urls]`)

**Interfaces:**
- Consumes: the existing `[project]` table and `LICENSE` file.
- Produces: a wheel/sdist whose metadata carries description, authors, license, urls, classifiers, keywords. No new dependencies.

- [ ] **Step 1: Rewrite the description**

In `pyproject.toml`, replace the line:

```toml
description = "Simulation-Based Inference Benchmark — JAX rewrite"
```

with:

```toml
description = "A JAX/NumPyro rewrite of the Simulation-Based Inference Benchmark (sbibm): benchmark tasks — priors, simulators, reference posteriors — plus ready-to-stream HuggingFace datasets for evaluating SBI methods."
```

- [ ] **Step 2: Add authors, license, keywords, classifiers**

Immediately after the `readme = "README.md"` line in `[project]`, insert:

```toml
authors = [{ name = "Aurelio Amerio", email = "dev@gensbi.com" }]
license = "MIT"
license-files = ["LICENSE"]
keywords = [
    "simulation-based-inference",
    "sbi",
    "benchmark",
    "jax",
    "numpyro",
    "bayesian-inference",
    "datasets",
    "machine-learning",
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
```

(No `License ::` classifier — under PEP 639 it would duplicate the SPDX `license` expression.)

- [ ] **Step 3: Add the project URLs**

Add a new `[project.urls]` table directly after the `[project.optional-dependencies]` block closes (before `[dependency-groups]`):

```toml
[project.urls]
Homepage = "https://github.com/aurelio-amerio/sbibm-jax"
Repository = "https://github.com/aurelio-amerio/sbibm-jax"
Issues = "https://github.com/aurelio-amerio/sbibm-jax/issues"
```

- [ ] **Step 4: Validate the TOML parses and carries the new metadata**

Run:
```bash
.venv/bin/python -c "import tomllib; p=tomllib.load(open('pyproject.toml','rb'))['project']; assert p['license']=='MIT'; assert p['authors'][0]['email']=='dev@gensbi.com'; assert p['keywords'] and p['classifiers']; assert 'sbibm' in p['description'] or 'sbi' in p['description'].lower(); print('pyproject metadata OK')"
```
Expected: `pyproject metadata OK`

- [ ] **Step 5: Build the distribution and confirm the backend accepts the metadata**

Run: `uv build` (if it fails with a sandbox/cache permission error, re-run with the sandbox disabled).
Expected: a wheel and sdist appear under `dist/`. 

Then inspect the wheel metadata:
```bash
.venv/bin/python -c "
import glob, zipfile, sys
whl = sorted(glob.glob('dist/sbibm_jax-*.whl'))[-1]
z = zipfile.ZipFile(whl)
meta = next(n for n in z.namelist() if n.endswith('METADATA'))
text = z.read(meta).decode()
for needle in ('License-Expression: MIT', 'Author-email', 'Classifier: Intended Audience :: Science/Research', 'Project-URL: Homepage'):
    assert needle in text, f'missing: {needle}'
print('wheel METADATA OK:', whl)
"
```
Expected: `wheel METADATA OK: dist/sbibm_jax-0.1.1-...whl`

**Contingency:** if `uv build` rejects the SPDX form with a license/metadata error, replace the two lines from Step 2 with `license = { file = "LICENSE" }` (and delete the `license-files` line), then re-run Steps 4–5; the wheel METADATA will then show a `License:` block instead of `License-Expression`. Adjust the Step 5 assertion's `License-Expression: MIT` needle to `License:` accordingly.

- [ ] **Step 6: Remove build artifacts and commit**

```bash
rm -rf dist
git add pyproject.toml
git commit -m "build: add full PyPI metadata (authors, license, urls, classifiers)"
```

---

### Task 4: README with dataset-usage instructions

**Files:**
- Modify: `README.md` (currently empty)

**Interfaces:**
- Consumes: the verified public API — `get_task`, `get_available_tasks`; `Task.get_prior(key, num_samples)`, `get_simulator(key)`, `Simulator.__call__(key, parameters)`, `get_observation(n)`, `get_true_parameters(n)`, `get_reference_posterior_samples(n)`; `TaskDataset(name, *, repo, normalize)` with `get_train_loader(batch_size)` / `get_reference(num_observation)`; `OnlineTaskDataset(...)` with `get_online_train_loader(batch_size, seed=, num_workers=)`. Badge image paths produced by Task 2 (`img/badges/*.svg`).
- Produces: the rendered PyPI/GitHub front page.

- [ ] **Step 1: Write the README**

Overwrite `README.md` with exactly:

````markdown
# sbibm-jax

A JAX/NumPyro rewrite of the Simulation-Based Inference Benchmark (sbibm): benchmark tasks — priors, simulators, and reference posteriors — plus ready-to-stream HuggingFace datasets for evaluating SBI methods.

[![Build](https://github.com/aurelio-amerio/sbibm-jax/actions/workflows/python-app.yml/badge.svg)](https://github.com/aurelio-amerio/sbibm-jax/actions/workflows/python-app.yml)
![Tests](https://raw.githubusercontent.com/aurelio-amerio/sbibm-jax/refs/heads/main/img/badges/tests.svg)
![Coverage](https://raw.githubusercontent.com/aurelio-amerio/sbibm-jax/refs/heads/main/img/badges/coverage.svg)
[![PyPI](https://img.shields.io/pypi/v/sbibm-jax.svg)](https://pypi.org/project/sbibm-jax/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`sbibm-jax` ports the [Simulation-Based Inference Benchmark](https://github.com/sbi-benchmark/sbibm) (sbibm) from PyTorch/Pyro to JAX, NumPyro, and diffrax. Each **task** defines a prior, a simulator, reference observations, and reference posterior samples for benchmarking SBI methods. Tasks can be used directly, or consumed as pre-generated / on-the-fly **HuggingFace datasets**. It is the benchmark companion to [GenSBI](https://github.com/aurelio-amerio/GenSBI).

## Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv add sbibm-jax
```

Or using pip:

```bash
pip install sbibm-jax
```

The default JAX dependency is the CUDA 12 build (`jax[cuda12]`) for GPU support; on a CPU-only machine install a CPU build of JAX instead.

Optional extras:

```bash
pip install sbibm-jax[hf]       # build/export HuggingFace datasets
pip install sbibm-jax[loader]   # consume datasets via grain: TaskDataset / OnlineTaskDataset
pip install sbibm-jax[pypesto]  # the beer_molbiosystems PEtab task (compiles AMICI)
```

## Usage

### 1. Benchmark tasks

Use a task's prior, simulator, and reference data directly:

```python
import jax
from sbibm_jax import get_task, get_available_tasks

print(get_available_tasks())                      # every task name

task = get_task("two_moons")
key = jax.random.PRNGKey(0)

theta = task.get_prior(key, num_samples=1000)     # (1000, dim_theta)
simulator = task.get_simulator(key)
x = simulator(key, theta)                          # (1000, dim_x)

# Reference data for observation #1
x_o = task.get_observation(num_observation=1)                        # (1, dim_x)
theta_o = task.get_true_parameters(num_observation=1)                # (1, dim_theta)
posterior = task.get_reference_posterior_samples(num_observation=1)  # (N, dim_theta)
```

### 2. Offline datasets (pre-generated)

`TaskDataset` streams the pre-generated benchmark splits from the Hub with [grain](https://github.com/google/grain). Requires the `[loader]` extra. Loaders yield `(theta, x)` already tokenized to shape `(batch, dim, 1)`:

```python
from sbibm_jax.data import TaskDataset

ds = TaskDataset(
    "two_moons",
    repo="aurelio-amerio/SBI-benchmarks",   # published dataset (the in-code default is the -test repo)
    normalize=True,                          # apply gen-time mean/std from metadata.json
)

train = ds.get_train_loader(batch_size=256)   # infinite: shuffle -> repeat -> batch
theta, x = next(iter(train))                  # theta: (256, dim_theta, 1), x: (256, dim_x, 1)

posterior = ds.get_reference(num_observation=1)
```

`kind="joint"` concatenates `(theta, x)` along the feature axis; `get_val_loader` / `get_test_loader` serve the validation and test splits.

### 3. Online datasets (simulate on the fly)

`OnlineTaskDataset` reads the same `metadata.json` (shapes + normalization stats) but draws fresh `(theta, x)` from the task's prior and simulator each batch — the splits are never downloaded. Finite-simulator, vector-`theta` tasks only:

```python
from sbibm_jax.data import OnlineTaskDataset

ds = OnlineTaskDataset(
    "two_moons",
    repo="aurelio-amerio/SBI-benchmarks",
    normalize=True,
)

loader = ds.get_online_train_loader(batch_size=256, seed=0, num_workers=4)  # num_workers=0 disables prefetch workers
theta, x = next(iter(loader))   # a fresh draw every batch
```

## Available tasks

Call `get_available_tasks()` for the full list — analytical, ODE, image, and time-series tasks. Each lives under `src/sbibm_jax/tasks/<name>/`.

## License

MIT — see [LICENSE](LICENSE). If you use `sbibm-jax`, please also consider citing the original [sbibm](https://github.com/sbi-benchmark/sbibm) benchmark.
````

- [ ] **Step 2: Verify the README is non-empty and structured**

Run: `grep -cE "^#{1,3} " README.md && grep -E "pip install sbibm-jax|get_task|TaskDataset|OnlineTaskDataset" README.md | wc -l`
Expected: ≥7 headers; ≥4 matched API/install lines.

- [ ] **Step 3: Re-run the task snippet end-to-end to confirm the documented API is accurate**

Run:
```bash
JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python -c "
import jax
from sbibm_jax import get_task, get_available_tasks
task = get_task('two_moons'); key = jax.random.PRNGKey(0)
theta = task.get_prior(key, num_samples=8)
x = task.get_simulator(key)(key, theta)
task.get_observation(1); task.get_true_parameters(1); task.get_reference_posterior_samples(1)
assert len(get_available_tasks()) > 0
print('README task snippet OK', theta.shape, x.shape)
"
```
Expected: `README task snippet OK (8, 2) (8, 2)` (a CUDA "NO_DEVICE" warning line on CPU-only hosts is harmless).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add README with install + dataset usage (task/offline/online)"
```

---

## Self-Review

**Spec coverage** (against `2026-06-22-pypi-refresh-design.md`):
- Deliverable 1 publish workflow → Task 1. ✓
- Deliverable 1 Build CI (lint/test/badges, docs jobs dropped) → Task 2. ✓
- Deliverable 2 full metadata (description/authors/license/keywords/classifiers/urls) + uv build verify + SPDX fallback → Task 3. ✓
- Deliverable 3 README (title/badges/overview/install/3 usage modes/tasks/license) → Task 4. ✓
- Out-of-scope items (no code change, no version bump, no live upload, no docs site, no pypesto in CI) → respected; version untouched, pypesto omitted from CI install. ✓

**Placeholder scan:** no TBD/TODO; every file's full content is inline; the one branch (SPDX license) is a concrete contingency with exact replacement text. ✓

**Type/name consistency:** API names in Task 4 (`get_prior(key, num_samples)`, `get_simulator(key)`, `get_observation`, `get_true_parameters`, `get_reference_posterior_samples`, `TaskDataset`/`get_train_loader`/`get_reference`, `OnlineTaskDataset`/`get_online_train_loader`) match the signatures verified in the spec. Group/marker names in Task 2 (`lint`/`test`/`hf`/`loader`, `slow`/`experimental`) and badge paths (`img/badges/*.svg`) are consistent between Task 2 (producer) and Task 4 (consumer). ✓
