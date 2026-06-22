# Design: refresh `sbibm-jax` for PyPI

**Date:** 2026-06-22
**Status:** Approved (design phase)

## Goal

Make `sbibm-jax` publishable to PyPI and usable by newcomers, by adding:

1. GitHub Actions to publish to PyPI on release + run CI (tests/lint/coverage) on push/PR.
2. Full PyPI metadata in `pyproject.toml` (authors, license, urls, classifiers, keywords, a better description).
3. A `README.md` with basic instructions for the three dataset-usage modes (benchmark **tasks**, **offline** HF datasets, **online** HF datasets).

These are copied/adapted from the sibling **GenSBI** / **GenSBI-examples** repos, which are already PyPI-published with the same author and tooling (`uv`, OIDC trusted publishing).

## Context / constraints

- The repo is `github.com/aurelio-amerio/sbibm-jax`, MIT licensed, author Aurelio Amerio.
- `/lhome/ific/a/aamerio/data/github/sbibm-jax` is a **symlink** to the primary lustre checkout — one edit covers both paths.
- Build backend is `uv_build`; Python pinned to 3.12.
- Tests are CPU-forced via `pytest-env` (`JAX_PLATFORMS=cpu`) and run under xdist (`-n 2` in `addopts`).
- The `test` dependency group already ships `genbadge[all]` + `junitparser` — the coverage/test-badge pipeline is anticipated.
- Test guards confirmed:
  - `tests/tasks/test_petab.py` uses `skipif(not HAS_PYPESTO)` → the heavy `pypesto`/AMICI tests **skip cleanly** when the extra is absent (no AMICI compile needed in CI).
  - `tests/data/` and `tests/hf/` use `importorskip("grain"/"datasets")` → run only when the `loader`/`hf` groups are installed.
  - Data/HF loader tests **mock the Hub** (`monkeypatch` of `hf_hub_download` / `load_dataset` with local fixtures) → CI needs no network and no live Hub repo.
- Data API surface (verified against `src/sbibm_jax/data/dataset.py`):
  - `TaskDataset(name, *, kind="conditional", repo=None, normalize=False, seed=42, …)` — offline; default repo is `config.TEST_REPO`.
  - `OnlineTaskDataset(name, *, kind="conditional", repo=None, normalize=False, seed=42)` — online; `get_online_train_loader(batch_size, *, seed=None, num_workers=0)`.
  - Collate yields `(theta, x)` for `kind="conditional"`, each tokenized to shape `(batch, dim, 1)`; concatenated for `kind="joint"`.
- Repo constants live in `sbibm_jax.hf.config`: production `aurelio-amerio/SBI-benchmarks`, default-in-code test repo `aurelio-amerio/SBI-benchmarks-test`. (The TEST repo has been refreshed with the new schema + stats, so the in-code default works; the README still points users at production.)
- Task API surface (verified against `src/sbibm_jax/tasks/task.py`): `get_prior(key, num_samples)`, `get_simulator(key, max_calls)`, `get_observation(num_observation)`, `get_reference_posterior_samples(num_observation, num_samples=...)`, `get_true_parameters(num_observation)`. Simulator `__call__(key, parameters)`.

## Deliverable 1 — GitHub Actions (`.github/workflows/`)

### `python-publish.yml`

Copied from `GenSBI-examples/.github/workflows/python-publish.yml`, with edits:

- `environment.url` → `https://pypi.org/p/sbibm-jax`.
- `uv python install 3.12` (project is pinned to 3.12; GenSBI used 3.13).
- Otherwise unchanged: trigger `release: [published]`, `permissions: id-token: write` + `contents: read` (OIDC trusted publishing, no API token), steps `checkout → setup-uv → uv build → uv publish`.

PyPI side (manual, documented for the user — not automated here): configure a Trusted Publisher for `sbibm-jax` pointing at this workflow, and create the `pypi` GitHub environment.

### `python-app.yml` (name: `Build`)

Adapted from `GenSBI/.github/workflows/python-app.yml`, **simplified** for sbibm-jax (no Sphinx docs site, no GitHub App). Jobs:

- **`lint`** — `setup-uv`, `uv python install 3.12`, `uv sync --group lint`, then GenSBI's safe two-pass flake8:
  - pass 1 `--select=E9,F63,F7,F82 --show-source` (real errors → fail build),
  - pass 2 `--exit-zero --max-complexity=10 --max-line-length=127` (warnings only; pre-existing E501 cannot break CI).
- **`test`** — `uv python install 3.12`, `uv sync --group test --group hf --group loader` (installs grain/datasets/HF so the data/HF tests run; the `pypesto` extra is intentionally omitted, so petab tests skip and no AMICI compile happens). Run pytest in three marker passes (fast = `not slow and not experimental`, then `slow`, then `experimental`) writing `reports/junit/*.xml` and accumulating `--cov=sbibm_jax`, emit `reports/coverage/coverage.xml`; merge junit with `junitparser`; `genbadge coverage` + `genbadge tests` → `reports/badges/*.svg`; upload `reports/` as an artifact. (Tests inherit the repo's `-n 2`; CPU is forced by `pytest-env`.)
- **`publish-badges`** — `needs: test`, `if: github.event_name == 'push' && github.ref == 'refs/heads/main'`. Download the artifact, copy SVGs to `img/badges/`, commit with `stefanzweifel/git-auto-commit-action@v6` using the **default `GITHUB_TOKEN`** and commit message `Update badges [skip ci]`. (No GitHub App, unlike GenSBI; `permissions: contents: write` on the job.)

**Dropped** from GenSBI: the `docs` (Sphinx `make html`) and `deploy-docs` (gh-pages + `gensbi.com` CNAME) jobs — sbibm-jax has no docs site.

## Deliverable 2 — `pyproject.toml` full PyPI metadata

Add to `[project]` (keeping `name`, `version = "0.1.1"`, `requires-python`, deps, optional-deps, groups untouched):

- **description** (rewritten): `A JAX/NumPyro rewrite of the Simulation-Based Inference Benchmark (sbibm): benchmark tasks — priors, simulators, reference posteriors — plus ready-to-stream HuggingFace datasets for evaluating SBI methods.`
- `authors = [{ name = "Aurelio Amerio", email = "dev@gensbi.com" }]` (matches GenSBI's published identity).
- `license = "MIT"` + `license-files = ["LICENSE"]` (PEP 639 SPDX form). **Verify** with `uv build` that the produced metadata is clean; fall back to `license = { file = "LICENSE" }` only if `uv_build` rejects the SPDX form.
- `keywords = ["simulation-based-inference", "sbi", "benchmark", "jax", "numpyro", "bayesian-inference", "datasets", "machine-learning"]`.
- `classifiers` — Development Status :: 4 - Beta; Intended Audience :: Science/Research; Operating System :: OS Independent; Programming Language :: Python :: 3 and :: 3.12; Topic :: Scientific/Engineering and :: Artificial Intelligence. (No `License ::` classifier — it would duplicate the SPDX expression under PEP 639.)
- `[project.urls]` — `Homepage` / `Repository` = `https://github.com/aurelio-amerio/sbibm-jax`, `Issues` = `…/issues`.

Version stays `0.1.1`; the user bumps it and tags a GitHub Release to trigger the publish workflow.

## Deliverable 3 — `README.md` (Standard)

Sections:

1. **Title** `# sbibm-jax` + one-line subtitle.
2. **Badges** — Build (Actions `python-app.yml`), Coverage (`img/badges/coverage.svg` raw URL, populated by CI), PyPI version (`shields.io/pypi/v/sbibm-jax`), License.
3. **Overview** — 2–3 sentences: JAX/NumPyro port of sbibm; benchmark tasks + HuggingFace datasets; companion to GenSBI.
4. **Installation** — `uv add sbibm-jax` / `pip install sbibm-jax`; optional extras `[hf]`, `[loader]`, `[pypesto]`; GPU note (default JAX is `jax[cuda12]`).
5. **Usage** — three runnable snippets:
   - **Benchmark tasks** (`from sbibm_jax import get_task`): `get_prior`, `get_simulator` + call, `get_observation`, `get_reference_posterior_samples`.
   - **Offline datasets** (`from sbibm_jax.data import TaskDataset`): construct with `repo="aurelio-amerio/SBI-benchmarks"`, `normalize=True`; `get_train_loader(batch_size)` yielding `(theta, x)` tokens of shape `(batch, dim, 1)`; `get_reference`. Note the in-code default is the `-test` repo.
   - **Online datasets** (`from sbibm_jax.data import OnlineTaskDataset`): `get_online_train_loader(batch_size, seed=, num_workers=)`; note finite-simulator, vector-theta tasks only.
6. **Available tasks** — `get_available_tasks()` + brief pointer to the task catalog.
7. **License & citation** — MIT; brief note.

The dataset snippets pass `repo="aurelio-amerio/SBI-benchmarks"` explicitly and state that the loader defaults to the `-test` repo.

## Out of scope

- No change to package code behaviour (loader default repo stays the test repo).
- No live PyPI upload, no GitHub Release, no version bump (the user does these).
- No Sphinx docs site.
- No `pypesto`/AMICI in CI.

## Verification

- `uv build` succeeds and produces a wheel + sdist with valid metadata (license/classifiers/urls render).
- `uv run flake8` style check: no **new** violations vs HEAD (bare flake8 is never clean here).
- The two workflow YAMLs are syntactically valid (yaml parse) and reference existing groups/markers.
- README code snippets match the verified API signatures above.
