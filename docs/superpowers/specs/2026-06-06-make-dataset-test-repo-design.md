# Design: test-repo default + `--prod` guardrail for `make_dataset.py`

**Date:** 2026-06-06
**Status:** Approved (pending spec review)
**Scope:** `scripts/make_dataset.py`, `src/sbibm_jax/hf/config.py`, docs, driver tests.

## Problem

`scripts/make_dataset.py` builds and uploads HuggingFace datasets for sbibm_jax
tasks. Today its only repo control is `--repo`, which defaults to the production
repo `aurelio-amerio/SBI-benchmarks`. A bare `make_dataset.py --all` therefore
pushes straight to production — a single typo or stray run can corrupt the main
dataset repo. A dedicated test repo now exists
(`https://huggingface.co/datasets/aurelio-amerio/SBI-benchmarks-test`) for
validating uploads of some or all of the new datasets, but nothing in the CLI
steers runs toward it or guards production.

## Goal

Make the **test repo the safe default** and require an **explicit `--prod`
flag** to write to production. Remove the arbitrary `--repo` flag in favor of the
simplest, hardest-to-misfire surface. Every run must print an unmissable banner
naming the resolved target and whether it is TEST or PRODUCTION.

Non-goals: arbitrary third-repo targeting from the CLI (requires editing config,
accepted trade-off); interactive y/N confirmation; changing build/split logic.

## Design

### Config (`src/sbibm_jax/hf/config.py`)

- Add `TEST_REPO: str = "aurelio-amerio/SBI-benchmarks-test"`.
- Keep `DEFAULT_REPO = "aurelio-amerio/SBI-benchmarks"` unchanged in both name
  and value. It is now the *production* target. Keeping the name avoids churn:
  `tests/hf/test_import_guard.py` asserts `config.DEFAULT_REPO ==
  "aurelio-amerio/SBI-benchmarks"`. Add a one-line comment marking it as the
  production repo and noting `TEST_REPO` is the CLI default.

### CLI (`scripts/make_dataset.py`)

- **Remove** the `--repo` argument and its references.
- **Add** `--prod` (`action="store_true"`). Help text: "Upload to the PRODUCTION
  repo (config.DEFAULT_REPO). Without it, uploads target the test repo
  (config.TEST_REPO)."
- Resolve the target exactly once, early in `main`:
  `repo = config.DEFAULT_REPO if args.prod else config.TEST_REPO`.
- Replace every prior use of `args.repo` with the resolved `repo`.
- Print a banner immediately after resolving the target — **before** the dry-run
  early return — so dry runs also reveal where a real run would push:
  `Target repo: <repo>  (PRODUCTION)` when `--prod`, else `... (TEST)`.
- Update the module docstring usage examples to show the test default and
  `--prod` (drop the `--repo` example).

### Resulting behavior

| Command | Target |
|---|---|
| `make_dataset.py --all` | test repo (safe default) |
| `make_dataset.py --all --prod` | production repo |
| `make_dataset.py --tasks two_moons --dry-run` | no upload; banner shows `(TEST)` |
| `make_dataset.py --tasks two_moons --prod --dry-run` | no upload; banner shows `(PRODUCTION)` |

All existing flags (`--tasks`, `--all`, `--metadata-path`, `--train-size`,
`--val-size`, `--test-size`, `--master-seed`, `--dry-run`, `--verbose`) are
unchanged.

### Docs (`CLAUDE.md`)

Update the `make_dataset.py` example block so the bare `--all` line reads as a
test-repo upload and add a `--prod` line for production, e.g.:

```bash
uv run python scripts/make_dataset.py --tasks two_moons --train-size 1000 --dry-run
uv run python scripts/make_dataset.py --all            # all tasks -> TEST repo
uv run python scripts/make_dataset.py --all --prod     # all tasks -> PRODUCTION repo
```

### Tests (`tests/hf/test_driver.py`)

- Existing smoke tests (`test_driver_help_runs`, `test_dry_run_writes_metadata`)
  remain valid — neither used `--repo`.
- Add two cheap, network-free assertions via `--dry-run`:
  - default dry run prints `(TEST)` and the test-repo id in stdout;
  - `--prod` dry run prints `(PRODUCTION)` and the production-repo id.

## Trade-offs / accepted risks

- **No arbitrary `--repo`.** Targeting a repo other than the two configured ones
  later requires editing `config.py`. Chosen deliberately for the simplest, safest
  surface.
- **`DEFAULT_REPO` name now means "production", not "CLI default".** Mitigated by
  a clarifying comment; renaming was rejected to keep the import-guard test and
  any external references stable.

## Out of scope

- Interactive confirmation prompts / `--yes`.
- Changes to `build_dataset`, exporters, split logic, or `upload.py` internals.
