# Design: test-repo default, `--prod` guardrail, and non-destructive metadata for `make_dataset.py`

**Date:** 2026-06-06
**Status:** Approved (pending spec review)
**Scope:** `scripts/make_dataset.py`, `src/sbibm_jax/hf/config.py`,
`src/sbibm_jax/hf/upload.py`, `src/sbibm_jax/hf/metadata.py`, docs, hf tests.

## Problem

`scripts/make_dataset.py` builds and uploads HuggingFace datasets for sbibm_jax
tasks. Two problems motivate this work:

1. **No guardrail against production.** The only repo control is `--repo`, which
   defaults to the production repo `aurelio-amerio/SBI-benchmarks`. A bare
   `make_dataset.py --all` pushes straight to production; one stray run or typo
   can corrupt the main dataset repo. A dedicated test repo now exists
   (`aurelio-amerio/SBI-benchmarks-test`) for validating uploads of some or all
   tasks, but nothing in the CLI steers runs toward it.

2. **Subset uploads clobber `metadata.json`.** `make_metadata(task_names, ...)`
   builds a dict keyed only by the selected tasks and `json.dumps` overwrites the
   local `metadata.json`; `upload_metadata` then pushes it with
   `upload_file(path_in_repo="metadata.json")`, which **replaces** the remote
   file wholesale. So `--tasks two_moons` against a repo that already documents 14
   tasks leaves `metadata.json` with only `two_moons` — the other 13 entries are
   lost. (Dataset *payloads* are safe: each task is pushed under its own
   `config_name=task_name`, so they don't clobber each other. Only the single
   shared `metadata.json` is at risk.)

## Goal

- Make the **test repo the safe default** and require an explicit **`--prod`**
  flag to write to production. Drop the arbitrary `--repo` flag for the simplest,
  hardest-to-misfire surface. Print an unmissable banner naming the resolved
  target and whether it is TEST or PRODUCTION.
- Make `metadata.json` uploads **non-destructive**: a subset run updates only the
  selected tasks' entries and preserves every other task's entry, by merging
  against the remote file before upload.

Non-goals: arbitrary third-repo targeting from the CLI (requires editing config,
accepted); interactive y/N confirmation; changing build/split/exporter logic;
changing the HF dataset-card (`README.md`) writing, which is owned by the
`datasets` library (see Verification).

## Design — Part 1: test-repo default + `--prod`

### Config (`src/sbibm_jax/hf/config.py`)

- Add `TEST_REPO: str = "aurelio-amerio/SBI-benchmarks-test"`.
- Keep `DEFAULT_REPO = "aurelio-amerio/SBI-benchmarks"` unchanged in name and
  value — it is now the *production* target. Keeping the name avoids churn:
  `tests/hf/test_import_guard.py` asserts that exact value. Add a one-line
  comment marking it as production and noting `TEST_REPO` is the CLI default.

### CLI (`scripts/make_dataset.py`)

- **Remove** the `--repo` argument and all references to `args.repo`.
- **Add** `--prod` (`action="store_true"`). Help: "Upload to the PRODUCTION repo
  (config.DEFAULT_REPO). Without it, uploads target the test repo
  (config.TEST_REPO)."
- Resolve the target once, early: `repo = config.DEFAULT_REPO if args.prod else
  config.TEST_REPO`.
- Print a banner immediately after resolving the target, **before** the dry-run
  early return: `Target repo: <repo>  (PRODUCTION)` when `--prod`, else
  `... (TEST)`.
- Update the module docstring usage examples (test default + `--prod`; drop the
  `--repo` example).

### Resulting behavior

| Command | Target |
|---|---|
| `make_dataset.py --all` | test repo (safe default) |
| `make_dataset.py --all --prod` | production repo |
| `--tasks two_moons --dry-run` | no upload; banner shows `(TEST)` |
| `--tasks two_moons --prod --dry-run` | no upload; banner shows `(PRODUCTION)` |

All other flags (`--tasks`, `--all`, `--metadata-path`, `--train-size`,
`--val-size`, `--test-size`, `--master-seed`, `--dry-run`, `--verbose`) are
unchanged.

## Design — Part 2: non-destructive metadata merge

Factored into three single-purpose pieces; the driver orchestrates them. This
keeps `upload_metadata` a pure uploader (its existing test stays valid) and keeps
all network access inside `upload.py`.

### `src/sbibm_jax/hf/upload.py` — `fetch_remote_metadata(repo_name) -> dict`

- New helper. Downloads the repo's existing `metadata.json` via
  `huggingface_hub.hf_hub_download(repo_id=repo_name, filename="metadata.json",
  repo_type="dataset", force_download=True)` (force to avoid a stale local cache),
  reads and `json.loads` it, returns the dict.
- On `EntryNotFoundError` (repo exists, no `metadata.json` yet) **or**
  `RepositoryNotFoundError` (fresh repo) → return `{}`.
- Let any other error (auth, HTTP/connection) **propagate** — a transient failure
  must never be silently treated as "empty remote", which would merge into `{}`
  and wipe the sibling entries we are protecting.
- Import `hf_hub_download` (and the two exception classes) at module scope so
  tests can monkeypatch them, mirroring the existing `upload_file` pattern.
- `upload_metadata` is **unchanged** (still a thin `upload_file` wrapper).

### `src/sbibm_jax/hf/metadata.py` — `merge_metadata(remote, local) -> dict`

- New pure function: `return {**remote, **local}`. Selected tasks (in `local`)
  overwrite their own entries; all other entries from `remote` are preserved.
  No I/O, no network — unit-testable with zero monkeypatching.
- `make_metadata` is unchanged (it already returns the built dict and optionally
  writes the file).

### Driver orchestration (`scripts/make_dataset.py`)

Real-upload path (non-dry-run):

1. `local = make_metadata(task_names, output_path=metadata_path, split_sizes=...)`
   — builds fresh entries for the selected tasks and writes the subset file.
2. Print the target banner (already done in Part 1).
3. `remote = fetch_remote_metadata(repo)`.
4. `merged = merge_metadata(remote, local)`.
5. Overwrite the local file with the merged dict
   (`metadata_path.write_text(json.dumps(merged, indent=4))`) so the on-disk file
   matches what is pushed.
6. `upload_metadata(str(metadata_path), repo)`.
7. `upload_dataset(repo, name, **build_opts)` for each task.
8. On successful completion, `metadata_path.unlink(missing_ok=True)` — the file is
   a transient build artifact; deleting it leaves a clean working tree.

Dry-run path: steps 3–8 are skipped. `make_metadata` still writes the subset
file and it is **kept** for inspection (no fetch, no merge, no delete).

Note: metadata is uploaded before the dataset payloads (unchanged order). A
mid-run dataset failure can therefore leave `metadata.json` over-claiming
relative to what was pushed; this is pre-existing and orthogonal, left as-is.

### Docs (`CLAUDE.md`)

Update the `make_dataset.py` example block: bare `--all` → test repo, plus a
`--prod` line for production, e.g.:

```bash
uv run python scripts/make_dataset.py --tasks two_moons --train-size 1000 --dry-run
uv run python scripts/make_dataset.py --all            # all tasks -> TEST repo
uv run python scripts/make_dataset.py --all --prod     # all tasks -> PRODUCTION repo
```

Add a sentence noting subset uploads now merge into the remote `metadata.json`
(non-destructive) and that the local `metadata.json` is deleted after a
successful real upload. Mention in `--metadata-path` help that this delete
removes whatever the path points at.

## Tests (`tests/hf/`)

- `test_upload.py`: existing `test_calls_upload_file` stays valid
  (`upload_metadata` is unchanged). Add tests for `fetch_remote_metadata`:
  monkeypatch `hf_hub_download` to (a) return a temp file with `{"a": 1}` → dict
  returned; (b) raise `EntryNotFoundError` → `{}`; (c) raise
  `RepositoryNotFoundError` → `{}`; (d) raise a generic error → propagates.
- `test_metadata.py`: add `merge_metadata` tests — disjoint keys union; shared key
  takes the local value; empty remote → local; empty local → remote.
- `test_driver.py`: existing smoke tests stay valid (none used `--repo`;
  `test_dry_run_writes_metadata` still holds since dry-run keeps the file). Add
  dry-run assertions that the banner prints `(TEST)` by default and
  `(PRODUCTION)` with `--prod`. (Merge + delete happen only on real upload, so
  they are covered by unit tests above rather than a networked driver test.)

## Verification (plan must include)

**Dataset-card (`README.md`) preservation on the test repo.** `push_to_hub`
makes the `datasets` library read-modify-write the repo's `README.md` YAML
(`configs:` / `dataset_info:` listing every config). This is library-managed and
outside our `metadata.json` merge. Recent `datasets` versions are expected to
merge config entries (multi-config repos are supported and we always pass
`config_name`), but this is version-dependent and must be confirmed empirically —
it is the one path that could still corrupt the repo despite the metadata fix.

Manual check against `aurelio-amerio/SBI-benchmarks-test`:
1. Upload task A (e.g. `gaussian_linear`), then separately upload task B (e.g.
   `two_moons`).
2. Confirm the repo `README.md` still lists **both** configs and that
   `load_dataset(test_repo, "gaussian_linear")` still resolves.
3. Confirm `metadata.json` on the repo contains **both** entries.

If the card drops configs on a subset push, that is a follow-up (card-merge step
or alternate upload strategy) — out of scope for this change unless verification
fails.

## Trade-offs / accepted risks

- **No arbitrary `--repo`.** Targeting a repo other than the two configured ones
  later requires editing `config.py`. Chosen for the simplest, safest surface.
- **`DEFAULT_REPO` name now means "production", not "CLI default".** Mitigated by
  a clarifying comment; renaming was rejected to keep the import-guard test and
  external references stable.
- **Metadata uploaded before payloads.** Pre-existing ordering; not addressed
  here.

## Out of scope

- Interactive confirmation prompts / `--yes`.
- Changes to `build_dataset`, exporters, split logic, or the HF dataset-card
  writer.
