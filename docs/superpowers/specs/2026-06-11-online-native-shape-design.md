# OnlineTaskDataset phase 2 — native-shape (non-vector) tasks

**Date:** 2026-06-11
**Status:** Approved (design)

## Goal

Lift the vector-only restriction on `OnlineTaskDataset` so non-vector tasks —
primarily `gaussian_random_field` (image), and any future timeseries task with
a simulator — can be simulated online with tokenization and normalization
identical to the offline `TaskDataset`.

## Current behavior (verified empirically on the merged phase-1 code)

- `gaussian_random_field` is online-eligible in every way except shape:
  cheap vmapped-FFT closure simulator, always-finite output
  (`hf_resample_invalid=False`), `dim_x=1024`, `hf_x_shape=(32, 32)`.
- The `Simulator` wrapper always emits **flat** rows
  (`task.flatten_data`, `tasks/simulator.py:80`): GRF yields `(n, 1024)`.
- The offline dataset stores **native** rows — the HF `ImageExporter`
  reshapes flat → `(n, 32, 32)` at generation time — so offline tokens are
  `(n, 32, 32, 1)`.
- Without the phase-1 guard, the online path would tokenize flat:
  `(n, 1024, 1)`; with `normalize=True`, the global-scalar image stats
  (tokenized to `(1, 1, 1, 1)`) broadcast against those rank-3 tokens into a
  silently wrong-rank `(1, n, 1024, 1)` array. Hence the current guard:
  `OnlineTaskDataset` raises `NotImplementedError` at construction for
  non-vector x or theta (commit 98a3f36).

## Decision 1: reshape in the worker source, from metadata `x_shape`

`_SimIterDataset` gains a trailing optional `x_shape=None` parameter. When
set, `_SimIterator.__next__` emits `np.asarray(x).reshape(-1, *x_shape)`
instead of the flat simulator output. `OnlineTaskDataset.get_online_train_loader`
passes `x_shape=self.x_shape` (parsed from the published `metadata.json` by
`_init_metadata` — the same contract the offline loader and the gen-time
stats use).

Why the worker source (vs a main-process reshape step):

- Raw source batches become layout-identical to offline HF rows, so
  `make_collate_jax` and normalization need **zero changes** — native tokens
  broadcast correctly against the global-scalar image stats; the wrong-rank
  bug only existed because tokens were flat.
- The reshape is a free view on the already-materialized numpy array; the
  bytes crossing the pickle boundary are unchanged.
- `x_shape=None` keeps the flat behavior, so the existing `_SimIterDataset`
  unit tests stay valid unchanged.

Theta is untouched: every task has vector theta, and `get_prior` already
returns `(n, dim_theta)`.

## Decision 2 (rejected alternative): `simulator.unflatten_data`

`Task.unflatten_data` (mirrored onto `Simulator`) looks purpose-built, and six
tasks override it. Survey of override vs published metadata `x_shape`:

| Task | `unflatten_data` | metadata `x_shape` | Agree? |
|---|---|---|---|
| gaussian_random_field | `(-1, N, N)` | `(N, N)` | yes |
| toy_lensing | `(-1, res, res)` | `(res, res)` | yes |
| gravitational_waves | `(-1, 8192, 2)` | `(8192, 2)` | yes |
| slcp | `(-1, num_data, 2)` | `(16,)` flat vector | **no** |
| lotka_volterra (raw) | `(-1, 2, T)` | flat vector | **no** |
| slcp_distractors | raises `NotImplementedError` | `(100,)` vector | **no** |

`unflatten_data` answers a different question — the *native simulation
shape* — which coincides with the published dataset shape only for tasks that
export non-flat. Applied blindly it would break vector tasks (SLCP online
tokens `(n, 8, 2, 1)` vs offline `(n, 16, 1)`; distractors crashes), so it
would have to be gated on metadata `x_kind != "vector"` anyway — swapping an
explicit shape for an unenforced convention that the override matches the
published `x_shape`, with silent drift whenever element counts happen to
match. Metadata `x_shape` is the single binding contract; offline parity is
guaranteed by construction. Rejected (including the gated + parity-check
variant, judged not worth the extra code).

## Decision 3: generic any-rank reshape; guard shrinks to theta

- `reshape(-1, *x_shape)` is a no-op for vector tasks (`x_shape == (dim_x,)`),
  native for image (`(H, W)`) AND timeseries (`(T, C)`). The x-kind
  construction guard is removed entirely; a future reworked
  `gravitational_waves` with a simulator works online for free.
- The guard keeps only `theta_kind != "vector"` → `NotImplementedError`
  (no task has non-vector theta today; the online path makes no provision for
  it).
- A mismatch between metadata `x_shape` and the simulator's flat output dim
  fails naturally in the reshape with a clear numpy error — no extra
  validation code. (`dim_x` is *derived from* `x_shape` in `_init_metadata`,
  so the published contract is internally consistent by construction.)

## Changes

| File | Change |
|---|---|
| `src/sbibm_jax/data/dataset.py` | `_SimIterDataset(..., x_shape=None)` + iterator reshape; `OnlineTaskDataset`: guard → theta-only, pass `x_shape` in `get_online_train_loader`; docstrings |
| `tests/data/test_online_dataset.py` | repurpose guard test (theta-kind), GRF loader tests, timeseries-shape test, GRF mp smoke |
| `CLAUDE.md` | update the `OnlineTaskDataset` sentence (drop vector-only) |

## Testing (metadata faked locally; no Hub)

- **Source unit:** `_SimIterDataset(task, sim, seed, bs, x_shape=(32, 32))`
  for GRF emits `{"xs": (bs, 32, 32)}` numpy; `x_shape=None` stays flat
  (existing tests unchanged).
- **Guard:** fake metadata with `theta_kind: "image"` → construction raises
  `NotImplementedError` matching "vector"; the phase-1 image-x guard test is
  repurposed to assert image-x now *constructs and loads* successfully.
- **GRF conditional loader** (`num_workers=0`, fake metadata
  `x_kind: "image"`, `x_shape: [32, 32]`, global-scalar stats): token shapes
  `theta (4, 2, 1)`, `x (4, 32, 32, 1)`; `normalize=True` output equals a
  manual `make_collate_jax` applied to the same raw draw, and differs from
  the unnormalized tokens (stats non-trivial).
- **Timeseries rank-generality:** `two_moons` with fake
  `x_kind: "timeseries"`, `x_shape: [2, 1]` (element count matches
  `dim_x=2`) → x tokens `(4, 2, 1, 1)`. Proves any-rank without a heavy task.
- **GRF mp smoke** (`num_workers=1`, one batch, ~15 s spawn + CPU FFT in the
  worker): shapes `(2, 32, 32, 1)`, finite. GRF is the primary online use
  case, so the end-to-end worker path gets direct coverage; same
  close-in-finally pattern as the existing smoke test.
- `kind="joint"` stays vector-only via the inherited `make_collate_jax`
  guard — no change, no new test.

## Non-goals

- No non-vector theta support (guard remains).
- No change to `unflatten_data` or its semantics.
- No online support for diverging simulators (`hf_resample_invalid` ODE/PEtab
  tasks) — unchanged phase-1 stance.
