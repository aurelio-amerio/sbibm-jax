# Design: `beer_molbiosystems` PEtab benchmark task

**Date:** 2026-06-05
**Status:** Approved (pre-implementation)

## Goal

Port the `Beer_MolBioSystems2014` PEtab benchmark model (from
`diffusion-experiments/case_study2`) into `sbibm-jax` as a benchmark `Task`,
mirroring how `case_study3` was ported as `gaussian_random_field`.

Unlike the existing JAX-native tasks, this one is backed by external libraries
and is **wrapped, not rewritten**: the pypesto/AMICI machinery from the case
study is copied in verbatim (stripped of unused code), not re-derived in JAX.

- **Prior** comes from the PEtab parameter table (truncated-normal / uniform on
  scaled space, with bound tweaks applied by `load_problem`).
- **Simulator** is a compiled AMICI ODE model (not JAX-vmappable).
- **Reference posterior** comes from live pypesto parallel-tempering MCMC.

The heavy machinery (`pypesto`, `petab`, `amici`, `benchmark-models-petab`,
`joblib`, `scipy`) is gated behind an optional **`pypesto` extra** so the rest of
the library stays JAX-only and installable without it.

## Decisions (from brainstorming)

1. **Scope:** Beer-only, hardcoded — not a generic PEtab wrapper. New PEtab
   models would get their own task directory later.
2. **Port, don't rewrite:** copy the SBI-relevant functions from
   `helper_pypesto.py` / `run_mcmc.py` verbatim into a task-local module,
   dropping the `bayesflow`/metrics code (which cannot even import without
   bayesflow). "Don't rewrite" = don't re-derive the algorithms.
3. **Prior:** delegate to the ported `sample_from_prior`; **no numpyro
   `prior_dist`**. `get_prior_dist()` raises an informative error. Rationale:
   anything generative (simulator, MCMC) needs the extra anyway, so requiring it
   to sample the prior adds no burden, and a hand-maintained numpyro prior would
   risk drifting from the PEtab table.
4. **Optional dependency:** a single `pypesto` extra
   (`pip install sbibm-jax[pypesto]` / `uv sync --extra pypesto`).
5. **Reference posterior:** implement `_sample_reference_posterior` via live
   pypesto MCMC. **Do not generate the CSVs now**; only verify the live MCMC
   path runs end-to-end with small settings.
6. **Structured-data reconstruction:** the reference MCMC conditions on a PEtab
   **measurement dataframe**, richer than the flat `dim_data` vector. Recover it
   from the stored flat observation via the fixed Beer PEtab template + the
   NaN/value pattern — **no sidecar metadata files**. This mirrors the LV/SIR
   philosophy where the richer internal representation is recomputable, not
   stored.
7. **Simulator batching:** joblib `Parallel` with a configurable `n_jobs`
   (default -1), mirroring the case study.
8. **Observations:** 10 observations whose true parameters are prior draws at
   fixed `observation_seeds`; observation = noisy AMICI simulation. Data files
   are deferred (generation routine provided, not run now).

## Architecture

### Placement & registry
- New directory `src/sbibm_jax/tasks/beer_molbiosystems/`:
  - `task.py` — `BeerMolBioSystems(Task)` subclass.
  - `petab_helpers.py` — ported subset of `case_study2/helper_pypesto.py` and
    `run_mcmc.py`.
- `name` = directory name (`beer_molbiosystems`),
  `name_display = "Beer (MolBioSystems2014)"`.
- Registered in `src/sbibm_jax/tasks/__init__.py` `get_task()` with a lazy
  per-branch import, like the other tasks. `get_available_tasks()` picks it up
  by directory discovery.

### Dependencies
- Add a PEP 621 `[project.optional-dependencies]` table with a `pypesto` extra:
  `pypesto`, `petab`, `amici`, `benchmark-models-petab`, `joblib`, `scipy`.
- `amici` compiles a C++ model on first use → the extra needs a working
  compiler / SWIG / BLAS. Documented in CLAUDE.md and the task docstring.
- All heavy deps are **lazily imported inside methods**, raising a clear
  "install the `pypesto` extra" error when missing, so the task still
  *constructs* without them.

### Ported helpers (`beer_molbiosystems/petab_helpers.py`)
Copy the needed subset verbatim, stripped of the bayesflow/metrics imports:
- From `helper_pypesto.py`: `load_problem`, `sample_from_prior`,
  `get_samples_from_dict`, `simulator_amici`, `amici_pred_to_df`,
  `amici_df_to_array`, `apply_noise_to_data`, `scale_values`,
  `values_to_linear_scale`, `create_pypesto_problem`.
- From `run_mcmc.py`: `run_mcmc`, `get_mcmc_posterior_samples`,
  `run_mcmc_single`.
- **Dropped** (bayesflow-dependent): `compute_metrics`, `sample_in_batches`,
  `compute_likelihood`, `compute_likelihood_parallel`, and the top-level
  bayesflow imports.

`diffusion-experiments/` is untracked and not part of the package, so the code
is copied in, not imported from there.

### Parameter space
- Canonical parameters = **free/estimated parameters in scaled space**.
  `dim_parameters` = number of free params.
- The simulator and MCMC need the **full** parameter vector → reconstruct via
  pypesto (fixed params filled with nominal values).
- `true_parameters`, reference posterior draws, and recovery are all in the same
  free-scaled space.

### Metadata constants
- `dim_parameters`, `dim_data`, `num_observations=10`, `observation_seeds`,
  `num_simulations`, `num_posterior_samples` hardcoded so the task constructs
  **without** the extra (required by `get_available_tasks()`).
- Exact `dim_parameters` (# free params) and `dim_data` (flattened
  `n_timepoints x n_series`, NaN-padded) values filled in during implementation
  by introspecting the loaded Beer problem (one-time AMICI compile).

### Prior
- `get_prior(key, num_samples)` lazily builds & caches the Beer problem, seeds
  numpy's RNG from the JAX `key`, calls the ported `sample_from_prior`
  `num_samples` times, stacks the free-scaled params, returns
  `(num_samples, dim_parameters)` `jnp` array.
- `get_prior_dist()` raises an informative error (no numpyro prior; use
  `get_prior`, install the extra). No `self.prior_dist`.

### Simulator
- `get_simulator(key, max_calls, n_jobs=-1)` lazily builds & caches the AMICI
  model + predictor once, returns a `Simulator`.
- Closure `(key, parameters)`:
  - reconstruct full parameter vectors from free-scaled params,
  - seed numpy RNG from `key` so measurement noise is reproducible,
  - run the ported `simulator_amici` over the batch with joblib
    `Parallel(n_jobs)`,
  - flatten each result via `amici_df_to_array`, stack to `(batch, dim_data)`,
  - **propagate NaN rows** for failed sims (matches source + ODE-task
    convention),
  - return `jnp`.
- `dim_data` = flattened AMICI output — fixed constant.

### Data representation
- The flat `dim_data` vector is the NaN-padded `(n_timepoints x n_series)`
  array (the summarized/observed form), analogous to LV's 20D summary.
- The structured PEtab measurement df (needed by MCMC) is recoverable from this
  flat array via the fixed Beer template + the NaN/value pattern. No sidecar
  metadata files.

### Reference posterior
- `_sample_reference_posterior(key, num_samples, num_observation=None,
  observation=None)`:
  - Asserts exactly one of `num_observation` / `observation` is provided
    (GRF convention).
  - **`num_observation` path:** deterministically regenerate from
    `observation_seeds[N-1]` — seed numpy, draw `true_parameters` via
    `sample_from_prior`, simulate (with noise) to get the measurement df.
    (Forward-compatible: once observation CSVs exist, this path can instead load
    `observation.csv` and reconstruct.)
  - **`observation` path:** take the flat `(1, dim_data)` array and reconstruct
    the measurement df from the Beer template + NaN/value pattern.
  - Runs live pypesto MCMC via the ported `run_mcmc_single`
    (`AdaptiveParallelTemperingSampler` + `AdaptiveMetropolisSampler`, Geweke
    burn-in via `get_mcmc_posterior_samples`), seeded from `key`. Returns
    `num_samples` free-scaled draws `(num_samples, dim_parameters)`. NaN-filled
    if the sim/MCMC fails (matches source).
- `get_reference_posterior_samples()` (base class) loads shipped CSVs when
  present. CSVs are not generated now.

### Observations
- 10 observations; `true_parameters` = prior draws at fixed `observation_seeds`,
  `observation` = noisy AMICI simulation.
- A generation routine produces the `files/num_observation_<N>/...` tree
  (`observation.csv`, `true_parameters.csv`,
  `reference_posterior_samples.csv.bz2`) but is **provided, not run now**. The
  smoke test generates one observation on the fly.

## Testing
New `tests/tasks/test_petab.py`, marked `slow` / `experimental`, **skipped when
the `pypesto` extra isn't importable**:
- construction-only metadata assertions (run **without** the extra),
- prior sampling shape,
- one simulator call (shape + dtype, NaN handling),
- a tiny live MCMC run (few chains, few samples) verifying the reference
  posterior path end-to-end.

## Out of scope
- Generating/shipping reference posterior CSVs and observation files.
- Generic multi-problem PEtab support.
- Any bayesflow / diffusion-model / metrics code from the case study.

## Implementation notes
- Installing the extra triggers a one-time AMICI compile of the Beer model
  (minutes).
- Two hardcoded dimension constants (`dim_parameters`, `dim_data`) are filled in
  from that introspection during implementation.
