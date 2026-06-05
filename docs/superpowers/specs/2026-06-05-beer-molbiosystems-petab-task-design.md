# Design: `beer_molbiosystems` PEtab benchmark task

**Date:** 2026-06-05
**Status:** Approved (pre-implementation)

## Goal

Integrate the `Beer_MolBioSystems2014` PEtab benchmark model (from
`diffusion-experiments/case_study2`) into `sbibm-jax` as a benchmark `Task`.
Unlike the existing JAX-native tasks, this one is backed by external libraries:

- **Prior** comes from the PEtab parameter table (truncated-normal on scaled
  space, with bound tweaks applied by `load_problem`).
- **Simulator** is a compiled AMICI ODE model (not JAX-vmappable).
- **Reference posterior** comes from pypesto parallel-tempering MCMC.

The heavy machinery (`pypesto`, `petab`, `amici`, `benchmark-models-petab`) is
gated behind an optional **`pypesto` extra** so the rest of the library stays
JAX-only and installable without it.

## Decisions (from brainstorming)

1. **Scope:** Beer-only, hardcoded — not a generic PEtab wrapper. New PEtab
   models would get their own task directory later.
2. **Reference posterior:** ship-precomputed-CSVs + live-MCMC fallback, but
   **do not generate the CSVs now**; only verify the live MCMC path runs
   end-to-end with small settings.
3. **Prior:** delegate to pypesto's `sample_from_prior`; **no numpyro
   `prior_dist`**. Rationale: anything generative (simulator, MCMC, training
   data) requires the extra anyway, so requiring it to sample the prior adds no
   burden, and a hand-maintained numpyro prior would risk drifting from the
   PEtab table.
4. **Simulator batching:** joblib `Parallel` with a configurable `n_jobs`
   (default -1), mirroring the case study. Can iterate later.
5. **Observations:** 10 observations whose true parameters are prior draws at
   fixed `observation_seeds`; observation = noisy AMICI simulation. Data files
   are deferred (generation routine provided, not run now).

## Architecture

### Placement & registry
- New directory `src/sbibm_jax/tasks/beer_molbiosystems/`.
- `name` = directory name (`beer_molbiosystems`),
  `name_display = "Beer (MolBioSystems2014)"`.
- Registered in `src/sbibm_jax/tasks/__init__.py` `get_task()` with a lazy
  per-branch import, like the other tasks. `get_available_tasks()` picks it up
  by directory discovery.

### Dependencies
- Add a PEP 621 `[project.optional-dependencies]` table with a `pypesto` extra:
  `pypesto`, `petab`, `amici`, `benchmark-models-petab`, `joblib`, `scipy`.
- Installable via `uv sync --extra pypesto` / `pip install sbibm-jax[pypesto]`.
- `amici` compiles a C++ model on first use → the extra needs a working
  compiler / SWIG / BLAS. Documented in CLAUDE.md and the task docstring.
- All heavy deps are **lazily imported inside methods**, raising a clear
  "install the `pypesto` extra" error when missing.

### Ported helpers (`beer_molbiosystems/petab_helpers.py`)
Port the needed subset of `case_study2/helper_pypesto.py`, stripped of the
BayesFlow/metrics imports:
- `load_problem`, `sample_from_prior`, `get_samples_from_dict`
- `simulator_amici` + `amici_pred_to_df`, `amici_df_to_array`,
  `apply_noise_to_data`, `scale_values`, `values_to_linear_scale`
- `create_pypesto_problem`
- From `run_mcmc.py`: `run_mcmc`, `get_mcmc_posterior_samples`,
  `run_mcmc_single`

`diffusion-experiments/` is untracked and not part of the package, so code is
copied in, not imported from there.

### Parameter space
- Canonical parameters = **free/estimated parameters in scaled space**.
  `dim_parameters` = number of free params.
- The simulator and MCMC need the **full** parameter vector → reconstruct via
  pypesto `get_full_vector` (fixed params filled with nominal values).
- `true_parameters` stored as free scaled params; reference posterior draws and
  recovery are in the same free-scaled space.

### Prior
- `get_prior(key, num_samples)` lazily calls ported `sample_from_prior`,
  seeding numpy's RNG from the JAX `key` for reproducibility, returns
  `(num_samples, dim_parameters)` `jnp` array of free scaled params.
- `get_prior_dist()` raises an informative error (no numpyro prior; use
  `get_prior`, install the extra).

### Simulator
- `get_simulator(key, max_calls, n_jobs=-1)` lazily builds & caches the AMICI
  model once, returns a `Simulator`.
- Closure `(key, parameters)`:
  - reconstruct full parameter vectors from free params,
  - run `simulator_amici` over the batch with joblib `Parallel(n_jobs)`,
  - seed numpy RNG from `key` so measurement noise is reproducible,
  - stack to `(batch, dim_data)`, propagate NaN rows for failed sims,
  - return `jnp`.
- `dim_data` = flattened AMICI output (`n_timepoints x n_series`, NaN-padded for
  missing condition/observable/time combos) — fixed constant.

### Reference posterior
- `_sample_reference_posterior(key, num_samples, num_observation=None,
  observation=None)` runs live pypesto MCMC
  (`AdaptiveParallelTemperingSampler` + `AdaptiveMetropolisSampler`, Geweke
  burn-in via `get_mcmc_posterior_samples`), returns `num_samples` free-param
  draws. Exactly one of `num_observation` / `observation` must be provided
  (matching the GRF convention).
- `get_reference_posterior_samples()` (base class) loads shipped CSVs when
  present. CSVs are not generated now.

### Observations
- 10 observations; `true_parameters` = prior draws at fixed `observation_seeds`,
  `observation` = noisy AMICI simulation.
- A generation routine produces the `files/num_observation_<N>/...` tree but is
  **not run now**. Smoke test generates one observation on the fly.

### Metadata constants
- `dim_parameters`, `dim_data`, `observation_seeds`, `num_simulations`,
  `num_posterior_samples` hardcoded so the task constructs **without** the
  extra (required by `get_available_tasks()` / `get_task_name_display()`).
- Exact `dim_parameters` / `dim_data` values filled in during implementation by
  introspecting the loaded Beer problem.

## Testing
- New `tests/tasks/test_petab.py`, marked `slow` / `experimental`, **skipped
  when the extra isn't importable**:
  - construction-only metadata assertions (run without the extra),
  - prior sampling shape,
  - one simulator call (shape + dtype, NaN handling),
  - a tiny live MCMC run (few chains, few samples) verifying the reference
    posterior path end-to-end.

## Out of scope
- Generating/shipping reference posterior CSVs and observation files.
- Generic multi-problem PEtab support.
- Any BayesFlow / diffusion-model / metrics code from the case study.

## Implementation notes
- Installing the extra triggers a one-time AMICI compile of the Beer model
  (minutes).
- Two hardcoded dimension constants are filled in from that introspection.
