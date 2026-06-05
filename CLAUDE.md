# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`sbibm-jax` is a JAX/NumPyro rewrite of the Simulation-Based Inference Benchmark
(sbibm). It provides a set of benchmark *tasks* — each defining a prior, a
simulator, reference observations, and reference posterior samples — for
evaluating SBI methods. The original sbibm was PyTorch/Pyro based; this port
replaces those with JAX, `numpyro.distributions`, and `diffrax` (for ODE tasks).

## Commands

This project uses `uv`. Python is pinned to 3.12.

```bash
uv sync --all-groups            # install all dependency groups (dev work)
uv run pytest                   # run the test suite (CPU-forced, 2 workers via xdist)
uv run pytest tests/tasks/test_analytical.py            # run one file
uv run pytest tests/tasks/test_ode.py::TestLotkaVolterra # run one class
uv run pytest -k two_moons                              # run by keyword
uv run pytest -m "not slow"     # skip slow-marked tests
uv run flake8 src tests         # lint
```

Test configuration lives in `pyproject.toml` (`[tool.pytest.ini_options]`):
`JAX_PLATFORMS=cpu` is injected via `pytest-env` so tests never touch the GPU,
and `-n 2` (pytest-xdist) runs them in parallel. Markers: `slow`, `experimental`.

The default JAX install is the CUDA 12 build (`jax[cuda12]`). PyTorch is only
needed for the one-time data-conversion script and is pinned to the CPU index;
pull it in with the `torch` group: `uv run --group torch python scripts/convert_torch_to_npz.py`.

The `beer_molbiosystems` PEtab task needs the optional `pypesto` extra
(`pypesto`, `petab`, `amici`, `benchmark-models-petab`, `joblib`, `scipy`):
`uv sync --extra pypesto`. Installing it triggers a one-time AMICI compile of
the Beer model (needs a C/C++ compiler, SWIG, and BLAS). The task constructs
without the extra (for registry discovery) but raises an informative error when
the prior/simulator/reference-posterior methods are called without it. Its
helper code is a verbatim port of `diffusion-experiments/case_study2`.

## Architecture

**Task abstraction.** `src/sbibm_jax/tasks/task.py` defines the abstract `Task`
base class. Every benchmark is a subclass living in
`src/sbibm_jax/tasks/<name>/task.py`. A task carries dimensionality/budget
metadata and implements three abstract methods:
- `get_prior(key, num_samples)` → parameter samples `(num_samples, dim_parameters)`
- `get_simulator(key, max_calls)` → a `Simulator` instance
- `_sample_reference_posterior(...)` → reference posterior (closed-form where possible)

Priors are `numpyro.distributions` objects stored on `self.prior_dist`. The base
class also provides CSV loaders for observations, true parameters, and reference
posterior samples.

**Simulator wrapper.** `src/sbibm_jax/tasks/simulator.py` wraps each task's raw
simulator function `(key, parameters) -> data`. It enforces a simulation budget
(`max_calls`, raising `SimulationBudgetExceeded`), counts calls, normalizes input
shapes, and flattens output via the task's `flatten_data`. Tasks define the
simulator as a closure inside `get_simulator` and return
`Simulator(task=self, simulator=fn, max_calls=...)`.

**Registry.** `src/sbibm_jax/tasks/__init__.py` maps task-name strings to classes
in `get_task()`, with lazy per-branch imports. Some names are aliases/variants of
the same class passing different kwargs (e.g. `slcp_distractors` → `SLCP(distractors=True)`,
`bernoulli_glm_raw` → `BernoulliGLM(summary="raw")`, `gaussian_nonlinear` → `SLCP`).
`get_available_tasks()` discovers task directories on disk and appends these
extra variant names. The top-level `sbibm_jax` package re-exports `get_task` and
`get_available_tasks`.

**Task data files.** Each task directory has a `files/` subtree:
`files/num_observation_<N>/{observation.csv, true_parameters.csv,
reference_posterior_samples.csv.bz2}`. These are read by the base-class loaders
via `sbibm_jax/utils/io.py` (pandas → numpy → JAX, default dtype `float32`,
`atleast_2d`). Some tasks also have task-specific data (design matrices, GMM
params) stored as `.npz`, converted from the original PyTorch `.pt`/`.torch`
files by `scripts/convert_torch_to_npz.py`.

**ODE tasks** (Lotka-Volterra, SIR) use `diffrax` for integration — vector field
function + `diffeqsolve` with `Tsit5`/`PIDController`, `jax.vmap`ed over the
parameter batch. They may produce NaNs for divergent parameters; simulators
propagate NaN rows rather than failing.

### Conventions

- All array ops use `jax.numpy`; randomness is explicit PRNG keys split with
  `jax.random.split`. Functions threading keys take `key` as the first argument.
- Task names equal their directory name (set via `Path(__file__).parent.name`),
  and `name_display` carries the human-readable label.
- Tasks are grouped into "phases" (analytical, ODE, …) reflected in test files,
  not in the package layout.

## Note on `diffusion-experiments/`

The `diffusion-experiments/` directory is untracked reference/research code
(case studies, separate from the `sbibm_jax` package). It is not part of the
installable package or the test suite.
