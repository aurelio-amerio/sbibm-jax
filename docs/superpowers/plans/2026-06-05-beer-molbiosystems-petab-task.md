# Beer (MolBioSystems2014) PEtab Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the `Beer_MolBioSystems2014` PEtab benchmark model from `diffusion-experiments/case_study2` into `sbibm-jax` as a `beer_molbiosystems` benchmark `Task`, wrapping pypesto/AMICI rather than rewriting in JAX.

**Architecture:** A `BeerMolBioSystems(Task)` subclass lazily loads a cached pypesto/AMICI Beer problem. The prior delegates to a ported `sample_from_prior`; the simulator runs the compiled AMICI model batched over joblib; the reference posterior runs live pypesto parallel-tempering MCMC. The PEtab measurement dataframe needed by MCMC is reconstructed from the flat observation via the fixed Beer template + NaN/value pattern. All heavy deps live behind an optional `pypesto` extra and are lazily imported.

**Tech Stack:** JAX, numpyro (base class only), pypesto, petab, amici, benchmark-models-petab, joblib, scipy, pytest.

**Spec:** `docs/superpowers/specs/2026-06-05-beer-molbiosystems-petab-task-design.md`

---

## File Structure

| File | Responsibility |
| --- | --- |
| `pyproject.toml` (modify) | Add the `pypesto` optional-dependency extra. |
| `src/sbibm_jax/tasks/beer_molbiosystems/__init__.py` (create) | Empty package marker. |
| `src/sbibm_jax/tasks/beer_molbiosystems/petab_helpers.py` (create) | Verbatim-ported pypesto/AMICI helper functions (bayesflow/metrics stripped). |
| `src/sbibm_jax/tasks/beer_molbiosystems/task.py` (create) | The `BeerMolBioSystems` task: metadata, prior, simulator, reference posterior, df reconstruction, observation generation. |
| `src/sbibm_jax/tasks/__init__.py` (modify) | Register `beer_molbiosystems` in `get_task()`. |
| `tests/tasks/test_petab.py` (create) | Construction/registry tests (no extra) + prior/simulator/MCMC tests (skipped without extra). |
| `CLAUDE.md` (modify) | Document the optional extra and the AMICI compile requirement. |

---

## Task 1: Add the `pypesto` optional-dependency extra — THEN STOP

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the optional-dependencies table**

Insert a `[project.optional-dependencies]` table immediately after the `dependencies = [...]` array (after line 15, before `[dependency-groups]`):

```toml
[project.optional-dependencies]
pypesto = [
    "pypesto",
    "petab",
    "amici",
    "benchmark-models-petab",
    "joblib",
    "scipy",
]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "build: add optional pypesto extra for beer_molbiosystems task"
```

- [ ] **Step 3: HARD STOP — hand off to the user for manual install**

The implementing agent runs in a sandbox and **cannot install packages or compile the AMICI C++ model**. Stop here and tell the user:

> "The `pypesto` extra is added and committed. Please install it manually — this triggers a one-time AMICI compile of the Beer model (takes minutes and needs a working C/C++ compiler, SWIG, and BLAS):
> ```
> uv sync --extra pypesto
> ```
> Let me know once it finishes (and report any compile errors) so I can continue with Task 2."

Do not proceed to Task 2 until the user confirms the install succeeded.

---

## Task 2: Introspect the Beer problem to fix the dimension constants

The task must construct **without** the extra (for registry discovery), so `dim_parameters` and `dim_data` must be hardcoded literals. This task obtains those literals by loading the problem once. **Requires the extra installed (Task 1).** May need to run outside the sandbox.

**Files:** none (introspection only — record the printed numbers for Task 4).

- [ ] **Step 1: Run the introspection snippet**

Run:

```bash
uv run python - <<'PY'
import logging, numpy as np
import benchmark_models_petab as benchmark_models
import pypesto.petab

logging.getLogger("pypesto").setLevel(logging.ERROR)
logging.getLogger("petab").setLevel(logging.ERROR)

petab_problem = benchmark_models.get_problem("Beer_MolBioSystems2014")
importer = pypesto.petab.PetabImporter(petab_problem, simulator_type="amici")
factory = importer.create_objective_creator()
factory.create_model(verbose=False)
pypesto_problem = importer.create_problem()

n_free = len(pypesto_problem.x_free_indices)

m = petab_problem.measurement_df
n_timepoints = m["time"].nunique()
n_series = m.groupby(["simulationConditionId", "observableId"]).ngroups
dim_data = n_timepoints * n_series

free_names = [pypesto_problem.x_names[i] for i in pypesto_problem.x_free_indices]
print("DIM_PARAMETERS =", n_free)
print("N_TIMEPOINTS   =", n_timepoints)
print("N_SERIES       =", n_series)
print("DIM_DATA       =", dim_data)
print("FREE_NAMES     =", free_names)
PY
```

Expected: prints integer values for `DIM_PARAMETERS`, `N_TIMEPOINTS`, `N_SERIES`, `DIM_DATA`, and the list of free parameter names.

- [ ] **Step 2: Record the four integers**

Write the printed `DIM_PARAMETERS`, `N_TIMEPOINTS`, `N_SERIES`, `DIM_DATA` values down — they are substituted as literals into `task.py` in Task 4. No code change in this task.

---

## Task 3: Port the pypesto/AMICI helpers verbatim

Copy the SBI-relevant functions from the case study into a task-local module, stripping the bayesflow/metrics code. `diffusion-experiments/` is untracked and not importable, so the code is copied in.

**Files:**
- Create: `src/sbibm_jax/tasks/beer_molbiosystems/__init__.py`
- Create: `src/sbibm_jax/tasks/beer_molbiosystems/petab_helpers.py`

- [ ] **Step 1: Create the empty package marker**

Create `src/sbibm_jax/tasks/beer_molbiosystems/__init__.py` with no content (empty file).

- [ ] **Step 2: Create `petab_helpers.py` with this exact module header**

Create `src/sbibm_jax/tasks/beer_molbiosystems/petab_helpers.py` starting with these imports (this **replaces** the original file's top imports at lines 1-19, dropping the bayesflow import and the `mad`/`median_abs_deviation` helpers which are only used by the dropped metrics code):

```python
"""pypesto/AMICI helpers for the Beer (MolBioSystems2014) PEtab task.

Ported verbatim from diffusion-experiments/case_study2/helper_pypesto.py and
run_mcmc.py, with the bayesflow-dependent metrics code removed. Heavy
dependencies (pypesto, petab, amici, benchmark_models_petab, joblib, scipy) are
imported here; this module is only imported lazily by task.py, so importing the
task package does not require the `pypesto` extra.
"""

import logging
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
import petab
import pypesto
import pypesto.petab
import pypesto.optimize as optimize
import pypesto.sample as sample
import benchmark_models_petab as benchmark_models
from joblib import Parallel, delayed  # noqa: F401  (kept for parity with source)
from scipy import stats
```

- [ ] **Step 3: Append the ported functions from `helper_pypesto.py`**

Copy these functions **verbatim** from `diffusion-experiments/case_study2/helper_pypesto.py` into `petab_helpers.py`, in this order. Use the listed source line ranges; do not alter the bodies:

- `load_problem` — source lines 52-110
- `get_samples_from_dict` — source lines 113-115
- `sample_from_prior` — source lines 118-170
- `simulator_amici` — source lines 173-182
- `create_pypesto_problem` — source lines 185-205
- `scale_values` — source lines 208-231
- `values_to_linear_scale` — source lines 234-257
- `apply_noise_to_data` — source lines 260-307
- `amici_pred_to_df` — source lines 310-346
- `amici_df_to_array` — source lines 349-383

Do **not** copy: `sample_and_simulate` (21-30), `simulate_parallel` (33-48), `compute_likelihood` (386-390), `compute_likelihood_parallel` (393-408), `sample_in_batches` (411-425), `compute_metrics` (428-472) — these are unused or bayesflow-dependent.

- [ ] **Step 4: Append the ported MCMC functions from `run_mcmc.py`**

Copy these functions **verbatim** from `diffusion-experiments/case_study2/run_mcmc.py` into `petab_helpers.py`, in this order, with the noted edit:

- `run_mcmc` — source lines 47-88
- `get_mcmc_posterior_samples` — source lines 91-98
- `run_mcmc_single` — source lines 102-127

Edit only inside `run_mcmc_single`: the source begins with `import amici` / `import logging` and a `pypesto.logging.log(...)` call (lines 104-107). Keep the `import amici` and the amici logger silencing, but they are already covered by module imports — leave the function body byte-for-byte as in the source. Make no other changes.

- [ ] **Step 5: Verify the module imports cleanly**

Run: `uv run python -c "from sbibm_jax.tasks.beer_molbiosystems import petab_helpers; print('ok')"`
Expected: prints `ok` (requires the `pypesto` extra).

- [ ] **Step 6: Lint and commit**

```bash
uv run flake8 src/sbibm_jax/tasks/beer_molbiosystems/petab_helpers.py
git add src/sbibm_jax/tasks/beer_molbiosystems/__init__.py src/sbibm_jax/tasks/beer_molbiosystems/petab_helpers.py
git commit -m "feat: port pypesto/amici helpers for beer_molbiosystems task"
```

If flake8 flags unused imports that are genuinely unused after the verbatim copy (e.g. `optimize`, `petab`, `Union`), keep them only if a copied function references them; otherwise remove the specific unused name. `run_mcmc` uses `optimize`, `Union`, `petab`; `sample_from_prior` uses `stats`, `pd`, `np`; all are referenced.

---

## Task 4: Scaffold `task.py` — metadata, prior, lazy loader

**Files:**
- Create: `src/sbibm_jax/tasks/beer_molbiosystems/task.py`
- Test: `tests/tasks/test_petab.py`

Substitute the four integers recorded in Task 2 for `DIM_PARAMETERS`, `N_TIMEPOINTS`, `N_SERIES`, `DIM_DATA` below.

- [ ] **Step 1: Write the failing construction + prior tests**

Create `tests/tasks/test_petab.py`:

```python
"""Tests for the Beer (MolBioSystems2014) PEtab task."""

import importlib.util

import jax
import jax.numpy as jnp
import pytest

from sbibm_jax.tasks.beer_molbiosystems.task import BeerMolBioSystems

HAS_PYPESTO = importlib.util.find_spec("pypesto") is not None
requires_pypesto = pytest.mark.skipif(
    not HAS_PYPESTO, reason="pypesto extra not installed"
)


class TestMetadata:
    def test_constructs_without_extra(self):
        # Must construct without importing pypesto (registry discovery).
        task = BeerMolBioSystems()
        assert task.name == "beer_molbiosystems"
        assert task.name_display == "Beer (MolBioSystems2014)"
        assert task.dim_parameters > 0
        assert task.dim_data > 0
        assert task.num_observations == 10
        assert len(task.observation_seeds) == 10

    def test_prior_dist_raises(self):
        task = BeerMolBioSystems()
        with pytest.raises(NotImplementedError):
            task.get_prior_dist()


@requires_pypesto
class TestPrior:
    def test_shape(self):
        task = BeerMolBioSystems()
        key = jax.random.PRNGKey(0)
        samples = task.get_prior(key, num_samples=5)
        assert samples.shape == (5, task.dim_parameters)
        assert jnp.isrealobj(samples)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/tasks/test_petab.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` for `BeerMolBioSystems` (task.py does not exist yet).

- [ ] **Step 3: Write `task.py` (metadata + lazy loader + prior)**

Create `src/sbibm_jax/tasks/beer_molbiosystems/task.py`:

```python
"""Beer (MolBioSystems2014) PEtab benchmark task.

Wraps the pypesto/AMICI Beer_MolBioSystems2014 benchmark model. All heavy
dependencies live behind the optional `pypesto` extra and are imported lazily,
so the task constructs (for registry discovery) without them. Install with:

    uv sync --extra pypesto      # or: pip install sbibm-jax[pypesto]

Installing the extra triggers a one-time AMICI compile of the Beer model.
"""

from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from sbibm_jax.tasks.simulator import Simulator
from sbibm_jax.tasks.task import Task

# Filled in from Task 2 introspection of the Beer problem.
DIM_PARAMETERS = DIM_PARAMETERS  # <-- replace with the integer from Task 2
N_TIMEPOINTS = N_TIMEPOINTS      # <-- replace with the integer from Task 2
N_SERIES = N_SERIES              # <-- replace with the integer from Task 2
DIM_DATA = DIM_DATA              # <-- replace with the integer from Task 2

_PROBLEM_NAME = "Beer_MolBioSystems2014"

_EXTRA_MSG = (
    "The beer_molbiosystems task requires the optional `pypesto` extra. "
    "Install it with `uv sync --extra pypesto` or "
    "`pip install sbibm-jax[pypesto]` (this triggers a one-time AMICI compile)."
)


def _seed_from_key(key: jax.random.PRNGKey) -> int:
    """Derive a 32-bit numpy seed from a JAX key (for reproducible numpy RNG)."""
    return int(jax.random.randint(key, (), 0, 2**31 - 1))


class BeerMolBioSystems(Task):
    def __init__(self, n_jobs: int = -1):
        """Beer (MolBioSystems2014) PEtab task.

        Args:
            n_jobs: joblib parallelism for the AMICI simulator batch (default -1,
                all cores).
        """
        self.n_jobs = n_jobs
        super().__init__(
            dim_parameters=DIM_PARAMETERS,
            dim_data=DIM_DATA,
            name=Path(__file__).parent.name,
            name_display="Beer (MolBioSystems2014)",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
        )
        # Lazily built, cached pypesto/AMICI handles.
        self._loaded = None

    # --- lazy pypesto/AMICI loading -------------------------------------

    def _load(self):
        """Build & cache the pypesto/AMICI Beer problem (one-time AMICI compile)."""
        if self._loaded is None:
            try:
                from sbibm_jax.tasks.beer_molbiosystems import petab_helpers
            except ImportError as e:  # pragma: no cover
                raise ImportError(_EXTRA_MSG) from e
            pypesto_problem, petab_problem, factory, amici_predictor = (
                petab_helpers.load_problem(_PROBLEM_NAME, create_amici_model=True)
            )
            self._loaded = {
                "helpers": petab_helpers,
                "pypesto_problem": pypesto_problem,
                "petab_problem": petab_problem,
                "factory": factory,
                "amici_predictor": amici_predictor,
            }
        return self._loaded

    # --- prior ----------------------------------------------------------

    def get_prior(
        self, key: jax.random.PRNGKey, num_samples: int = 1
    ) -> jnp.ndarray:
        L = self._load()
        helpers = L["helpers"]
        pp = L["pypesto_problem"]
        petab_problem = L["petab_problem"]

        np.random.seed(_seed_from_key(key))
        rows = []
        for _ in range(num_samples):
            prior = helpers.sample_from_prior(petab_problem, pp)
            full_scaled = np.asarray(prior["amici_params"]).reshape(-1)
            free = pp.get_reduced_vector(full_scaled)
            rows.append(np.asarray(free, dtype=float).reshape(-1))
        return jnp.asarray(np.stack(rows, axis=0))

    def get_prior_dist(self):
        raise NotImplementedError(
            "beer_molbiosystems has no numpyro prior_dist; the prior is defined "
            "by the PEtab parameter table. Use get_prior(key, num_samples). "
            + _EXTRA_MSG
        )

    def get_labels_parameters(self) -> List[str]:
        L = self._load()
        pp = L["pypesto_problem"]
        return [pp.x_names[i] for i in pp.x_free_indices]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/tasks/test_petab.py -v`
Expected: PASS for `TestMetadata` (no extra needed) and `TestPrior` (extra installed). Without the extra, `TestPrior` is skipped.

- [ ] **Step 5: Lint and commit**

```bash
uv run flake8 src/sbibm_jax/tasks/beer_molbiosystems/task.py tests/tasks/test_petab.py
git add src/sbibm_jax/tasks/beer_molbiosystems/task.py tests/tasks/test_petab.py
git commit -m "feat: scaffold beer_molbiosystems task with prior"
```

---

## Task 5: Implement the AMICI simulator

**Files:**
- Modify: `src/sbibm_jax/tasks/beer_molbiosystems/task.py`
- Test: `tests/tasks/test_petab.py`

- [ ] **Step 1: Write the failing simulator test**

Add to `tests/tasks/test_petab.py`:

```python
@requires_pypesto
class TestSimulator:
    def test_shape_and_dtype(self):
        task = BeerMolBioSystems()
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(1), 3)
        theta = task.get_prior(k1, num_samples=3)
        sim = task.get_simulator(k2)
        data = sim(k3, theta)
        assert data.shape == (3, task.dim_data)
        assert jnp.isrealobj(data)

    def test_budget_exceeded(self):
        from sbibm_jax.tasks.simulator import SimulationBudgetExceeded

        task = BeerMolBioSystems()
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(2), 3)
        theta = task.get_prior(k1, num_samples=3)
        sim = task.get_simulator(k2, max_calls=1)
        with pytest.raises(SimulationBudgetExceeded):
            sim(k3, theta)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/tasks/test_petab.py::TestSimulator -v`
Expected: FAIL — `AttributeError`/`NotImplementedError` (`get_simulator` not defined on the subclass).

- [ ] **Step 3: Implement `get_simulator`**

Add these methods to `BeerMolBioSystems` in `task.py` (after `get_labels_parameters`):

```python
    # --- simulator ------------------------------------------------------

    def _full_scaled(self, free_scaled: np.ndarray) -> np.ndarray:
        """Reconstruct a full scaled parameter vector from free-scaled params."""
        pp = self._load()["pypesto_problem"]
        return np.asarray(pp.get_full_vector(np.asarray(free_scaled).reshape(-1)))

    def get_simulator(
        self, key: jax.random.PRNGKey, max_calls: Optional[int] = None
    ) -> Simulator:
        from joblib import Parallel, delayed

        L = self._load()
        helpers = L["helpers"]
        factory = L["factory"]
        amici_predictor = L["amici_predictor"]
        petab_problem = L["petab_problem"]
        pp = L["pypesto_problem"]
        n_jobs = self.n_jobs
        dim_data = self.dim_data

        def _simulate_one(full_scaled):
            out = helpers.simulator_amici(
                full_scaled, amici_predictor, factory,
                petab_problem, pp, return_df=False,
            )
            return np.asarray(out["sim_data"], dtype=float).reshape(-1)

        def simulator(key, parameters):
            params = np.asarray(parameters, dtype=float)
            np.random.seed(_seed_from_key(key))  # reproducible measurement noise
            full = [self._full_scaled(params[i]) for i in range(params.shape[0])]
            results = Parallel(n_jobs=n_jobs)(
                delayed(_simulate_one)(f) for f in full
            )
            rows = []
            for r in results:
                if r.shape[0] != dim_data:
                    rows.append(np.full(dim_data, np.nan))
                else:
                    rows.append(r)
            return jnp.asarray(np.stack(rows, axis=0))

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/tasks/test_petab.py::TestSimulator -v`
Expected: PASS (extra installed) or SKIP (no extra).

- [ ] **Step 5: Lint and commit**

```bash
uv run flake8 src/sbibm_jax/tasks/beer_molbiosystems/task.py tests/tasks/test_petab.py
git add src/sbibm_jax/tasks/beer_molbiosystems/task.py tests/tasks/test_petab.py
git commit -m "feat: add AMICI simulator to beer_molbiosystems task"
```

---

## Task 6: Reference posterior + measurement-df reconstruction + observation generation

**Files:**
- Modify: `src/sbibm_jax/tasks/beer_molbiosystems/task.py`
- Test: `tests/tasks/test_petab.py`

- [ ] **Step 1: Write the failing reference-posterior test**

Add to `tests/tasks/test_petab.py`:

```python
@requires_pypesto
@pytest.mark.slow
@pytest.mark.experimental
class TestReferencePosterior:
    def test_reconstruct_roundtrip(self):
        # Generating an observation yields a flat array whose reconstructed
        # measurement df matches the generated df on the measured rows.
        task = BeerMolBioSystems()
        true_params, flat_obs, sim_df = task._generate_observation(
            task.observation_seeds[0]
        )
        assert flat_obs.shape == (task.dim_data,)
        recon = task._flat_to_measurement_df(flat_obs)
        import numpy as np
        a = sim_df.sort_values(
            ["simulationConditionId", "observableId", "time"]
        )["simulation"].to_numpy()
        b = recon.sort_values(
            ["simulationConditionId", "observableId", "time"]
        )["simulation"].to_numpy()
        finite = np.isfinite(a) & np.isfinite(b)
        assert finite.sum() > 0
        assert np.allclose(a[finite], b[finite], atol=1e-6)

    def test_live_mcmc_runs(self):
        # Tiny MCMC: verify the reference path runs end-to-end and returns
        # free-scaled draws of the right shape.
        task = BeerMolBioSystems()
        samples = task._sample_reference_posterior(
            jax.random.PRNGKey(0),
            num_samples=8,
            num_observation=1,
            n_starts=0,
            n_mcmc_samples=200,
            n_chains=2,
        )
        assert samples.shape == (8, task.dim_parameters)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/tasks/test_petab.py::TestReferencePosterior -v`
Expected: FAIL — `AttributeError` (`_generate_observation` / `_flat_to_measurement_df` / `_sample_reference_posterior` not yet defined).

- [ ] **Step 3: Implement reconstruction, observation generation, and the reference sampler**

Add these methods to `BeerMolBioSystems` in `task.py`:

```python
    # --- reference posterior --------------------------------------------

    def _series_and_timepoints(self):
        """Fixed (sorted) series keys and timepoints defining the flat layout.

        Matches the layout produced by petab_helpers.amici_df_to_array:
        rows = sorted unique times, columns = sorted (condition, observable)
        pairs, flattened row-major into dim_data.
        """
        petab_problem = self._load()["petab_problem"]
        m = petab_problem.measurement_df
        timepoints = np.sort(m["time"].unique())
        series = sorted(
            m.groupby(["simulationConditionId", "observableId"]).groups.keys()
        )
        return series, timepoints

    def _flat_to_measurement_df(self, flat_obs):
        """Reconstruct a PEtab measurement df from a flat observation vector.

        Uses the fixed Beer template (measured rows) + the NaN/value pattern of
        the flat array. Sets both 'simulation' and 'measurement' columns so the
        result is consumable by run_mcmc / run_mcmc_single.
        """
        from copy import deepcopy

        petab_problem = self._load()["petab_problem"]
        series, timepoints = self._series_and_timepoints()
        arr = np.asarray(flat_obs, dtype=float).reshape(
            len(timepoints), len(series)
        )
        series_index = {s: i for i, s in enumerate(series)}
        time_index = {float(t): i for i, t in enumerate(timepoints)}

        df = deepcopy(petab_problem.measurement_df)
        vals = []
        for _, row in df.iterrows():
            si = series_index[
                (row["simulationConditionId"], row["observableId"])
            ]
            ti = time_index[float(row["time"])]
            vals.append(arr[ti, si])
        df["simulation"] = vals
        df["measurement"] = vals
        return df

    def _generate_observation(self, seed: int):
        """Generate one observation deterministically from an integer seed.

        Returns (true_params_free_scaled, flat_obs, sim_data_df).
        """
        L = self._load()
        helpers = L["helpers"]
        petab_problem = L["petab_problem"]
        pp = L["pypesto_problem"]
        factory = L["factory"]
        amici_predictor = L["amici_predictor"]

        np.random.seed(int(seed))
        prior = helpers.sample_from_prior(petab_problem, pp)
        full_scaled = np.asarray(prior["amici_params"]).reshape(-1)
        true_free = np.asarray(pp.get_reduced_vector(full_scaled), dtype=float)

        out = helpers.simulator_amici(
            full_scaled, amici_predictor, factory,
            petab_problem, pp, return_df=True,
        )
        flat_obs = np.asarray(out["sim_data"], dtype=float).reshape(-1)
        return true_free, flat_obs, out["sim_data_df"]

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
        n_starts: int = 10,
        n_mcmc_samples: int = 100000,
        n_chains: int = 5,
    ) -> jnp.ndarray:
        assert (num_observation is None) != (observation is None), (
            "Provide exactly one of num_observation or observation."
        )
        L = self._load()
        helpers = L["helpers"]
        petab_problem = L["petab_problem"]
        pp = L["pypesto_problem"]

        if num_observation is not None:
            seed = self.observation_seeds[num_observation - 1]
            _, _, sim_df = self._generate_observation(seed)
        else:
            flat = np.asarray(observation, dtype=float).reshape(-1)
            sim_df = self._flat_to_measurement_df(flat)

        np.random.seed(_seed_from_key(key))
        samples = helpers.run_mcmc_single(
            petab_prob=petab_problem,
            pypesto_prob=pp,
            sim_data_df=sim_df,
            n_starts=n_starts,
            n_mcmc_samples=n_mcmc_samples,
            n_final_samples=num_samples,
            n_chains=n_chains,
        )
        return jnp.asarray(np.asarray(samples, dtype=float))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/tasks/test_petab.py::TestReferencePosterior -v`
Expected: PASS (extra installed) or SKIP (no extra). The live-MCMC test is slow (a minute or two).

- [ ] **Step 5: Lint and commit**

```bash
uv run flake8 src/sbibm_jax/tasks/beer_molbiosystems/task.py tests/tasks/test_petab.py
git add src/sbibm_jax/tasks/beer_molbiosystems/task.py tests/tasks/test_petab.py
git commit -m "feat: add live MCMC reference posterior for beer_molbiosystems"
```

---

## Task 7: Register the task in the registry

**Files:**
- Modify: `src/sbibm_jax/tasks/__init__.py:60-67`
- Test: `tests/tasks/test_petab.py`

- [ ] **Step 1: Write the failing registry test**

Add to `tests/tasks/test_petab.py`:

```python
class TestRegistry:
    def test_get_task_returns_instance(self):
        from sbibm_jax import get_task

        task = get_task("beer_molbiosystems")
        assert isinstance(task, BeerMolBioSystems)

    def test_available_tasks_includes_beer(self):
        from sbibm_jax import get_available_tasks

        assert "beer_molbiosystems" in get_available_tasks()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/tasks/test_petab.py::TestRegistry -v`
Expected: FAIL — `NotImplementedError: Task 'beer_molbiosystems' not found.`

- [ ] **Step 3: Add the registry branch**

In `src/sbibm_jax/tasks/__init__.py`, insert this branch after the `gaussian_random_field` branch (after line 64, before the `else`):

```python
    elif task_name == "beer_molbiosystems":
        from sbibm_jax.tasks.beer_molbiosystems.task import BeerMolBioSystems
        return BeerMolBioSystems(*args, **kwargs)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/tasks/test_petab.py::TestRegistry -v`
Expected: PASS. `get_available_tasks()` already discovers the directory automatically.

- [ ] **Step 5: Lint and commit**

```bash
uv run flake8 src/sbibm_jax/tasks/__init__.py
git add src/sbibm_jax/tasks/__init__.py tests/tasks/test_petab.py
git commit -m "feat: register beer_molbiosystems in task registry"
```

---

## Task 8: Observation/CSV generation routine (provided, not run)

Provide a method that writes the `files/num_observation_<N>/...` tree, mirroring the base-class file layout, but do **not** execute it (CSVs are generated later).

**Files:**
- Modify: `src/sbibm_jax/tasks/beer_molbiosystems/task.py`
- Test: `tests/tasks/test_petab.py`

- [ ] **Step 1: Write the failing test (dry-run into a tmp dir, one observation)**

Add to `tests/tasks/test_petab.py`:

```python
@requires_pypesto
@pytest.mark.slow
@pytest.mark.experimental
class TestObservationGeneration:
    def test_generate_one_observation_files(self, tmp_path):
        task = BeerMolBioSystems()
        task.generate_observation_files(
            num_observation=1,
            out_dir=tmp_path,
            num_reference_samples=0,  # skip the expensive MCMC for the dry run
        )
        obs = tmp_path / "num_observation_1" / "observation.csv"
        tp = tmp_path / "num_observation_1" / "true_parameters.csv"
        assert obs.exists()
        assert tp.exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/tasks/test_petab.py::TestObservationGeneration -v`
Expected: FAIL — `AttributeError` (`generate_observation_files` not defined).

- [ ] **Step 3: Implement `generate_observation_files`**

Add this method to `BeerMolBioSystems` in `task.py`:

```python
    # --- data-file generation (provided; run later, not now) ------------

    def generate_observation_files(
        self,
        num_observation: int,
        out_dir: Optional[Path] = None,
        num_reference_samples: Optional[int] = None,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Write the files/num_observation_<N>/ tree for one observation.

        Creates observation.csv and true_parameters.csv, and (when
        num_reference_samples > 0) reference_posterior_samples.csv.bz2. This is
        provided for later batch generation; it is not run as part of the port.

        Args:
            num_observation: 1-indexed observation number.
            out_dir: Base directory (defaults to <task>/files).
            num_reference_samples: Reference draws to generate (default
                self.num_reference_posterior_samples; 0 to skip MCMC).
            key: PRNG key for the reference MCMC (defaults to a key seeded from
                the observation seed).
        """
        base = Path(out_dir) if out_dir is not None else (self.path / "files")
        obs_dir = base / f"num_observation_{num_observation}"
        obs_dir.mkdir(parents=True, exist_ok=True)

        seed = self.observation_seeds[num_observation - 1]
        true_free, flat_obs, _ = self._generate_observation(seed)

        self.save_data(obs_dir / "observation.csv", jnp.asarray(flat_obs)[None, :])
        self.save_parameters(
            obs_dir / "true_parameters.csv", jnp.asarray(true_free)[None, :]
        )

        n_ref = (
            self.num_reference_posterior_samples
            if num_reference_samples is None
            else num_reference_samples
        )
        if n_ref and n_ref > 0:
            ref_key = key if key is not None else jax.random.PRNGKey(int(seed))
            ref = self._sample_reference_posterior(
                ref_key, num_samples=n_ref, num_observation=num_observation
            )
            self.save_parameters(
                obs_dir / "reference_posterior_samples.csv.bz2", ref
            )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/tasks/test_petab.py::TestObservationGeneration -v`
Expected: PASS (extra installed) or SKIP (no extra).

- [ ] **Step 5: Lint and commit**

```bash
uv run flake8 src/sbibm_jax/tasks/beer_molbiosystems/task.py tests/tasks/test_petab.py
git add src/sbibm_jax/tasks/beer_molbiosystems/task.py tests/tasks/test_petab.py
git commit -m "feat: add observation-file generation routine for beer_molbiosystems"
```

---

## Task 9: Document the optional extra in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add a note under the Commands section**

In `CLAUDE.md`, after the paragraph describing the `torch` group (the `uv run --group torch ...` paragraph), add:

```markdown
The `beer_molbiosystems` PEtab task needs the optional `pypesto` extra
(`pypesto`, `petab`, `amici`, `benchmark-models-petab`, `joblib`, `scipy`):
`uv sync --extra pypesto`. Installing it triggers a one-time AMICI compile of
the Beer model (needs a C/C++ compiler, SWIG, and BLAS). The task constructs
without the extra (for registry discovery) but raises an informative error when
the prior/simulator/reference-posterior methods are called without it. Its
helper code is a verbatim port of `diffusion-experiments/case_study2`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document pypesto extra for beer_molbiosystems task"
```

---

## Task 10: Full verification sweep

**Files:** none.

- [ ] **Step 1: Lint the whole package**

Run: `uv run flake8 src tests`
Expected: no output (clean).

- [ ] **Step 2: Run the task tests without the extra logic exercised by markers**

Run: `uv run pytest tests/tasks/test_petab.py -v`
Expected: with the extra installed, all pass; the `slow`/`experimental` MCMC and generation tests run (may take a few minutes). Without the extra, only `TestMetadata` and `TestRegistry` run and pass; the rest skip.

- [ ] **Step 3: Confirm the rest of the suite still passes**

Run: `uv run pytest -m "not slow"`
Expected: PASS (no regressions; the construction/registry beer tests run, the extra-dependent ones skip if the extra is absent).

- [ ] **Step 4: Final commit (if any lint fixes were needed)**

```bash
git add -A
git commit -m "chore: lint fixes for beer_molbiosystems task" || echo "nothing to commit"
```

---

## Self-Review Notes

- **Spec coverage:** placement/registry (Task 7), `pypesto` extra (Task 1), ported helpers with bayesflow stripped (Task 3), free-scaled parameter space + full-vector reconstruction (Tasks 4-5), prior delegating to pypesto with `get_prior_dist` raising (Task 4), AMICI simulator with NaN propagation + joblib batching (Task 5), live MCMC reference posterior with both conditioning paths + template/NaN reconstruction (Task 6), 10 observations + deferred generation routine (Task 8), metadata constants for extra-free construction (Tasks 2, 4), tests skipped without the extra (Tasks 4-8), CLAUDE.md docs (Task 9). CSV generation is provided but not run — matches the spec's "do not generate CSVs now."
- **Deferred constants:** `DIM_PARAMETERS`/`N_TIMEPOINTS`/`N_SERIES`/`DIM_DATA` are not placeholders — they are concrete integers produced by the Task 2 introspection command and substituted as literals in Task 4.
- **Type consistency:** `_load()` returns a dict with keys `helpers`, `pypesto_problem`, `petab_problem`, `factory`, `amici_predictor`, used consistently across Tasks 4-8. `_generate_observation` returns `(true_free, flat_obs, sim_data_df)` consumed by Tasks 6 and 8. `_flat_to_measurement_df` and `_series_and_timepoints` share the same series/timepoint layout that mirrors `amici_df_to_array`.
