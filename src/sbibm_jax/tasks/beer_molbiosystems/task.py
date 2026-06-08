"""Beer (MolBioSystems2014) PEtab benchmark task.

Wraps the pypesto/AMICI Beer_MolBioSystems2014 benchmark model. All heavy
dependencies live behind the optional `pypesto` extra and are imported lazily,
so the task constructs (for registry discovery) without them. Install with:

    uv sync --extra pypesto      # or: pip install sbibm-jax[pypesto]

Installing the extra triggers a one-time AMICI compile of the Beer model.
"""

import logging
from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from sbibm_jax.tasks.simulator import Simulator
from sbibm_jax.tasks.task import Task

# Filled in from Task 2 introspection of the Beer problem.
DIM_PARAMETERS = 72
N_TIMEPOINTS = 714
N_SERIES = 38
DIM_DATA = 27132

_PROBLEM_NAME = "Beer_MolBioSystems2014"

_EXTRA_MSG = (
    "The beer_molbiosystems task requires the optional `pypesto` extra. "
    "Install it with `uv sync --extra pypesto` or "
    "`pip install sbibm-jax[pypesto]` (this triggers a one-time AMICI compile)."
)


def _seed_from_key(key: jax.random.PRNGKey) -> int:
    """Derive a 32-bit numpy seed from a JAX key (for reproducible numpy RNG)."""
    return int(jax.random.randint(key, (), 0, 2**31 - 1))


def _silence_amici_logging(level: int = logging.WARNING) -> None:
    """Raise every ``amici.*`` logger to at least ``level``.

    AMICI pins an intermediate logger under ``amici.sim.*`` to DEBUG and
    dumps a PEtab parameter-mapping summary on *every* simulation; with a
    permissive root handler that floods bulk generation with millions of
    lines. Raising only the top-level ``amici`` logger doesn't help (the
    leaf loggers inherit the intermediate's DEBUG level), so walk every
    amici.* logger that currently exists. Never *lowers* a logger — pypesto
    pins ``amici`` itself to CRITICAL, and WARNING+ (incl. real AMICI
    failures) must keep surfacing.
    """
    for name in list(logging.Logger.manager.loggerDict):
        if name == "amici" or name.startswith("amici."):
            logger = logging.getLogger(name)
            if logger.getEffectiveLevel() < level:
                logger.setLevel(level)


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
            path=Path(__file__).parent.absolute(),
        )
        # AMICI failures emit full NaN rows; rejection-resample at HF export time.
        self.hf_resample_invalid = True
        # Cap HF generation at 100k train (expensive simulator); consumers
        # subsample smaller budgets by indexing the dataset prefix.
        self.hf_split_sizes = {
            "train": 100_000, "validation": 10_000, "test": 10_000,
        }
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
            # AMICI's loggers exist only after the model is loaded; silence
            # them now so the per-simulation PEtab-mapping DEBUG dump doesn't
            # flood bulk dataset generation with millions of lines.
            _silence_amici_logging()
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
