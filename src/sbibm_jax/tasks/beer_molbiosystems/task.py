"""Beer (MolBioSystems2014) PEtab benchmark task.

Wraps the pypesto/AMICI Beer_MolBioSystems2014 benchmark model. All heavy
dependencies live behind the optional `pypesto` extra and are imported lazily,
so the task constructs (for registry discovery) without them. Install with:

    uv sync --extra pypesto      # or: pip install sbibm-jax[pypesto]

Installing the extra triggers a one-time AMICI compile of the Beer model.
"""

from pathlib import Path
from typing import List, Optional  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from sbibm_jax.tasks.simulator import Simulator  # noqa: F401
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
