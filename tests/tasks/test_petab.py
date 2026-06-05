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


class TestRegistry:
    def test_get_task_returns_instance(self):
        from sbibm_jax import get_task

        task = get_task("beer_molbiosystems")
        assert isinstance(task, BeerMolBioSystems)

    def test_available_tasks_includes_beer(self):
        from sbibm_jax import get_available_tasks

        assert "beer_molbiosystems" in get_available_tasks()
