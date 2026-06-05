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
