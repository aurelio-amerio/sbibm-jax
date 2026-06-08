"""Tests for analytical tasks (Phase 1)."""

import jax
import jax.numpy as jnp
import pandas as pd
import pytest

from sbibm_jax import get_task


# -- Parametrised fixtures for all Phase 1 tasks --

ANALYTICAL_TASKS = [
    ("gaussian_linear", 10, 10),
    ("gaussian_linear_uniform", 10, 10),
    ("gaussian_mixture", 2, 2),
    ("two_moons", 2, 2),
]


@pytest.fixture(params=ANALYTICAL_TASKS, ids=[t[0] for t in ANALYTICAL_TASKS])
def task_info(request):
    """Yields (task_instance, dim_params, dim_x) for each analytical task."""
    name, dim_params, dim_x = request.param
    task = get_task(name)
    return task, dim_params, dim_x


class TestPrior:
    def test_shape(self, task_info):
        task, dim_params, _ = task_info
        key = jax.random.PRNGKey(42)
        samples = task.get_prior(key, num_samples=50)
        assert samples.shape == (50, dim_params)

    def test_single_sample(self, task_info):
        task, dim_params, _ = task_info
        key = jax.random.PRNGKey(0)
        sample = task.get_prior(key, num_samples=1)
        assert sample.shape == (1, dim_params)

    def test_different_keys_give_different_samples(self, task_info):
        task, _, _ = task_info
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        s1 = task.get_prior(k1, num_samples=5)
        s2 = task.get_prior(k2, num_samples=5)
        assert not jnp.allclose(s1, s2)


class TestSimulator:
    def test_shape(self, task_info):
        task, dim_params, dim_x = task_info
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
        samples = task.get_prior(k1, num_samples=20)
        sim = task.get_simulator(k2)
        data = sim(k3, samples)
        assert data.shape == (20, dim_x)

    def test_single_sample(self, task_info):
        task, _, dim_x = task_info
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(1), 3)
        sample = task.get_prior(k1, num_samples=1)
        sim = task.get_simulator(k2)
        data = sim(k3, sample)
        assert data.shape == (1, dim_x)


class TestDataLoading:
    def test_observation_shape(self, task_info):
        task, _, dim_x = task_info
        obs = task.get_observation(1)
        assert obs.shape == (1, dim_x)

    def test_true_parameters_shape(self, task_info):
        task, dim_params, _ = task_info
        tp = task.get_true_parameters(1)
        assert tp.shape == (1, dim_params)

    def test_reference_posterior_shape(self, task_info):
        task, dim_params, _ = task_info
        ref = task.get_reference_posterior_samples(1)
        assert ref.shape[1] == dim_params
        assert ref.shape[0] > 0

    def test_all_observations_loadable(self, task_info):
        task, _, dim_x = task_info
        for i in range(1, min(task.num_observations + 1, 4)):  # test first 3
            obs = task.get_observation(i)
            assert obs.shape == (1, dim_x)

    def test_observation_matches_raw_csv(self, task_info):
        """Cross-validate observation loading against raw pandas read."""
        task, _, _ = task_info
        path = (
            task.path / "files" / "num_observation_1" / "observation.csv"
        )
        raw = pd.read_csv(path).values
        obs = task.get_observation(1)
        assert jnp.allclose(obs, jnp.array(raw, dtype=jnp.float32), atol=1e-6)


class TestTaskRegistry:
    def test_get_available_tasks(self):
        from sbibm_jax.tasks import get_available_tasks
        tasks = get_available_tasks()
        assert "gaussian_linear" in tasks
        assert "two_moons" in tasks

    def test_get_task_returns_correct_type(self):
        task = get_task("gaussian_linear")
        assert task.name == "gaussian_linear"
        assert task.dim_theta == 10

    def test_unknown_task_raises(self):
        with pytest.raises(NotImplementedError):
            get_task("nonexistent_task")
