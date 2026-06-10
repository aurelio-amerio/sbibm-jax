"""Tests for the file-backed Gravitational Waves task."""

import jax
import pytest

from sbibm_jax.tasks.gravitational_waves.task import GravitationalWaves


class TestMetadata:
    def test_dims_and_name(self):
        task = GravitationalWaves()
        assert task.dim_theta == 2
        assert task.dim_x == 8192 * 2
        assert task.name == "gravitational_waves"
        assert task.name_display == "Gravitational Waves"

    def test_hf_hints(self):
        task = GravitationalWaves()
        assert task.hf_x_kind == "timeseries"
        assert task.hf_x_shape == (8192, 2)
        assert task.hf_stats_axes == {"theta": (0,), "x": (0, 1)}
        assert task.hf_external is True


class TestMocksRaise:
    def test_get_prior_raises(self):
        task = GravitationalWaves()
        with pytest.raises(NotImplementedError):
            task.get_prior(jax.random.PRNGKey(0), num_samples=1)

    def test_get_simulator_raises(self):
        task = GravitationalWaves()
        with pytest.raises(NotImplementedError):
            task.get_simulator(jax.random.PRNGKey(0))

    def test_reference_posterior_raises(self):
        task = GravitationalWaves()
        with pytest.raises(NotImplementedError):
            task._sample_reference_posterior(
                jax.random.PRNGKey(0), num_samples=10, num_observation=1,
            )


class TestRegistry:
    def test_get_task_returns_instance(self):
        from sbibm_jax import get_task
        task = get_task("gravitational_waves")
        assert isinstance(task, GravitationalWaves)

    def test_available_tasks_includes_gw(self):
        from sbibm_jax import get_available_tasks
        assert "gravitational_waves" in get_available_tasks()
