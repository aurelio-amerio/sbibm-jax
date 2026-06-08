# tests/hf/test_stats.py
"""Tests for hf.stats: streaming reduction matches numpy; axis resolution."""

import numpy as np
import pytest

from sbibm_jax.hf.stats import StatsAccumulator, resolve_stats_axes


class TestStatsAccumulator:
    def test_per_feature_matches_numpy(self):
        rng = np.random.default_rng(0)
        theta = rng.normal(size=(1000, 3)).astype(np.float32)
        x = rng.normal(size=(1000, 5)).astype(np.float32)
        acc = StatsAccumulator(theta_axes=(0,), x_axes=(0,))
        for i in range(0, 1000, 256):                       # stream in chunks
            acc.update(theta[i:i + 256], x[i:i + 256])
        res = acc.result()
        np.testing.assert_allclose(
            np.array(res["theta_mean"]), theta.mean(0, keepdims=True),
            rtol=1e-5, atol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(res["x_std"]), x.std(0, keepdims=True),
            rtol=1e-5, atol=1e-4,
        )
        assert np.array(res["theta_mean"]).shape == (1, 3)
        assert np.array(res["x_mean"]).shape == (1, 5)

    def test_global_scalar_image_reduction(self):
        rng = np.random.default_rng(1)
        theta = rng.normal(size=(64, 2)).astype(np.float32)
        x = rng.normal(size=(64, 8, 8)).astype(np.float32)   # native image
        acc = StatsAccumulator(theta_axes=(0,), x_axes=(0, 1, 2))
        acc.update(theta, x)
        res = acc.result()
        assert np.array(res["x_mean"]).shape == (1, 1, 1)
        np.testing.assert_allclose(
            np.array(res["x_mean"]).item(), x.mean(), rtol=1e-5, atol=1e-5,
        )


class TestResolveStatsAxes:
    def test_default_is_reduce_batch_only(self):
        class T:  # no hf_stats_axes
            pass
        theta_axes, x_axes = resolve_stats_axes(T())
        assert theta_axes == (0,)
        assert x_axes == (0,)

    def test_task_override_wins(self):
        class T:
            hf_stats_axes = {"theta": (0,), "x": (0, 1, 2)}
        theta_axes, x_axes = resolve_stats_axes(T())
        assert x_axes == (0, 1, 2)
