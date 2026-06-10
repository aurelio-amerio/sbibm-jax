# tests/data/test_process.py
"""Collate: tokenization, joint concat, normalization, dtype, joint guard."""

import numpy as np
import pytest

from sbibm_jax.data.process import make_collate


def _batch():
    return {
        "thetas": np.arange(2 * 3, dtype=np.float32).reshape(2, 3),  # (B=2, 3)
        "xs": np.ones((2, 5), dtype=np.float32),                     # (B=2, 5) vector
    }


class TestConditional:
    def test_tokenizes_to_trailing_channel(self):
        collate = make_collate(kind="conditional", x_kind="vector")
        theta, x = collate(_batch())
        assert theta.shape == (2, 3, 1)
        assert x.shape == (2, 5, 1)

    def test_normalize_applies_stats(self):
        stats = {"theta_mean": [[1.0, 1.0, 1.0]], "theta_std": [[1.0, 1.0, 1.0]],
                 "x_mean": [[1.0, 1.0, 1.0, 1.0, 1.0]],
                 "x_std": [[2.0, 2.0, 2.0, 2.0, 2.0]]}
        collate = make_collate(kind="conditional", x_kind="vector",
                               normalize=True, stats=stats)
        theta, x = collate(_batch())
        # x all ones, mean 1, std 2 -> 0
        np.testing.assert_allclose(np.asarray(x), 0.0, atol=1e-6)


class TestJoint:
    def test_joint_concats_along_feature_axis(self):
        collate = make_collate(kind="joint", x_kind="vector")
        out = collate(_batch())
        assert out.shape == (2, 3 + 5, 1)

    def test_joint_raises_for_image_x(self):
        with pytest.raises(ValueError, match="joint.*vector"):
            make_collate(kind="joint", x_kind="image")


class TestTimeSeriesConditional:
    def _ts_batch(self):
        xs = np.empty((2, 4, 2), dtype=np.float32)
        xs[..., 0] = 1.0
        xs[..., 1] = 3.0
        return {"thetas": np.ones((2, 2), dtype=np.float32), "xs": xs}

    def test_tokenizes_with_channel(self):
        collate = make_collate(kind="conditional", x_kind="timeseries")
        theta, x = collate(self._ts_batch())
        assert theta.shape == (2, 2, 1)
        assert x.shape == (2, 4, 2, 1)

    def test_normalize_per_channel(self):
        stats = {"theta_mean": [[1.0, 1.0]], "theta_std": [[1.0, 1.0]],
                 "x_mean": [[[1.0, 3.0]]], "x_std": [[[1.0, 2.0]]]}
        collate = make_collate(kind="conditional", x_kind="timeseries",
                               normalize=True, stats=stats)
        _, x = collate(self._ts_batch())
        # channel 0: (1-1)/1=0 ; channel 1: (3-3)/2=0
        np.testing.assert_allclose(np.asarray(x), 0.0, atol=1e-6)

    def test_joint_raises_for_timeseries(self):
        with pytest.raises(ValueError, match="joint.*vector"):
            make_collate(kind="joint", x_kind="timeseries")

    def test_joint_raises_for_non_vector_theta(self):
        with pytest.raises(ValueError, match="joint.*vector"):
            make_collate(kind="joint", x_kind="vector", theta_kind="image")
