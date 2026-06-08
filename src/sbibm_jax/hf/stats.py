# src/sbibm_jax/hf/stats.py
"""Streaming normalization statistics over native-shaped (theta, x).

Stats are accumulated in float64 (sum + sum-of-squares) so the mean/std are
exact over ~1e6 rows without holding them in RAM. Reduction axes refer to the
native BATCH-INCLUSIVE shape (axis 0 = batch); output keeps reduced dims so the
result broadcasts against a single native row. The published shapes are e.g.
(1, dim_theta) for per-feature theta, (1, 1, 1) for a global-scalar image x,
(1, 1, C) for a per-channel time-series x.
"""

from typing import Tuple

import numpy as np


class _Reducer:
    def __init__(self, axes: Tuple[int, ...]):
        self.axes = tuple(axes)
        self._sum = None
        self._sumsq = None
        self._count = 0

    def update(self, arr: np.ndarray) -> None:
        a = np.asarray(arr, dtype=np.float64)
        s = a.sum(axis=self.axes, keepdims=True)
        ss = (a * a).sum(axis=self.axes, keepdims=True)
        n = 1
        for ax in self.axes:
            n *= a.shape[ax]
        if self._sum is None:
            self._sum, self._sumsq = s, ss
        else:
            self._sum += s
            self._sumsq += ss
        self._count += n

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        mean = self._sum / self._count
        var = np.maximum(self._sumsq / self._count - mean * mean, 0.0)
        std = np.sqrt(var)
        return mean.astype(np.float32), std.astype(np.float32)


class StatsAccumulator:
    """Accumulate mean/std for theta and x over their reduction axes."""

    def __init__(self, theta_axes, x_axes):
        self.theta_axes = tuple(theta_axes)
        self.x_axes = tuple(x_axes)
        self._t = _Reducer(self.theta_axes)
        self._x = _Reducer(self.x_axes)

    def update(self, theta_native: np.ndarray, x_native: np.ndarray) -> None:
        self._t.update(theta_native)
        self._x.update(x_native)

    def result(self) -> dict:
        tm, ts = self._t.finalize()
        xm, xs = self._x.finalize()
        return {
            "theta_mean": tm.tolist(),
            "theta_std": ts.tolist(),
            "x_mean": xm.tolist(),
            "x_std": xs.tolist(),
            "theta_axes": list(self.theta_axes),
            "x_axes": list(self.x_axes),
        }


def resolve_stats_axes(task) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return (theta_axes, x_axes) from task.hf_stats_axes, default reduce-batch.

    Default reduces only the batch axis (per-feature stats). Tasks whose x is an
    image / time-series set hf_stats_axes to avoid per-pixel stats, e.g.
    {"theta": (0,), "x": (0, 1, 2)} for a global-scalar image.
    """
    spec = getattr(task, "hf_stats_axes", None)
    if spec is None:
        return (0,), (0,)
    return tuple(spec["theta"]), tuple(spec["x"])
