"""Dataset exporter base + data-kind subclasses.

The exporter owns the HF Features schema and the shape transformation from the
flat (batch, dim_data) block emitted by sbibm_jax.hf.generate to the
native-storage shape (vector / image / time-series). Concrete subclasses
override three things: `data_kind`, `x_feature()`, and `shape_x(x_flat)`.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from datasets import Array2D, Features, List, Value

from sbibm_jax.hf import config
from sbibm_jax.tasks.task import Task


class DatasetExporter(ABC):
    """Abstract base. Subclasses set `data_kind` and override `x_feature`/`shape_x`."""

    data_kind: str = ""

    def __init__(
        self,
        task: Task,
        *,
        train_size: int,
        val_size: int,
        test_size: int,
        dtype=config.DEFAULT_DTYPE,
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        max_factor: float = config.DEFAULT_MAX_FACTOR,
        data_shape: Optional[Tuple[int, ...]] = None,
        resample_invalid: bool = False,
    ):
        if type(self) is DatasetExporter:
            raise TypeError(
                "DatasetExporter is abstract; instantiate VectorExporter / "
                "ImageExporter / TimeSeriesExporter."
            )
        self.task = task
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.max_factor = max_factor
        self.data_shape = data_shape if data_shape is not None else (task.dim_data,)
        self.resample_invalid = resample_invalid

    @abstractmethod
    def x_feature(self):
        """The HF feature for a single `x` row."""

    @abstractmethod
    def shape_x(self, x_flat: np.ndarray) -> np.ndarray:
        """Reshape a flat `(batch, dim_data)` block to native storage shape."""

    def theta_feature(self):
        return List(Value("float32"))

    def features(self) -> Features:
        return Features({"xs": self.x_feature(), "thetas": self.theta_feature()})


class VectorExporter(DatasetExporter):
    """Flat parameter-vector storage. The default for analytical tasks."""

    data_kind = "vector"

    def x_feature(self):
        return List(Value("float32"))

    def shape_x(self, x_flat: np.ndarray) -> np.ndarray:
        return np.asarray(x_flat, dtype=self.dtype)


class ImageExporter(DatasetExporter):
    """`Array2D`-backed image storage (e.g. Gaussian Random Field 32x32)."""

    data_kind = "image"

    def __init__(self, task: Task, *, data_shape: Tuple[int, int], **kwargs):
        if data_shape is None or len(data_shape) != 2:
            raise ValueError(
                f"ImageExporter requires a 2-D data_shape (H, W); got {data_shape!r}."
            )
        super().__init__(task, data_shape=data_shape, **kwargs)

    def x_feature(self):
        return Array2D(shape=self.data_shape, dtype="float32")

    def shape_x(self, x_flat: np.ndarray) -> np.ndarray:
        h, w = self.data_shape
        return np.asarray(x_flat, dtype=self.dtype).reshape(-1, h, w)


class TimeSeriesExporter(DatasetExporter):
    """`Array2D`-backed time-series storage (T, C)."""

    data_kind = "timeseries"

    def __init__(self, task: Task, *, data_shape: Tuple[int, int], **kwargs):
        if data_shape is None or len(data_shape) != 2:
            raise ValueError(
                f"TimeSeriesExporter requires a 2-D data_shape (T, C); got {data_shape!r}."
            )
        super().__init__(task, data_shape=data_shape, **kwargs)

    def x_feature(self):
        return Array2D(shape=self.data_shape, dtype="float32")

    def shape_x(self, x_flat: np.ndarray) -> np.ndarray:
        t, c = self.data_shape
        return np.asarray(x_flat, dtype=self.dtype).reshape(-1, t, c)
