"""Tests for DatasetExporter and its data-kind subclasses."""

import numpy as np
import pytest
from datasets import Array2D, Features, List, Value

from sbibm_jax import get_task
from sbibm_jax.hf.exporter import (
    DatasetExporter,
    ImageExporter,
    TimeSeriesExporter,
    VectorExporter,
)


class TestVectorExporter:
    def test_data_kind(self):
        task = get_task("gaussian_linear")
        exp = VectorExporter(task, train_size=4, val_size=2, test_size=2)
        assert exp.data_kind == "vector"

    def test_features_schema(self):
        task = get_task("gaussian_linear")
        exp = VectorExporter(task, train_size=4, val_size=2, test_size=2)
        feats = exp.features()
        assert isinstance(feats, Features)
        assert isinstance(feats["xs"], List)
        assert isinstance(feats["xs"].feature, Value)
        assert feats["xs"].feature.dtype == "float32"
        assert isinstance(feats["thetas"], List)
        assert feats["thetas"].feature.dtype == "float32"

    def test_shape_x_identity(self):
        task = get_task("gaussian_linear")
        exp = VectorExporter(task, train_size=4, val_size=2, test_size=2)
        flat = np.zeros((3, task.dim_data), dtype=np.float32)
        out = exp.shape_x(flat)
        assert out.shape == (3, task.dim_data)
        assert out.dtype == np.float32

    def test_base_class_is_abstract(self):
        task = get_task("gaussian_linear")
        with pytest.raises(TypeError):
            DatasetExporter(task, train_size=1, val_size=1, test_size=1)


class TestImageExporter:
    def test_data_kind(self):
        task = get_task("gaussian_random_field", field_size=8)
        exp = ImageExporter(
            task, data_shape=(8, 8), train_size=4, val_size=2, test_size=2,
        )
        assert exp.data_kind == "image"

    def test_features_schema(self):
        task = get_task("gaussian_random_field", field_size=8)
        exp = ImageExporter(
            task, data_shape=(8, 8), train_size=4, val_size=2, test_size=2,
        )
        feats = exp.features()
        assert isinstance(feats["xs"], Array2D)
        assert feats["xs"].shape == (8, 8)
        assert feats["xs"].dtype == "float32"

    def test_shape_x_reshapes_to_image(self):
        task = get_task("gaussian_random_field", field_size=8)
        exp = ImageExporter(
            task, data_shape=(8, 8), train_size=4, val_size=2, test_size=2,
        )
        flat = np.zeros((5, 8 * 8), dtype=np.float32)
        out = exp.shape_x(flat)
        assert out.shape == (5, 8, 8)
        assert out.dtype == np.float32

    def test_rejects_non_2d_shape(self):
        task = get_task("gaussian_random_field", field_size=8)
        with pytest.raises(ValueError, match="2-D data_shape"):
            ImageExporter(
                task,
                data_shape=(8, 8, 3),
                train_size=4,
                val_size=2,
                test_size=2,
            )


class TestTimeSeriesExporter:
    def test_data_kind(self):
        task = get_task("gaussian_linear")  # any task; data_shape is what counts
        exp = TimeSeriesExporter(
            task, data_shape=(5, 2), train_size=4, val_size=2, test_size=2,
        )
        assert exp.data_kind == "timeseries"

    def test_features_schema(self):
        task = get_task("gaussian_linear")
        exp = TimeSeriesExporter(
            task, data_shape=(5, 2), train_size=4, val_size=2, test_size=2,
        )
        feats = exp.features()
        assert isinstance(feats["xs"], Array2D)
        assert feats["xs"].shape == (5, 2)
        assert feats["xs"].dtype == "float32"

    def test_shape_x_reshapes_to_tc(self):
        task = get_task("gaussian_linear")
        exp = TimeSeriesExporter(
            task, data_shape=(5, 2), train_size=4, val_size=2, test_size=2,
        )
        flat = np.zeros((7, 5 * 2), dtype=np.float32)
        out = exp.shape_x(flat)
        assert out.shape == (7, 5, 2)
        assert out.dtype == np.float32

    def test_rejects_non_2d_shape(self):
        task = get_task("gaussian_linear")
        with pytest.raises(ValueError, match="2-D data_shape"):
            TimeSeriesExporter(
                task,
                data_shape=(5,),
                train_size=4,
                val_size=2,
                test_size=2,
            )
