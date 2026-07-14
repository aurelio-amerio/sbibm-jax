"""Tests for hf.registry.get_exporter."""

import pytest

from sbibm_jax import get_task
from sbibm_jax.hf import config
from sbibm_jax.hf.exporter import HealpixExporter, ImageExporter, VectorExporter
from sbibm_jax.hf.registry import X_KIND_REGISTRY, get_exporter


class TestRegistry:
    def test_known_kinds(self):
        assert set(X_KIND_REGISTRY) == {
            "vector", "image", "timeseries", "healpix",
        }

    def test_default_is_vector(self):
        task = get_task("gaussian_linear")
        exp = get_exporter(task)
        assert isinstance(exp, VectorExporter)
        assert exp.train_size == config.DEFAULT_SPLIT_SIZES["train"]
        assert exp.val_size == config.DEFAULT_SPLIT_SIZES["validation"]
        assert exp.test_size == config.DEFAULT_SPLIT_SIZES["test"]

    def test_split_size_overrides(self):
        task = get_task("gaussian_linear")
        exp = get_exporter(task, train_size=10, val_size=2, test_size=2)
        assert exp.train_size == 10
        assert exp.val_size == 2
        assert exp.test_size == 2

    def test_hf_x_kind_hint_selects_image(self):
        task = get_task("gaussian_linear")
        task.hf_x_kind = "image"
        task.hf_x_shape = (4, 4)
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert isinstance(exp, ImageExporter)
        assert exp.x_shape == (4, 4)

    def test_hf_theta_hints_propagate(self):
        task = get_task("gaussian_linear")
        task.hf_theta_kind = "timeseries"
        task.hf_theta_shape = (3, 4)
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert exp.theta_kind == "timeseries"
        assert exp.theta_shape == (3, 4)

    def test_theta_hints_default_to_vector(self):
        task = get_task("gaussian_linear")
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert exp.theta_kind == "vector"
        assert exp.theta_shape == (task.dim_theta,)

    def test_resample_invalid_hint_propagates(self):
        task = get_task("gaussian_linear")
        task.hf_resample_invalid = True
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert exp.resample_invalid is True

    def test_unknown_x_kind_raises(self):
        task = get_task("gaussian_linear")
        task.hf_x_kind = "tensor4d"
        with pytest.raises(ValueError, match="Unknown x_kind"):
            get_exporter(task)


class TestRegistryRealTasks:
    def test_grf_selects_image_exporter(self):
        task = get_task("gaussian_random_field", field_size=8)
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert isinstance(exp, ImageExporter)
        assert exp.x_shape == (8, 8)

    def test_grf_default_field_size_32(self):
        task = get_task("gaussian_random_field")
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert isinstance(exp, ImageExporter)
        assert exp.x_shape == (32, 32)

    def test_toy_lensing_selects_image_exporter(self):
        task = get_task("toy_lensing", resolution=8)
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert isinstance(exp, ImageExporter)
        assert exp.x_shape == (8, 8)

    def test_toy_lensing_default_resolution_32(self):
        task = get_task("toy_lensing")
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert isinstance(exp, ImageExporter)
        assert exp.x_shape == (32, 32)

    def test_grf_256_selects_image_exporter(self):
        task = get_task("gaussian_random_field_256")
        exp = get_exporter(task)
        assert isinstance(exp, ImageExporter)
        assert exp.x_shape == (256, 256)
        assert exp.train_size == 100_000
        assert exp.val_size == 10_000
        assert exp.test_size == 10_000

    def test_spherical_grf_dispatches_healpix(self):
        exp = get_exporter(get_task("spherical_grf"))
        assert isinstance(exp, HealpixExporter)
        assert exp.x_shape == (49152,)
        assert exp.train_size == 100_000


class TestResampleHints:
    @pytest.mark.parametrize(
        "name",
        ["lotka_volterra", "sir", "beer_molbiosystems"],
    )
    def test_resample_invalid_set(self, name):
        try:
            task = get_task(name)
        except ImportError as e:
            pytest.skip(f"task {name} requires an extra: {e}")
        assert getattr(task, "hf_resample_invalid", False) is True
