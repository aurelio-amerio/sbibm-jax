"""Tests for hf.reference.load_reference."""

import numpy as np
import pytest
from datasets import Dataset

from sbibm_jax import get_task
from sbibm_jax.hf.reference import load_reference
from sbibm_jax.hf.registry import get_exporter


class TestLoadReference:
    def test_two_moons_present(self):
        task = get_task("two_moons")
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        ref = load_reference(task, exp)
        assert isinstance(ref, Dataset)
        assert len(ref) == task.num_observations  # 10
        cols = set(ref.column_names)
        assert cols == {"reference_samples", "observations", "true_parameters"}

    def test_two_moons_shapes(self):
        task = get_task("two_moons")
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        ref = load_reference(task, exp)
        # observations: each row is (1, dim_data) -> flat list of dim_data floats.
        row = ref[0]
        assert len(row["observations"]) == task.dim_data
        assert len(row["true_parameters"]) == task.dim_parameters
        # reference_samples is a (num_ref_posterior_samples, dim_parameters) block
        # -> list-of-lists. Each inner list has length dim_parameters.
        rs = row["reference_samples"]
        assert len(rs) == task.num_reference_posterior_samples
        assert len(rs[0]) == task.dim_parameters

    def test_grf_absent_returns_none(self):
        task = get_task("gaussian_random_field", field_size=8)
        exp = get_exporter(task, train_size=4, val_size=2, test_size=2)
        assert load_reference(task, exp) is None
