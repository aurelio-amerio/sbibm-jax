"""Smoke tests for scripts/make_dataset.py (no real HF calls)."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "scripts" / "make_dataset.py"


def test_driver_help_runs():
    result = subprocess.run(
        [sys.executable, str(DRIVER), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "usage" in result.stdout.lower()


def test_dry_run_writes_metadata(tmp_path):
    out = tmp_path / "metadata.json"
    result = subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--tasks",
            "gaussian_linear",
            "--metadata-path",
            str(out),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.exists()
    assert "gaussian_linear" in out.read_text()


def test_dry_run_banner_defaults_to_test(tmp_path):
    out = tmp_path / "metadata.json"
    result = subprocess.run(
        [sys.executable, str(DRIVER), "--tasks", "gaussian_linear",
         "--metadata-path", str(out), "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "(TEST)" in result.stdout
    assert "aurelio-amerio/SBI-benchmarks-test" in result.stdout


def test_dry_run_banner_prod(tmp_path):
    out = tmp_path / "metadata.json"
    result = subprocess.run(
        [sys.executable, str(DRIVER), "--tasks", "gaussian_linear",
         "--metadata-path", str(out), "--prod", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "(PRODUCTION)" in result.stdout
    # the production repo id, without the -test suffix, must appear
    assert "aurelio-amerio/SBI-benchmarks\n" in result.stdout \
        or "aurelio-amerio/SBI-benchmarks " in result.stdout
