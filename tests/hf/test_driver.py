"""Smoke tests for scripts/make_dataset.py (no real HF calls)."""

import importlib.util
import json
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


def _load_driver():
    spec = importlib.util.spec_from_file_location("make_dataset_mod", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_real_upload_merges_remote_and_deletes_local(monkeypatch, tmp_path):
    mod = _load_driver()
    out = tmp_path / "metadata.json"

    # Remote already documents a *different* task; merge must preserve it.
    monkeypatch.setattr(
        mod, "fetch_remote_metadata", lambda repo: {"two_moons": {"x": 1}})

    captured = {}

    def fake_upload_metadata(path, repo):
        captured["content"] = json.loads(open(path).read())
        captured["repo"] = repo

    monkeypatch.setattr(mod, "upload_metadata", fake_upload_metadata)
    monkeypatch.setattr(mod, "upload_dataset", lambda repo, name, **o: None)

    mod.main(["--tasks", "gaussian_linear", "--metadata-path", str(out)])

    # merged: remote task preserved + selected task added
    assert "two_moons" in captured["content"]
    assert "gaussian_linear" in captured["content"]
    # default target is the test repo
    assert captured["repo"] == "aurelio-amerio/SBI-benchmarks-test"
    # local artifact deleted -> clean state
    assert not out.exists()


def test_partial_size_override_respects_task_cap(monkeypatch, tmp_path):
    """A partial CLI override must not clobber a task's hf_split_sizes cap.

    gaussian_random_field caps train at 100k. Passing only --val-size must
    leave train/test at the task cap in BOTH the recorded metadata and the
    actual upload — never the global 1M default. Metadata and the generated
    dataset must agree on every split size.
    """
    mod = _load_driver()
    out = tmp_path / "metadata.json"

    monkeypatch.setattr(mod, "fetch_remote_metadata", lambda repo: {})

    captured = {}

    def fake_upload_metadata(path, repo):
        captured["meta"] = json.loads(open(path).read())

    upload_opts = {}

    def fake_upload_dataset(repo, name, **opts):
        upload_opts[name] = opts

    monkeypatch.setattr(mod, "upload_metadata", fake_upload_metadata)
    monkeypatch.setattr(mod, "upload_dataset", fake_upload_dataset)

    mod.main(["--tasks", "gaussian_random_field",
              "--val-size", "5000", "--metadata-path", str(out)])

    # Metadata: the unspecified train/test keep the 100k/10k task cap; only
    # validation reflects the explicit override.
    splits = captured["meta"]["gaussian_random_field"]["splits"]
    assert splits == {"train": 100_000, "validation": 5000, "test": 10_000}

    # Upload: no train_size/test_size override is forwarded, so the exporter
    # resolves them from hf_split_sizes -> matches the metadata above.
    opts = upload_opts["gaussian_random_field"]
    assert "train_size" not in opts
    assert "test_size" not in opts
    assert opts["val_size"] == 5000


def test_dry_run_keeps_local_and_skips_network(monkeypatch, tmp_path):
    mod = _load_driver()
    out = tmp_path / "metadata.json"

    def boom(repo):
        raise AssertionError("network must not be touched on --dry-run")

    monkeypatch.setattr(mod, "fetch_remote_metadata", boom)
    mod.main(["--tasks", "gaussian_linear", "--metadata-path", str(out),
              "--dry-run"])
    assert out.exists()  # kept for inspection


def test_chunk_size_forwarded_to_upload(monkeypatch, tmp_path):
    mod = _load_driver()
    out = tmp_path / "metadata.json"

    monkeypatch.setattr(mod, "fetch_remote_metadata", lambda repo: {})
    monkeypatch.setattr(mod, "upload_metadata", lambda path, repo: None)

    upload_opts = {}

    def fake_upload_dataset(repo, name, **opts):
        upload_opts[name] = opts

    monkeypatch.setattr(mod, "upload_dataset", fake_upload_dataset)

    mod.main(["--tasks", "gaussian_linear", "--chunk-size", "256",
              "--metadata-path", str(out)])

    assert upload_opts["gaussian_linear"]["chunk_size"] == 256


def test_chunk_size_absent_by_default(monkeypatch, tmp_path):
    mod = _load_driver()
    out = tmp_path / "metadata.json"

    monkeypatch.setattr(mod, "fetch_remote_metadata", lambda repo: {})
    monkeypatch.setattr(mod, "upload_metadata", lambda path, repo: None)

    upload_opts = {}

    def fake_upload_dataset(repo, name, **opts):
        upload_opts[name] = opts

    monkeypatch.setattr(mod, "upload_dataset", fake_upload_dataset)

    mod.main(["--tasks", "gaussian_linear", "--metadata-path", str(out)])

    assert "chunk_size" not in upload_opts["gaussian_linear"]
