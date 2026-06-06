"""Tests for hf.upload — monkeypatched, no real HF calls."""

import json
import pytest

from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

import sbibm_jax.hf.upload as upload_mod
from sbibm_jax.hf.upload import (
    fetch_remote_metadata,
    upload_dataset,
    upload_metadata,
)


class _FakeDataset:
    def __init__(self, name):
        self.name = name
        self.push_calls = []

    def push_to_hub(self, repo_name, **kwargs):
        self.push_calls.append((repo_name, kwargs))


class TestUploadMetadata:
    def test_calls_upload_file(self, monkeypatch, tmp_path):
        path = tmp_path / "metadata.json"
        path.write_text("{}")

        calls = []

        def fake_upload_file(**kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(upload_mod, "upload_file", fake_upload_file)
        upload_metadata(str(path), "user/repo")

        assert len(calls) == 1
        kw = calls[0]
        assert kw["path_or_fileobj"] == str(path)
        assert kw["path_in_repo"] == "metadata.json"
        assert kw["repo_id"] == "user/repo"
        assert kw["repo_type"] == "dataset"


class TestUploadDataset:
    def test_pushes_each_split_with_right_config(self, monkeypatch):
        train = _FakeDataset("train")
        val = _FakeDataset("val")
        test = _FakeDataset("test")
        ref = _FakeDataset("ref")

        def fake_build(task_name, **opts):
            return {
                "train": train,
                "validation": val,
                "test": test,
                "reference": ref,
            }

        monkeypatch.setattr(upload_mod, "build_dataset", fake_build)
        upload_dataset("user/repo", "two_moons")

        assert train.push_calls == [
            ("user/repo", {
                "config_name": "two_moons",
                "split": "train",
                "private": False,
            }),
        ]
        assert val.push_calls == [
            ("user/repo", {
                "config_name": "two_moons",
                "split": "validation",
                "private": False,
            }),
        ]
        assert test.push_calls == [
            ("user/repo", {
                "config_name": "two_moons",
                "split": "test",
                "private": False,
            }),
        ]
        assert ref.push_calls == [
            ("user/repo", {
                "config_name": "two_moons_posterior",
                "split": "reference_posterior",
                "private": False,
            }),
        ]

    def test_skips_reference_when_absent(self, monkeypatch):
        train = _FakeDataset("train")
        val = _FakeDataset("val")
        test = _FakeDataset("test")

        def fake_build(task_name, **opts):
            return {
                "train": train,
                "validation": val,
                "test": test,
                "reference": None,
            }

        monkeypatch.setattr(upload_mod, "build_dataset", fake_build)
        upload_dataset("user/repo", "gaussian_random_field")

        assert train.push_calls[0][1]["config_name"] == "gaussian_random_field"
        assert val.push_calls[0][1]["split"] == "validation"
        # No assertion needed for "ref" - it does not exist (would have raised).


class TestFetchRemoteMetadata:
    def test_returns_parsed_dict(self, monkeypatch, tmp_path):
        f = tmp_path / "metadata.json"
        f.write_text(json.dumps({"two_moons": {"dim": 2}}))
        monkeypatch.setattr(
            upload_mod, "hf_hub_download", lambda **kw: str(f))
        assert fetch_remote_metadata("user/repo") == {"two_moons": {"dim": 2}}

    def test_entry_not_found_returns_empty(self, monkeypatch):
        def boom(**kw):
            raise EntryNotFoundError.__new__(EntryNotFoundError)
        monkeypatch.setattr(upload_mod, "hf_hub_download", boom)
        assert fetch_remote_metadata("user/repo") == {}

    def test_repo_not_found_returns_empty(self, monkeypatch):
        def boom(**kw):
            raise RepositoryNotFoundError.__new__(RepositoryNotFoundError)
        monkeypatch.setattr(upload_mod, "hf_hub_download", boom)
        assert fetch_remote_metadata("user/repo") == {}

    def test_other_error_propagates(self, monkeypatch):
        def boom(**kw):
            raise ValueError("transient network error")
        monkeypatch.setattr(upload_mod, "hf_hub_download", boom)
        with pytest.raises(ValueError):
            fetch_remote_metadata("user/repo")
