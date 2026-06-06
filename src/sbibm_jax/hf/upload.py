"""HuggingFace upload helpers — isolated so tests can monkeypatch.

`upload_file` and `push_to_hub` are the only network surface. Both are imported
at module scope so monkeypatching `sbibm_jax.hf.upload.upload_file` (and the
shadowing of `build_dataset` here) is sufficient for any test.
"""

from huggingface_hub import upload_file

from sbibm_jax.hf.build import build_dataset


def upload_metadata(file_path: str, repo_name: str) -> None:
    """Push a metadata.json file to the dataset repo."""
    upload_file(
        path_or_fileobj=file_path,
        path_in_repo="metadata.json",
        repo_id=repo_name,
        repo_type="dataset",
    )


def upload_dataset(repo_name: str, task_name: str, **build_opts) -> None:
    """Build the dataset for `task_name` and push each split to `repo_name`.

    The dataset is pushed under config_name=task_name with splits train /
    validation / test. If the task ships a reference block, it is pushed under
    config_name=f"{task_name}_posterior" with split "reference_posterior".
    """
    bundle = build_dataset(task_name, **build_opts)
    bundle["train"].push_to_hub(
        repo_name, config_name=task_name, split="train", private=False,
    )
    bundle["validation"].push_to_hub(
        repo_name, config_name=task_name, split="validation", private=False,
    )
    bundle["test"].push_to_hub(
        repo_name, config_name=task_name, split="test", private=False,
    )
    if bundle.get("reference") is not None:
        bundle["reference"].push_to_hub(
            repo_name,
            config_name=f"{task_name}_posterior",
            split="reference_posterior",
            private=False,
        )
