"""HuggingFace upload helpers — isolated so tests can monkeypatch.

`upload_file`, `hf_hub_download`, and `push_to_hub` are the network surface.
`upload_file` and `hf_hub_download` are imported at module scope so
monkeypatching them on `sbibm_jax.hf.upload` (and the shadowing of
`build_dataset` here) is sufficient for any test.
"""

import json

from huggingface_hub import hf_hub_download, upload_file
from huggingface_hub.utils import EntryNotFoundError

from sbibm_jax.hf.build import build_dataset


def upload_metadata(file_path: str, repo_name: str) -> None:
    """Push a metadata.json file to the dataset repo."""
    upload_file(
        path_or_fileobj=file_path,
        path_in_repo="metadata.json",
        repo_id=repo_name,
        repo_type="dataset",
    )


def fetch_remote_metadata(repo_name: str) -> dict:
    """Return the repo's existing metadata.json as a dict, or {} if absent.

    Downloads with force_download to avoid a stale local cache. Only a missing
    metadata.json in an existing repo (EntryNotFoundError) yields {} — a fresh
    start. Every other failure propagates: a non-existent repo, an auth error,
    or a transient HTTP/connection error must never be silently treated as an
    empty remote, which would drop sibling task entries on merge. (Note: under
    force_download=True, hf_hub_download wraps a missing-repo error as a
    generic error rather than RepositoryNotFoundError, so it propagates too.)
    """
    try:
        local_path = hf_hub_download(
            repo_id=repo_name,
            filename="metadata.json",
            repo_type="dataset",
            force_download=True,
        )
    except EntryNotFoundError:
        return {}
    with open(local_path) as f:
        return json.load(f)


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
