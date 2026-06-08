# src/sbibm_jax/data/dataset.py
"""TaskDataset: load an SBI-benchmarks task from the Hub and serve theta/x.

Driven entirely by the published metadata.json (dims, data_kind/shape, splits,
stats) — no per-task code. Default repo is the TEST repo (config.TEST_REPO);
pass repo=config.DEFAULT_REPO for production.
"""

import json

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from sbibm_jax.hf import config
from sbibm_jax.data.process import make_collate

_MAX_WORKERS_CAP = 8  # shared node; never exceed (see CLAUDE.md / memory).


class TaskDataset:
    def __init__(
        self,
        name,
        *,
        kind="conditional",
        repo=None,
        normalize=False,
        dtype=np.float32,
        seed=42,
        use_prefetching=True,
        max_workers=None,
    ):
        self.name = name
        self.kind = kind
        self.repo = repo if repo is not None else config.TEST_REPO
        self.normalize = normalize
        self.dtype = dtype
        self.seed = seed
        self.use_prefetching = use_prefetching
        self.max_workers = (
            None if max_workers is None else min(int(max_workers), _MAX_WORKERS_CAP)
        )

        meta_path = hf_hub_download(
            repo_id=self.repo, filename="metadata.json", repo_type="dataset",
        )
        with open(meta_path) as f:
            entry = json.load(f)[name]

        self.dim_theta = int(entry["dim_theta"])
        self.dim_x = int(entry["dim_x"])
        self.data_kind = entry["data_kind"]
        self.data_shape = tuple(entry["data_shape"])
        self.num_observations = int(entry["num_observations"])
        self.has_reference = bool(entry["has_reference"])
        self.dim_joint = self.dim_theta + self.dim_x if kind == "joint" else None

        stats = entry.get("stats")
        if stats is not None:
            self.theta_mean = stats["theta_mean"]
            self.theta_std = stats["theta_std"]
            self.x_mean = stats["x_mean"]
            self.x_std = stats["x_std"]
        else:
            self.theta_mean = self.theta_std = self.x_mean = self.x_std = None

        self._collate = make_collate(
            kind=kind, data_kind=self.data_kind,
            normalize=normalize, stats=stats, dtype=dtype,
        )

        self.dataset = load_dataset(self.repo, name).with_format("numpy")
        self.df_train = self.dataset["train"]
        self.df_val = self.dataset["validation"]
        self.df_test = self.dataset["test"]
        self.max_samples = self.df_train.num_rows
        self._posterior = None  # lazily loaded in get_reference
