# src/sbibm_jax/data/dataset.py
"""TaskDataset: load an SBI-benchmarks task from the Hub and serve theta/x.

Driven entirely by the published metadata.json (dims, data_kind/shape, splits,
stats) — no per-task code. Default repo is the TEST repo (config.TEST_REPO);
pass repo=config.DEFAULT_REPO for production.
"""

import json

import grain
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from sbibm_jax.hf import config
from sbibm_jax.data.process import make_collate, _stat_array

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

    def _loader(self, split, batch_size, num_samples=None):
        if num_samples is not None:
            if num_samples > split.num_rows:
                raise ValueError(
                    f"num_samples={num_samples} exceeds split size {split.num_rows}."
                )
            split = split.select(range(int(num_samples)))
        pipe = (
            grain.MapDataset.source(split)
            .shuffle(self.seed)
            .repeat()
            .to_iter_dataset()
            .batch(batch_size)
            .map(self._collate)
        )
        if self.use_prefetching and self.max_workers:
            cfg = grain.experimental.pick_performance_config(
                ds=pipe, ram_budget_mb=1024, max_workers=self.max_workers,
                max_buffer_size=None,
            )
            pipe = pipe.mp_prefetch(cfg.multiprocessing_options)
        return pipe

    def get_train_loader(self, batch_size, num_samples=None):
        return self._loader(self.df_train, batch_size, num_samples)

    def get_val_loader(self, batch_size):
        return self._loader(self.df_val, batch_size)

    def get_test_loader(self, batch_size):
        return self._loader(self.df_test, batch_size)

    def _ensure_posterior(self):
        if not self.has_reference:
            raise ValueError(
                f"Task {self.name!r} has no reference posterior "
                f"({self.name}_posterior config absent)."
            )
        if self._posterior is None:
            self._posterior = load_dataset(
                self.repo, f"{self.name}_posterior",
            ).with_format("numpy")["reference_posterior"]
        return self._posterior

    def _check_obs(self, num_observation):
        if not 1 <= num_observation <= self.num_observations:
            raise ValueError(
                f"num_observation must be in [1, {self.num_observations}]."
            )

    def get_reference(self, num_observation=1):
        self._check_obs(num_observation)
        post = self._ensure_posterior()
        i = num_observation - 1
        return post["observations"][i], post["reference_samples"][i]

    def get_true_parameters(self, num_observation=1):
        self._check_obs(num_observation)
        return self._ensure_posterior()["true_parameters"][num_observation - 1]

    def _norm(self, arr, mean, std):
        m = _stat_array(mean, self.dtype)
        s = _stat_array(std, self.dtype)
        return (np.asarray(arr, dtype=self.dtype) - m) / s

    def _unnorm(self, arr, mean, std):
        m = _stat_array(mean, self.dtype)
        s = _stat_array(std, self.dtype)
        return np.asarray(arr, dtype=self.dtype) * s + m

    def normalize_theta(self, theta):
        return self._norm(theta, self.theta_mean, self.theta_std)

    def unnormalize_theta(self, theta):
        return self._unnorm(theta, self.theta_mean, self.theta_std)

    def normalize_x(self, x):
        return self._norm(x, self.x_mean, self.x_std)

    def unnormalize_x(self, x):
        return self._unnorm(x, self.x_mean, self.x_std)
