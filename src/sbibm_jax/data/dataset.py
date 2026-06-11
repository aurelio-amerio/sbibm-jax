# src/sbibm_jax/data/dataset.py
"""TaskDataset: load an SBI-benchmarks task from the Hub and serve theta/x.

Driven entirely by the published metadata.json (x_kind/x_shape,
theta_kind/theta_shape, splits, stats) — no per-task code. Default repo is the
TEST repo (config.TEST_REPO); pass repo=config.DEFAULT_REPO for production.
"""

import json

import grain
import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from sbibm_jax.hf import config
from sbibm_jax.data.process import make_collate, make_collate_jax, _stat_array
from sbibm_jax.tasks import get_task

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

        self._init_metadata(self._load_metadata_entry())
        self._init_splits()

    def _load_metadata_entry(self):
        """Download metadata.json from the Hub and return this task's entry."""
        meta_path = hf_hub_download(
            repo_id=self.repo, filename="metadata.json", repo_type="dataset",
        )
        with open(meta_path) as f:
            return json.load(f)[self.name]

    def _init_metadata(self, entry):
        """Shapes, kinds, dims, reference info, stats, and the collate fn.

        Shared by TaskDataset and OnlineTaskDataset.
        """
        self.x_kind = entry["x_kind"]
        self.x_shape = tuple(entry["x_shape"])
        self.theta_kind = entry["theta_kind"]
        self.theta_shape = tuple(entry["theta_shape"])
        self.dim_x = int(np.prod(self.x_shape))
        self.dim_theta = int(np.prod(self.theta_shape))
        self.num_observations = int(entry["num_observations"])
        self.has_reference = bool(entry["has_reference"])
        self.dim_joint = (
            self.dim_theta + self.dim_x if self.kind == "joint" else None
        )

        stats = entry.get("stats")
        self._stats = stats
        if stats is not None:
            self.theta_mean = stats["theta_mean"]
            self.theta_std = stats["theta_std"]
            self.x_mean = stats["x_mean"]
            self.x_std = stats["x_std"]
        else:
            self.theta_mean = self.theta_std = self.x_mean = self.x_std = None

        self._collate = make_collate(
            kind=self.kind, x_kind=self.x_kind, theta_kind=self.theta_kind,
            normalize=self.normalize, stats=stats, dtype=self.dtype,
        )
        self._posterior = None  # lazily loaded in get_reference

    def _init_splits(self):
        """Offline-only: download the HF splits."""
        self.dataset = load_dataset(self.repo, self.name).with_format("numpy")
        self.df_train = self.dataset["train"]
        self.df_val = self.dataset["validation"]
        self.df_test = self.dataset["test"]
        self.max_samples = self.df_train.num_rows

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


def _worker_init(worker_index, worker_count):
    """grain mp_prefetch worker init: force JAX onto CPU in the worker.

    Must be jax.config.update, not os.environ["JAX_PLATFORMS"]: jax captures
    the env var at import time, and jax is already imported here (cloudpickle
    loads this function by reference, importing this module first). The
    update is effective because grain runs worker_init_fn before unpickling
    the dataset — i.e. before anything touches a JAX backend. Spawn start
    method is guaranteed by grain itself.
    """
    del worker_index, worker_count
    jax.config.update("jax_platforms", "cpu")


class _SimIterDataset(grain.IterDataset):
    """Infinite source IterDataset drawing (theta, x) from prior + simulator.

    Implements grain's SupportsInPlaceSlicing protocol: under mp_prefetch,
    grain calls set_slice(slice(worker_index, None, num_workers)) on each
    worker's copy — required for a parentless source IterDataset, and it
    doubles as the per-worker stream id (folded into the PRNG key so workers
    produce independent streams).

    With x_shape set (the published metadata x_shape), the simulator's flat
    rows are reshaped to (n, *x_shape) before crossing the pickle boundary,
    so raw batches are layout-identical to offline HF rows (a no-op for
    vector tasks, native for image/timeseries). None keeps flat output.
    """

    def __init__(self, task, simulator, seed, batch_size, x_shape=None):
        super().__init__()
        self._task = task
        self._simulator = simulator
        self._seed = int(seed)  # plain int; keys built lazily in the iterator
        self._batch_size = int(batch_size)
        self._x_shape = None if x_shape is None else tuple(x_shape)
        self._worker_index = 0
        self._worker_count = 1

    def set_slice(self, sl, sequential_slice=False):
        del sequential_slice
        self._worker_index = sl.start or 0
        self._worker_count = sl.step or 1

    def __iter__(self):
        return _SimIterator(self)


class _SimIterator(grain.DatasetIterator):
    """Iterator holding the running PRNG key as checkpointable state."""

    def __init__(self, parent):
        super().__init__()
        self._p = parent
        base = jax.random.PRNGKey(parent._seed)
        self._key = jax.random.fold_in(base, parent._worker_index)

    def __next__(self):
        self._key, sub = jax.random.split(self._key)
        kt, ks = jax.random.split(sub)
        theta = self._p._task.get_prior(kt, self._p._batch_size)
        x = self._p._simulator(ks, theta)
        xs = np.asarray(x)
        if self._p._x_shape is not None:
            # Native layout matching the offline HF rows (metadata x_shape);
            # a free view — same bytes across the pickle boundary.
            xs = xs.reshape(-1, *self._p._x_shape)
        # Raw numpy across the pickle boundary; tokenization happens in the
        # main process (make_collate_jax).
        return {"thetas": np.asarray(theta), "xs": xs}

    # Abstract on DatasetIterator and genuinely called by grain's worker
    # loop (checkpoint/seek) — real implementations, not stubs. PRNGKey is a
    # raw uint32 (2,) array, so the key IS the state.
    def get_state(self):
        return {"key": np.asarray(self._key).tolist()}

    def set_state(self, state):
        self._key = jnp.asarray(state["key"], dtype=jnp.uint32)


class OnlineTaskDataset(TaskDataset):
    """Simulate-on-the-fly variant of TaskDataset.

    Serves fresh (theta, x) batches from the task's prior + simulator instead
    of the pre-generated HF splits; metadata-driven shapes, stats, and
    tokenization are identical to TaskDataset (same metadata.json). The HF
    splits are never downloaded; get_reference/get_true_parameters still work
    (separate {name}_posterior config).

    Assumes the simulator always yields finite rows: tasks whose simulators
    legitimately diverge (hf_resample_invalid=True, i.e. ODE/PEtab) are not
    intended for online use, and hf_external tasks without a simulator
    (gravitational_waves) fail at construction. Vector x/theta only for now:
    simulators emit flat rows and the online path has no flat->native reshape.

    Simulator.num_simulations is only meaningful with num_workers=0: under
    mp_prefetch each worker counts on its own pickled copy.
    """

    def __init__(
        self,
        name,
        *,
        kind="conditional",
        repo=None,
        normalize=False,
        dtype=jnp.float32,
        seed=42,
    ):
        self.name = name
        self.kind = kind
        self.repo = repo if repo is not None else config.TEST_REPO
        self.normalize = normalize
        self.dtype = dtype
        self.seed = seed

        self._init_metadata(self._load_metadata_entry())
        # Replace the numpy collate set by _init_metadata: the online path
        # collates in the main process, after the pickle boundary, so jnp is
        # safe (and saves a host round-trip before the training step).
        self._collate = make_collate_jax(
            kind=kind, x_kind=self.x_kind, theta_kind=self.theta_kind,
            normalize=normalize, stats=self._stats, dtype=dtype,
        )

        self.task = get_task(name)
        # Eager build: tasks without a simulator raise NotImplementedError
        # here, at construction, instead of on the first next().
        self.simulator = self.task.get_simulator(
            jax.random.PRNGKey(self.seed), max_calls=None,
        )
        # The Simulator wrapper emits flat rows (task.flatten_data); the
        # offline path reshapes flat -> native at HF-generation time, but the
        # online path has no reshape step yet, so non-vector tokenization
        # (image/timeseries) would be silently wrong. Vector-only for now.
        if self.x_kind != "vector" or self.theta_kind != "vector":
            raise NotImplementedError(
                f"OnlineTaskDataset is vector-only for now (simulators emit "
                f"flat rows); task {name!r} has x_kind={self.x_kind!r}, "
                f"theta_kind={self.theta_kind!r}."
            )

    def _offline_error(self):
        return NotImplementedError(
            "OnlineTaskDataset generates batches on the fly; use "
            "get_online_train_loader."
        )

    def get_train_loader(self, batch_size, num_samples=None):
        raise self._offline_error()

    def get_val_loader(self, batch_size):
        raise self._offline_error()

    def get_test_loader(self, batch_size):
        raise self._offline_error()

    def get_online_train_loader(self, batch_size, *, seed=None, num_workers=0):
        """Infinite loader of freshly simulated, tokenized jnp batches.

        Reproducible for a fixed (seed, num_workers); changing num_workers
        changes the stream (grain stateful-transform caveat) but stays
        deterministic. num_workers=0 simulates in-process (on the default JAX
        device); num_workers>=1 simulates in CPU spawn workers, leaving the
        GPU to the training step. Pass a distinct seed for independent
        concurrent loaders.
        """
        seed = self.seed if seed is None else seed
        num_workers = min(int(num_workers), _MAX_WORKERS_CAP)
        ds = _SimIterDataset(self.task, self.simulator, seed, batch_size)
        if num_workers > 0:
            ds = ds.mp_prefetch(
                grain.MultiprocessingOptions(num_workers=num_workers),
                worker_init_fn=_worker_init,
            )
        return ds.map(self._collate)
