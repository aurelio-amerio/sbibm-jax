"""Configuration defaults for the sbibm_jax.hf pipeline."""

import os

import numpy as np

# Production dataset repo. This is what the consumer loaders read by default.
# NOTE: it is NOT the upload default — make_dataset.py targets TEST_REPO
# unless --prod is passed.
DEFAULT_REPO: str = "aurelio-amerio/SBI-benchmarks"

# Default target for make_dataset.py (safe; use --prod to hit DEFAULT_REPO).
TEST_REPO: str = "aurelio-amerio/SBI-benchmarks-test"

# Set this env var to a truthy value ("1", "true", "yes", "on") to make the
# loaders read from TEST_REPO instead of DEFAULT_REPO.
USE_TEST_ENV_VAR: str = "SBIBM_JAX_USE_TEST"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"", "0", "false", "no", "off"})


def use_test_repo() -> bool:
    """Whether the test repo is selected via the environment."""
    raw = os.environ.get(USE_TEST_ENV_VAR, "")
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(
        f"{USE_TEST_ENV_VAR}={raw!r} is not a valid boolean; use one of "
        f"{sorted(_TRUTHY)} or {sorted(_FALSY - {''})}."
    )


def get_default_repo() -> str:
    """Repo the loaders default to: production unless the env flag is set.

    Read at call time (not import time) so tests and notebooks can flip
    ``SBIBM_JAX_USE_TEST`` without reimporting the package.
    """
    return TEST_REPO if use_test_repo() else DEFAULT_REPO


DEFAULT_SPLIT_SIZES: dict = {
    "train": 1_000_000,
    "validation": 10_000,
    "test": 10_000,
}

DEFAULT_DTYPE = np.float32

DEFAULT_CHUNK_SIZE: int = 4096

DEFAULT_MAX_FACTOR: float = 10.0

DEFAULT_MASTER_SEED: int = 42
