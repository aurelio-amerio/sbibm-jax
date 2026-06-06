"""Configuration defaults for the sbibm_jax.hf pipeline."""

import numpy as np

DEFAULT_REPO: str = "aurelio-amerio/SBI-benchmarks"

DEFAULT_SPLIT_SIZES: dict = {
    "train": 1_000_000,
    "validation": 10_000,
    "test": 10_000,
}

DEFAULT_DTYPE = np.float32

DEFAULT_CHUNK_SIZE: int = 4096

DEFAULT_MAX_FACTOR: float = 10.0

DEFAULT_MASTER_SEED: int = 0
