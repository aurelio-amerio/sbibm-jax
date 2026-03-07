"""I/O utilities for reading and writing arrays from/to CSV files."""

from pathlib import Path
from typing import Iterable, Optional, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd


def get_array_from_csv(
    path: Union[str, Path],
    dtype: type = np.float32,
    atleast_2d: bool = True,
) -> jnp.ndarray:
    """Load a JAX array from a CSV file.

    Args:
        path: Path to CSV file.
        dtype: Numpy dtype for loading.
        atleast_2d: If True, ensure result is at least 2D.

    Returns:
        JAX array loaded from the CSV.
    """
    data = pd.read_csv(path).values.astype(dtype)
    if atleast_2d:
        data = np.atleast_2d(data)
    return jnp.array(data)


def save_array_to_csv(
    path: Union[str, Path],
    data: jnp.ndarray,
    columns: Optional[Iterable[str]] = None,
    dtype: type = np.float32,
    index: bool = False,
) -> None:
    """Save a JAX array to a CSV file.

    Args:
        path: Path to save CSV.
        data: JAX array to save.
        columns: Column names for the CSV.
        dtype: Numpy dtype for saving.
        index: Whether to write row index.
    """
    pd.DataFrame(
        np.asarray(data).astype(dtype),
        columns=columns,
    ).to_csv(path, index=index)
