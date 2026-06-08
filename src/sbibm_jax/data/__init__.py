# src/sbibm_jax/data/__init__.py
"""Consumer-side loading of the SBI-benchmarks HuggingFace datasets.

Requires the optional `[loader]` extra (`grain`, `datasets`, `huggingface_hub`).
Importing without it raises an informative ImportError, mirroring the `[hf]`
pattern. The loader serves theta/x pairs via grain, exposes task dims and
gen-time normalization stats from metadata.json, and (opt-in) graph masks via
`sbibm_jax.data.masks`.
"""

try:
    import grain  # noqa: F401
    import datasets  # noqa: F401
    import huggingface_hub  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The sbibm_jax.data subpackage requires the optional `[loader]` extra. "
        "Install it with `uv sync --extra loader` or `pip install sbibm-jax[loader]`."
    ) from e

from sbibm_jax.data.dataset import TaskDataset  # noqa: E402

__all__ = ["TaskDataset"]
