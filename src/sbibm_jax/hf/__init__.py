"""HuggingFace dataset pipeline for sbibm_jax.

Requires the optional `[hf]` extra (`datasets`, `huggingface_hub`). Importing
this subpackage without the extra raises an informative ImportError that points
at `pip install sbibm-jax[hf]`, mirroring the existing `pypesto` extra pattern.

Public API (re-exported below): build_dataset, upload_dataset, make_metadata,
get_exporter.
"""

try:
    import datasets  # noqa: F401
    import huggingface_hub  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The sbibm_jax.hf subpackage requires the optional `[hf]` extra. "
        "Install it with `uv sync --extra hf` or `pip install sbibm-jax[hf]`."
    ) from e

from sbibm_jax.hf import config  # noqa: E402
from sbibm_jax.hf.build import build_dataset  # noqa: E402
from sbibm_jax.hf.metadata import make_metadata  # noqa: E402
from sbibm_jax.hf.registry import get_exporter  # noqa: E402
from sbibm_jax.hf.upload import upload_dataset, upload_metadata  # noqa: E402

__all__ = [
    "build_dataset",
    "config",
    "get_exporter",
    "make_metadata",
    "upload_dataset",
    "upload_metadata",
]
