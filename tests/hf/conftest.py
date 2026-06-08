"""Skip the whole hf test subdir if `datasets` isn't importable."""

import pytest

pytest.importorskip(
    "datasets",
    reason="The [hf] extra is not installed (uv sync --extra hf).",
)
