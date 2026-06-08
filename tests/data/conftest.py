"""Skip the whole data test subdir if grain/datasets aren't importable."""
import pytest
pytest.importorskip("grain", reason="The [loader] extra is not installed.")
pytest.importorskip("datasets", reason="The [loader] extra is not installed.")
