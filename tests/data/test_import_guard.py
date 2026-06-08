"""The data subpackage exposes TaskDataset when the extra is present."""

def test_taskdataset_importable():
    from sbibm_jax.data import TaskDataset
    assert TaskDataset is not None
