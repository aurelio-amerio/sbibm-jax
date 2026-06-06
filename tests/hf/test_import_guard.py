"""Smoke tests for the sbibm_jax.hf package boundary."""


def test_package_imports():
    import sbibm_jax.hf

    assert hasattr(sbibm_jax.hf, "__all__")


def test_config_defaults():
    from sbibm_jax.hf import config

    assert config.DEFAULT_REPO == "aurelio-amerio/SBI-benchmarks"
    assert config.DEFAULT_SPLIT_SIZES == {
        "train": 1_000_000,
        "validation": 10_000,
        "test": 10_000,
    }
    assert config.DEFAULT_CHUNK_SIZE == 4096
    assert config.DEFAULT_MAX_FACTOR == 10.0
    assert config.DEFAULT_MASTER_SEED == 0


def test_test_repo_constant():
    from sbibm_jax.hf import config

    assert config.TEST_REPO == "aurelio-amerio/SBI-benchmarks-test"
    # production constant is unchanged
    assert config.DEFAULT_REPO == "aurelio-amerio/SBI-benchmarks"
