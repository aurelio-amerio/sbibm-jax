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
    assert config.DEFAULT_MASTER_SEED == 42


def test_test_repo_constant():
    from sbibm_jax.hf import config

    assert config.TEST_REPO == "aurelio-amerio/SBI-benchmarks-test"
    # production constant is unchanged
    assert config.DEFAULT_REPO == "aurelio-amerio/SBI-benchmarks"


def test_default_repo_helper_prefers_production(monkeypatch):
    from sbibm_jax.hf import config

    monkeypatch.delenv(config.USE_TEST_ENV_VAR, raising=False)
    assert config.use_test_repo() is False
    assert config.get_default_repo() == config.DEFAULT_REPO


def test_default_repo_helper_honors_env_flag(monkeypatch):
    from sbibm_jax.hf import config

    for truthy in ("1", "true", "TRUE", " yes ", "on"):
        monkeypatch.setenv(config.USE_TEST_ENV_VAR, truthy)
        assert config.use_test_repo() is True
        assert config.get_default_repo() == config.TEST_REPO

    for falsy in ("", "0", "false", "no", "off"):
        monkeypatch.setenv(config.USE_TEST_ENV_VAR, falsy)
        assert config.use_test_repo() is False
        assert config.get_default_repo() == config.DEFAULT_REPO


def test_default_repo_helper_rejects_garbage(monkeypatch):
    import pytest

    from sbibm_jax.hf import config

    monkeypatch.setenv(config.USE_TEST_ENV_VAR, "maybe")
    with pytest.raises(ValueError, match=config.USE_TEST_ENV_VAR):
        config.get_default_repo()


def test_new_helpers_reexported():
    import sbibm_jax.hf as hf

    assert hasattr(hf, "merge_metadata")
    assert hasattr(hf, "fetch_remote_metadata")
    assert "merge_metadata" in hf.__all__
    assert "fetch_remote_metadata" in hf.__all__
