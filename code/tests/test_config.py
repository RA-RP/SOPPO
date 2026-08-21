from pathlib import Path

import pytest

from src.config import apply_overrides, load_config, validate_config


ROOT = Path(__file__).parents[1]


def server_paths(config):
    return apply_overrides(
        config,
        [
            "model.name_or_path=/server/models/Qwen3-4B",
            "model.manifest_path=/server/models/Qwen3-4B/model_manifest.json",
            "data.data_dir=/server/data/ultrafeedback/mvp-v0.5-30k",
            "output.run_dir=/server/runs/test",
        ],
    )


def test_joint_config_freezes_30k_lora_and_8_56_batch_contract():
    config = server_paths(load_config(ROOT / "configs" / "mvp" / "soppo_pe_exp.yaml"))
    validate_config(config, world_size=2)
    assert config["training"]["global_batch_size"] == 64
    assert config["training"]["joint_labeled_global_batch_size"] == 8
    assert config["training"]["joint_unlabeled_global_batch_size"] == 56
    assert sum(config["training"]["joint_unlabeled_microbatch_pattern"]) == 28
    assert config["model"]["lora"]["r"] == 8
    assert config["model"]["lora"]["alpha"] == 16
    assert config["data"]["total_samples"] == 30000


def test_dpo_requires_reference_and_global_batch_64():
    config = server_paths(load_config(ROOT / "configs" / "mvp" / "dpo10.yaml"))
    with pytest.raises(ValueError, match="reference_cache"):
        validate_config(config, world_size=2)
    config = apply_overrides(config, ["data.reference_cache=/server/cache/reference"])
    validate_config(config, world_size=2)


def test_joint_batch_contract_fails_closed():
    config = server_paths(load_config(ROOT / "configs" / "mvp" / "soppo_pe_exp.yaml"))
    config = apply_overrides(config, ["training.joint_unlabeled_global_batch_size=54"])
    with pytest.raises(ValueError, match="joint_unlabeled"):
        validate_config(config, world_size=2)


def test_static_lambda_is_preregistered():
    config = server_paths(load_config(ROOT / "configs" / "mvp" / "soppo_pe_static.yaml"))
    config = apply_overrides(config, ["method.fixed_lambda=0.2"])
    with pytest.raises(ValueError, match="lambda"):
        validate_config(config, world_size=2)


def test_static_lambda_requires_normalized_weighting():
    config = server_paths(load_config(ROOT / "configs" / "mvp" / "soppo_pe_static.yaml"))
    config = apply_overrides(config, ["method.weighting=exponential_gamma"])
    with pytest.raises(ValueError, match="normalized_fixed_lambda"):
        validate_config(config, world_size=2)
