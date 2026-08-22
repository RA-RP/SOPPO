from pathlib import Path

import pytest

from src.config import (
    apply_distributed_training_profile,
    apply_overrides,
    distributed_training_profile,
    load_config,
    validate_config,
)


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


@pytest.mark.parametrize(
    ("devices", "accumulation", "local_unlabeled"),
    [(1, 16, 56), (2, 8, 28), (4, 4, 14)],
)
def test_joint_config_freezes_30k_lora_and_global_8_56_contract(
    devices, accumulation, local_unlabeled
):
    config = server_paths(load_config(ROOT / "configs" / "mvp" / "soppo_pe_exp.yaml"))
    config = apply_distributed_training_profile(config, devices)
    validate_config(config, world_size=devices)
    assert config["training"]["global_batch_size"] == 64
    assert config["training"]["gradient_accumulation_steps"] == accumulation
    assert config["training"]["joint_labeled_global_batch_size"] == 8
    assert config["training"]["joint_unlabeled_global_batch_size"] == 56
    assert sum(config["training"]["joint_unlabeled_microbatch_pattern"]) == local_unlabeled
    assert len(config["training"]["joint_labeled_microsteps"]) * devices == 8
    assert config["training"]["backward_subbatch_size_per_device"] == 2
    assert config["model"]["lora"]["r"] == 8
    assert config["model"]["lora"]["alpha"] == 16
    assert config["data"]["total_samples"] == 30000


@pytest.mark.parametrize(("devices", "accumulation"), [(1, 16), (2, 8), (4, 4)])
def test_dpo_requires_reference_and_global_batch_64(devices, accumulation):
    config = server_paths(load_config(ROOT / "configs" / "mvp" / "dpo10.yaml"))
    config = apply_distributed_training_profile(config, devices)
    with pytest.raises(ValueError, match="reference_cache"):
        validate_config(config, world_size=devices)
    config = apply_overrides(config, ["data.reference_cache=/server/cache/reference"])
    validate_config(config, world_size=devices)
    assert config["training"]["dpo_batch_size_per_device"] == 4
    assert config["training"]["gradient_accumulation_steps"] == accumulation
    assert config["training"]["backward_subbatch_size_per_device"] == 2


def test_distributed_profile_rejects_unsupported_device_count():
    with pytest.raises(ValueError, match="1, 2, or 4"):
        distributed_training_profile(3)


def test_config_rejects_profile_world_size_mismatch():
    config = server_paths(load_config(ROOT / "configs" / "mvp" / "dpo10.yaml"))
    config = apply_overrides(config, ["data.reference_cache=/server/cache/reference"])
    with pytest.raises(ValueError, match="does not match"):
        validate_config(config, world_size=1)


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
