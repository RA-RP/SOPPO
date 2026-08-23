import copy
import json
import sys
from pathlib import Path

import pytest

from src.round2.config import load_round2_config, validate_round2_config
from src.round2.queue_protocol import (
    REQUEST_SCHEMA_VERSION,
    RESPONSE_SCHEMA_VERSION,
    atomic_write_json,
    validate_request,
    validate_response,
)
from src.round2.run_rollout import _build_pairs
from src.round2.sft_schema import SFT_SCHEMA_VERSION, validate_sft_corpus
from src.round2.tp_backend import build_tp_command, launch_spec_from_config


ROOT = Path(__file__).parents[1]


def round2_config(method="soppo_pe_sft_rollout_exp"):
    path = ROOT / "configs" / "round2" / f"{method}.yaml"
    source = "sft_rollout" if method.endswith("sft_rollout_exp") else "rollout_only"
    config = load_round2_config(
        path,
        [
            f"provenance.git_commit={'a' * 40}",
            "model.name_or_path=/server/models/Qwen3-4B",
            "model.manifest_path=/server/models/Qwen3-4B/model_manifest.json",
            "data.data_dir=/server/data/round2",
            f"output.run_dir=/server/runs/{method}",
            f"rollout.artifact_dir=/server/runs/{method}/rollouts",
            "rollout.sft_data_file=/server/data/round2/sft.jsonl",
            "rollout.temperature=0.7",
            "rollout.top_p=0.9",
            f"rollout.source={source}",
        ],
    )
    validate_round2_config(config)
    return config


def test_round2_config_requires_tp2_lora_and_separate_rollout_gpu():
    config = round2_config()
    invalid = copy.deepcopy(config)
    invalid["tensor_parallel"]["tensor_model_parallel_size"] = 1
    with pytest.raises(ValueError, match="tensor_model_parallel_size=2"):
        validate_round2_config(invalid)

    invalid = copy.deepcopy(config)
    invalid["rollout"]["gpu_ids"] = "1"
    with pytest.raises(ValueError, match="disjoint"):
        validate_round2_config(invalid)

    invalid = copy.deepcopy(config)
    invalid["training"]["save_steps"] = 40
    with pytest.raises(ValueError, match="save_steps=1"):
        validate_round2_config(invalid)


def test_round2_strong_smoke_contract_is_not_a_small_batch():
    config = round2_config()
    config["training"].update(
        {
            "smoke_mode": True,
            "max_steps": 1,
            "eval_max_samples": 8,
            "smoke_objective_step": 1,
        }
    )
    config["rollout"]["min_new_tokens"] = 512
    validate_round2_config(config)
    assert config["training"]["joint_labeled_global_batch_size"] == 8
    assert config["training"]["joint_unlabeled_global_batch_size"] == 56
    assert config["model"]["max_seq_len"] == 2048


def _request(checkpoint: Path, method: str):
    generation = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 512,
        "min_new_tokens": 0,
        "max_model_len": 2048,
    }
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "step": 3,
        "method": method,
        "policy_checkpoint": str(checkpoint),
        "generation": generation,
        "samples": [
            {
                "sample_id": f"sample-{index:03d}",
                "prompt": f"prompt {index}",
            }
            for index in range(56)
        ],
    }
    if method == "soppo_pe_sft_rollout_exp":
        for index, sample in enumerate(request["samples"]):
            sample["sft_response"] = f"sft {index}"
    return request


@pytest.mark.parametrize(
    "method,candidate_count,expected_sources",
    [
        ("soppo_pe_sft_rollout_exp", 1, {"sft", "rollout"}),
        ("soppo_pe_rollout_only_exp", 2, {"rollout_0", "rollout_1"}),
    ],
)
def test_round2_online_pair_contract(
    tmp_path, method, candidate_count, expected_sources
):
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    request = _request(checkpoint, method)
    validate_request(request)
    if method == "soppo_pe_rollout_only_exp":
        assert all("sft_response" not in sample for sample in request["samples"])
    generated = [
        [f"rollout-{candidate}-{index}" for candidate in range(candidate_count)]
        for index in range(56)
    ]
    pairs = _build_pairs(request, generated, seed=42)
    assert len(pairs) == 56
    assert all(
        {pair["response_a_source"], pair["response_b_source"]} == expected_sources
        for pair in pairs
    )
    assert all("label" not in pair for pair in pairs)
    response = {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "step": request["step"],
        "method": method,
        "policy_checkpoint": str(checkpoint),
        "generation": request["generation"],
        "generation_seconds": 1.0,
        "statistics": {
            "generated_sequences": 56 * candidate_count,
        },
        "pairs": pairs,
    }
    validate_response(response, request)
    if method == "soppo_pe_sft_rollout_exp":
        tampered = copy.deepcopy(response)
        first = tampered["pairs"][0]
        side = (
            "response_a"
            if first["response_a_source"] == "sft"
            else "response_b"
        )
        first[side] = "different SFT response"
        with pytest.raises(ValueError, match="SFT response changed"):
            validate_response(tampered, request)


def test_round2_sft_corpus_matches_every_unlabeled_prompt(tmp_path):
    unlabeled = tmp_path / "unlabeled.jsonl"
    sft = tmp_path / "sft.jsonl"
    rows = [
        {"sample_id": "one", "prompt": "p1"},
        {"sample_id": "two", "prompt": "p2"},
    ]
    unlabeled.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sft.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": SFT_SCHEMA_VERSION,
                    "sample_id": row["sample_id"],
                    "prompt": row["prompt"],
                    "response": f"response-{row['sample_id']}",
                }
            )
            + "\n"
            for row in rows
        )
    )
    summary = validate_sft_corpus(sft, unlabeled, expected_rows=2)
    assert summary["rows"] == 2
    assert summary["matches_unlabeled_split"] is True

    leaked = json.loads(sft.read_text().splitlines()[0])
    leaked["metadata"] = "not-preregistered"
    sft.write_text(json.dumps(leaked) + "\n")
    with pytest.raises(ValueError, match="unregistered fields"):
        validate_sft_corpus(sft, unlabeled, expected_rows=2)


def test_round2_queue_refuses_overwrite(tmp_path):
    path = tmp_path / "request.json"
    atomic_write_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        atomic_write_json(path, {"value": 2})


def test_round2_tp_command_uses_two_torchrun_processes():
    config = round2_config()
    spec = launch_spec_from_config(config, "/server/config.yaml", sys.executable)
    command = build_tp_command(spec)
    assert spec.nproc_per_node == 2
    assert "--nproc_per_node=2" in command
    assert "src.round2.tp_trainer" in command
