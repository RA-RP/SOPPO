import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

from src.round2.config import load_round2_config, validate_round2_config
from src.round2.queue_protocol import (
    REQUEST_SCHEMA_VERSION,
    RESPONSE_SCHEMA_VERSION,
    atomic_write_json,
    validate_request,
    validate_response,
)
from src.round2.prepare_sft_anchor import SELECTION_RULE, prepare_sft_anchor
from src.round2.run_rollout import _build_pairs
from src.round2.sft_schema import SFT_SCHEMA_VERSION, validate_sft_corpus
from src.round2.tp_backend import (
    _build_peft_tp_hook_compatibility,
    _expected_local_shape,
    _verify_local_tp_shapes,
    build_tp_command,
    launch_spec_from_config,
)
from src.round2.tp_trainer import _local_tp_squared_norms


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

    invalid = copy.deepcopy(config)
    invalid["rollout"]["top_p"] = 0.9
    with pytest.raises(ValueError, match="top_p=0.8"):
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
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
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
        {
            "sample_id": "one",
            "prompt": "p1",
            "response_a": "response-one",
            "response_b": "other-one",
        },
        {
            "sample_id": "two",
            "prompt": "p2",
            "response_a": "response-two",
            "response_b": "other-two",
        },
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
    assert summary["matches_public_response_a"] is True

    leaked = json.loads(sft.read_text().splitlines()[0])
    leaked["metadata"] = "not-preregistered"
    sft.write_text(json.dumps(leaked) + "\n")
    with pytest.raises(ValueError, match="unregistered fields"):
        validate_sft_corpus(sft, unlabeled, expected_rows=2)


def test_round2_sft_anchor_is_deterministically_derived_and_reused(tmp_path):
    unlabeled = tmp_path / "unlabeled.jsonl"
    output_dir = tmp_path / "anchor"
    rows = [
        {
            "sample_id": "one",
            "prompt": "p1",
            "response_a": "a1",
            "response_b": "b1",
            "is_truncated": False,
        },
        {
            "sample_id": "two",
            "prompt": "p2",
            "response_a": "a2",
            "response_b": "b2",
            "is_truncated": True,
        },
    ]
    unlabeled.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "manifest_public.json").write_text(
        json.dumps(
            {
                "dataset": "openbmb/UltraFeedback",
                "unlabeled_train": 2,
                "split_ratios": {"unlabeled_train": 0.8},
                "position_randomization_ratio": {"unlabeled": 0.5},
                "checksums": {
                    "unlabeled.jsonl": hashlib.sha256(
                        unlabeled.read_bytes()
                    ).hexdigest()
                },
            }
        )
    )
    anchor, evidence = prepare_sft_anchor(unlabeled, output_dir, expected_rows=2)
    actual = [json.loads(line) for line in anchor.read_text().splitlines()]
    assert [row["response"] for row in actual] == ["a1", "a2"]
    assert evidence["selection_rule"] == SELECTION_RULE
    assert evidence["reused"] is False

    reused_anchor, reused = prepare_sft_anchor(
        unlabeled, output_dir, expected_rows=2
    )
    assert reused_anchor == anchor
    assert reused["reused"] is True


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


def test_round2_peft_tp_hook_compatibility_supplies_current_plan():
    calls = []
    model = type("FakeTPModel", (), {"tp_plan": {"layer": "colwise"}})()

    def transformers_hook(
        model,
        module,
        tp_plan,
        layer_name,
        current_module_plan,
        device_mesh,
        parameter_name=None,
    ):
        calls.append(
            {
                "model": model,
                "module": module,
                "tp_plan": tp_plan,
                "layer_name": layer_name,
                "current_module_plan": current_module_plan,
                "device_mesh": device_mesh,
                "parameter_name": parameter_name,
            }
        )

    compatible, evidence = _build_peft_tp_hook_compatibility(transformers_hook)
    compatible(model, "module", "colwise", ("layer",), "mesh")
    assert evidence["compatibility_installed"] is True
    assert evidence["legacy_peft_call_count"] == 1
    assert calls == [
        {
            "model": model,
            "module": "module",
            "tp_plan": {"layer": "colwise"},
            "layer_name": ("layer",),
            "current_module_plan": "colwise",
            "device_mesh": "mesh",
            "parameter_name": None,
        }
    ]


def test_round2_peft_tp_hook_compatibility_preserves_new_api_calls():
    calls = []

    def transformers_hook(
        model,
        module,
        tp_plan,
        layer_name,
        current_module_plan,
        device_mesh,
        parameter_name=None,
    ):
        calls.append((current_module_plan, device_mesh, parameter_name))

    compatible, _ = _build_peft_tp_hook_compatibility(transformers_hook)
    compatible(
        "model",
        "module",
        {"layer": "rowwise"},
        "layer",
        "rowwise",
        "mesh",
        "weight",
    )
    assert calls == [("rowwise", "mesh", "weight")]


class _FakeMesh:
    def __init__(self, size=2, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank


class _FakeTPModel(torch.nn.Module):
    def __init__(self, replicated_q_proj=False, omit_q_proj_hooks=False):
        super().__init__()
        self._tp_size = 2
        self._device_mesh = _FakeMesh()
        self.tp_plan = {
            "layers.*.q_proj": "colwise",
            "layers.*.o_proj": "rowwise",
        }
        layer = torch.nn.Module()
        layer.q_proj = torch.nn.Linear(
            4, 8 if replicated_q_proj else 4, bias=False
        )
        layer.o_proj = torch.nn.Linear(4, 8, bias=False)
        for module, plan in (
            (layer.q_proj, "colwise"),
            (layer.o_proj, "rowwise"),
        ):
            if omit_q_proj_hooks and module is layer.q_proj:
                continue
            module._hf_tp_plan = plan
            module._hf_device_mesh = self._device_mesh
        self.layers = torch.nn.ModuleList([layer])


def test_round2_tp_verifier_accepts_checkpoint_backed_local_tensor_slices():
    shapes = {
        "layers.0.q_proj.weight": (8, 4),
        "layers.0.o_proj.weight": (8, 8),
    }
    evidence = _verify_local_tp_shapes(_FakeTPModel(), shapes)
    assert evidence["sharding_representation"] == (
        "checkpoint-verified-local-tensor-slices"
    )
    assert evidence["sharded_parameter_count"] == 2
    assert _expected_local_shape((8, 4), "colwise", 2, 0) == (4, 4)
    assert _expected_local_shape((8, 8), "rowwise", 2, 1) == (8, 4)


def test_round2_tp_verifier_rejects_replicated_weight_shape():
    shapes = {
        "layers.0.q_proj.weight": (8, 4),
        "layers.0.o_proj.weight": (8, 8),
    }
    with pytest.raises(RuntimeError, match="local shard shape mismatch"):
        _verify_local_tp_shapes(_FakeTPModel(replicated_q_proj=True), shapes)


def test_round2_tp_verifier_rejects_missing_module_hooks():
    shapes = {
        "layers.0.q_proj.weight": (8, 4),
        "layers.0.o_proj.weight": (8, 8),
    }
    with pytest.raises(RuntimeError, match="TP hooks are missing"):
        _verify_local_tp_shapes(_FakeTPModel(omit_q_proj_hooks=True), shapes)


def test_round2_tp_norm_separates_shards_from_replicas():
    sharded = torch.nn.Parameter(torch.zeros(2))
    replicated = torch.nn.Parameter(torch.zeros(2))
    sharded.grad = torch.tensor([3.0, 4.0])
    replicated.grad = torch.tensor([6.0, 8.0])
    squared = _local_tp_squared_norms(
        [sharded, replicated], {id(sharded)}, torch.device("cpu")
    )
    assert squared.tolist() == [25.0, 100.0]


def test_round2_rollout_worker_has_an_isolated_cleanup_group():
    script = (ROOT / "scripts" / "round2" / "run_method.sh").read_text()
    assert 'setsid "$ROUND2_ROLLOUT_PYTHON"' in script
    assert 'kill -TERM -- "-$ROLLOUT_PGID"' in script
    assert 'kill -KILL -- "-$ROLLOUT_PGID"' in script

    status_script = (ROOT / "scripts" / "round2" / "status_all.sh").read_text()
    assert '"$ROUND2_RUN_ROOT/strong_smoke"' in status_script
