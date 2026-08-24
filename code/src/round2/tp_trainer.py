"""Online round2 trainer: TP=2 Qwen3 LoRA plus exact 8/56 PE updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import jsonlines
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import canonical_json
from ..data.dataset import PreferenceCollator, PreferenceDataset, TOKENIZATION_CONTRACT
from ..model.model_manifest import verify_manifest
from ..model.pe_loss import PELoss
from ..model.sspo_loss import objective_weights, pe_pair_probabilities, simpo_pair_losses
from .config import load_round2_config, validate_round2_config
from .queue_protocol import (
    REQUEST_SCHEMA_VERSION,
    atomic_write_json,
    file_sha256,
    validate_response,
    wait_for_json,
)
from .sft_schema import load_sft_jsonl, validate_sft_corpus
from .tp_backend import describe_tp_parameters, validate_training_runtime


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _replace_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _initialize(config: Dict[str, Any]) -> Tuple[int, int, int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 2 or local_rank not in {0, 1}:
        raise RuntimeError(
            "Round2 must be launched by torchrun with exactly two local TP ranks"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    expected = str(config["tensor_parallel"]["gpu_ids"])
    if visible != expected:
        raise RuntimeError(
            f"Training CUDA_VISIBLE_DEVICES mismatch: actual={visible}, expected={expected}"
        )
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _lora_config(config: Dict[str, Any]) -> LoraConfig:
    values = config["model"]["lora"]
    return LoraConfig(
        r=int(values["r"]),
        lora_alpha=int(values["alpha"]),
        lora_dropout=float(values["dropout"]),
        bias=str(values["bias"]),
        task_type=str(values["task_type"]),
        target_modules=list(values["target_modules"]),
    )


def _verify_trainable_lora(policy, target_modules: Sequence[str]) -> Dict[str, Any]:
    expected = set(target_modules)
    observed = set()
    unexpected = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        pieces = set(name.split("."))
        targets = pieces & expected
        if len(targets) != 1 or not ({"lora_A", "lora_B"} & pieces):
            unexpected.append(name)
        else:
            observed.update(targets)
    if unexpected:
        raise ValueError(f"Non-LoRA trainable parameters found: {unexpected[:8]}")
    if observed != expected:
        raise ValueError(
            f"TP LoRA target coverage mismatch: actual={sorted(observed)}, "
            f"expected={sorted(expected)}"
        )
    tp_coverage = {target: 0 for target in expected}
    tp_plans = set()
    for module_name, module in policy.named_modules():
        pieces = set(module_name.split("."))
        targets = pieces & expected
        if len(targets) != 1 or not hasattr(module, "get_base_layer"):
            continue
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        target = next(iter(targets))
        base_layer = module.get_base_layer()
        tp_plan = getattr(base_layer, "_hf_tp_plan", None)
        device_mesh = getattr(base_layer, "_hf_device_mesh", None)
        if tp_plan not in {"colwise", "rowwise"} or device_mesh is None:
            raise RuntimeError(
                f"PEFT LoRA target did not inherit a supported TP plan: {module_name}"
            )
        if int(device_mesh.size()) != 2:
            raise RuntimeError(f"PEFT LoRA target has the wrong TP mesh: {module_name}")
        tp_coverage[target] += 1
        tp_plans.add(str(tp_plan))
    if any(count == 0 for count in tp_coverage.values()):
        raise RuntimeError(f"PEFT TP-LoRA module coverage is incomplete: {tp_coverage}")
    return {
        "target_module_counts": dict(sorted(tp_coverage.items())),
        "target_tp_plans": sorted(tp_plans),
    }


def _load_tp_policy(config: Dict[str, Any]):
    model = config["model"]
    model_path = Path(model["name_or_path"]).resolve()
    verify_manifest(model_path, Path(model["manifest_path"]).resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=DTYPES[model["torch_dtype"]],
        attn_implementation=model["attention_implementation"],
        low_cpu_mem_usage=True,
        tp_plan="auto",
    )
    base.config.use_cache = False
    tp_evidence = describe_tp_parameters(base, model_path)
    if bool(model["gradient_checkpointing"]):
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base.enable_input_require_grads()
    policy = get_peft_model(base, _lora_config(config))
    tp_evidence["lora"] = _verify_trainable_lora(
        policy, model["lora"]["target_modules"]
    )
    policy.train()
    return policy, tokenizer, tp_evidence


def _chat_ids(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    enable_thinking: bool,
) -> Dict[str, List[int]]:
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response + eos, add_special_tokens=False)["input_ids"]
    if not response_ids:
        raise ValueError("Round2 response tokenization is empty")
    full_ids = prompt_ids + response_ids
    response_start = len(prompt_ids)
    if len(full_ids) > int(max_length):
        removed = len(full_ids) - int(max_length)
        full_ids = full_ids[removed:]
        response_start = max(0, response_start - removed)
    loss_mask = [0] * response_start + [1] * (len(full_ids) - response_start)
    if sum(loss_mask[1:]) == 0:
        raise ValueError("Round2 response was fully truncated")
    return {"input_ids": full_ids, "loss_mask": loss_mask}


def _pair_to_cpu_batch(pair: Dict[str, str], tokenizer, config: Dict[str, Any]) -> Dict:
    row: Dict[str, Any] = {"sample_id": pair["sample_id"]}
    for side in ("a", "b"):
        encoded = _chat_ids(
            tokenizer,
            pair["prompt"],
            pair[f"response_{side}"],
            int(config["model"]["max_seq_len"]),
            bool(config["model"]["enable_thinking"]),
        )
        row[f"input_ids_{side}"] = encoded["input_ids"]
        row[f"loss_mask_{side}"] = encoded["loss_mask"]
    return PreferenceCollator(tokenizer.pad_token_id)([row])


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _response_mean_logp(policy, batch: Dict[str, torch.Tensor], side: str) -> torch.Tensor:
    """Compute only response-position logits for one physical pair."""
    if int(batch[f"input_ids_{side}"].shape[0]) != 1:
        raise ValueError("Round2 TP profile requires a physical batch of one pair")
    response_positions = torch.nonzero(
        batch[f"loss_mask_{side}"][0, 1:].bool(), as_tuple=False
    ).reshape(-1)
    if response_positions.numel() == 0:
        raise ValueError("Round2 response has no scored token")
    outputs = policy(
        input_ids=batch[f"input_ids_{side}"],
        attention_mask=batch[f"attention_mask_{side}"],
        use_cache=False,
        return_dict=True,
        logits_to_keep=response_positions,
    )
    labels = batch[f"input_ids_{side}"][:, 1:].index_select(1, response_positions)
    token_logps = F.log_softmax(outputs.logits.float(), dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    return token_logps.mean(dim=-1).reshape(())


def _first_pass(
    policy,
    batches: Sequence[Dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    score_a = []
    score_b = []
    policy.eval()
    with torch.no_grad():
        for cpu_batch in batches:
            batch = _move_batch(cpu_batch, device)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                score_a.append(_response_mean_logp(policy, batch, "a").float())
                score_b.append(_response_mean_logp(policy, batch, "b").float())
    policy.train()
    return torch.stack(score_a), torch.stack(score_b)


def _backward_responses(
    policy,
    batches: Sequence[Dict[str, Any]],
    coefficient_a: torch.Tensor,
    coefficient_b: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if len(batches) != coefficient_a.numel() or len(batches) != coefficient_b.numel():
        raise ValueError("Round2 response coefficient population is incomplete")
    for index, cpu_batch in enumerate(batches):
        batch = _move_batch(cpu_batch, device)
        for side, coefficients in (("a", coefficient_a), ("b", coefficient_b)):
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                live_score = _response_mean_logp(policy, batch, side)
                contribution = coefficients[index].detach() * live_score
            if not torch.isfinite(contribution):
                raise FloatingPointError("Non-finite round2 response contribution")
            contribution.backward()


def _cosine_factor(step: int, planned_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(1, planned_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def _shuffled_batches(
    size: int, batch_size: int, epochs: int, seed: int
) -> Iterable[Tuple[int, List[int]]]:
    for epoch in range(int(epochs)):
        indices = list(range(int(size)))
        random.Random(int(seed) + epoch).shuffle(indices)
        usable = len(indices) - len(indices) % int(batch_size)
        for start in range(0, usable, int(batch_size)):
            yield epoch, indices[start : start + int(batch_size)]


def _cycle_labeled_indices(size: int, count: int, seed: int):
    cycle = 0
    position = 0
    indices: List[int] = []
    while True:
        if position + int(count) > len(indices):
            indices = list(range(int(size)))
            random.Random(int(seed) + 100000 + cycle).shuffle(indices)
            position = 0
            cycle += 1
        selected = indices[position : position + int(count)]
        position += int(count)
        yield selected


def _longest_sft_indices(
    rows: Sequence[Dict[str, str]], tokenizer, count: int
) -> List[int]:
    """Select real prompts that most strongly exercise the rollout context cap."""
    lengths = []
    for index, row in enumerate(rows):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        lengths.append((prompt_tokens, len(row.get("response", "")), index))
    return [index for _, _, index in sorted(lengths, reverse=True)[: int(count)]]


def _longest_preference_indices(
    dataset: PreferenceDataset, count: int
) -> List[int]:
    lengths = []
    for index in range(len(dataset)):
        row = dataset[index]
        lengths.append(
            (
                max(len(row["input_ids_a"]), len(row["input_ids_b"])),
                index,
            )
        )
    return [index for _, index in sorted(lengths, reverse=True)[: int(count)]]


def _publish_adapter(
    policy: PeftModel,
    tokenizer,
    config: Dict[str, Any],
    policy_root: Path,
    step: int,
    rank: int,
) -> Path:
    final = policy_root / f"step_{step:06d}"
    partial = policy_root / f".step_{step:06d}.partial"
    if rank == 0:
        policy_root.mkdir(parents=True, exist_ok=True)
        if final.exists() or partial.exists():
            raise FileExistsError(f"Refuse to overwrite policy publication: {final}")
    dist.barrier()
    # PEFT's Transformers-native TP save gathers local LoRA shards, so every
    # TP rank must enter it. Supplying only LoRA entries prevents PEFT from
    # gathering the frozen 4B base state before its adapter-only filter.
    adapter_state = {
        name: value
        for name, value in policy.state_dict().items()
        if "lora_" in name
    }
    if not adapter_state:
        raise RuntimeError("Round2 TP adapter state is empty")
    policy.save_pretrained(
        partial,
        safe_serialization=True,
        save_embedding_layers=False,
        is_main_process=rank == 0,
        state_dict=adapter_state,
    )
    dist.barrier()
    if rank == 0:
        tokenizer.save_pretrained(partial)
        (partial / "run_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        metadata = {
            "format": "peft_lora_adapter",
            "base_model": str(Path(config["model"]["name_or_path"]).resolve()),
            "model_manifest": str(Path(config["model"]["manifest_path"]).resolve()),
            "step": int(step),
            "tp_size": 2,
            "adapter_state_entries": len(adapter_state),
            "config_sha256": hashlib.sha256(
                canonical_json(config).encode("utf-8")
            ).hexdigest(),
        }
        (partial / "checkpoint_meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        ready = {
            **metadata,
            "adapter_sha256": file_sha256(partial / "adapter_model.safetensors"),
        }
        (partial / "READY.json").write_text(
            json.dumps(ready, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        partial.replace(final)
    dist.barrier()
    if not (final / "READY.json").is_file():
        raise RuntimeError(f"TP adapter publication did not complete: {final}")
    return final


def _request_dynamic_pairs(
    config: Dict[str, Any],
    samples: Sequence[Dict[str, str]],
    checkpoint: Path,
    step: int,
    rank: int,
) -> Tuple[List[Dict[str, str]], float, Dict[str, Any]]:
    rollout = config["rollout"]
    artifact_dir = Path(rollout["artifact_dir"]).resolve()
    request_path = artifact_dir / "requests" / f"step_{step:06d}.json"
    response_path = artifact_dir / "responses" / f"step_{step:06d}.json"
    generation = {
        "temperature": float(rollout["temperature"]),
        "top_p": float(rollout["top_p"]),
        "top_k": int(rollout["top_k"]),
        "min_p": float(rollout["min_p"]),
        "max_new_tokens": int(rollout["max_new_tokens"]),
        "min_new_tokens": int(rollout["min_new_tokens"]),
        "max_model_len": int(rollout["max_model_len"]),
    }
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "step": int(step),
        "method": config["method"]["name"],
        "policy_checkpoint": str(checkpoint),
        "generation": generation,
        "samples": [],
    }
    for row in samples:
        sample = {"sample_id": row["sample_id"], "prompt": row["prompt"]}
        if config["method"]["name"] == "soppo_pe_sft_rollout_exp":
            sample["sft_response"] = row["response"]
        request["samples"].append(sample)
    if rank == 0:
        atomic_write_json(request_path, request)
    dist.barrier()
    started = time.monotonic()
    response = wait_for_json(
        response_path,
        float(rollout["request_timeout_seconds"]),
        float(rollout["poll_interval_seconds"]),
        artifact_dir / "worker.failed.json",
    )
    validate_response(response, request)
    dist.barrier()
    return (
        response["pairs"],
        float(response["generation_seconds"]),
        response["statistics"],
    )


def _evaluate(
    policy,
    dataset: PreferenceDataset,
    collator: PreferenceCollator,
    device: torch.device,
    dtype: torch.dtype,
    beta: float,
    indices: Sequence[int] | None = None,
) -> Dict[str, Any]:
    correct = 0
    brier = 0.0
    finite = 0
    policy.eval()
    with torch.no_grad():
        selected = list(indices) if indices is not None else list(range(len(dataset)))
        for index in selected:
            batch = _move_batch(collator([dataset[index]]), device)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                score_a = _response_mean_logp(policy, batch, "a")
                score_b = _response_mean_logp(policy, batch, "b")
            delta = float(beta) * (score_a.float() - score_b.float())
            probability = torch.sigmoid(delta)
            label = int(batch["labels"].item())
            correct += int(bool(probability > 0.5) == bool(label))
            brier += float((probability - label) ** 2)
            finite += int(torch.isfinite(delta))
    policy.train()
    if finite != len(selected):
        raise FloatingPointError("Round2 validation produced non-finite scores")
    return {
        "val_accuracy": correct / len(selected),
        "val_brier": brier / len(selected),
        "val_samples": len(selected),
        "score_type": "simpo_mean_logp_delta_margin_free",
    }


def _scalar(value: torch.Tensor) -> float:
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(value, DTensor):
            value = value.full_tensor()
    except ImportError:
        pass
    return float(value.detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_round2_config(args.config)
    validate_round2_config(config)
    versions = validate_training_runtime(config)
    rank, local_rank, world_size, device = _initialize(config)
    _seed_everything(int(config["training"]["seed"]))

    output_dir = Path(config["output"]["run_dir"]).resolve()
    training_dir = output_dir / "training"
    logs_dir = output_dir / "logs"
    state_path = output_dir / "state.json"
    if rank == 0:
        if training_dir.exists() or state_path.exists():
            raise FileExistsError(f"Refuse to reuse round2 training attempt: {output_dir}")
        training_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        _replace_json(
            state_path,
            {"state": "initializing", "step": 0, "backend": "transformers-native-tp"},
        )
    dist.barrier()

    policy, tokenizer, tp_evidence = _load_tp_policy(config)
    dtype = DTYPES[config["model"]["torch_dtype"]]
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in policy.parameters())
    if not 0 < trainable_count < total_count:
        raise ValueError("Round2 TP policy is not LoRA-only trainable")

    data_dir = Path(config["data"]["data_dir"]).resolve()
    labeled_train = PreferenceDataset(
        str(data_dir / "labeled_train.jsonl"),
        tokenizer,
        max_length=int(config["model"]["max_seq_len"]),
        require_labels=True,
        enable_thinking=bool(config["model"]["enable_thinking"]),
    )
    labeled_val = PreferenceDataset(
        str(data_dir / "labeled_val.jsonl"),
        tokenizer,
        max_length=int(config["model"]["max_seq_len"]),
        require_labels=True,
        enable_thinking=bool(config["model"]["enable_thinking"]),
    )
    sft_summary = validate_sft_corpus(
        config["rollout"]["sft_data_file"],
        data_dir / "unlabeled_train.jsonl",
        int(config["data"]["unlabeled_train_samples"]),
    )
    sft_rows = load_sft_jsonl(config["rollout"]["sft_data_file"])
    sft_rows = sorted(sft_rows, key=lambda row: row["sample_id"])
    if config["method"]["name"] == "soppo_pe_rollout_only_exp":
        # Preserve the exact prompt universe/order while making the rollout-only
        # trainer structurally unable to consume SFT response text.
        sft_rows = [
            {"sample_id": row["sample_id"], "prompt": row["prompt"]}
            for row in sft_rows
        ]
    collator = PreferenceCollator(tokenizer.pad_token_id)
    training = config["training"]
    method = config["method"]
    labeled_size = int(training["joint_labeled_global_batch_size"])
    dynamic_size = int(training["joint_unlabeled_global_batch_size"])
    if labeled_size != 8 or dynamic_size != 56:
        raise ValueError("Round2 trainer requires the exact 8/56 population")

    steps_per_epoch = len(sft_rows) // dynamic_size
    planned_steps = int(training["epochs"]) * steps_per_epoch
    if training.get("max_steps") is not None:
        planned_steps = min(planned_steps, int(training["max_steps"]))
    if planned_steps < 1:
        raise ValueError("Round2 training plan contains zero optimizer steps")
    smoke_mode = bool(training.get("smoke_mode"))
    if smoke_mode:
        smoke_sft_indices = _longest_sft_indices(sft_rows, tokenizer, dynamic_size)
        smoke_labeled_indices = _longest_preference_indices(
            labeled_train, labeled_size
        )
        validation_indices = _longest_preference_indices(
            labeled_val, int(training["eval_max_samples"])
        )
        batch_plan = iter([(0, smoke_sft_indices)])
        labeled_plan = iter([smoke_labeled_indices])
    else:
        batch_plan = _shuffled_batches(
            len(sft_rows),
            dynamic_size,
            int(training["epochs"]),
            int(training["seed"]),
        )
        labeled_plan = _cycle_labeled_indices(
            len(labeled_train), labeled_size, int(training["seed"])
        )
        validation_indices = None

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["lr"]),
        betas=(float(training["adam_beta1"]), float(training["adam_beta2"])),
        weight_decay=float(training["weight_decay"]),
        foreach=False,
    )
    warmup_steps = int(planned_steps * float(training["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _cosine_factor(step, planned_steps, warmup_steps)
    )

    rank_evidence = {
        **tp_evidence,
        "rank": rank,
        "local_rank": local_rank,
        "device_name": torch.cuda.get_device_name(device),
        "device_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
    }
    gathered_evidence: List[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_evidence, rank_evidence)
    if rank == 0:
        _replace_json(
            output_dir / "tp_evidence.json",
            {
                "world_size": world_size,
                "git_commit": config["provenance"]["git_commit"],
                "config_sha256": hashlib.sha256(
                    canonical_json(config).encode("utf-8")
                ).hexdigest(),
                "versions": versions,
                "ranks": gathered_evidence,
                "tokenization_contract": TOKENIZATION_CONTRACT,
                "sft": sft_summary,
            },
        )

    policy_root = output_dir / "policy"
    checkpoint = _publish_adapter(policy, tokenizer, config, policy_root, 0, rank)
    metrics_path = logs_dir / "metrics.jsonl"
    best_accuracy = -1.0
    best_brier = float("inf")
    completed_steps = 0

    for epoch, sft_indices in batch_plan:
        if completed_steps >= planned_steps:
            break
        step_started = time.monotonic()
        selected_sft = [sft_rows[index] for index in sft_indices]
        dynamic_pairs, generation_seconds, generation_statistics = _request_dynamic_pairs(
            config, selected_sft, checkpoint, completed_steps, rank
        )
        if smoke_mode:
            if int(generation_statistics["min_generated_tokens"]) != int(
                config["rollout"]["max_new_tokens"]
            ):
                raise RuntimeError(
                    "Strong smoke did not force max-length rollout responses"
                )
        dynamic_batches = [
            _pair_to_cpu_batch(pair, tokenizer, config) for pair in dynamic_pairs
        ]
        labeled_indices = next(labeled_plan)
        labeled_batches = [collator([labeled_train[index]]) for index in labeled_indices]
        if smoke_mode:
            longest_labeled_sequence = max(
                int(batch[key].shape[1])
                for batch in labeled_batches
                for key in ("input_ids_a", "input_ids_b")
            )
            if longest_labeled_sequence != int(config["model"]["max_seq_len"]):
                raise RuntimeError(
                    "Strong smoke did not exercise a max-length labeled sequence"
                )

        labeled_a, labeled_b = _first_pass(
            policy, labeled_batches, device, dtype
        )
        dynamic_a, dynamic_b = _first_pass(
            policy, dynamic_batches, device, dtype
        )
        objective_scheduler_step = (
            int(training["smoke_objective_step"])
            if smoke_mode
            else completed_steps
        )
        supervised_weight, auxiliary_weight = objective_weights(
            method, objective_scheduler_step
        )

        labeled_leaf_a = labeled_a.detach().requires_grad_(True)
        labeled_leaf_b = labeled_b.detach().requires_grad_(True)
        labels = torch.tensor(
            [int(labeled_train[index]["label"]) for index in labeled_indices],
            dtype=torch.long,
            device=device,
        )
        labeled_losses, _ = simpo_pair_losses(
            labeled_leaf_a,
            labeled_leaf_b,
            labels,
            float(training["simpo_beta"]),
            float(training["simpo_margin"]),
        )
        labeled_loss = labeled_losses.mean()
        labeled_coefficient_a, labeled_coefficient_b = torch.autograd.grad(
            labeled_loss, (labeled_leaf_a, labeled_leaf_b)
        )

        probabilities = pe_pair_probabilities(
            dynamic_a.detach(), dynamic_b.detach(), float(training["simpo_beta"])
        ).requires_grad_(True)
        pe_loss, pe_info = PELoss(
            float(training["epsilon"]),
            method["pe_distance"],
            bool(method["detach_denominator"]),
        ).to(device)(probabilities)
        probability_coefficient = torch.autograd.grad(pe_loss, probabilities)[0]
        probability_jacobian = (
            float(training["simpo_beta"]) * probabilities.detach() * (1 - probabilities.detach())
        )
        dynamic_coefficient_a = probability_coefficient * probability_jacobian
        dynamic_coefficient_b = -dynamic_coefficient_a

        optimizer.zero_grad(set_to_none=True)
        _backward_responses(
            policy,
            labeled_batches,
            float(supervised_weight) * labeled_coefficient_a,
            float(supervised_weight) * labeled_coefficient_b,
            device,
            dtype,
        )
        _backward_responses(
            policy,
            dynamic_batches,
            float(auxiliary_weight) * dynamic_coefficient_a,
            float(auxiliary_weight) * dynamic_coefficient_b,
            device,
            dtype,
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(training["max_grad_norm"]), foreach=False
        )
        grad_norm_value = _scalar(grad_norm)
        if not math.isfinite(grad_norm_value):
            raise FloatingPointError("Round2 produced a non-finite gradient norm")
        optimizer.step()
        scheduler.step()
        completed_steps += 1
        checkpoint = _publish_adapter(
            policy, tokenizer, config, policy_root, completed_steps, rank
        )

        record = {
            "step": completed_steps,
            "epoch": epoch,
            "method": method["name"],
            "objective_scheduler_step": objective_scheduler_step,
            "loss_supervised": float(labeled_loss.detach()),
            "loss_pe": float(pe_loss.detach()),
            "supervised_weight": float(supervised_weight),
            "aux_weight": float(auxiliary_weight),
            "weighted_objective_value": (
                float(supervised_weight) * float(labeled_loss.detach())
                + float(auxiliary_weight) * float(pe_loss.detach())
            ),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm_value,
            "global_batch_size": 64,
            "labeled_pairs": labeled_size,
            "dynamic_pairs": dynamic_size,
            "physical_pair_subbatch": 1,
            "policy_checkpoint": str(checkpoint),
            "rollout_generation_seconds": generation_seconds,
            "rollout_statistics": generation_statistics,
            "optimizer_step_seconds": time.monotonic() - step_started,
            "pe": pe_info,
        }
        if rank == 0:
            with jsonlines.open(metrics_path, "a") as writer:
                writer.write(record)
            _replace_json(
                state_path,
                {
                    "state": "training",
                    "step": completed_steps,
                    "planned_steps": planned_steps,
                    "policy_checkpoint": str(checkpoint),
                },
            )

        should_evaluate = (
            completed_steps % int(training["eval_steps"]) == 0
            or completed_steps == planned_steps
        )
        if should_evaluate:
            validation = _evaluate(
                policy,
                labeled_val,
                collator,
                device,
                dtype,
                float(training["simpo_beta"]),
                validation_indices,
            )
            if rank == 0:
                with jsonlines.open(metrics_path, "a") as writer:
                    writer.write({"step": completed_steps, **validation})
                accuracy = float(validation["val_accuracy"])
                brier = float(validation["val_brier"])
                if accuracy > best_accuracy or (
                    accuracy == best_accuracy and brier < best_brier
                ):
                    best_accuracy = accuracy
                    best_brier = brier
                    _replace_json(
                        output_dir / "best.json",
                        {
                            "step": completed_steps,
                            "policy_checkpoint": str(checkpoint),
                            "selection_split": "labeled_validation",
                            **validation,
                        },
                    )

    local_memory = {
        "rank": rank,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    memory_by_rank: List[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(memory_by_rank, local_memory)
    completion = {
        "status": "succeeded",
        "git_commit": config["provenance"]["git_commit"],
        "config_sha256": hashlib.sha256(
            canonical_json(config).encode("utf-8")
        ).hexdigest(),
        "steps": completed_steps,
        "method": method["name"],
        "backend": "transformers-native-tp",
        "tensor_parallel_size": 2,
        "data_parallel_size": 1,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "final_policy_checkpoint": str(checkpoint),
        "best_val_accuracy": best_accuracy,
        "best_val_brier": best_brier,
        "memory_by_rank": memory_by_rank,
    }
    if rank == 0:
        _replace_json(output_dir / "complete.json", completion)
        _replace_json(state_path, completion)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
