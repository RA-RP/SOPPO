"""Single-GPU Round3 trainer for all five approved methods.

Logical losses are evaluated on the complete registered population. The
resolved physical subbatch controls only activation memory: a detached first
pass obtains exact logical coefficients and a second pass accumulates their
vector-Jacobian product before one optimizer update.
"""

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
import numpy as np
import torch
import torch.nn.functional as F

from ..model.dpo_loss import compute_sequence_logprob, response_token_count
from ..model.model_utils import DTYPES, load_tokenizer, load_trainable_policy
from .checkpoint import load_training_state, publish_staging_adapter, save_durable_checkpoint
from .config import (
    DPO_METHODS,
    DYNAMIC_METHODS,
    SSPO_METHOD,
    load_round3_config,
    validate_round3_config,
)
from .data import PairCollator, PairDataset, Round3TextEncoder, SingleCollator, SingleDataset
from .losses import (
    GitHubSSPOState,
    dpo_objective,
    github_sspo_objective,
    joint_dpo_pe_objective,
    pe_objective,
    rollout_anchor_statistics,
)
from .queue_protocol import (
    atomic_write_json,
    canonical_json,
    make_request,
    merge_replica_responses,
    read_json,
    wait_for_replica_responses,
)


def _seed_everything(seed: int) -> None:
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, ":4096:8"):
        raise RuntimeError(
            "Round3 requires CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic replay"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _replace_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def _chunks(values: Sequence[Any], size: int) -> Iterable[Tuple[int, Sequence[Any]]]:
    for start in range(0, len(values), int(size)):
        yield start, values[start : start + int(size)]


def _move(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _response_scores(policy, batch: Dict[str, torch.Tensor], side: str | None) -> Tuple[torch.Tensor, torch.Tensor]:
    suffix = f"_{side}" if side is not None else ""
    outputs = policy(
        input_ids=batch[f"input_ids{suffix}"],
        attention_mask=batch[f"attention_mask{suffix}"],
        use_cache=False,
        return_dict=True,
    )
    total = compute_sequence_logprob(
        outputs.logits,
        batch[f"input_ids{suffix}"],
        batch[f"loss_mask{suffix}"],
    )
    mean = total / response_token_count(batch[f"loss_mask{suffix}"])
    return total, mean


def _first_pass_pairs(
    policy,
    examples: Sequence[Dict[str, Any]],
    collator: PairCollator,
    physical: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    totals_a, totals_b, means_a, means_b = [], [], [], []
    policy.eval()
    with torch.no_grad():
        for _, rows in _chunks(examples, physical):
            batch = _move(collator(list(rows)), device)
            with torch.autocast("cuda", dtype=dtype):
                total_a, mean_a = _response_scores(policy, batch, "a")
                total_b, mean_b = _response_scores(policy, batch, "b")
            totals_a.append(total_a.float())
            totals_b.append(total_b.float())
            means_a.append(mean_a.float())
            means_b.append(mean_b.float())
    policy.train()
    return tuple(torch.cat(values) for values in (totals_a, totals_b, means_a, means_b))


def _first_pass_singles(
    policy,
    examples: Sequence[Dict[str, Any]],
    collator: SingleCollator,
    physical: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    means = []
    policy.eval()
    with torch.no_grad():
        for _, rows in _chunks(examples, physical):
            batch = _move(collator(list(rows)), device)
            with torch.autocast("cuda", dtype=dtype):
                _, mean = _response_scores(policy, batch, None)
            means.append(mean.float())
    policy.train()
    return torch.cat(means)


def _backward_pairs(
    policy,
    examples: Sequence[Dict[str, Any]],
    collator: PairCollator,
    total_coeff_a: torch.Tensor,
    total_coeff_b: torch.Tensor,
    mean_coeff_a: torch.Tensor,
    mean_coeff_b: torch.Tensor,
    physical: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    expected = len(examples)
    coefficients = (total_coeff_a, total_coeff_b, mean_coeff_a, mean_coeff_b)
    if any(value.numel() != expected for value in coefficients):
        raise ValueError("Round3 pair coefficient population is incomplete")
    for start, rows in _chunks(examples, physical):
        stop = start + len(rows)
        batch = _move(collator(list(rows)), device)
        contribution = torch.zeros((), device=device, dtype=torch.float32)
        for side, total_coeff, mean_coeff in (
            ("a", total_coeff_a, mean_coeff_a),
            ("b", total_coeff_b, mean_coeff_b),
        ):
            with torch.autocast("cuda", dtype=dtype):
                total, mean = _response_scores(policy, batch, side)
                contribution = contribution + (
                    total.float() * total_coeff[start:stop].detach()
                    + mean.float() * mean_coeff[start:stop].detach()
                ).sum()
        if not torch.isfinite(contribution):
            raise FloatingPointError("Non-finite Round3 pair backward contribution")
        contribution.backward()


def _backward_singles(
    policy,
    examples: Sequence[Dict[str, Any]],
    collator: SingleCollator,
    mean_coefficients: torch.Tensor,
    physical: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if mean_coefficients.numel() != len(examples):
        raise ValueError("Round3 single coefficient population is incomplete")
    for start, rows in _chunks(examples, physical):
        stop = start + len(rows)
        batch = _move(collator(list(rows)), device)
        with torch.autocast("cuda", dtype=dtype):
            _, means = _response_scores(policy, batch, None)
            contribution = (means.float() * mean_coefficients[start:stop].detach()).sum()
        if not torch.isfinite(contribution):
            raise FloatingPointError("Non-finite Round3 single backward contribution")
        contribution.backward()


def _zero_like(value: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(value, dtype=torch.float32)


def _cosine_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def _load_reference_cache(path: Path, expected_ids: Sequence[str]) -> Dict[str, Dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Round3 reference cache not found: {path}")
    records: Dict[str, Dict[str, float]] = {}
    with jsonlines.open(path) as reader:
        for row in reader:
            sample_id = row.get("sample_id")
            if sample_id in records:
                raise ValueError(f"Duplicate Round3 reference-cache ID: {sample_id}")
            records[sample_id] = {
                "ref_logp_a": float(row["ref_logp_a"]),
                "ref_logp_b": float(row["ref_logp_b"]),
            }
    if set(records) != set(expected_ids) or len(records) != len(expected_ids):
        raise ValueError("Round3 reference cache does not exactly match its data view")
    return records


def _data_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    root = Path(config["data"]["data_dir"]).resolve()
    cache = Path(config["data"]["reference_cache_dir"]).resolve()
    paths = {
        "paired_1k": root / "paired_train_1k.jsonl",
        "paired_8k": root / "paired_train_8k.jsonl",
        "unpaired": root / "unpaired_train_7k.jsonl",
        "validation": root / "validation_1k.jsonl",
        "ref_1k": cache / "paired_train_1k.reference.jsonl",
        "ref_8k": cache / "paired_train_8k.reference.jsonl",
        "ref_validation": cache / "validation_1k.reference.jsonl",
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Round3 input missing ({key}): {path}")
    return paths


def _dataset_rows(path: Path) -> List[Dict[str, Any]]:
    with jsonlines.open(path) as reader:
        return list(reader)


def _dynamic_examples(rows: Sequence[Dict[str, Any]], encoder: Round3TextEncoder) -> List[Dict[str, Any]]:
    examples = []
    for row in rows:
        item: Dict[str, Any] = {"sample_id": row["sample_id"]}
        for side in ("a", "b"):
            encoded = encoder.encode(row["prompt"], row[f"response_{side}"])
            for key, value in encoded.items():
                item[f"{key}_{side}"] = value
        examples.append(item)
    return examples


def _pair_truncation(examples: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return {
        key: sum(int(row[key]) for row in examples)
        for key in (
            "prompt_tokens_removed_a",
            "prompt_tokens_removed_b",
            "response_tokens_removed_a",
            "response_tokens_removed_b",
        )
    }


def _single_truncation(examples: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return {
        key: sum(int(row[key]) for row in examples)
        for key in ("prompt_tokens_removed", "response_tokens_removed")
    }


def _longest_pair_indices(dataset: PairDataset, count: int) -> List[int]:
    ranked = []
    for index in range(len(dataset)):
        row = dataset[index]
        ranked.append(
            (
                max(len(row["input_ids_a"]), len(row["input_ids_b"])),
                row["response_tokens_a"] + row["response_tokens_b"],
                -index,
                index,
            )
        )
    return [item[-1] for item in sorted(ranked, reverse=True)[: int(count)]]


def _longest_single_indices(dataset: SingleDataset, count: int) -> List[int]:
    ranked = []
    for index in range(len(dataset)):
        row = dataset[index]
        ranked.append((len(row["input_ids"]), row["response_tokens"], -index, index))
    return [item[-1] for item in sorted(ranked, reverse=True)[: int(count)]]


def _rollout_generation_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    rollout = config["rollout"]
    return {
        "enable_thinking": False,
        "do_sample": True,
        "temperature": float(rollout["temperature"]),
        "top_p": float(rollout["top_p"]),
        "top_k": int(rollout["top_k"]),
        "min_p": float(rollout["min_p"]),
        "repetition_penalty": float(rollout["repetition_penalty"]),
        "presence_penalty": float(rollout["presence_penalty"]),
        "max_prompt_tokens": 1024,
        "max_new_tokens": 1024,
        "max_model_len": 2048,
        "eos_token_id": list(rollout["eos_token_id"]),
        "pad_token_id": int(rollout["pad_token_id"]),
    }


def _obtain_dynamic_pairs(
    policy,
    tokenizer,
    config: Dict[str, Any],
    source_rows: Sequence[Dict[str, Any]],
    step: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    artifact_dir = Path(config["rollout"]["artifact_dir"]).resolve()
    staging_root = artifact_dir / "policy"
    checkpoint = staging_root / f"step_{int(step):06d}"
    if not checkpoint.is_dir():
        checkpoint = publish_staging_adapter(policy, tokenizer, staging_root, config, step)
    request = make_request(
        config["method"]["name"],
        step,
        checkpoint,
        source_rows,
        _rollout_generation_contract(config),
        int(config["training"]["seed"]),
    )
    request_path = artifact_dir / "requests" / f"step_{int(step):06d}.json"
    if request_path.is_file():
        if read_json(request_path) != request:
            raise ValueError("Existing Round3 rollout request differs from exact resume request")
    else:
        atomic_write_json(request_path, request)
    responses = wait_for_replica_responses(
        artifact_dir,
        request,
        float(config["rollout"]["request_timeout_seconds"]),
        float(config["rollout"]["poll_interval_seconds"]),
    )
    return merge_replica_responses(request, responses)


def _selection_loss(
    policy,
    dataset: PairDataset,
    collator: PairCollator,
    physical: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[float, Dict[str, int]]:
    if len(dataset) != 1000:
        raise ValueError("Round3 selection view must contain exactly 1,000 pairs")
    total_loss = 0.0
    seen = 0
    truncation = {
        "prompt_tokens_removed_a": 0,
        "prompt_tokens_removed_b": 0,
        "response_tokens_removed_a": 0,
        "response_tokens_removed_b": 0,
    }
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), 4):
            examples = [dataset[index] for index in range(start, start + 4)]
            for key, value in _pair_truncation(examples).items():
                truncation[key] += value
            policy_a, policy_b, _, _ = _first_pass_pairs(
                policy, examples, collator, physical, device, dtype
            )
            reference_a = torch.tensor([row["ref_logp_a"] for row in examples], device=device)
            reference_b = torch.tensor([row["ref_logp_b"] for row in examples], device=device)
            labels = torch.tensor([row["label"] for row in examples], device=device)
            delta = 0.1 * ((policy_a - reference_a) - (policy_b - reference_b))
            direction = labels.to(delta.dtype).mul(2).sub(1)
            values = -F.logsigmoid(direction * delta)
            if not torch.isfinite(values).all():
                raise FloatingPointError("Non-finite Round3 selection batch")
            total_loss += float(values.sum())
            seen += len(examples)
    policy.train()
    loss = total_loss / seen
    if not math.isfinite(loss):
        raise FloatingPointError("Non-finite Round3 selection aggregate")
    return loss, truncation


def _state_hash(state: GitHubSSPOState | None) -> str | None:
    if state is None:
        return None
    return hashlib.sha256(canonical_json(state.state_dict()).encode("utf-8")).hexdigest()


def _run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-checkpoint")
    args = parser.parse_args()
    config = load_round3_config(args.config)
    validate_round3_config(config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(config["training"]["train_gpu"]):
        raise RuntimeError("Round3 trainer CUDA_VISIBLE_DEVICES differs from its resolved physical GPU")
    _seed_everything(int(config["training"]["seed"]))
    device = torch.device("cuda:0")
    dtype = DTYPES[config["model"]["torch_dtype"]]
    method = config["method"]["name"]
    output_dir = Path(config["output"]["run_dir"]).resolve()
    metrics_path = output_dir / "logs" / "metrics.jsonl"
    state_path = output_dir / "state.json"
    fresh = args.resume_checkpoint is None
    if fresh:
        if output_dir.exists():
            existing = {path.name for path in output_dir.iterdir()}
            if method not in DYNAMIC_METHODS or not existing <= {"rollouts"}:
                raise FileExistsError(f"Refuse to reuse Round3 run directory: {output_dir}")
        (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    elif not output_dir.is_dir():
        raise FileNotFoundError("Round3 resume requires the original run directory")

    paths = _data_paths(config)
    tokenizer = load_tokenizer(config["model"]["name_or_path"])
    policy = load_trainable_policy(config, adapter_checkpoint=args.resume_checkpoint)
    # PEFT normally promotes trainable adapters, but enforce the approved mixed
    # precision contract instead of relying on version-specific defaults.
    trainable = []
    for parameter in policy.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
            trainable.append(parameter)
    if not trainable or any(parameter.dtype != torch.float32 for parameter in trainable):
        raise ValueError("Round3 requires FP32 trainable LoRA parameters")
    policy.to(device).train()
    physical = int(config["training"]["physical_pair_subbatch"])
    pair_collator = PairCollator(tokenizer.pad_token_id)
    single_collator = SingleCollator(tokenizer.pad_token_id)

    paired_path = paths["paired_8k"] if method == "dpo_8k" else paths["paired_1k"]
    expected_pairs = 8000 if method == "dpo_8k" else 1000
    paired_ids = [row["sample_id"] for row in _dataset_rows(paired_path)]
    pair_cache = None
    if method != SSPO_METHOD:
        cache_path = paths["ref_8k"] if method == "dpo_8k" else paths["ref_1k"]
        pair_cache = _load_reference_cache(cache_path, paired_ids)
    paired = PairDataset(paired_path, tokenizer, require_labels=True, reference_cache=pair_cache)
    if len(paired) != expected_pairs:
        raise ValueError(f"Round3 {method} paired view has {len(paired)} rows, expected {expected_pairs}")
    unpaired = SingleDataset(paths["unpaired"], tokenizer) if method in ({SSPO_METHOD} | DYNAMIC_METHODS) else None
    if unpaired is not None and len(unpaired) != 7000:
        raise ValueError("Round3 unpaired view must contain exactly 7,000 singles")
    validation_ids = [row["sample_id"] for row in _dataset_rows(paths["validation"])]
    validation_cache = _load_reference_cache(paths["ref_validation"], validation_ids)
    validation = PairDataset(paths["validation"], tokenizer, require_labels=True, reference_cache=validation_cache)

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"]["learning_rate"]),
        betas=(float(config["training"]["adam_beta1"]), float(config["training"]["adam_beta2"])),
        eps=float(config["training"]["adam_epsilon"]),
        weight_decay=float(config["training"]["weight_decay"]),
        foreach=False,
    )
    total_steps = 250
    strong_smoke = config["execution"]["mode"] == "strong_smoke"
    target_steps = 1 if strong_smoke else total_steps
    warmup_steps = int(total_steps * float(config["training"]["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _cosine_factor(step, total_steps, warmup_steps)
    )
    sspo_state = GitHubSSPOState() if method == SSPO_METHOD else None
    completed_steps = 0
    if args.resume_checkpoint:
        payload = load_training_state(
            args.resume_checkpoint,
            config,
            optimizer,
            scheduler,
            require_sspo=method == SSPO_METHOD,
        )
        completed_steps = int(payload["global_step"])
        if method == SSPO_METHOD:
            sspo_state = GitHubSSPOState.from_state_dict(payload["sspo"])
    if completed_steps < 0 or completed_steps >= target_steps:
        raise ValueError("Round3 resume checkpoint step is outside this execution profile")

    batch_labeled = int(config["method"]["labeled_pairs"])
    batch_unpaired = int(config["method"]["unpaired_units"])
    if len(paired) // batch_labeled != total_steps:
        raise ValueError("Round3 paired data does not form exactly 250 optimizer steps")
    if unpaired is not None and len(unpaired) // batch_unpaired != total_steps:
        raise ValueError("Round3 unpaired data does not form exactly 250 optimizer steps")
    if fresh:
        _replace_json(state_path, {"state": "initializing", "step": 0, "method_id": method})

    best: Tuple[float, int, str] | None = None
    best_path = output_dir / "best.json"
    if best_path.is_file():
        prior_best = read_json(best_path)
        best = (
            float(prior_best["eval_selection_loss"]),
            int(prior_best["checkpoint_step"]),
            str(prior_best["checkpoint"]),
        )
    encoder = Round3TextEncoder(tokenizer)
    smoke_labeled_indices = (
        _longest_pair_indices(paired, batch_labeled) if strong_smoke else None
    )
    smoke_unpaired_indices = (
        _longest_single_indices(unpaired, batch_unpaired)
        if strong_smoke and unpaired is not None
        else None
    )
    for step in range(completed_steps, target_steps):
        started = time.monotonic()
        labeled_indices = (
            smoke_labeled_indices
            if strong_smoke
            else list(range(step * batch_labeled, (step + 1) * batch_labeled))
        )
        assert labeled_indices is not None
        labeled_examples = [paired[index] for index in labeled_indices]
        policy_total_a, policy_total_b, policy_mean_a, policy_mean_b = _first_pass_pairs(
            policy, labeled_examples, pair_collator, physical, device, dtype
        )
        telemetry: Dict[str, Any]
        optimizer.zero_grad(set_to_none=True)

        if method in DPO_METHODS:
            leaf_a = policy_total_a.detach().requires_grad_(True)
            leaf_b = policy_total_b.detach().requires_grad_(True)
            ref_a = torch.tensor([row["ref_logp_a"] for row in labeled_examples], device=device)
            ref_b = torch.tensor([row["ref_logp_b"] for row in labeled_examples], device=device)
            labels = torch.tensor([row["label"] for row in labeled_examples], device=device)
            loss, telemetry = dpo_objective(leaf_a, leaf_b, ref_a, ref_b, labels)
            coeff_a, coeff_b = torch.autograd.grad(loss, (leaf_a, leaf_b))
            _backward_pairs(
                policy, labeled_examples, pair_collator,
                coeff_a, coeff_b, _zero_like(coeff_a), _zero_like(coeff_b),
                physical, device, dtype,
            )
        elif method == SSPO_METHOD:
            assert unpaired is not None and sspo_state is not None
            single_indices = (
                smoke_unpaired_indices
                if strong_smoke
                else list(range(step * 28, (step + 1) * 28))
            )
            assert single_indices is not None
            single_examples = [unpaired[index] for index in single_indices]
            unpaired_means = _first_pass_singles(policy, single_examples, single_collator, physical, device, dtype)
            labels = torch.tensor([row["label"] for row in labeled_examples], dtype=torch.bool, device=device)
            chosen = torch.where(labels, policy_mean_a, policy_mean_b).detach().requires_grad_(True)
            rejected = torch.where(labels, policy_mean_b, policy_mean_a).detach().requires_grad_(True)
            single_leaf = unpaired_means.detach().requires_grad_(True)
            loss, telemetry = github_sspo_objective(chosen, rejected, single_leaf, sspo_state, step)
            chosen_coeff, rejected_coeff, single_coeff = torch.autograd.grad(
                loss, (chosen, rejected, single_leaf)
            )
            coeff_a = torch.where(labels, chosen_coeff, rejected_coeff)
            coeff_b = torch.where(labels, rejected_coeff, chosen_coeff)
            _backward_pairs(
                policy, labeled_examples, pair_collator,
                _zero_like(coeff_a), _zero_like(coeff_b), coeff_a, coeff_b,
                physical, device, dtype,
            )
            _backward_singles(
                policy, single_examples, single_collator, single_coeff,
                physical, device, dtype,
            )
            telemetry["truncation_labeled"] = _pair_truncation(labeled_examples)
            telemetry["truncation_unpaired"] = _single_truncation(single_examples)
        else:
            assert unpaired is not None
            source_indices = (
                smoke_unpaired_indices
                if strong_smoke
                else list(range(step * 28, (step + 1) * 28))
            )
            assert source_indices is not None
            source_rows = [unpaired.rows[index] for index in source_indices]
            dynamic_pairs, rollout_statistics = _obtain_dynamic_pairs(
                policy, tokenizer, config, source_rows, step
            )
            dynamic_examples = _dynamic_examples(dynamic_pairs, encoder)
            _, _, dynamic_mean_a, dynamic_mean_b = _first_pass_pairs(
                policy, dynamic_examples, pair_collator, physical, device, dtype
            )
            labeled_a = policy_total_a.detach().requires_grad_(True)
            labeled_b = policy_total_b.detach().requires_grad_(True)
            dynamic_a = dynamic_mean_a.detach().requires_grad_(True)
            dynamic_b = dynamic_mean_b.detach().requires_grad_(True)
            ref_a = torch.tensor([row["ref_logp_a"] for row in labeled_examples], device=device)
            ref_b = torch.tensor([row["ref_logp_b"] for row in labeled_examples], device=device)
            labels = torch.tensor([row["label"] for row in labeled_examples], device=device)
            dpo_loss, dpo_info = dpo_objective(labeled_a, labeled_b, ref_a, ref_b, labels)
            pe_loss, pe_info = pe_objective(dynamic_a, dynamic_b)
            rollout_vs_sft = None
            if method == "dpo_pe_sft_rollout":
                rollout_is_a = []
                for row in dynamic_pairs:
                    if {row["response_a_source"], row["response_b_source"]} != {
                        "sft",
                        "rollout_0",
                    }:
                        raise ValueError("Round3 SFT+rollout pair source contract changed")
                    rollout_is_a.append(row["response_a_source"] == "rollout_0")
                rollout_vs_sft = rollout_anchor_statistics(
                    dynamic_a,
                    dynamic_b,
                    torch.tensor(rollout_is_a, device=device, dtype=torch.bool),
                )
            loss = joint_dpo_pe_objective(dpo_loss, pe_loss, float(config["method"]["lambda_pe"]))
            coefficients = torch.autograd.grad(loss, (labeled_a, labeled_b, dynamic_a, dynamic_b))
            _backward_pairs(
                policy, labeled_examples, pair_collator,
                coefficients[0], coefficients[1], _zero_like(coefficients[0]), _zero_like(coefficients[1]),
                physical, device, dtype,
            )
            _backward_pairs(
                policy, dynamic_examples, pair_collator,
                _zero_like(coefficients[2]), _zero_like(coefficients[3]), coefficients[2], coefficients[3],
                physical, device, dtype,
            )
            token_counts = rollout_statistics.pop("response_tokens")
            telemetry = {
                "loss_joint": float(loss.detach()),
                **dpo_info,
                "pe": pe_info,
                **({"rollout_vs_sft": rollout_vs_sft} if rollout_vs_sft is not None else {}),
                "rollout": {
                    **rollout_statistics,
                    "response_tokens_mean": float(np.mean(token_counts)),
                    "response_tokens_p50": float(np.percentile(token_counts, 50)),
                    "response_tokens_p95": float(np.percentile(token_counts, 95)),
                    "response_tokens_min": int(min(token_counts)),
                    "response_tokens_max": int(max(token_counts)),
                },
                "truncation_labeled": _pair_truncation(labeled_examples),
                "truncation_dynamic_pairs": _pair_truncation(dynamic_examples),
            }

        if method in DPO_METHODS:
            telemetry["truncation_labeled"] = _pair_truncation(labeled_examples)

        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise FloatingPointError("Round3 produced a non-finite train loss")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(config["training"]["max_grad_norm"]))
        grad_norm_value = float(grad_norm)
        if not math.isfinite(grad_norm_value) or any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in trainable
        ):
            raise FloatingPointError("Round3 produced a non-finite gradient")
        optimizer.step()
        scheduler.step()
        completed_steps = step + 1
        record = {
            "record_type": "train",
            "method_id": method,
            "optimizer_step": completed_steps,
            "objective_step": step,
            "loss": loss_value,
            "grad_norm": grad_norm_value,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "labeled_pairs": batch_labeled,
            "unpaired_units": batch_unpaired,
            "physical_pair_subbatch": physical,
            "optimizer_step_seconds": time.monotonic() - started,
            **telemetry,
        }
        with jsonlines.open(metrics_path, "a") as writer:
            writer.write(record)
        _replace_json(
            state_path,
            {"state": "training", "method_id": method, "step": completed_steps, "planned_steps": total_steps},
        )

        if strong_smoke and completed_steps == 1:
            smoke_checkpoint = save_durable_checkpoint(
                policy,
                tokenizer,
                optimizer,
                scheduler,
                output_dir / "smoke_checkpoint",
                config,
                completed_steps,
                sspo_state.state_dict() if sspo_state is not None else None,
                allow_smoke_step=True,
            )
            _replace_json(
                output_dir / "smoke_complete.json",
                {
                    "status": "succeeded",
                    "method_id": method,
                    "optimizer_steps": 1,
                    "logical_labeled_pairs": batch_labeled,
                    "logical_unpaired_units": batch_unpaired,
                    "checkpoint": str(smoke_checkpoint),
                    "checkpoint_training_state": "optimizer_scheduler_rng_global_step_and_optional_sspo",
                },
            )
        elif completed_steps % 25 == 0:
            durable = save_durable_checkpoint(
                policy,
                tokenizer,
                optimizer,
                scheduler,
                output_dir / "checkpoints",
                config,
                completed_steps,
                sspo_state.state_dict() if sspo_state is not None else None,
            )
            before_hash = _state_hash(sspo_state)
            try:
                selection, selection_truncation = _selection_loss(
                    policy, validation, pair_collator, physical, device, dtype
                )
                valid = True
                invalid_reason = None
            except FloatingPointError as exc:
                selection = None
                valid = False
                invalid_reason = str(exc)
                selection_truncation = None
            after_hash = _state_hash(sspo_state)
            if before_hash != after_hash:
                raise RuntimeError("Common selection mutated Round3 SSPO running state")
            selection_record = {
                "record_type": "selection",
                "method_id": method,
                "checkpoint_step": completed_steps,
                "checkpoint": str(durable),
                "valid": valid,
                "eval_selection_loss": selection,
                "invalid_reason": invalid_reason,
                "truncation": selection_truncation,
                "selection_pairs": 1000,
                "score_type": "dpo_reference_delta_beta_0.1",
                "sspo_state_sha256_before": before_hash,
                "sspo_state_sha256_after": after_hash,
            }
            with jsonlines.open(metrics_path, "a") as writer:
                writer.write(selection_record)
            if valid:
                candidate = (float(selection), completed_steps, str(durable))
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
                    _replace_json(
                        best_path,
                        {
                            "method_id": method,
                            "checkpoint_step": completed_steps,
                            "checkpoint": str(durable),
                            "eval_selection_loss": float(selection),
                            "selection_pairs": 1000,
                            "score_type": "dpo_reference_delta_beta_0.1",
                        },
                    )

    if strong_smoke:
        completion = {
            "state": "succeeded",
            "profile": "strong_smoke",
            "method_id": method,
            "steps": completed_steps,
            "git_commit": config["provenance"]["git_commit"],
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        _replace_json(output_dir / "complete.json", completion)
        _replace_json(state_path, completion)
        return
    if best is None:
        raise RuntimeError("All ten Round3 checkpoints have invalid selection loss")
    checkpoint_count = len(list((output_dir / "checkpoints").glob("step_*")))
    if checkpoint_count != 10:
        raise RuntimeError(f"Round3 must retain exactly ten durable checkpoints, found {checkpoint_count}")
    completion = {
        "state": "succeeded",
        "method_id": method,
        "steps": completed_steps,
        "durable_checkpoints": checkpoint_count,
        "best_checkpoint": best[2],
        "best_step": best[1],
        "best_eval_selection_loss": best[0],
        "git_commit": config["provenance"]["git_commit"],
        "config_sha256": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    _replace_json(output_dir / "complete.json", completion)
    _replace_json(state_path, completion)


def main() -> None:
    _run()


if __name__ == "__main__":
    main()
