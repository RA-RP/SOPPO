"""Server-only LoRA trainer for the five preregistered v0.6 configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import jsonlines
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from ..config import (
    apply_distributed_training_profile,
    apply_overrides,
    canonical_json,
    load_config,
    save_config,
    validate_config,
)
from ..data.dataset import (
    PreferenceCollator,
    PreferenceDataset,
    TOKENIZATION_CONTRACT,
    create_dataloader,
    data_file_sha256,
)
from ..model.dpo_loss import (
    DPOLoss,
    model_pair_logps,
    model_pair_mean_logps,
    preference_delta,
    response_token_count,
)
from ..model.model_utils import (
    DTYPES,
    count_parameters,
    load_tokenizer,
    load_trainable_policy,
    save_adapter_checkpoint,
    unwrap_model,
)
from ..model.pe_loss import exact_global_pe_coefficients, pe_surrogate
from ..model.sspo_loss import (
    SSPOThresholdState,
    gather_equal_vector,
    hard_pseudo_response_losses,
    objective_weights,
    pe_pair_probabilities,
    simpo_pair_losses,
)


JOINT_METHODS = {"sspo_hard_exp", "soppo_pe_exp", "soppo_pe_static"}
PE_METHODS = {"soppo_pe_exp", "soppo_pe_static"}
DPO_METHODS = {"dpo10", "dpo100"}


def distributed_initialize() -> Tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def seed_everything(seed: int, rank: int) -> None:
    value = int(seed) + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def synchronized_rank_zero_action(rank: int, description: str, action: Callable[[], None]) -> None:
    """Run a filesystem mutation on rank zero and propagate failure to every rank."""
    error = None
    if rank == 0:
        try:
            action()
        except Exception as exc:  # preserve the useful failure text for every worker
            error = f"{type(exc).__name__}: {exc}"
    payload = [error]
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(f"Rank-zero {description} failed: {payload[0]}")


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def split_cpu_batch(batch: Dict, maximum_size: int) -> List[Dict]:
    """Split a collated CPU batch without changing sample order or normalization."""
    maximum_size = int(maximum_size)
    if maximum_size < 1:
        raise ValueError("backward_subbatch_size_per_device must be positive")
    sample_ids = batch.get("sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ValueError("A collated preference batch must contain non-empty sample_ids")
    size = len(sample_ids)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.ndim == 0 or value.shape[0] != size:
                raise ValueError(f"Batch field has inconsistent leading dimension: {key}")
        elif isinstance(value, list):
            if len(value) != size:
                raise ValueError(f"Batch field has inconsistent list length: {key}")
        else:
            raise TypeError(f"Unsupported collated batch field: {key}={type(value).__name__}")
    pieces = []
    for start in range(0, size, maximum_size):
        stop = min(size, start + maximum_size)
        pieces.append(
            {
                key: value[start:stop]
                for key, value in batch.items()
            }
        )
    return pieces


def split_cpu_batches(batches: Sequence[Dict], maximum_size: int) -> List[Dict]:
    return [piece for batch in batches for piece in split_cpu_batch(batch, maximum_size)]


def batch_sample_count(batches: Sequence[Dict]) -> int:
    return sum(len(batch["sample_ids"]) for batch in batches)


def cache_for(cache_root: str, data_file: Path) -> Path:
    return Path(cache_root).resolve() / f"{data_file.name}.ref.jsonl"


def verify_cache_contract(
    cache_file: Path,
    data_file: Path,
    model_manifest: Path,
    max_length: int,
    enable_thinking: bool,
) -> None:
    manifest_path = cache_file.with_suffix(".manifest.json")
    if not cache_file.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete reference cache: {cache_file}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_model = hashlib.sha256(model_manifest.read_bytes()).hexdigest()
    if manifest.get("model_manifest_sha256") != expected_model:
        raise ValueError(f"Reference cache/model mismatch: {cache_file}")
    if manifest.get("input_sha256") != data_file_sha256(data_file):
        raise ValueError(f"Reference cache/data mismatch: {cache_file}")
    if manifest.get("cache_sha256") != data_file_sha256(cache_file):
        raise ValueError(f"Reference cache checksum mismatch: {cache_file}")
    if int(manifest.get("max_length", -1)) != int(max_length):
        raise ValueError(f"Reference cache max-length mismatch: {cache_file}")
    if bool(manifest.get("enable_thinking")) != bool(enable_thinking):
        raise ValueError(f"Reference cache thinking-mode mismatch: {cache_file}")
    if manifest.get("response_only") is not True:
        raise ValueError(f"Reference cache is not response-only: {cache_file}")
    if manifest.get("tokenization_contract") != TOKENIZATION_CONTRACT:
        raise ValueError(f"Reference cache tokenization-contract mismatch: {cache_file}")


class PatternBatchSampler:
    """Batch distributed-sampler indices in a repeating, fail-closed size pattern."""

    def __init__(self, sampler: DistributedSampler, pattern: Sequence[int]):
        self.sampler = sampler
        self.pattern = tuple(int(value) for value in pattern)
        if not self.pattern or any(value < 1 for value in self.pattern):
            raise ValueError("Every joint unlabeled microbatch size must be positive")

    def __iter__(self):
        iterator = iter(self.sampler)
        while True:
            cycle = []
            try:
                for size in self.pattern:
                    cycle.append([next(iterator) for _ in range(size)])
            except StopIteration:
                return
            yield from cycle

    def __len__(self) -> int:
        return (len(self.sampler) // sum(self.pattern)) * len(self.pattern)


def make_dataset(
    data_file: Path,
    tokenizer,
    config: Dict,
    cache_file: Optional[Path],
    require_labels: bool,
) -> PreferenceDataset:
    return PreferenceDataset(
        str(data_file),
        tokenizer,
        max_length=int(config["model"]["max_seq_len"]),
        reference_cache_path=str(cache_file) if cache_file else None,
        require_labels=require_labels,
        enable_thinking=bool(config["model"]["enable_thinking"]),
    )


def regular_loader(
    dataset: PreferenceDataset,
    tokenizer,
    config: Dict,
    rank: int,
    world_size: int,
    batch_size: int,
    shuffle: bool,
):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=shuffle)
    loader = create_dataloader(
        dataset,
        int(batch_size),
        PreferenceCollator(tokenizer.pad_token_id),
        shuffle=shuffle,
        num_workers=int(config["data"]["num_workers"]),
        sampler=sampler,
    )
    return loader, sampler


def patterned_loader(
    dataset: PreferenceDataset,
    tokenizer,
    config: Dict,
    rank: int,
    world_size: int,
):
    if world_size not in {1, 2, 4}:
        raise ValueError("The v0.6 joint population contract requires 1, 2, or 4 ranks")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    batch_sampler = PatternBatchSampler(
        sampler, config["training"]["joint_unlabeled_microbatch_pattern"]
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=PreferenceCollator(tokenizer.pad_token_id),
        pin_memory=True,
    )
    return loader, sampler


def infinite_batches(loader, sampler) -> Iterator[Dict]:
    cycle_index = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(cycle_index)
        yield from loader
        cycle_index += 1


def evaluate_validation(
    model,
    loader,
    device: torch.device,
    method: str,
    dpo_beta: float,
    simpo_beta: float,
    dtype: torch.dtype,
) -> Dict[str, float]:
    model.eval()
    totals = torch.zeros(7, dtype=torch.float64, device=device)
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                if method in DPO_METHODS:
                    policy_a, policy_b = model_pair_logps(model, batch)
                    delta = preference_delta(
                        policy_a,
                        policy_b,
                        batch["ref_logp_a"],
                        batch["ref_logp_b"],
                        dpo_beta,
                    )
                    raw_delta = float(simpo_beta) * (
                        policy_a / response_token_count(batch["loss_mask_a"])
                        - policy_b / response_token_count(batch["loss_mask_b"])
                    )
                else:
                    mean_a, mean_b = model_pair_mean_logps(model, batch)
                    delta = float(simpo_beta) * (mean_a - mean_b)
                    raw_delta = delta
            probs = torch.sigmoid(delta.float())
            labels = batch["labels"]
            totals[0] += ((probs > 0.5) == labels.bool()).sum()
            totals[1] += ((probs - labels.float()) ** 2).sum()
            totals[2] += labels.numel()
            totals[3] += torch.isfinite(delta).sum()
            raw_probs = torch.sigmoid(raw_delta.float())
            totals[4] += ((raw_probs > 0.5) == labels.bool()).sum()
            totals[5] += ((raw_probs - labels.float()) ** 2).sum()
            totals[6] += torch.isfinite(raw_delta).sum()
    if dist.is_initialized():
        dist.all_reduce(totals)
    count = float(totals[2])
    if count == 0 or float(totals[3]) != count or float(totals[6]) != count:
        raise ValueError("Validation produced missing or non-finite scores")
    model.train()
    result = {
        "val_accuracy": float(totals[0] / count),
        "val_brier": float(totals[1] / count),
        "val_samples": int(count),
        "score_type": "dpo_reference_delta" if method in DPO_METHODS else "simpo_mean_logp_delta",
    }
    if method in DPO_METHODS:
        result.update(
            {
                "raw_mean_logp_val_accuracy": float(totals[4] / count),
                "raw_mean_logp_val_brier": float(totals[5] / count),
                "raw_mean_logp_score_type": "simpo_mean_logp_delta_margin_free",
            }
        )
    return result


def write_json(path: Path, payload: Dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def cosine_factor(step: int, planned_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(1, planned_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def maybe_no_sync(model, synchronize: bool):
    if isinstance(model, DistributedDataParallel) and not synchronize:
        return model.no_sync()
    from contextlib import nullcontext

    return nullcontext()


def collect_joint_first_pass(
    policy,
    labeled_batches: List[Dict],
    unlabeled_batches: List[Dict],
    device: torch.device,
    dtype: torch.dtype,
    include_labeled: bool,
):
    labeled_scores = []
    unlabeled_scores = []
    policy.eval()
    with torch.no_grad():
        if include_labeled:
            for cpu_batch in labeled_batches:
                batch = move_batch(cpu_batch, device)
                with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                    mean_a, mean_b = model_pair_mean_logps(policy, batch)
                labeled_scores.append((mean_a.float(), mean_b.float(), batch["labels"]))
        for cpu_batch in unlabeled_batches:
            batch = move_batch(cpu_batch, device)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                mean_a, mean_b = model_pair_mean_logps(policy, batch)
            unlabeled_scores.append((mean_a.float(), mean_b.float()))
    policy.train()
    return labeled_scores, unlabeled_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--init-checkpoint")
    args = parser.parse_args()
    rank, local_rank, world_size, device = distributed_initialize()
    config = apply_overrides(load_config(args.config), args.set)
    if not bool(config["training"].get("smoke_mode", False)):
        config = apply_distributed_training_profile(config, world_size)
    validate_config(config, world_size=world_size)
    seed_everything(int(config["training"]["seed"]), rank)

    output_dir = Path(config["output"]["run_dir"]).resolve()
    def prepare_output() -> None:
        if output_dir.exists():
            raise FileExistsError(f"Refuse to overwrite run directory: {output_dir}")
        output_dir.mkdir(parents=True)
        (output_dir / "checkpoints").mkdir()
        (output_dir / "logs").mkdir()
        save_config(config, output_dir / "config.resolved.yaml")
        write_json(output_dir / "state.json", {"status": "running", "step": 0})

    synchronized_rank_zero_action(rank, "run-directory initialization", prepare_output)

    model_cfg = config["model"]
    training = config["training"]
    method_cfg = config["method"]
    method = method_cfg["name"]
    tokenizer = load_tokenizer(model_cfg["name_or_path"])
    policy = load_trainable_policy(config, adapter_checkpoint=args.init_checkpoint).to(device)
    if world_size > 1:
        policy = DistributedDataParallel(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    trainable_parameters, total_parameters = count_parameters(policy)

    data_dir = Path(config["data"]["data_dir"]).resolve()
    train_file = (
        Path(config["data"].get("train_file") or data_dir / "oracle_train.private.jsonl")
        if method == "dpo100"
        else data_dir / "labeled_train.jsonl"
    )
    val_file = data_dir / "labeled_val.jsonl"
    unlabeled_file = data_dir / "unlabeled_train.jsonl"
    model_manifest = Path(model_cfg["manifest_path"])
    cache_root = config["data"].get("reference_cache")
    train_cache = val_cache = None
    if method in DPO_METHODS:
        train_cache = cache_for(cache_root, train_file)
        val_cache = cache_for(cache_root, val_file)
        verify_cache_contract(
            train_cache,
            train_file,
            model_manifest,
            int(model_cfg["max_seq_len"]),
            bool(model_cfg["enable_thinking"]),
        )
        verify_cache_contract(
            val_cache,
            val_file,
            model_manifest,
            int(model_cfg["max_seq_len"]),
            bool(model_cfg["enable_thinking"]),
        )

    train_dataset = make_dataset(train_file, tokenizer, config, train_cache, True)
    val_dataset = make_dataset(val_file, tokenizer, config, val_cache, True)
    val_loader, _ = regular_loader(
        val_dataset,
        tokenizer,
        config,
        rank,
        world_size,
        int(training["dpo_batch_size_per_device"]),
        False,
    )

    train_loader = train_sampler = unlabeled_loader = unlabeled_sampler = None
    if method in DPO_METHODS:
        train_loader, train_sampler = regular_loader(
            train_dataset,
            tokenizer,
            config,
            rank,
            world_size,
            int(training["dpo_batch_size_per_device"]),
            True,
        )
        steps_per_epoch = len(train_loader) // int(training["gradient_accumulation_steps"])
    else:
        train_loader, train_sampler = regular_loader(
            train_dataset,
            tokenizer,
            config,
            rank,
            world_size,
            int(training["joint_labeled_batch_size_per_device"]),
            True,
        )
        unlabeled_dataset = make_dataset(unlabeled_file, tokenizer, config, None, False)
        unlabeled_loader, unlabeled_sampler = patterned_loader(
            unlabeled_dataset, tokenizer, config, rank, world_size
        )
        steps_per_epoch = len(unlabeled_loader) // int(training["gradient_accumulation_steps"])

    planned_steps = int(training["epochs"]) * steps_per_epoch
    if training.get("max_steps") is not None:
        planned_steps = min(planned_steps, int(training["max_steps"]))
    if planned_steps < 1:
        raise ValueError("Training plan contains zero optimizer steps")

    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if training["optimizer"] != "adamw":
        raise ValueError("v0.6 formal contract supports AdamW only")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["lr"]),
        betas=(float(training["adam_beta1"]), float(training["adam_beta2"])),
        weight_decay=float(training["weight_decay"]),
    )
    warmup_steps = int(planned_steps * float(training["warmup_ratio"]))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_factor(step, planned_steps, warmup_steps)
    )
    dtype = DTYPES[model_cfg["torch_dtype"]]
    dpo_objective = DPOLoss(float(training["dpo_beta"]))
    threshold_state = SSPOThresholdState(
        momentum=float(method_cfg["ema_momentum"]),
        prior=float(method_cfg["pseudo_prior"]),
        epsilon=float(method_cfg["reward_std_epsilon"]),
        grid_points=int(method_cfg["kde_grid_points"]),
    ) if method == "sspo_hard_exp" else None

    metrics_path = output_dir / "logs" / "metrics.jsonl"
    global_step = 0
    best_accuracy = -1.0
    best_brier = float("inf")
    accumulation = int(training["gradient_accumulation_steps"])
    backward_subbatch_size = int(training["backward_subbatch_size_per_device"])
    labeled_iterator = iter(infinite_batches(train_loader, train_sampler)) if method in JOINT_METHODS else None

    if method == "dpo10":
        with unwrap_model(policy).disable_adapter():
            initial_validation = evaluate_validation(
                policy,
                val_loader,
                device,
                method,
                float(training["dpo_beta"]),
                float(training["simpo_beta"]),
                dtype,
            )

        def write_initial_validation() -> None:
            write_json(
                output_dir / "initial_validation.json",
                {
                    **initial_validation,
                    "checkpoint": "frozen_qwen3_base_before_training",
                    "headroom_score_type": "simpo_mean_logp_delta_margin_free",
                },
            )

        synchronized_rank_zero_action(
            rank, "DPO-10 frozen-base validation write", write_initial_validation
        )

    for epoch in range(int(training["epochs"])):
        if method in DPO_METHODS:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            source_iterator = iter(train_loader)
            optimizer_steps_this_epoch = len(train_loader) // accumulation
        else:
            unlabeled_sampler.set_epoch(epoch)
            source_iterator = iter(unlabeled_loader)
            optimizer_steps_this_epoch = len(unlabeled_loader) // accumulation

        for _ in range(optimizer_steps_this_epoch):
            if global_step >= planned_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            running_supervised = 0.0
            running_aux = 0.0
            aux_info: Dict = {}
            if method in DPO_METHODS:
                logical_batches = [next(source_iterator) for _micro in range(accumulation)]
                backward_batches = split_cpu_batches(logical_batches, backward_subbatch_size)
                local_dpo_pairs = int(training["dpo_batch_size_per_device"]) * accumulation
                if batch_sample_count(backward_batches) != local_dpo_pairs:
                    raise ValueError("DPO logical batch was not preserved during backward subbatching")
                for index, cpu_batch in enumerate(backward_batches):
                    synchronize = index == len(backward_batches) - 1
                    with maybe_no_sync(policy, synchronize):
                        batch = move_batch(cpu_batch, device)
                        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                            policy_a, policy_b = model_pair_logps(policy, batch)
                            losses, _ = dpo_objective.per_sample(
                                policy_a,
                                policy_b,
                                batch["ref_logp_a"],
                                batch["ref_logp_b"],
                                batch["labels"],
                            )
                            contribution = losses.sum() / local_dpo_pairs
                        if not torch.isfinite(contribution):
                            raise FloatingPointError(f"Non-finite DPO loss at step {global_step}")
                        contribution.backward()
                        running_supervised += float(losses.sum().detach()) / local_dpo_pairs
                supervised_weight, aux_weight = 1.0, 0.0
            else:
                unlabeled_batches = [next(source_iterator) for _micro in range(accumulation)]
                labeled_steps = set(int(value) for value in training["joint_labeled_microsteps"])
                labeled_batches = [next(labeled_iterator) for index in range(accumulation) if index in labeled_steps]
                first_labeled, first_unlabeled = collect_joint_first_pass(
                    policy,
                    labeled_batches,
                    unlabeled_batches,
                    device,
                    dtype,
                    method == "sspo_hard_exp",
                )
                supervised_weight, aux_weight = objective_weights(method_cfg, global_step)
                pe_coefficients = None
                pe_actual = 0.0
                if method in PE_METHODS:
                    local_probabilities = torch.cat([
                        pe_pair_probabilities(a, b, float(training["simpo_beta"]))
                        for a, b in first_unlabeled
                    ])
                    expected = int(training["joint_unlabeled_global_batch_size"]) // world_size
                    if local_probabilities.numel() != expected:
                        raise ValueError("PE population is incomplete; patterned batch contract failed")
                    pe_coefficients, pe_actual, aux_info = exact_global_pe_coefficients(
                        local_probabilities,
                        float(training["epsilon"]),
                        method_cfg["pe_distance"],
                        bool(method_cfg["detach_denominator"]),
                    )
                else:
                    local_winning = []
                    local_losing = []
                    local_all = []
                    for mean_a, mean_b, labels in first_labeled:
                        reward_a = float(training["simpo_beta"]) * mean_a
                        reward_b = float(training["simpo_beta"]) * mean_b
                        local_winning.append(torch.where(labels.bool(), reward_a, reward_b))
                        local_losing.append(torch.where(labels.bool(), reward_b, reward_a))
                        local_all.extend([reward_a, reward_b])
                    for mean_a, mean_b in first_unlabeled:
                        local_all.extend([
                            float(training["simpo_beta"]) * mean_a,
                            float(training["simpo_beta"]) * mean_b,
                        ])
                    global_winning = gather_equal_vector(torch.cat(local_winning))
                    global_losing = gather_equal_vector(torch.cat(local_losing))
                    global_all = gather_equal_vector(torch.cat(local_all))
                    aux_info = threshold_state.update(global_all, global_winning, global_losing)

                local_labeled_total = int(training["joint_labeled_global_batch_size"]) // world_size
                local_unlabeled_pairs = int(training["joint_unlabeled_global_batch_size"]) // world_size
                labeled_backward_batches = split_cpu_batches(
                    labeled_batches, backward_subbatch_size
                )
                unlabeled_backward_batches = split_cpu_batches(
                    unlabeled_batches, backward_subbatch_size
                )
                if batch_sample_count(labeled_backward_batches) != local_labeled_total:
                    raise ValueError("Joint labeled population changed during backward subbatching")
                if batch_sample_count(unlabeled_backward_batches) != local_unlabeled_pairs:
                    raise ValueError("Joint unlabeled population changed during backward subbatching")
                backward_count = len(labeled_backward_batches) + len(unlabeled_backward_batches)
                backward_index = 0
                for cpu_batch in labeled_backward_batches:
                    with maybe_no_sync(policy, backward_index == backward_count - 1):
                        batch = move_batch(cpu_batch, device)
                        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                            mean_a, mean_b = model_pair_mean_logps(policy, batch)
                            losses, _ = simpo_pair_losses(
                                mean_a,
                                mean_b,
                                batch["labels"],
                                float(training["simpo_beta"]),
                                float(training["simpo_margin"]),
                            )
                            contribution = supervised_weight * losses.sum() / local_labeled_total
                        if not torch.isfinite(contribution):
                            raise FloatingPointError(f"Non-finite SimPO loss at step {global_step}")
                        contribution.backward()
                        running_supervised += float(losses.sum().detach()) / local_labeled_total
                    backward_index += 1

                coefficient_offset = 0
                pseudo_positive_weighted = 0.0
                local_unlabeled_responses = 2 * local_unlabeled_pairs
                for cpu_batch in unlabeled_backward_batches:
                    with maybe_no_sync(policy, backward_index == backward_count - 1):
                        batch = move_batch(cpu_batch, device)
                        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                            mean_a, mean_b = model_pair_mean_logps(policy, batch)
                            if method in PE_METHODS:
                                probabilities = pe_pair_probabilities(
                                    mean_a, mean_b, float(training["simpo_beta"])
                                )
                                count = probabilities.numel()
                                coefficients = pe_coefficients[coefficient_offset : coefficient_offset + count]
                                coefficient_offset += count
                                contribution = aux_weight * pe_surrogate(
                                    probabilities, coefficients, world_size
                                )
                                running_aux = pe_actual
                            else:
                                rewards = float(training["simpo_beta"]) * torch.cat([mean_a, mean_b])
                                losses, hard_info = hard_pseudo_response_losses(rewards, threshold_state)
                                contribution = aux_weight * losses.sum() / local_unlabeled_responses
                                running_aux += float(losses.sum().detach()) / local_unlabeled_responses
                                pseudo_positive_weighted += (
                                    hard_info["pseudo_positive_rate"] * rewards.numel() / local_unlabeled_responses
                                )
                        if not torch.isfinite(contribution):
                            raise FloatingPointError(f"Non-finite auxiliary loss at step {global_step}")
                        contribution.backward()
                    backward_index += 1
                if method in PE_METHODS and coefficient_offset != pe_coefficients.numel():
                    raise ValueError("PE second pass did not consume every population coefficient")
                if method == "sspo_hard_exp":
                    aux_info["pseudo_positive_rate"] = pseudo_positive_weighted

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(training["max_grad_norm"]))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at step {global_step}")
            optimizer.step()
            lr_scheduler.step()
            global_step += 1

            record = {
                "step": global_step,
                "epoch": epoch,
                "method": method,
                "loss_supervised": running_supervised,
                "loss_aux": running_aux,
                "supervised_weight": supervised_weight,
                "aux_weight": aux_weight,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm),
                "global_batch_size": int(training["global_batch_size"]),
                "backward_subbatch_size_per_device": backward_subbatch_size,
                "joint_labeled_global_batch_size": (
                    int(training["joint_labeled_global_batch_size"]) if method in JOINT_METHODS else None
                ),
                "joint_unlabeled_global_batch_size": (
                    int(training["joint_unlabeled_global_batch_size"]) if method in JOINT_METHODS else None
                ),
                "aux": aux_info or None,
            }
            if rank == 0:
                with jsonlines.open(metrics_path, "a") as writer:
                    writer.write(record)
                write_json(output_dir / "state.json", {"status": "running", "step": global_step})

            should_evaluate = global_step % int(training["eval_steps"]) == 0 or global_step == planned_steps
            if should_evaluate:
                validation = evaluate_validation(
                    policy,
                    val_loader,
                    device,
                    method,
                    float(training["dpo_beta"]),
                    float(training["simpo_beta"]),
                    dtype,
                )
                if rank == 0:
                    with jsonlines.open(metrics_path, "a") as writer:
                        writer.write({"step": global_step, **validation})
                    accuracy = float(validation["val_accuracy"])
                    brier = float(validation["val_brier"])
                    if accuracy > best_accuracy or (accuracy == best_accuracy and brier < best_brier):
                        best_accuracy, best_brier = accuracy, brier
                        write_json(
                            output_dir / "best.json",
                            {"step": global_step, **validation, "selection_split": "validation"},
                        )

            should_save = bool(config["output"].get("save_checkpoints", True)) and (
                global_step % int(training["save_steps"]) == 0 or global_step == planned_steps
            )
            if should_save:
                checkpoint = output_dir / "checkpoints" / f"step_{global_step:06d}"
                synchronized_rank_zero_action(
                    rank,
                    f"checkpoint save at step {global_step}",
                    lambda: save_adapter_checkpoint(
                        policy,
                        tokenizer,
                        str(checkpoint),
                        config,
                        0,
                        training_state={
                            "step": global_step,
                            "threshold": threshold_state.state_dict() if threshold_state else None,
                        },
                    ),
                )
            if global_step >= planned_steps:
                break
        if global_step >= planned_steps:
            break

    memory = torch.tensor(
        [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)

    def write_completion() -> None:
        completion = {
            "status": "succeeded",
            "steps": global_step,
            "method": method,
            "config_sha256": hashlib.sha256(canonical_json(config).encode()).hexdigest(),
            "best_val_accuracy": best_accuracy,
            "best_val_brier": best_brier,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "checkpoint_format": "peft_lora_adapter",
            "peak_memory_allocated_bytes": int(memory[0]),
            "peak_memory_reserved_bytes": int(memory[1]),
        }
        write_json(output_dir / "complete.json", completion)
        write_json(output_dir / "state.json", completion)

    synchronized_rank_zero_action(rank, "completion write", write_completion)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
