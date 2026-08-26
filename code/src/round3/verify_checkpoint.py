"""Load an actual strong-smoke adapter and its complete Round3 training state."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict

import torch

from ..model.model_utils import DTYPES, load_tokenizer, load_trainable_policy
from .checkpoint import load_training_state
from .config import SSPO_METHOD, load_round3_config, validate_round3_config
from .data import PairCollator, PairDataset, SingleCollator, SingleDataset
from .losses import GitHubSSPOState, github_sspo_objective
from .trainer import (
    _backward_pairs,
    _backward_singles,
    _cosine_factor,
    _first_pass_pairs,
    _first_pass_singles,
    _seed_everything,
    _zero_like,
)


LOSS_ABS_TOLERANCE = 1e-7
PARAMETER_MAX_ABS_TOLERANCE = 1e-7
PARAMETER_MAX_REL_TOLERANCE = 1e-6


def _optimizer_and_scheduler(policy, config: Dict[str, Any]):
    trainable = []
    for parameter in policy.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
            trainable.append(parameter)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"]["learning_rate"]),
        betas=(float(config["training"]["adam_beta1"]), float(config["training"]["adam_beta2"])),
        eps=float(config["training"]["adam_epsilon"]),
        weight_decay=float(config["training"]["weight_decay"]),
        foreach=False,
    )
    warmup = int(250 * float(config["training"]["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _cosine_factor(step, 250, warmup)
    )
    return trainable, optimizer, scheduler


def _sspo_next_batch_replay(
    config: Dict[str, Any], checkpoint: Path, labeled_examples, single_examples, tokenizer
) -> Dict[str, Any]:
    device = torch.device("cuda:0")
    dtype = DTYPES[config["model"]["torch_dtype"]]
    policy = load_trainable_policy(config, adapter_checkpoint=str(checkpoint))
    trainable, optimizer, scheduler = _optimizer_and_scheduler(policy, config)
    policy.to(device).train()
    payload = load_training_state(
        checkpoint, config, optimizer, scheduler, require_sspo=True
    )
    sspo_state = GitHubSSPOState.from_state_dict(payload["sspo"])
    pair_collator = PairCollator(tokenizer.pad_token_id)
    single_collator = SingleCollator(tokenizer.pad_token_id)
    physical = int(config["training"]["physical_pair_subbatch"])
    _, _, mean_a, mean_b = _first_pass_pairs(
        policy, labeled_examples, pair_collator, physical, device, dtype
    )
    unpaired_means = _first_pass_singles(
        policy, single_examples, single_collator, physical, device, dtype
    )
    labels = torch.tensor(
        [row["label"] for row in labeled_examples], dtype=torch.bool, device=device
    )
    chosen = torch.where(labels, mean_a, mean_b).detach().requires_grad_(True)
    rejected = torch.where(labels, mean_b, mean_a).detach().requires_grad_(True)
    singles = unpaired_means.detach().requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    loss, _ = github_sspo_objective(
        chosen,
        rejected,
        singles,
        sspo_state,
        global_step=int(payload["global_step"]),
    )
    chosen_coeff, rejected_coeff, single_coeff = torch.autograd.grad(
        loss, (chosen, rejected, singles)
    )
    coeff_a = torch.where(labels, chosen_coeff, rejected_coeff)
    coeff_b = torch.where(labels, rejected_coeff, chosen_coeff)
    _backward_pairs(
        policy,
        labeled_examples,
        pair_collator,
        _zero_like(coeff_a),
        _zero_like(coeff_b),
        coeff_a,
        coeff_b,
        physical,
        device,
        dtype,
    )
    _backward_singles(
        policy,
        single_examples,
        single_collator,
        single_coeff,
        physical,
        device,
        dtype,
    )
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable, float(config["training"]["max_grad_norm"])
    )
    if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
        raise FloatingPointError("Non-finite SSPO checkpoint next-batch replay")
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize(device)
    parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }
    result = {
        "loss": float(loss.detach()),
        "sspo_state": sspo_state.state_dict(),
        "parameters": parameters,
        "scheduler_last_epoch": int(scheduler.last_epoch),
        "next_global_step": int(payload["global_step"]) + 1,
    }
    del policy, trainable, optimizer, scheduler, payload
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _compare_sspo_replays(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    loss_abs = abs(float(left["loss"]) - float(right["loss"]))
    if left["sspo_state"] != right["sspo_state"]:
        raise ValueError("SSPO checkpoint replay changed the restored running-state trajectory")
    if (
        left["scheduler_last_epoch"] != right["scheduler_last_epoch"]
        or left["next_global_step"] != right["next_global_step"]
        or set(left["parameters"]) != set(right["parameters"])
    ):
        raise ValueError("SSPO checkpoint replay state/parameter structure mismatch")
    max_abs = 0.0
    max_rel = 0.0
    for name in left["parameters"]:
        first = left["parameters"][name].float()
        second = right["parameters"][name].float()
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise FloatingPointError("Non-finite trainable parameter in SSPO checkpoint replay")
        difference = (first - second).abs()
        max_abs = max(max_abs, float(difference.max()))
        denominator = torch.maximum(first.abs(), second.abs()).clamp_min(1e-12)
        max_rel = max(max_rel, float((difference / denominator).max()))
    if (
        not math.isfinite(loss_abs)
        or loss_abs > LOSS_ABS_TOLERANCE
        or max_abs > PARAMETER_MAX_ABS_TOLERANCE
        or max_rel > PARAMETER_MAX_REL_TOLERANCE
    ):
        raise ValueError(
            "SSPO checkpoint next-batch replay exceeded the pre-registered numerical tolerance"
        )
    return {
        "status": "verified",
        "loss_abs_difference": loss_abs,
        "loss_abs_tolerance": LOSS_ABS_TOLERANCE,
        "parameter_max_abs_difference": max_abs,
        "parameter_max_abs_tolerance": PARAMETER_MAX_ABS_TOLERANCE,
        "parameter_max_rel_difference": max_rel,
        "parameter_max_rel_tolerance": PARAMETER_MAX_REL_TOLERANCE,
        "running_state_exact": True,
        "scheduler_and_global_step_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_round3_config(args.config)
    validate_round3_config(config)
    # The registered SSPO round-trip compares independent CUDA backward passes.
    # Configure the same deterministic backend contract as formal training
    # before either policy is loaded or any CUDA context is initialized.
    _seed_everything(int(config["training"]["seed"]))
    checkpoint = Path(args.checkpoint).resolve()
    policy = load_trainable_policy(config, adapter_checkpoint=str(checkpoint)).cuda().train()
    trainable, optimizer, scheduler = _optimizer_and_scheduler(policy, config)
    payload = load_training_state(
        checkpoint,
        config,
        optimizer,
        scheduler,
        require_sspo=config["method"]["name"] == SSPO_METHOD,
    )
    sspo = None
    if config["method"]["name"] == SSPO_METHOD:
        sspo = GitHubSSPOState.from_state_dict(payload["sspo"]).state_dict()
    result = {
        "status": "verified",
        "method_id": config["method"]["name"],
        "global_step": int(payload["global_step"]),
        "optimizer_state_entries": len(optimizer.state),
        "scheduler_last_epoch": int(scheduler.last_epoch),
        "rng_state_restored": True,
        "sspo_state": sspo,
        "adapter_load": "peft_trainable_fp32",
    }
    if int(payload["global_step"]) != 1 or not optimizer.state:
        raise ValueError("Round3 strong-smoke checkpoint did not restore step/optimizer state")
    if config["method"]["name"] == SSPO_METHOD:
        del policy, trainable, optimizer, scheduler, payload
        gc.collect()
        torch.cuda.empty_cache()
        tokenizer = load_tokenizer(config["model"]["name_or_path"])
        data_root = Path(config["data"]["data_dir"]).resolve()
        paired = PairDataset(
            data_root / "paired_train_1k.jsonl", tokenizer, require_labels=True
        )
        unpaired = SingleDataset(data_root / "unpaired_train_7k.jsonl", tokenizer)
        labeled_examples = [paired[index] for index in range(4, 8)]
        single_examples = [unpaired[index] for index in range(28, 56)]
        first = _sspo_next_batch_replay(
            config, checkpoint, labeled_examples, single_examples, tokenizer
        )
        second = _sspo_next_batch_replay(
            config, checkpoint, labeled_examples, single_examples, tokenizer
        )
        result["sspo_next_batch_roundtrip"] = _compare_sspo_replays(first, second)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite Round3 checkpoint verification: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
