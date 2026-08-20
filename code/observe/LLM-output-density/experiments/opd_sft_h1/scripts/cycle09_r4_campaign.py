#!/usr/bin/env python3
"""Cycle 09 Round 4 window-v2 generation and three-layer GPU campaign."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import cycle09_r4_common as c

GETSLICE = c.REPO / "GetSlice"
if str(GETSLICE) not in sys.path:
    sys.path.insert(0, str(GETSLICE))

from utils.profiling_utils import _gram_to_svdllm_scaling_diag_matrix  # noqa: E402


def parse_ints(value: str, default: tuple[int, ...]) -> list[int]:
    if not value:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_names(value: str, default: tuple[str, ...]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def load_model(path: Path, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.eval().to(device)
    return model


def unload_model(model) -> None:
    if model is None:
        return
    model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def complete_corpus(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    try:
        rows = c.read_jsonl(path)
    except Exception:
        return False
    return (
        len(rows) == expected
        and all("full_token_ids" in row and "generation_token_ids" in row for row in rows)
    )


def formatted_prompt(tokenizer, prompt: str, instruction: str, domain: str) -> str:
    if domain == "bos":
        token = tokenizer.bos_token or tokenizer.eos_token
        if not token:
            raise ValueError("BOS probe requires a BOS or EOS tokenizer token")
        return token
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt + instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generation_requests(
    tokenizer,
    bank: list[dict[str, str]],
    domain: str,
    batch_seed: int,
) -> tuple[list[str], list[Any]]:
    from vllm import SamplingParams

    prompts = [
        formatted_prompt(tokenizer, item["prompt"], item["instruction"], domain)
        for item in bank
    ]
    params = [
        SamplingParams(
            temperature=c.TEMPERATURE,
            top_p=c.TOP_P,
            max_tokens=c.MAX_NEW_TOKENS,
            seed=c.stable_seed(batch_seed, domain, item["sample_id"]),
        )
        for item in bank
    ]
    return prompts, params


def generate_targets(
    *,
    model_path: Path,
    targets: list[tuple[Path, str, str, int, str | None, int | None]],
    banks: dict[str, list[dict[str, str]]],
    n_samples: int,
    max_new_tokens: int,
    max_model_len: int,
    gpu_mem: float,
) -> None:
    pending = [
        item for item in targets if not complete_corpus(item[0], n_samples)
    ]
    if not pending:
        return

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    try:
        for target, probe_type, domain, batch_seed, arm, step in pending:
            bank = banks[domain][:n_samples]
            prompts = [
                formatted_prompt(tokenizer, item["prompt"], item["instruction"], domain)
                for item in bank
            ]
            sampling = [
                SamplingParams(
                    temperature=c.TEMPERATURE,
                    top_p=c.TOP_P,
                    max_tokens=max_new_tokens,
                    seed=c.stable_seed(batch_seed, probe_type, domain, item["sample_id"]),
                )
                for item in bank
            ]
            generated = llm.generate(prompts, sampling)
            rows = []
            for item, request in zip(bank, generated):
                completion = request.outputs[0]
                prompt_ids = [int(value) for value in request.prompt_token_ids]
                generation_ids = [int(value) for value in completion.token_ids]
                rows.append(
                    {
                        "sample_id": item["sample_id"],
                        "probe_type": probe_type,
                        "domain": domain,
                        "source_kind": (
                            "base_generation" if probe_type == "S"
                            else "checkpoint_training_signal_rollout" if probe_type == "X"
                            else "checkpoint_nontraining_generation"
                        ),
                        "arm": arm,
                        "step": step,
                        "generation_seed": batch_seed,
                        "per_request_seed": c.stable_seed(
                            batch_seed, probe_type, domain, item["sample_id"]
                        ),
                        "prompt_text": item["prompt"] + item["instruction"],
                        "formatted_prompt": prompts[len(rows)],
                        "generation_text": completion.text,
                        "prompt_token_ids": prompt_ids,
                        "generation_token_ids": generation_ids,
                        "full_token_ids": prompt_ids + generation_ids,
                        "eligible_start": len(prompt_ids),
                        "eligible_end": len(prompt_ids) + len(generation_ids),
                        "finish_reason": completion.finish_reason,
                        "generation_config": {
                            "max_new_tokens": max_new_tokens,
                            "temperature": c.TEMPERATURE,
                            "top_p": c.TOP_P,
                            "batch_seed": batch_seed,
                        },
                    }
                )
            c.write_jsonl_atomic(target, rows)
            print(f"[Generate] {target} n={len(rows)}", flush=True)
    finally:
        del llm, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_all(args, tokenizer) -> None:
    banks = c.prompt_banks(args.n_samples)
    seeds = args.generation_seeds

    base_targets = []
    for seed in seeds:
        for domain in ("math", "ood", "general", "bos"):
            base_targets.append(
                (
                    c.generated_corpus_path("S", domain, seed, run_root=args.run_root),
                    "S",
                    domain,
                    seed,
                    "base",
                    0,
                )
            )
        base_targets.append(
            (
                c.generated_corpus_path(
                    "X", "math", seed, "opd", 0, args.run_root
                ),
                "X",
                "math",
                seed,
                "opd",
                0,
            )
        )
        for arm in c.ARMS:
            for domain in ("ood", "general", "bos"):
                base_targets.append(
                    (
                        c.generated_corpus_path(
                            "H", domain, seed, arm, 0, args.run_root
                        ),
                        "H",
                        domain,
                        seed,
                        arm,
                        0,
                    )
                )
    generate_targets(
        model_path=c.BASE_MODEL,
        targets=base_targets,
        banks=banks,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        gpu_mem=args.gpu_mem,
    )

    for arm in args.arms:
        for step in args.steps:
            if step == 0:
                continue
            targets = []
            for seed in seeds:
                if arm == "opd":
                    targets.append(
                        (
                            c.generated_corpus_path(
                                "X", "math", seed, arm, step, args.run_root
                            ),
                            "X",
                            "math",
                            seed,
                            arm,
                            step,
                        )
                    )
                for domain in ("ood", "general", "bos"):
                    targets.append(
                        (
                            c.generated_corpus_path(
                                "H", domain, seed, arm, step, args.run_root
                            ),
                            "H",
                            domain,
                            seed,
                            arm,
                            step,
                        )
                    )
            generate_targets(
                model_path=c.model_path(arm, step),
                targets=targets,
                banks=banks,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                max_model_len=args.max_model_len,
                gpu_mem=args.gpu_mem,
            )


def module_at(model, layer: int, name: str):
    value = model.model.layers[int(layer)]
    for part in name.split("."):
        value = getattr(value, part)
    return value


def _sample_bin_weights(sample: c.PreparedSample, device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    length = sample.input_ids.shape[1]
    for bin_name in ("early", "mid", "late"):
        windows = [window for window in sample.windows if window.position_bin == bin_name]
        if not windows:
            continue
        weights = torch.zeros(length, dtype=torch.float32, device=device)
        for window in windows:
            weights[window.start:window.end] += 1.0 / (
                len(windows) * max(window.end - window.start, 1)
            )
        result[bin_name] = weights
    return result


class _ProbeForwardComplete(RuntimeError):
    """Internal control flow used after the highest requested layer is captured."""


def _probe_pad_token_id(model) -> int:
    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "eos_token_id", None)
    if isinstance(pad_token_id, (list, tuple)):
        pad_token_id = pad_token_id[0] if pad_token_id else None
    return int(pad_token_id) if pad_token_id is not None else 0


def _probe_batches(
    samples: list[c.PreparedSample],
    forward_batch_size: int,
    max_batch_tokens: int | None,
):
    """Yield stable-order batches bounded by sample count and padded tokens."""
    if forward_batch_size < 1:
        raise ValueError("forward_batch_size must be positive")
    if max_batch_tokens is not None and max_batch_tokens < 1:
        raise ValueError("max_batch_tokens must be positive when provided")

    pending: list[c.PreparedSample] = []
    pending_max_length = 0
    for sample in samples:
        sample_length = int(sample.input_ids.shape[1])
        candidate_max = max(pending_max_length, sample_length)
        candidate_size = len(pending) + 1
        exceeds_samples = candidate_size > forward_batch_size
        exceeds_tokens = (
            max_batch_tokens is not None
            and candidate_max * candidate_size > max_batch_tokens
        )
        if pending and (exceeds_samples or exceeds_tokens):
            yield pending
            pending = []
            pending_max_length = 0
        pending.append(sample)
        pending_max_length = max(pending_max_length, sample_length)
    if pending:
        yield pending


def _pad_probe_batch(model, batch: list[c.PreparedSample], device: torch.device):
    width = max(int(sample.input_ids.shape[1]) for sample in batch)
    input_ids = torch.full(
        (len(batch), width),
        _probe_pad_token_id(model),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long, device=device)
    token_weights = torch.zeros((len(batch), width), dtype=torch.float32, device=device)
    bin_weights: list[dict[str, torch.Tensor]] = []
    for row, sample in enumerate(batch):
        length = int(sample.input_ids.shape[1])
        input_ids[row, :length] = sample.input_ids[0].to(device)
        attention_mask[row, :length] = sample.attention_mask[0].to(device)
        token_weights[row, :length] = sample.token_weights.to(device)
        sample_bins = _sample_bin_weights(sample, device)
        bin_weights.append(
            {
                name: torch.nn.functional.pad(weights, (0, width - length))
                for name, weights in sample_bins.items()
            }
        )
    return input_ids, attention_mask, token_weights, bin_weights


@torch.no_grad()
def collect_profile(
    model,
    samples: list[c.PreparedSample],
    layers: list[int],
    device: str,
    *,
    keep_factors: bool,
    keep_residual_samples: bool,
    keep_input_sample_means: bool = False,
    factor_layers: tuple[int, ...] = (18,),
    forward_batch_size: int = 1,
    max_batch_tokens: int | None = None,
    early_stop: bool = False,
) -> dict[str, Any]:
    dev = torch.device(device)
    n_samples = len(samples)
    if not layers:
        raise ValueError("at least one probe layer is required")
    grams: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in layers}
    residual_second: dict[int, torch.Tensor] = {}
    residual_mean: dict[int, torch.Tensor] = {}
    position_second: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in layers}
    position_mean: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in layers}
    position_counts = {name: 0 for name in ("early", "mid", "late")}
    sample_factors: list[dict[int, dict[str, torch.Tensor]]] = []
    input_sample_means: list[dict[int, dict[str, torch.Tensor]]] = []
    residual_samples: list[dict[int, torch.Tensor]] = []
    residual_sample_means: list[dict[int, torch.Tensor]] = []
    active: dict[str, Any] = {}

    handles = []
    stop_layer = max(layers)

    def make_linear_hook(layer: int, group: str):
        def hook(_module, inputs):
            raw = inputs[0].detach()
            if raw.dim() == 2:
                raw = raw.unsqueeze(0)
            weights = active["weights"]
            if raw.shape[:2] != weights.shape:
                raise RuntimeError(
                    f"probe hook shape drift for layer={layer} group={group}: "
                    f"raw={tuple(raw.shape)} weights={tuple(weights.shape)}"
                )
            factors = []
            for row in range(raw.shape[0]):
                index = weights[row] > 0
                selected = raw[row, index].float()
                selected_weights = weights[row, index]
                factor = selected * torch.sqrt(selected_weights).unsqueeze(1)
                factors.append(factor)
                if keep_input_sample_means and layer in factor_layers:
                    active["input_sample_means"][row][layer][group] = (
                        selected * selected_weights.unsqueeze(1)
                    ).sum(dim=0).to(dtype=torch.float32, device="cpu").contiguous()
                if keep_factors and layer in factor_layers:
                    active["factors"][row][layer][group] = factor.to(
                        dtype=torch.float16, device="cpu"
                    ).contiguous()
            combined = torch.cat(factors, dim=0)
            if group not in grams[layer]:
                width = combined.shape[1]
                grams[layer][group] = torch.zeros(
                    (width, width), dtype=torch.float32, device=dev
                )
            grams[layer][group].add_(combined.T @ combined, alpha=1.0 / n_samples)
        return hook

    def make_layer_hook(layer: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            weights = active["weights"]
            if hidden.shape[:2] != weights.shape:
                raise RuntimeError(
                    f"probe residual shape drift for layer={layer}: "
                    f"hidden={tuple(hidden.shape)} weights={tuple(weights.shape)}"
                )
            factors = []
            means = []
            position_factors = {name: [] for name in ("early", "mid", "late")}
            position_means = {name: [] for name in ("early", "mid", "late")}
            for row in range(hidden.shape[0]):
                index = weights[row] > 0
                selected = hidden[row, index].float()
                selected_weights = weights[row, index]
                factor = selected * torch.sqrt(selected_weights).unsqueeze(1)
                mean = (selected * selected_weights.unsqueeze(1)).sum(dim=0)
                factors.append(factor)
                means.append(mean)
                if keep_residual_samples:
                    active["residual"][row][layer] = factor.to(
                        dtype=torch.float16, device="cpu"
                    ).contiguous()
                    active["residual_sample_means"][row][layer] = mean.to(
                        dtype=torch.float32, device="cpu"
                    ).contiguous()

                for bin_name, bin_weights in active["bin_weights"][row].items():
                    bin_index = bin_weights > 0
                    bin_selected = hidden[row, bin_index].float()
                    selected_bin_weights = bin_weights[bin_index]
                    bin_factor = bin_selected * torch.sqrt(
                        selected_bin_weights
                    ).unsqueeze(1)
                    bin_mean = (
                        bin_selected * selected_bin_weights.unsqueeze(1)
                    ).sum(dim=0)
                    position_factors[bin_name].append(bin_factor)
                    position_means[bin_name].append(bin_mean)

            combined = torch.cat(factors, dim=0)
            if layer not in residual_second:
                width = combined.shape[1]
                residual_second[layer] = torch.zeros(
                    (width, width), dtype=torch.float32, device=dev
                )
                residual_mean[layer] = torch.zeros(
                    width, dtype=torch.float32, device=dev
                )
            residual_second[layer].add_(combined.T @ combined, alpha=1.0 / n_samples)
            residual_mean[layer].add_(
                torch.stack(means, dim=0).sum(dim=0), alpha=1.0 / n_samples
            )

            for bin_name, bin_factor_rows in position_factors.items():
                if not bin_factor_rows:
                    continue
                bin_factor = torch.cat(bin_factor_rows, dim=0)
                bin_mean = torch.stack(position_means[bin_name], dim=0).sum(dim=0)
                if bin_name not in position_second[layer]:
                    width = bin_factor.shape[1]
                    position_second[layer][bin_name] = torch.zeros(
                        (width, width), dtype=torch.float32, device=dev
                    )
                    position_mean[layer][bin_name] = torch.zeros(
                        width, dtype=torch.float32, device=dev
                    )
                position_second[layer][bin_name].add_(bin_factor.T @ bin_factor)
                position_mean[layer][bin_name].add_(bin_mean)
            if early_stop and layer == stop_layer:
                raise _ProbeForwardComplete
        return hook

    for layer in layers:
        for group, module_name in c.GROUP_CAPTURE_MODULE.items():
            handles.append(
                module_at(model, layer, module_name).register_forward_pre_hook(
                    make_linear_hook(layer, group)
                )
            )
        handles.append(model.model.layers[layer].register_forward_hook(make_layer_hook(layer)))

    forward_batches = 0
    try:
        for batch in _probe_batches(samples, forward_batch_size, max_batch_tokens):
            input_ids, attention_mask, weights, bin_weights = _pad_probe_batch(
                model, batch, dev
            )
            active["weights"] = weights
            active["bin_weights"] = bin_weights
            active["factors"] = [
                {layer: {} for layer in factor_layers if layer in layers}
                for _ in batch
            ]
            active["input_sample_means"] = [
                {layer: {} for layer in factor_layers if layer in layers}
                for _ in batch
            ]
            active["residual"] = [{} for _ in batch]
            active["residual_sample_means"] = [{} for _ in batch]
            for sample_bins in bin_weights:
                for bin_name in sample_bins:
                    position_counts[bin_name] += 1
            stopped = False
            try:
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            except _ProbeForwardComplete:
                stopped = True
            if early_stop and not stopped:
                raise RuntimeError(f"probe early-stop layer {stop_layer} was not reached")
            forward_batches += 1
            if keep_factors:
                sample_factors.extend(active["factors"])
            if keep_input_sample_means:
                input_sample_means.extend(active["input_sample_means"])
            if keep_residual_samples:
                residual_samples.extend(active["residual"])
                residual_sample_means.extend(active["residual_sample_means"])
            del input_ids, attention_mask, weights, bin_weights
    finally:
        for handle in handles:
            handle.remove()

    for layer in layers:
        for bin_name in list(position_second[layer]):
            count = max(position_counts[bin_name], 1)
            position_second[layer][bin_name].div_(count)
            position_mean[layer][bin_name].div_(count)

    def cpu_nested(value):
        if isinstance(value, torch.Tensor):
            return value.cpu()
        if isinstance(value, dict):
            return {key: cpu_nested(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_nested(item) for item in value]
        return value

    return {
        "n_samples": n_samples,
        "grams": cpu_nested(grams),
        "residual_second": cpu_nested(residual_second),
        "residual_mean": cpu_nested(residual_mean),
        "position_second": cpu_nested(position_second),
        "position_mean": cpu_nested(position_mean),
        "position_counts": position_counts,
        "sample_factors": sample_factors,
        "input_sample_means": input_sample_means,
        "residual_samples": residual_samples,
        "residual_sample_means": residual_sample_means,
        "forward_execution": {
            "requested_batch_size": int(forward_batch_size),
            "max_batch_tokens": max_batch_tokens,
            "batch_count": int(forward_batches),
            "early_stop": bool(early_stop),
            "early_stop_layer": int(stop_layer) if early_stop else None,
        },
    }


def scaling_by_group(profile: dict[str, Any], layers: list[int], device: str):
    result = {}
    for layer in layers:
        result[layer] = {}
        for group, gram in profile["grams"][layer].items():
            result[layer][group] = _gram_to_svdllm_scaling_diag_matrix(
                gram.to(device=device, dtype=torch.float32),
                cholesky_jitter=1e-5,
                singular_floor=0.0,
            ).to(device=device, dtype=torch.float32)
    return result


def adapter_scaling(adapter_dir: Path) -> float:
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    rank = float(config["r"])
    alpha = float(config["lora_alpha"])
    return alpha / math.sqrt(rank) if config.get("use_rslora", False) else alpha / rank


def load_adapter_state(step: int) -> tuple[dict[str, torch.Tensor], float]:
    adapter_dir = c.SFT_ADAPTERS / c.step_label(step)
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"missing SFT adapter: {adapter_dir}")
    return load_file(adapter_dir / "adapter_model.safetensors"), adapter_scaling(adapter_dir)


def sft_delta(
    state: dict[str, torch.Tensor],
    scaling: float,
    layer: int,
    module: str,
    device: str,
) -> torch.Tensor:
    prefix = f"base_model.model.model.layers.{layer}.{module}"
    left = state[f"{prefix}.lora_B.weight"].to(device=device, dtype=torch.float32)
    right = state[f"{prefix}.lora_A.weight"].to(device=device, dtype=torch.float32)
    return scaling * (left @ right)


def top32_approx(delta: torch.Tensor, seed: int) -> torch.Tensor:
    q = min(40, min(delta.shape))
    if q <= 32:
        left, singular, right = torch.linalg.svd(delta, full_matrices=False)
        return (left[:, :32] * singular[:32]) @ right[:32, :]
    device_index = delta.device.index if delta.is_cuda else None
    with torch.random.fork_rng(devices=[] if device_index is None else [device_index]):
        torch.manual_seed(seed)
        left, singular, right = torch.svd_lowrank(delta, q=q, niter=4)
    order = torch.argsort(singular, descending=True)
    return (left[:, order[:32]] * singular[order[:32]]) @ right[:, order[:32]].T


def update_matrix(
    arm: str,
    step: int,
    layer: int,
    module: str,
    current_model,
    base_model,
    adapter_state,
    adapter_scale,
    device: str,
) -> tuple[torch.Tensor, str]:
    base_weight = module_at(base_model, layer, module).weight.detach().float()
    if step == 0:
        return torch.zeros_like(base_weight), "base_identity"
    if arm == "sft":
        return (
            sft_delta(adapter_state, adapter_scale, layer, module, device),
            "sft_clean_fp32_ba",
        )
    merged = module_at(current_model, layer, module).weight.detach().float()
    delta = top32_approx(
        merged - base_weight,
        c.stable_seed(42, arm, step, layer, module),
    )
    return delta, "opd_top32_approx"


def raw_er_rows(profile: dict[str, Any], layers: list[int], device: str):
    rows = []
    for layer in layers:
        second = profile["residual_second"][layer].to(device=device, dtype=torch.float32)
        mean = profile["residual_mean"][layer].to(device=device, dtype=torch.float32)
        covariance = second - torch.outer(mean, mean)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).cpu().numpy()[::-1]
        rows.append(
            {
                "layer": layer,
                "position_bin": "all",
                "sample_count": profile["n_samples"],
                "raw_residual_er": c.effective_rank(eigenvalues),
                "raw_trace": float(eigenvalues.sum()),
            }
        )
        for bin_name, bin_second in profile["position_second"][layer].items():
            bin_mean = profile["position_mean"][layer][bin_name].to(
                device=device, dtype=torch.float32
            )
            covariance = bin_second.to(device=device, dtype=torch.float32) - torch.outer(
                bin_mean, bin_mean
            )
            eigenvalues = (
                torch.linalg.eigvalsh(covariance).clamp_min(0).cpu().numpy()[::-1]
            )
            rows.append(
                {
                    "layer": layer,
                    "position_bin": bin_name,
                    "sample_count": profile["position_counts"][bin_name],
                    "raw_residual_er": c.effective_rank(eigenvalues),
                    "raw_trace": float(eigenvalues.sum()),
                }
            )
    return rows


def representation_drift_rows(current, base, layers: list[int]) -> list[dict[str, Any]]:
    rows = []
    for layer in layers:
        numerator = 0.0
        denominator = 0.0
        count = 0
        for current_sample, base_sample in zip(
            current["residual_samples"], base["residual_samples"]
        ):
            current_factor = current_sample[layer].float()
            base_factor = base_sample[layer].float()
            width = min(current_factor.shape[0], base_factor.shape[0])
            numerator += float(torch.square(current_factor[:width] - base_factor[:width]).sum())
            denominator += float(torch.square(base_factor[:width]).sum())
            count += 1
        rows.append(
            {
                "layer": layer,
                "n_samples": count,
                "m2b_representation_drift": math.sqrt(
                    numerator / max(denominator, 1e-30)
                ),
            }
        )
    return rows


@torch.no_grad()
def spectral_metric_payload(
    *,
    arm: str,
    step: int,
    current_model,
    base_model,
    current_profile,
    base_profile,
    layers: list[int],
    device: str,
) -> dict[str, Any]:
    current_scales = scaling_by_group(current_profile, layers, device)
    base_scales = scaling_by_group(base_profile, layers, device)
    adapter_state = adapter_scale = None
    if arm == "sft" and step != 0:
        adapter_state, adapter_scale = load_adapter_state(step)

    spectra = {"per_checkpoint": {}, "frozen_base": {}}
    m2_rows = []
    m3_rows = []

    try:
        for layer in layers:
            layer_key = f"layer_{layer}"
            spectra["per_checkpoint"][layer_key] = {}
            spectra["frozen_base"][layer_key] = {}
            for module in c.MODULES:
                group = c.MODULE_TO_GROUP[module]
                current_scale = current_scales[layer][group]
                base_scale = base_scales[layer][group]
                actual_weight = module_at(current_model, layer, module).weight.detach().to(
                    device=device, dtype=torch.float32
                )
                base_weight = module_at(base_model, layer, module).weight.detach().to(
                    device=device, dtype=torch.float32
                )
                update, source_kind = update_matrix(
                    arm,
                    step,
                    layer,
                    module,
                    current_model,
                    base_model,
                    adapter_state,
                    adapter_scale,
                    device,
                )
                effective_weight = base_weight + update

                primary = actual_weight @ current_scale
                frozen_actual = actual_weight @ base_scale
                primary_sigma = torch.linalg.svdvals(primary)
                frozen_sigma = torch.linalg.svdvals(frozen_actual)
                spectra["per_checkpoint"][layer_key][module] = primary_sigma.cpu().tolist()
                spectra["frozen_base"][layer_key][module] = frozen_sigma.cpu().tolist()

                delta_x0 = update @ base_scale
                base_x0 = base_weight @ base_scale
                delta_xt = update @ current_scale
                base_xt = base_weight @ current_scale
                m2_rows.extend(
                    [
                        {
                            "layer": layer,
                            "module": module,
                            "reference": "X0_primary",
                            "source_kind": source_kind,
                            "m2_output_drift": float(
                                torch.linalg.vector_norm(delta_x0)
                                / torch.linalg.vector_norm(base_x0).clamp_min(1e-30)
                            ),
                        },
                        {
                            "layer": layer,
                            "module": module,
                            "reference": "Xt_secondary",
                            "source_kind": source_kind,
                            "m2_output_drift": float(
                                torch.linalg.vector_norm(delta_xt)
                                / torch.linalg.vector_norm(base_xt).clamp_min(1e-30)
                            ),
                        },
                    ]
                )

                base_whitened = base_weight @ base_scale
                u0, sigma0, vh0 = torch.linalg.svd(
                    base_whitened, full_matrices=False
                )
                current_effective_x0 = effective_weight @ base_scale
                delta_norm_sq = torch.square(delta_x0).sum().clamp_min(1e-30)
                for epsilon in (0.05, 0.01):
                    rank = c.functional_rank(sigma0.cpu().tolist(), epsilon)
                    u = u0[:, :rank]
                    v = vh0[:rank, :].T
                    base_u = u.T @ base_whitened
                    current_u = u.T @ current_effective_x0
                    delta_u = u.T @ delta_x0
                    base_v = base_whitened @ v
                    current_v = current_effective_x0 @ v
                    delta_v = delta_x0 @ v
                    m3_rows.append(
                        {
                            "layer": layer,
                            "module": module,
                            "epsilon": epsilon,
                            "rank": rank,
                            "source_kind": source_kind,
                            "e_keep_u": float(
                                torch.square(current_u).sum()
                                / torch.square(base_u).sum().clamp_min(1e-30)
                            ),
                            "phi_u": float(
                                torch.square(delta_u).sum() / delta_norm_sq
                            ),
                            "e_keep_v": float(
                                torch.square(current_v).sum()
                                / torch.square(base_v).sum().clamp_min(1e-30)
                            ),
                            "phi_v": float(
                                torch.square(delta_v).sum() / delta_norm_sq
                            ),
                        }
                    )
                del (
                    actual_weight,
                    base_weight,
                    update,
                    effective_weight,
                    primary,
                    frozen_actual,
                    primary_sigma,
                    frozen_sigma,
                    delta_x0,
                    base_x0,
                    delta_xt,
                    base_xt,
                    base_whitened,
                    u0,
                    sigma0,
                    vh0,
                    current_effective_x0,
                )
                torch.cuda.empty_cache()
    finally:
        current_scales.clear()
        base_scales.clear()
        if adapter_state is not None:
            adapter_state.clear()
        gc.collect()
        torch.cuda.empty_cache()

    return {"spectra": spectra, "m2": m2_rows, "m3": m3_rows}


def measurement_path(run_root: Path, arm: str, step: int, task_id: str) -> Path:
    return run_root / "measurements" / arm / c.step_label(step) / f"{task_id}.json"


def reference_path(run_root: Path, task_id: str) -> Path:
    return run_root / "scratch/references" / f"{task_id}.pt"


def reference_view(profile: dict[str, Any]) -> dict[str, Any]:
    """Drop bootstrap-only factors before caching the frozen-base reference."""
    return {
        key: ([] if key == "sample_factors" else value)
        for key, value in profile.items()
    }


def bundle_path(run_root: Path, arm: str, step: int, task_id: str) -> Path:
    return run_root / "scratch/bootstrap_factors" / arm / c.step_label(step) / f"{task_id}.pt"


def retain_bundle(task: c.ProbeTask) -> bool:
    if task.task_id in {"legacy_S_math", "E_ood"}:
        return True
    if task.probe_type == "S" and task.domain == "math" and task.generation_seed == 3:
        return True
    if task.probe_type == "H" and task.domain == "ood" and task.generation_seed == 3:
        return True
    return False


def task_complete(path: Path, layers: list[int]) -> bool:
    payload = c.read_json(path, {})
    if payload.get("schema_version") != 2:
        return False
    return payload.get("layers") == layers and "spectra" in payload


def save_reference(path: Path, profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(profile, tmp)
    os.replace(tmp, path)


def load_reference(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def window_manifest_rows(
    task: c.ProbeTask,
    arm: str,
    step: int,
    samples: list[c.PreparedSample],
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        for window in sample.windows:
            rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "task_id": task.task_id,
                    "probe_type": task.probe_type,
                    "domain": task.domain,
                    "generation_seed": task.generation_seed,
                    "window_seed": c.WINDOW_SEED,
                    "sample_id": sample.sample_id,
                    "actual_k": len(sample.windows),
                    "eligible_tokens": sample.eligible_end - sample.eligible_start,
                    "window_index": window.window_index,
                    "window_start": window.start,
                    "window_end": window.end,
                    "window_tokens": window.token_count,
                    "relative_start": window.relative_start,
                    "relative_center": window.relative_center,
                    "relative_end": window.relative_end,
                    "position_bin": window.position_bin,
                }
            )
    return rows


def probe_campaign(args, tokenizer) -> None:
    all_tasks = [
        task
        for task in c.build_task_index(args.run_root)
        if task.generation_seed is None or task.generation_seed in args.generation_seeds
        if task.target_arm is None or task.target_arm in args.arms
        if task.target_step is None or task.target_step in args.steps
        if not args.task_filter
        or any(token in task.task_id for token in args.task_filter)
    ]
    missing = [task.task_id for task in all_tasks if not Path(task.corpus_path).exists()]
    if missing:
        raise FileNotFoundError(
            f"generated/fixed corpora are incomplete ({len(missing)} missing); "
            f"examples={missing[:8]}"
        )

    base_model = load_model(c.BASE_MODEL, args.device)
    reference_cache: dict[str, dict[str, Any]] = {}
    window_rows_path = args.run_root / "window_manifest_rows.jsonl"
    window_handle = open(window_rows_path, "a", encoding="utf-8")
    try:
        for arm in args.arms:
            for step in args.steps:
                tasks = c.tasks_for_model(all_tasks, arm, step)
                tasks = [
                    task for task in tasks
                    if task.alias_of is None
                    and (not args.task_filter or any(
                        token in task.task_id for token in args.task_filter
                    ))
                ]
                pending = [
                    task for task in tasks
                    if not task_complete(
                        measurement_path(args.run_root, arm, step, task.task_id),
                        args.layers,
                    )
                ]
                if not pending:
                    print(f"[Model skip] {arm}/{c.step_label(step)}", flush=True)
                    continue

                current_model = base_model if step == 0 else load_model(
                    c.model_path(arm, step), args.device
                )
                print(
                    f"[Model] {arm}/{c.step_label(step)} pending={len(pending)}",
                    flush=True,
                )
                try:
                    for task in pending:
                        target = measurement_path(
                            args.run_root, arm, step, task.task_id
                        )
                        samples = c.prepare_samples(
                            Path(task.corpus_path),
                            tokenizer,
                            corpus_id=task.task_id,
                            window_seed=c.WINDOW_SEED,
                            max_context_tokens=args.max_context_tokens,
                        )
                        if args.measurement_n > 0:
                            samples = samples[: args.measurement_n]
                        keep = retain_bundle(task)
                        current_profile = collect_profile(
                            current_model,
                            samples,
                            args.layers,
                            args.device,
                            keep_factors=keep,
                            keep_residual_samples=True,
                        )

                        ref_path = reference_path(args.run_root, task.task_id)
                        if task.shared_across_arms:
                            if task.task_id not in reference_cache:
                                if ref_path.exists():
                                    reference_cache[task.task_id] = load_reference(ref_path)
                                elif step == 0:
                                    reference_cache[task.task_id] = reference_view(
                                        current_profile
                                    )
                                    save_reference(ref_path, reference_cache[task.task_id])
                                else:
                                    reference_cache[task.task_id] = collect_profile(
                                        base_model,
                                        samples,
                                        args.layers,
                                        args.device,
                                        keep_factors=False,
                                        keep_residual_samples=True,
                                    )
                                    save_reference(ref_path, reference_cache[task.task_id])
                            base_profile = reference_cache[task.task_id]
                        elif step == 0:
                            base_profile = current_profile
                        else:
                            base_profile = collect_profile(
                                base_model,
                                samples,
                                args.layers,
                                args.device,
                                keep_factors=False,
                                keep_residual_samples=True,
                            )

                        metrics = spectral_metric_payload(
                            arm=arm,
                            step=step,
                            current_model=current_model,
                            base_model=base_model,
                            current_profile=current_profile,
                            base_profile=base_profile,
                            layers=args.layers,
                            device=args.device,
                        )
                        payload = {
                            "schema_version": 2,
                            "arm": arm,
                            "step": step,
                            "task": c.task_to_dict(task),
                            "layers": args.layers,
                            "n_samples": len(samples),
                            "windows": window_manifest_rows(task, arm, step, samples),
                            "hierarchical_normalization": (
                                "window token mean -> sample window mean -> sample equal mean"
                            ),
                            "raw_residual": raw_er_rows(
                                current_profile, args.layers, args.device
                            ),
                            "m2b": representation_drift_rows(
                                current_profile, base_profile, args.layers
                            ),
                            **metrics,
                        }
                        c.write_json_atomic(target, payload)

                        if keep:
                            bundle = {
                                "schema_version": 2,
                                "arm": arm,
                                "step": step,
                                "task": c.task_to_dict(task),
                                "layers": args.layers,
                                "sample_ids": [sample.sample_id for sample in samples],
                                "sample_factors": current_profile["sample_factors"],
                                "residual_samples": current_profile["residual_samples"],
                                "residual_sample_means": current_profile[
                                    "residual_sample_means"
                                ],
                            }
                            factor_target = bundle_path(
                                args.run_root, arm, step, task.task_id
                            )
                            factor_target.parent.mkdir(parents=True, exist_ok=True)
                            tmp = factor_target.with_suffix(".pt.tmp")
                            torch.save(bundle, tmp)
                            os.replace(tmp, factor_target)
                            c.assert_scratch_budget(args.run_root, args.scratch_limit_gib)

                        for row in window_manifest_rows(task, arm, step, samples):
                            window_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        window_handle.flush()
                        print(
                            f"[Cell] {arm}/{c.step_label(step)}/{task.task_id}",
                            flush=True,
                        )
                        if not task.shared_across_arms:
                            del base_profile
                        del current_profile, metrics, payload
                        gc.collect()
                        torch.cuda.empty_cache()
                finally:
                    if current_model is not base_model:
                        unload_model(current_model)

        for arm in args.arms:
            for step in args.steps:
                alias = c.ProbeTask(
                    task_id=f"X_sft_math__{c.step_label(step)}",
                    probe_type="X",
                    domain="math",
                    corpus_path=str(args.run_root / "corpora/fixed/legacy_S_math.jsonl"),
                    source_kind="fixed_dataset_cot_question_masked",
                    generation_seed=None,
                    generated=False,
                    shared_across_arms=False,
                    target_arm="sft",
                    target_step=step,
                    alias_of="legacy_S_math",
                )
                if arm != "sft":
                    continue
                source = measurement_path(args.run_root, arm, step, "legacy_S_math")
                target = measurement_path(args.run_root, arm, step, alias.task_id)
                if source.exists() and not target.exists():
                    payload = c.read_json(source)
                    payload["task"] = c.task_to_dict(alias)
                    payload["alias_measurement_source"] = str(source)
                    c.write_json_atomic(target, payload)
    finally:
        window_handle.close()
        reference_cache.clear()
        unload_model(base_model)


def summarize(args) -> None:
    spectrum_rows = []
    m2_rows = []
    m3_rows = []
    position_rows = []
    for path in sorted((args.run_root / "measurements").rglob("*.json")):
        payload = c.read_json(path)
        if payload.get("schema_version") != 2:
            continue
        task = payload["task"]
        base = {
            "arm": payload["arm"],
            "step": payload["step"],
            "task_id": task["task_id"],
            "probe_type": task["probe_type"],
            "domain": task["domain"],
            "generation_seed": task["generation_seed"],
            "n_samples": payload["n_samples"],
        }
        for track, layers in payload["spectra"].items():
            for layer_key, modules in layers.items():
                layer = int(layer_key.split("_")[-1])
                for module, sigma in modules.items():
                    spectrum_rows.append(
                        {
                            **base,
                            "track": track,
                            "layer": layer,
                            "module": module,
                            "effective_rank": c.effective_rank(sigma),
                            "r_eps_005": c.functional_rank(sigma, 0.05),
                            "r_eps_001": c.functional_rank(sigma, 0.01),
                            "tail_energy_r32": c.tail_energy(sigma, 32),
                            "sigma_json": json.dumps(sigma),
                            "measurement_path": str(path),
                        }
                    )
        for row in payload["m2"]:
            m2_rows.append({**base, **row})
        for row in payload["m2b"]:
            m2_rows.append(
                {
                    **base,
                    "layer": row["layer"],
                    "module": "__representation__",
                    "reference": "paired_hidden_states",
                    "source_kind": "same_forward_text",
                    "m2_output_drift": row["m2b_representation_drift"],
                }
            )
        for row in payload["m3"]:
            m3_rows.append({**base, **row})
        for row in payload["raw_residual"]:
            position_rows.append({**base, **row})

    c.write_csv_atomic(args.mini_root / "R4_v2_spectra_all.csv", spectrum_rows)
    c.write_csv_atomic(args.mini_root / "R4_m2_output_drift.csv", m2_rows)
    c.write_csv_atomic(args.mini_root / "R4_m3_keep_aim.csv", m3_rows)
    c.write_csv_atomic(args.mini_root / "R4_v2_position_tertiles.csv", position_rows)

    tasks = c.build_task_index(args.run_root)
    manifest = {
        "schema_version": 2,
        "cycle": "mini_cycle09_round4",
        "layer_scope_user_override": [9, 18, 27],
        "handoff_line_50_full36_superseded": True,
        "probe_taxonomy": {
            "S": "base generated, frozen/shared",
            "E": "external fixed text",
            "X": "actual arm training signal",
            "H": "per-checkpoint nontraining generation, symmetric domains",
        },
        "generation_seeds": args.generation_seeds,
        "window_seed": c.WINDOW_SEED,
        "generation_and_window_seeds_separate": True,
        "window_tokens": c.WINDOW_TOKENS,
        "requested_k": c.WINDOW_K,
        "short_generation_policy": "single variable window when <512; zero-token missing",
        "position_bins": ["early", "mid", "late"],
        "hierarchical_normalization": (
            "window token mean -> sample window mean -> corpus equal sample mean"
        ),
        "bootstrap_unit": "sample; windows nested",
        "prompt_policy": "prompt enters forward context and has zero statistical weight",
        "E_math": "deferred_by_user",
        "E_math_hard": "included",
        "scratch_limit_gib": args.scratch_limit_gib,
        "tasks": [c.task_to_dict(task) for task in tasks],
        "output_counts": {
            "spectra_rows": len(spectrum_rows),
            "m2_rows": len(m2_rows),
            "m3_rows": len(m3_rows),
            "position_rows": len(position_rows),
        },
    }
    c.write_json_atomic(args.mini_root / "R4_v2_manifest.json", manifest)
    print(
        f"[Summarize] spectra={len(spectrum_rows)} m2={len(m2_rows)} "
        f"m3={len(m3_rows)} position={len(position_rows)}",
        flush=True,
    )


def prepare(args, tokenizer) -> None:
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.mini_root.mkdir(parents=True, exist_ok=True)
    c.prepare_fixed_corpora(tokenizer, args.run_root)
    state = {
        "schema_version": 1,
        "state": "PREPARED",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_root": str(args.run_root),
        "mini_root": str(args.mini_root),
        "layers": args.layers,
        "arms": args.arms,
        "steps": args.steps,
        "generation_seeds": args.generation_seeds,
    }
    c.write_json_atomic(args.run_root / "R4_CAMPAIGN_STATUS.json", state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-root", type=Path, default=c.RUN_ROOT)
    parser.add_argument("--mini-root", type=Path, default=c.MINI_ROOT)
    parser.add_argument("--arms", default=",".join(c.ARMS))
    parser.add_argument("--steps", default=",".join(map(str, c.STEPS)))
    parser.add_argument("--layers", default=",".join(map(str, c.LAYERS)))
    parser.add_argument(
        "--generation-seeds", default=",".join(map(str, c.GENERATION_SEEDS))
    )
    parser.add_argument("--n-samples", type=int, default=c.N_GENERATED)
    parser.add_argument("--measurement-n", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=c.MAX_NEW_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--max-context-tokens", type=int, default=c.MAX_CONTEXT_TOKENS
    )
    parser.add_argument("--gpu-mem", type=float, default=0.82)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scratch-limit-gib", type=int, default=c.SCRATCH_LIMIT_GIB)
    parser.add_argument("--task-filter", default="")
    args = parser.parse_args()

    if args.all:
        args.prepare = args.generate = args.probe = args.summarize = True
    if not any((args.prepare, args.generate, args.probe, args.summarize)):
        parser.print_help()
        return

    args.arms = parse_names(args.arms, c.ARMS)
    args.steps = parse_ints(args.steps, c.STEPS)
    args.layers = parse_ints(args.layers, c.LAYERS)
    args.generation_seeds = parse_ints(
        args.generation_seeds, c.GENERATION_SEEDS
    )
    args.task_filter = parse_names(args.task_filter, ())
    if set(args.layers).difference(c.LAYERS):
        raise ValueError("Round 4 user ruling permits only layers 9,18,27")

    if args.smoke:
        args.run_root = args.run_root / "smoke"
        args.mini_root = args.mini_root / "smoke_r4"
        args.prepare = args.generate = args.probe = args.summarize = True
        args.arms = ["opd"]
        args.steps = [0]
        args.layers = [18]
        args.generation_seeds = [3]
        args.n_samples = 2
        args.measurement_n = 1
        args.max_new_tokens = 32
        args.max_model_len = 1024
        args.max_context_tokens = 1024
        args.task_filter = ["S_math__g3"]

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(str(c.BASE_MODEL), trust_remote_code=True)

    if args.prepare:
        prepare(args, tokenizer)
    if args.generate:
        generate_all(args, tokenizer)
    if args.probe:
        probe_campaign(args, tokenizer)
    if args.summarize:
        summarize(args)


if __name__ == "__main__":
    main()
