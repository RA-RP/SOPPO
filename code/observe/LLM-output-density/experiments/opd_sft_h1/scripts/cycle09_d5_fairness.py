#!/usr/bin/env python3
"""D5: cumulative raw-W and current/fixed-S update energy on the frozen grids.

The paper-facing track is the serialized BF16 merged checkpoint.  This runner
does not compute merged-minus-base ranks/tails; it only emits the preregistered
raw-update and whitened-update energies needed for the fair Model-W/WS rows.
Base grams are kept in host RAM for one model at a time, while each current
checkpoint is materialized once and used for all four frozen probes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch


REPO = Path("/root/LLM-output-density")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
import sys

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cycle09_block3_common as b3  # noqa: E402
import cycle09_llama_geometry as lgeom  # noqa: E402
import cycle09_llama_model_export as lexport  # noqa: E402
import cycle09_qwen_d4_merged_state as qd4  # noqa: E402
import cycle09_r4_campaign as campaign  # noqa: E402
import cycle09_r4_common as c4  # noqa: E402
import cycle09_stage3_common as qstage  # noqa: E402


ROOT = b3.AUTODL / "cycle09_relative_functional_contraction/d5_fairness"
FINAL = b3.AUTODL / "cycle09_relative_functional_contraction/final"
MINI = b3.MINI
ARMS = ("opd", "sft", "offkd", "seqkd")
PROBES = ("E_general", "E_math", "E_ood", "E_if")
MODULES = tuple(b3.MODULES)
EPSILONS = (0.01, 0.025, 0.05, 0.10)
STEPS = {
    "llama": (5, 20, 40, 80, 160, 320),
    "qwen": (5, 10, 20, 40, 80, 160, 320, 480, 624),
}
LAYERS = {"llama": 14, "qwen": 18}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def label(step: int) -> str:
    return f"step_{int(step):03d}"


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


@contextmanager
def lock(path: Path):
    import fcntl

    target = path.with_suffix(path.suffix + ".lock")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def cell_path(tag: str, model: str, arm: str, step: int, probe: str) -> Path:
    return ROOT / tag / "cells" / model / arm / label(step) / f"{probe}.json"


def checkpoint_complete(tag: str, model: str, arm: str, step: int) -> bool:
    return all(read_json(cell_path(tag, model, arm, step, probe), {}).get("status") == "complete" for probe in PROBES)


def group_for(module: str) -> str:
    return c4.MODULE_TO_GROUP[module]


def samples_for(model: str, probe: str, limit: int) -> list[Any]:
    if model == "llama":
        samples = lgeom.prepare_samples(b3.load_llama_tokenizer(), probe, 0)
    else:
        samples = qd4.samples_for(probe, 0)
    return samples[:limit] if limit else samples


def base_path(model: str) -> Path:
    return b3.LLAMA_STUDENT_RUNTIME if model == "llama" else qstage.BASE_MODEL


def current_path(model: str, arm: str, step: int) -> Path:
    if model == "llama":
        return lexport.merged_target(arm, step)
    raise ValueError("Qwen non-base checkpoints require ephemeral adapter materialization")


def load_model(path: Path, device: str):
    from transformers import AutoModelForCausalLM

    value = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    value.config.use_cache = False
    value.eval().to(device)
    return value


def load_cpu_base(model: str):
    from transformers import AutoModelForCausalLM

    value = AutoModelForCausalLM.from_pretrained(
        str(base_path(model)), torch_dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    value.config.use_cache = False
    value.eval()
    return value


def unload(model: Any) -> None:
    if model is not None:
        model.to("cpu")
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def current_model(model: str, arm: str, step: int, device: str):
    if model == "llama":
        value = load_model(current_path(model, arm, step), device)
        try:
            yield value
        finally:
            unload(value)
        return
    with qd4.materialized_model(arm, step) as merged:
        value = load_model(merged, device)
        try:
            yield value
        finally:
            unload(value)


def collect_grams(model: Any, samples: list[Any], layer: int, device: str, batch_size: int, max_batch_tokens: int) -> dict[str, torch.Tensor]:
    profile = campaign.collect_profile(
        model, samples, [layer], device,
        keep_factors=False, keep_residual_samples=False, keep_input_sample_means=False,
        factor_layers=(layer,), forward_batch_size=batch_size, max_batch_tokens=max_batch_tokens,
        early_stop=True,
    )
    return {group: profile["grams"][layer][group].cpu() for group in set(c4.MODULE_TO_GROUP.values())}


def sqrt_gram(gram: torch.Tensor, device: str) -> torch.Tensor:
    symmetric = ((gram + gram.T) / 2).to(device=device, dtype=torch.float64)
    values, vectors = torch.linalg.eigh(symmetric)
    return (vectors * values.clamp_min(0).sqrt()) @ vectors.T


def build_base_grams(model_name: str, args: argparse.Namespace) -> dict[str, tuple[list[Any], dict[str, torch.Tensor]]]:
    base = load_model(base_path(model_name), args.device)
    try:
        return {
            probe: (
                samples_for(model_name, probe, args.sample_limit),
                collect_grams(base, samples_for(model_name, probe, args.sample_limit), LAYERS[model_name], args.device,
                              args.forward_batch_size, args.max_batch_tokens),
            )
            for probe in PROBES
        }
    finally:
        unload(base)


def calculate_probe(
    current: Any,
    base_cpu: Any,
    model_name: str,
    arm: str,
    step: int,
    probe: str,
    samples: list[Any],
    base_grams: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    target = cell_path(args.tag, model_name, arm, step, probe)
    with lock(target):
        cached = read_json(target, {})
        if cached.get("status") == "complete":
            return cached
        current_grams = collect_grams(current, samples, LAYERS[model_name], args.device, args.forward_batch_size, args.max_batch_tokens)
        rows: list[dict[str, Any]] = []
        for module in MODULES:
            group = group_for(module)
            current_weight = campaign.module_at(current, LAYERS[model_name], module).weight.detach().to(args.device, torch.float32)
            base_weight = campaign.module_at(base_cpu, LAYERS[model_name], module).weight.detach().to(args.device, torch.float32)
            delta = current_weight - base_weight
            current_scale = sqrt_gram(current_grams[group], args.device).to(torch.float32)
            fixed_scale = sqrt_gram(base_grams[group], args.device).to(torch.float32)
            raw_energy = float(delta.square().sum())
            current_energy = float((delta @ current_scale).square().sum())
            fixed_energy = float((delta @ fixed_scale).square().sum())
            for epsilon in EPSILONS:
                rows.append({
                    "model": model_name, "arm": arm, "checkpoint": step, "probe_name": probe,
                    "layer": LAYERS[model_name], "module": module, "epsilon": epsilon,
                    "raw_weight_energy": raw_energy,
                    "whitened_update_energy_current": current_energy,
                    "whitened_update_energy_fixed": fixed_energy,
                    "activation_exposure_ratio": current_energy / max(fixed_energy, 1e-30),
                    "sample_count": len(samples), "sample_ids_sha256": json_digest([item.sample_id for item in samples]),
                    "weight_object": "serialized_merged_bf16_effective_difference",
                    "checkpoint_storage_dtype": "bf16_safetensors_on_disk",
                    "subtraction_dtype": "fp32", "gram_and_whitening_dtype": "fp64_eigh_clamp_nonnegative",
                    "WS_matmul_dtype": "fp32", "window_protocol": "v2_three_level_equal_sample",
                })
            del current_weight, base_weight, delta, current_scale, fixed_scale
            torch.cuda.empty_cache()
        payload = {
            "schema_version": "cycle09_d5_fairness_v1", "status": "complete", "model": model_name,
            "arm": arm, "checkpoint": step, "probe_name": probe, "layer": LAYERS[model_name],
            "sample_count": len(samples), "sample_ids_sha256": json_digest([item.sample_id for item in samples]),
            "rows": rows, "created_utc": now(),
        }
        atomic_json(target, payload)
        return payload


def run_checkpoint(model_name: str, arm: str, step: int, args: argparse.Namespace, base_cpu: Any, base_cache: dict[str, tuple[list[Any], dict[str, torch.Tensor]]]) -> dict[str, Any]:
    if checkpoint_complete(args.tag, model_name, arm, step):
        return {"status": "complete", "cached": True, "model": model_name, "arm": arm, "checkpoint": step}
    with current_model(model_name, arm, step, args.device) as current:
        cells = [
            calculate_probe(current, base_cpu, model_name, arm, step, probe, *base_cache[probe], args)
            for probe in PROBES
        ]
    return {"status": "complete", "model": model_name, "arm": arm, "checkpoint": step, "cells": len(cells)}


def selected_models(value: str) -> tuple[str, ...]:
    result = ("llama", "qwen") if value == "all" else tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or set(result) - {"llama", "qwen"}:
        raise ValueError(f"invalid models={value}")
    return result


def formal(args: argparse.Namespace) -> dict[str, Any]:
    completed: list[dict[str, Any]] = []
    progress = ROOT / args.tag / "progress.json"
    for model_name in selected_models(args.models):
        base_cpu = load_cpu_base(model_name)
        try:
            base_cache = build_base_grams(model_name, args)
            for arm in ARMS:
                for step in STEPS[model_name]:
                    result = run_checkpoint(model_name, arm, step, args, base_cpu, base_cache)
                    completed.append(result)
                    atomic_json(progress, {
                        "schema_version": "cycle09_d5_fairness_v1", "status": "running", "completed": completed,
                        "remaining": sum(not checkpoint_complete(args.tag, m, a, s) for m in selected_models(args.models) for a in ARMS for s in STEPS[m]),
                        "updated_utc": now(),
                    })
            del base_cache
        finally:
            unload(base_cpu)
    result = finalize(args)
    atomic_json(progress, {"schema_version": "cycle09_d5_fairness_v1", "status": "complete", "completed": completed, "finalize": result, "completed_utc": now()})
    return result


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    payloads = [read_json(path) for path in sorted((ROOT / args.tag / "cells").rglob("*.json"))]
    rows = [row for payload in payloads if payload.get("status") == "complete" for row in payload["rows"]]
    module = pd.DataFrame(rows)
    if module.empty:
        raise RuntimeError("no D5 cells available")
    group = ["model", "arm", "checkpoint", "probe_name", "layer", "epsilon"]
    equal7 = module.groupby(group, dropna=False).agg(
        module_count=("module", "nunique"),
        raw_update_energy_equal7=("raw_weight_energy", "mean"),
        whitened_update_energy_equal7=("whitened_update_energy_current", "mean"),
        whitened_update_energy_fixed_equal7=("whitened_update_energy_fixed", "mean"),
        activation_exposure_ratio_equal7=("activation_exposure_ratio", "mean"),
    ).reset_index()
    atomic_csv(FINAL / "d5_fairness_update_module.csv", module)
    atomic_csv(FINAL / "d5_fairness_update_equal7.csv", equal7)
    manifest = {
        "schema_version": "cycle09_d5_fairness_v1", "status": "complete", "tag": args.tag,
        "module_rows": len(module), "equal7_rows": len(equal7), "created_utc": now(),
        "artifacts": [str(FINAL / "d5_fairness_update_module.csv"), str(FINAL / "d5_fairness_update_equal7.csv")],
        "main_track": "serialized_merged_bf16_effective_difference_with_current_S",
        "sensitivity": "same_delta_with_fixed_base_S",
    }
    atomic_json(ROOT / args.tag / "finalize_manifest.json", manifest)
    return manifest


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    smoke_args = argparse.Namespace(**vars(args))
    smoke_args.tag = "smoke"
    smoke_args.sample_limit = 2
    smoke_args.models = "all"
    results = []
    for model_name, arm, step in (("llama", "opd", 5), ("qwen", "offkd", 20)):
        base_cpu = load_cpu_base(model_name)
        try:
            base_cache = build_base_grams(model_name, smoke_args)
            results.append(run_checkpoint(model_name, arm, step, smoke_args, base_cpu, base_cache))
        finally:
            unload(base_cpu)
    return {"status": "complete", "results": results, "created_utc": now()}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "formal", "finalize"), required=True)
    parser.add_argument("--models", default="all")
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--forward-batch-size", type=int, default=1)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = {"smoke": smoke, "formal": formal, "finalize": finalize}[args.phase](args)
    print(json.dumps({"status": result.get("status"), "created_utc": now()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
