#!/usr/bin/env python3
"""Llama per-checkpoint whitened rank and native raw-representation suite."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import cycle09_block3_common as c
import cycle09_llama_model_export as export
import cycle09_llama_probe_prepare as probes
import cycle09_r4_campaign as campaign
import cycle09_r4_common as c4


ROOT = c.RUN_ROOT / "llama_geometry"
CELL_ROOT = ROOT / "cells"
EPSILONS = (0.01, 0.025, 0.05, 0.10)
PROBE_NAMES = ("S_math", "E_math", "E_math_hard_v2", "E_ood", "E_if", "E_general")


def corpus_path(probe: str) -> Path:
    return probes.CORPUS_ROOT / f"{probe}.jsonl"


def layers_for(step: int) -> list[int]:
    return [14] if step in {80, 320} else [7, 14, 21]


def reference_root(smoke: bool = False) -> Path:
    return ROOT / ("smoke/scratch/references" if smoke else "scratch/references")


def reference_path(probe: str, smoke: bool = False) -> Path:
    return reference_root(smoke) / f"{probe}.pt"


def base_spectra_path(probe: str, smoke: bool = False) -> Path:
    return reference_root(smoke) / f"{probe}_spectra.json"


def base_cell_path(smoke: bool = False) -> Path:
    return ROOT / ("cells/smoke/base.json" if smoke else "cells/base.json")


def cell_path(arm: str, step: int, smoke: bool) -> Path:
    branch = "smoke" if smoke else "formal"
    return CELL_ROOT / branch / arm / f"step_{step:03d}.json"


def save_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_profile(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def prepare_samples(tokenizer: Any, probe: str, measurement_n: int) -> list[c4.PreparedSample]:
    samples = c4.prepare_samples(corpus_path(probe), tokenizer, corpus_id=f"llama:{probe}")
    return samples[:measurement_n] if measurement_n else samples


def profile_model(
    model: Any,
    samples: list[c4.PreparedSample],
    layers: list[int],
    device: str,
    *,
    keep_sample_means: bool,
    forward_batch_size: int,
    max_batch_tokens: int,
) -> dict[str, Any]:
    profile = campaign.collect_profile(
        model,
        samples,
        layers,
        device,
        keep_factors=False,
        keep_residual_samples=keep_sample_means,
        factor_layers=(),
        forward_batch_size=forward_batch_size,
        max_batch_tokens=max_batch_tokens,
        early_stop=True,
    )
    if keep_sample_means:
        profile["residual_samples"] = []
    return profile


def summarize_spectra(
    spectra: dict[str, Any],
    reference: dict[str, Any],
    profile: dict[str, Any],
    probe: str,
    arm: str,
    step: int,
    layers: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    r_rows = []
    tail_rows = []
    for layer in layers:
        for module in c.MODULES:
            sigma_values = spectra[str(layer)][module]
            sigma = np.asarray(sigma_values, dtype=np.float64)
            base_sigma = reference[str(layer)][module]
            energy = np.square(sigma)
            total = max(float(energy.sum()), 1e-300)
            tail_curve = np.concatenate(([1.0], 1.0 - np.cumsum(energy) / total))
            tail_rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "probe": probe,
                    "track": "per_checkpoint",
                    "layer": layer,
                    "module": module,
                    "n_samples": int(profile["n_samples"]),
                    "tail_energy_curve_json": json.dumps(
                        tail_curve.tolist(), separators=(",", ":")
                    ),
                    "singular_count": len(sigma_values),
                }
            )
            for epsilon in EPSILONS:
                current_rank = c4.functional_rank(sigma_values, epsilon)
                base_rank = c4.functional_rank(base_sigma, epsilon)
                r_rows.append(
                    {
                        "arm": arm,
                        "step": step,
                        "probe": probe,
                        "track": "per_checkpoint",
                        "layer": layer,
                        "module": module,
                        "epsilon": epsilon,
                        "r_epsilon": current_rank,
                        "base_r_epsilon": base_rank,
                        "delta_from_base": current_rank - base_rank,
                        "n_samples": int(profile["n_samples"]),
                    }
                )
    return r_rows, tail_rows


def spectrum_rows(
    model: Any,
    profile: dict[str, Any],
    probe: str,
    arm: str,
    step: int,
    layers: list[int],
    device: str,
    *,
    smoke: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scales = campaign.scaling_by_group(profile, layers, device)
    reference = json.loads(base_spectra_path(probe, smoke).read_text(encoding="utf-8"))
    spectra: dict[str, Any] = {}
    try:
        for layer in layers:
            spectra[str(layer)] = {}
            for module in c.MODULES:
                group = c4.MODULE_TO_GROUP[module]
                weight = campaign.module_at(model, layer, module).weight.detach().to(
                    device=device, dtype=torch.float32
                )
                whitened = weight @ scales[layer][group]
                sigma = torch.linalg.svdvals(whitened).double().cpu().numpy()
                sigma_values = sigma.tolist()
                spectra[str(layer)][module] = sigma_values
                del weight, whitened, sigma
                torch.cuda.empty_cache()
    finally:
        scales.clear()
        gc.collect()
        torch.cuda.empty_cache()
    r_rows, tail_rows = summarize_spectra(
        spectra, reference, profile, probe, arm, step, layers
    )
    return r_rows, tail_rows, spectra


def sample_mean_matrix(profile: dict[str, Any], layer: int) -> torch.Tensor:
    rows = [sample[layer].float() for sample in profile["residual_sample_means"]]
    if not rows:
        raise RuntimeError(f"profile has no retained sample means at layer {layer}")
    return torch.stack(rows, dim=0)


def pairwise_cosine_mean(matrix: torch.Tensor) -> float:
    normalized = torch.nn.functional.normalize(matrix.float(), dim=1, eps=1e-12)
    gram = normalized @ normalized.T
    count = gram.shape[0]
    if count <= 1:
        return 0.0
    return float((gram.sum() - gram.diagonal().sum()) / (count * (count - 1)))


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    width = min(left.shape[0], right.shape[0])
    left = left[:width].float()
    right = right[:width].float()
    left -= left.mean(dim=0, keepdim=True)
    right -= right.mean(dim=0, keepdim=True)
    left_gram = left @ left.T
    right_gram = right @ right.T
    numerator = torch.sum(left_gram * right_gram)
    denominator = torch.linalg.vector_norm(left_gram) * torch.linalg.vector_norm(right_gram)
    return float(numerator / denominator.clamp_min(1e-30))


def raw_row(
    profile: dict[str, Any],
    base_profile: dict[str, Any],
    arm: str,
    step: int,
    probe: str,
    layer: int,
    device: str,
) -> dict[str, Any]:
    second = profile["residual_second"][layer].to(device=device, dtype=torch.float32)
    mean = profile["residual_mean"][layer].to(device=device, dtype=torch.float32)
    covariance = second - torch.outer(mean, mean)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).double().cpu().numpy()[::-1]
    trace = max(float(eigenvalues.sum()), 1e-300)
    probabilities = eigenvalues / trace
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, None))))
    participation = trace * trace / max(float(np.square(eigenvalues).sum()), 1e-300)
    current_means = sample_mean_matrix(profile, layer).to(device)
    base_means = sample_mean_matrix(base_profile, layer).to(device)
    centered = current_means - current_means.mean(dim=0, keepdim=True)
    row = {
        "arm": arm,
        "step": step,
        "probe": probe,
        "layer": layer,
        "native_object": "raw residual-stream sample means and centered covariance",
        "n_samples": int(profile["n_samples"]),
        "normalized_entropy_effective_rank": math.exp(entropy),
        "participation_ratio": participation,
        "top1_explained_share": float(eigenvalues[:1].sum() / trace),
        "top8_explained_share": float(eigenvalues[:8].sum() / trace),
        "top32_explained_share": float(eigenvalues[:32].sum() / trace),
        "raw_anisotropy": pairwise_cosine_mean(current_means),
        "centered_anisotropy": pairwise_cosine_mean(centered),
        "linear_cka_vs_step0": linear_cka(base_means, current_means),
    }
    del second, mean, covariance, current_means, base_means, centered
    torch.cuda.empty_cache()
    return row


def base_reference(args: argparse.Namespace) -> dict[str, Any]:
    probe_manifest = c.read_json(probes.MANIFEST, {})
    if probe_manifest.get("status") != "complete":
        raise RuntimeError(f"probe campaign incomplete: {probes.MANIFEST}")
    tokenizer = c.load_llama_tokenizer()
    model = campaign.load_model(c.LLAMA_STUDENT, args.device)
    all_r, all_tail, all_raw = [], [], []
    spectra_inventory = []
    try:
        for probe in PROBE_NAMES:
            samples = prepare_samples(tokenizer, probe, args.measurement_n)
            profile = profile_model(
                model,
                samples,
                [7, 14, 21],
                args.device,
                keep_sample_means=True,
                forward_batch_size=args.forward_batch_size,
                max_batch_tokens=args.max_batch_tokens,
            )
            save_torch(reference_path(probe, args.smoke), profile)
            # Bootstrap base spectra through the same per-checkpoint whitening path.
            scales = campaign.scaling_by_group(profile, [7, 14, 21], args.device)
            spectra: dict[str, Any] = {}
            try:
                for layer in (7, 14, 21):
                    spectra[str(layer)] = {}
                    for module in c.MODULES:
                        group = c4.MODULE_TO_GROUP[module]
                        weight = campaign.module_at(model, layer, module).weight.detach().to(
                            device=args.device, dtype=torch.float32
                        )
                        sigma = torch.linalg.svdvals(weight @ scales[layer][group]).double().cpu().tolist()
                        spectra[str(layer)][module] = sigma
            finally:
                scales.clear()
                torch.cuda.empty_cache()
            c.atomic_json(base_spectra_path(probe, args.smoke), spectra)
            r_rows, tail_rows = summarize_spectra(
                spectra,
                spectra,
                profile,
                probe,
                "base",
                0,
                [7, 14, 21],
            )
            all_r.extend(r_rows)
            all_tail.extend(tail_rows)
            spectra_inventory.append(
                {
                    "arm": "base",
                    "step": 0,
                    "probe": probe,
                    "track": "per_checkpoint",
                    "spectra_path": str(base_spectra_path(probe, args.smoke)),
                    "sha256": c.sha256_file(base_spectra_path(probe, args.smoke)),
                }
            )
            all_raw.extend(
                raw_row(profile, profile, "base", 0, probe, layer, args.device)
                for layer in (7, 14, 21)
            )
            del profile, samples
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        campaign.unload_model(model)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "arm": "base",
        "step": 0,
        "r_epsilon": all_r,
        "tail_energy": all_tail,
        "raw_representation": all_raw,
        "spectra_inventory": spectra_inventory,
        "forward_execution": {
            "batch_size": args.forward_batch_size,
            "max_batch_tokens": args.max_batch_tokens,
            "early_stop": True,
        },
        "created_utc": c.utc_now(),
    }
    c.atomic_json(base_cell_path(args.smoke), payload)
    return payload


def lock_for(path: Path):
    lock = path.with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    handle = lock.open("w", encoding="utf-8")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in c.ARMS or args.step not in c.MEASURED_CHECKPOINTS or args.step == 0:
        raise ValueError("geometry cell requires one arm and a nonzero measured checkpoint")
    target = cell_path(args.arm, args.step, args.smoke)
    lock = lock_for(target)
    try:
        cached = c.read_json(target, {})
        if cached.get("status") == "complete":
            return cached
        base_cell = base_cell_path(args.smoke)
        if not base_cell.is_file():
            raise FileNotFoundError(f"base references are incomplete: {base_cell}")
        model_path = export.merged_target(args.arm, args.step)
        if not c.model_check(model_path)["complete"]:
            raise FileNotFoundError(f"merged model missing: {model_path}")
        tokenizer = c.load_llama_tokenizer()
        model = campaign.load_model(model_path, args.device)
        layers = layers_for(args.step)
        need_raw = args.step in c.GEOMETRY_LANDMARKS
        all_r, all_tail, all_raw, inventories = [], [], [], []
        try:
            for probe in PROBE_NAMES:
                samples = prepare_samples(tokenizer, probe, args.measurement_n)
                profile = profile_model(
                    model,
                    samples,
                    layers,
                    args.device,
                    keep_sample_means=need_raw,
                    forward_batch_size=args.forward_batch_size,
                    max_batch_tokens=args.max_batch_tokens,
                )
                r_rows, tail_rows, spectra = spectrum_rows(
                    model,
                    profile,
                    probe,
                    args.arm,
                    args.step,
                    layers,
                    args.device,
                    smoke=args.smoke,
                )
                branch = "smoke" if args.smoke else "formal"
                spectra_path = (
                    ROOT
                    / "spectra"
                    / branch
                    / args.arm
                    / f"step_{args.step:03d}"
                    / f"{probe}.json"
                )
                c.atomic_json(spectra_path, spectra)
                inventories.append(
                    {
                        "arm": args.arm,
                        "step": args.step,
                        "probe": probe,
                        "track": "per_checkpoint",
                        "spectra_path": str(spectra_path),
                        "sha256": c.sha256_file(spectra_path),
                    }
                )
                all_r.extend(r_rows)
                all_tail.extend(tail_rows)
                if need_raw:
                    base_profile = load_profile(reference_path(probe, args.smoke))
                    all_raw.extend(
                        raw_row(
                            profile,
                            base_profile,
                            args.arm,
                            args.step,
                            probe,
                            layer,
                            args.device,
                        )
                        for layer in layers
                    )
                    del base_profile
                del profile, samples
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            campaign.unload_model(model)
        expected_cells = len(PROBE_NAMES) * len(layers) * len(c.MODULES)
        if len(all_r) != expected_cells * len(EPSILONS) or len(all_tail) != expected_cells:
            raise RuntimeError("Llama geometry cell row-count drift")
        if len(all_raw) != (len(PROBE_NAMES) * len(layers) if need_raw else 0):
            raise RuntimeError("Llama raw-representation row-count drift")
        payload = {
            "schema_version": 1,
            "status": "complete",
            "arm": args.arm,
            "step": args.step,
            "layers": layers,
            "track": "per_checkpoint",
            "r_epsilon": all_r,
            "tail_energy": all_tail,
            "raw_representation": all_raw,
            "spectra_inventory": inventories,
            "forward_execution": {
                "batch_size": args.forward_batch_size,
                "max_batch_tokens": args.max_batch_tokens,
                "early_stop": True,
                "stop_layer": max(layers),
            },
            "created_utc": c.utc_now(),
        }
        c.atomic_json(target, payload)
        return payload
    finally:
        lock.close()


def parse_names(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names or set(names).difference(allowed):
        raise ValueError(f"invalid names={value!r}; allowed={allowed}")
    return names


def parse_steps(value: str) -> tuple[int, ...]:
    steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not steps or set(steps).difference(c.MEASURED_CHECKPOINTS):
        raise ValueError(f"invalid measured steps={value!r}")
    return steps


def output_names(scope: str) -> dict[str, Path]:
    if scope == "full":
        prefix = "llama"
    else:
        prefix = f"llama_{scope}"
    return {
        "r_epsilon": ROOT / f"{prefix}_r_epsilon.csv",
        "tail_energy": ROOT / f"{prefix}_tail_energy.csv",
        "raw_representation": ROOT / f"{prefix}_raw_representation_suite.csv",
        "spectra_inventory": ROOT / f"{prefix}_full_spectra_inventory.csv",
    }


def finalize(arms: tuple[str, ...], steps: tuple[int, ...], scope: str) -> dict[str, Any]:
    payloads = [c.read_json(base_cell_path(False), {})]
    if payloads[0].get("status") != "complete":
        raise RuntimeError("incomplete Llama geometry base reference")
    for arm in arms:
        for step in steps:
            if step == 0:
                continue
            path = cell_path(arm, step, False)
            payload = c.read_json(path, {})
            if payload.get("status") != "complete":
                raise RuntimeError(f"incomplete Llama geometry cell: {arm}/{step}")
            payloads.append(payload)
    r_rows = [row for payload in payloads for row in payload["r_epsilon"]]
    tail_rows = [row for payload in payloads for row in payload["tail_energy"]]
    raw_rows = [row for payload in payloads for row in payload["raw_representation"]]
    spectra = [row for payload in payloads for row in payload["spectra_inventory"]]
    if not all((r_rows, tail_rows, raw_rows, spectra)):
        raise RuntimeError("L3 partial finalization produced an empty output")
    outputs = output_names(scope)
    c.atomic_csv(outputs["r_epsilon"], r_rows)
    c.atomic_csv(outputs["tail_energy"], tail_rows)
    c.atomic_csv(outputs["raw_representation"], raw_rows)
    c.atomic_csv(outputs["spectra_inventory"], spectra)
    for path in outputs.values():
        c.atomic_text(c.MINI / path.name, path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "task": "Cycle09 block3 L3 Llama whitened functional rank + native-space baseline",
        "track": "per_checkpoint_only",
        "layers": {"headline": 14, "robustness": [7, 21]},
        "scope": scope,
        "arms": list(arms),
        "headline_steps": list(steps),
        "robustness_steps": [step for step in steps if step in c.GEOMETRY_LANDMARKS],
        "epsilons": list(EPSILONS),
        "probes": list(PROBE_NAMES),
        "modules": list(c.MODULES),
        "forward_execution": {
            "batch_size": payloads[0]["forward_execution"]["batch_size"],
            "max_batch_tokens": payloads[0]["forward_execution"]["max_batch_tokens"],
            "early_stop": True,
        },
        "delta_w_policy": "adapter BA fp32 only for any update-space analysis; merge-subtract forbidden",
        "native_object_catalog": {
            "raw_representation": ["entropy ER", "PR", "top-k share", "raw/centered anisotropy", "CKA", "domain separability"],
            "weight_update": ["update norm/rank/stable-rank", "spectral concentration/sparsity", "near-zero mass", "principal/off-principal overlap", "subspace locking"],
            "singular_vector": ["principal angle", "U/V rotation", "intruder dimensions"],
            "output_probability": ["logit entropy", "ECE", "PPL", "KL", "task score"],
            "ours": ["per-checkpoint-whitened W_t S_D,t r_epsilon", "tail-energy curve"],
        },
        "instantiated_native_baseline": "raw representation suite only",
        "probe_manifest": c.artifact(probes.MANIFEST),
        "outputs": [c.artifact(path) for path in outputs.values()],
        "row_counts": {
            "r_epsilon": len(r_rows),
            "tail_energy": len(tail_rows),
            "raw_representation": len(raw_rows),
            "spectra_inventory": len(spectra),
        },
        "created_utc": c.utc_now(),
    }
    manifest_name = "llama_geometry_manifest.json" if scope == "full" else f"llama_{scope}_geometry_manifest.json"
    c.atomic_json(ROOT / manifest_name, manifest)
    c.atomic_json(c.MINI / manifest_name, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("reference", "cell", "finalize"), required=True)
    parser.add_argument("--arm", choices=c.ARMS, default="opd")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--measurement-n", type=int, default=0)
    parser.add_argument("--forward-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=16384)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arms", default=",".join(c.ARMS))
    parser.add_argument("--steps", default=",".join(map(str, c.MEASURED_CHECKPOINTS)))
    parser.add_argument("--scope", default="full")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.smoke and not arguments.measurement_n:
        arguments.measurement_n = 2
    if arguments.phase == "reference":
        result = base_reference(arguments)
    elif arguments.phase == "cell":
        result = run_cell(arguments)
    else:
        result = finalize(parse_names(arguments.arms, c.ARMS), parse_steps(arguments.steps), arguments.scope)
    print(json.dumps({"status": result.get("status"), "phase": arguments.phase}, indent=2))
