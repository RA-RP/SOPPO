#!/usr/bin/env python3
"""Cycle 09 C2 raw-ER CI and C3 OPD overcompression/rebound bootstrap.

C2 uses the frozen E_ood corpus and centered residual covariance at L18. C3 uses
the five static probe families, per-checkpoint whitening, and OPD steps 0/40/160.
Both resample samples with windows nested inside each sample.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

import cycle09_r4_campaign as campaign
import cycle09_r4_common as c4
import cycle09_stage3_common as s3
from utils.profiling_utils import _gram_to_svdllm_scaling_diag_matrix


C2_LAYER = 18
C2_STEPS = (0, 5, 10, 20, 40, 80)
C3_LAYER = 18
C3_STEPS = (0, 40, 160)
SBOS_SEEDS = (3, 17, 31)
C2_ROOT = s3.RUN_ROOT / "c2_raw_er"
C3_ROOT = s3.RUN_ROOT / "c3_opd_rebound"


def interval(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan"), float("nan"), float("nan")
    return (
        float(finite.mean()),
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
    )


@contextmanager
def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def normalize_layers(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {int(key): item for key, item in value.items()}


def load_bundle(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("sample_factors", "residual_samples", "residual_sample_means"):
        if key in payload:
            payload[key] = [normalize_layers(item) for item in payload[key]]
    return payload


def sample_ids(samples: list[c4.PreparedSample]) -> list[str]:
    return [sample.sample_id for sample in samples]


def c2_stage_bundle(arm: str, step: int) -> Path:
    label = "base" if step == 0 else arm
    return C2_ROOT / "residual_bundles" / label / s3.step_label(step) / "E_ood.pt"


def c2_existing_bundle(arm: str, step: int) -> Path | None:
    source_arm = "opd" if step == 0 else arm
    path = (
        c4.RUN_ROOT
        / "scratch/bootstrap_factors"
        / source_arm
        / s3.step_label(step)
        / "E_ood.pt"
    )
    return path if path.is_file() else None


def bundle_has_residuals(path: Path, expected_ids: list[str], layer: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_bundle(path)
        residuals = payload.get("residual_samples", [])
        ids = payload.get("sample_ids", expected_ids)
        valid = (
            list(ids) == expected_ids
            and len(residuals) == len(expected_ids)
            and all(layer in item for item in residuals)
            and len(payload.get("residual_sample_means", [])) == len(expected_ids)
        )
        del payload
        return bool(valid)
    except (OSError, KeyError, ValueError, RuntimeError):
        return False


def ensure_c2_bundle(
    arm: str,
    step: int,
    samples: list[c4.PreparedSample],
    device: str,
) -> Path:
    ids = sample_ids(samples)
    existing = c2_existing_bundle(arm, step)
    if existing is not None and bundle_has_residuals(existing, ids, C2_LAYER):
        print(f"[C2 source bundle] {arm}/{step} -> {existing}", flush=True)
        return existing
    target = c2_stage_bundle(arm, step)
    if bundle_has_residuals(target, ids, C2_LAYER):
        print(f"[C2 cached bundle] {arm}/{step}", flush=True)
        return target

    with lock(target.with_suffix(".lock")):
        if bundle_has_residuals(target, ids, C2_LAYER):
            return target
        path = s3.require_model(arm, step)
        print(f"[C2 collect] {arm}/{step} model={path}", flush=True)
        model = campaign.load_model(path, device)
        try:
            profile = campaign.collect_profile(
                model,
                samples,
                [C2_LAYER],
                device,
                keep_factors=False,
                keep_residual_samples=True,
            )
            payload = {
                "schema_version": 1,
                "task": "C2",
                "arm": arm,
                "step": int(step),
                "probe": "E_ood",
                "layer": C2_LAYER,
                "sample_ids": ids,
                "residual_samples": profile["residual_samples"],
                "residual_sample_means": profile["residual_sample_means"],
            }
            atomic_torch(target, payload)
            del profile, payload
        finally:
            campaign.unload_model(model)
            gc.collect()
            torch.cuda.empty_cache()
    return target


def bootstrap_indices(n: int, draws: int, *seed_parts: Any) -> np.ndarray:
    rng = np.random.default_rng(c4.stable_seed(42, *seed_parts))
    return rng.integers(0, n, size=(draws, n), dtype=np.int64)


def normalized_raw_er(eigenvalues: torch.Tensor) -> float:
    values = eigenvalues.clamp_min(0)
    total = values.sum()
    if not torch.isfinite(total) or float(total) <= 0:
        return float("nan")
    probabilities = values / total
    entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum()
    return float((torch.exp(entropy) / values.numel()).cpu())


@torch.inference_mode()
def raw_er_draws(
    bundle: dict[str, Any],
    indices: np.ndarray,
    layer: int,
    device: str,
) -> tuple[np.ndarray, float]:
    residuals = bundle["residual_samples"]
    means_cpu = bundle["residual_sample_means"]
    n = len(residuals)
    if indices.shape[1] != n:
        raise ValueError(f"bootstrap width {indices.shape[1]} != samples {n}")
    dimension = int(residuals[0][layer].shape[1])
    seconds = torch.empty(
        (n, dimension, dimension), dtype=torch.float32, device=device
    )
    means = torch.stack([item[layer].float() for item in means_cpu]).to(device)
    for sample_index, item in enumerate(residuals):
        factor = item[layer].to(device=device, dtype=torch.float32)
        seconds[sample_index] = factor.T @ factor
        del factor
        if (sample_index + 1) % 16 == 0:
            print(f"[raw ER moments] {sample_index + 1}/{n}", flush=True)

    def evaluate(weights: torch.Tensor) -> float:
        second = torch.tensordot(weights, seconds, dims=([0], [0]))
        mean = weights @ means
        covariance = second - torch.outer(mean, mean)
        eigenvalues = torch.linalg.eigvalsh(covariance)
        value = normalized_raw_er(eigenvalues)
        del second, mean, covariance, eigenvalues
        return value

    point = evaluate(torch.full((n,), 1.0 / n, device=device, dtype=torch.float32))
    output = np.empty(indices.shape[0], dtype=np.float64)
    for draw, sampled in enumerate(indices):
        counts = np.bincount(sampled, minlength=n).astype(np.float32) / n
        output[draw] = evaluate(torch.from_numpy(counts).to(device))
        if (draw + 1) % 32 == 0 or draw + 1 == len(output):
            print(f"[raw ER bootstrap] {draw + 1}/{len(output)}", flush=True)
    del seconds, means
    gc.collect()
    torch.cuda.empty_cache()
    return output, point


def c2_draw_path(arm: str, step: int, draws: int) -> Path:
    label = "base" if step == 0 else arm
    return C2_ROOT / "draws" / label / s3.step_label(step) / f"raw_er_L18_d{draws}.npz"


def run_c2_cells(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(s3.BASE_MODEL), trust_remote_code=True)
    corpus = c4.RUN_ROOT / "corpora/fixed/E_ood.jsonl"
    samples = c4.prepare_samples(
        corpus,
        tokenizer,
        corpus_id="E_ood",
        window_seed=c4.WINDOW_SEED,
        max_context_tokens=c4.MAX_CONTEXT_TOKENS,
    )
    ids = sample_ids(samples)
    indices = bootstrap_indices(len(samples), args.draws, "C2", "E_ood")
    cells = [("opd", 0)] + [
        (arm, step) for arm in args.arms for step in C2_STEPS if step != 0
    ]
    seen = set()
    for arm, step in cells:
        key = ("base", 0) if step == 0 else (arm, step)
        if key in seen:
            continue
        seen.add(key)
        target = c2_draw_path(arm, step, args.draws)
        metadata = target.with_suffix(".json")
        if target.is_file() and metadata.is_file():
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            if (
                payload.get("draws") == args.draws
                and payload.get("sample_ids_sha256") == s3.sha256_json(ids)
            ):
                print(f"[C2 cached draws] {key}", flush=True)
                continue
            raise RuntimeError(f"incompatible C2 cache: {target}")
        with lock(target.with_suffix(".lock")):
            if target.is_file() and metadata.is_file():
                continue
            bundle_path = ensure_c2_bundle(arm, step, samples, args.device)
            bundle = load_bundle(bundle_path)
            values, point = raw_er_draws(bundle, indices, C2_LAYER, args.device)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, draws=values, point=np.array([point]))
            os.replace(temporary, target)
            s3.atomic_json(
                metadata,
                {
                    "schema_version": 1,
                    "task": "C2",
                    "arm": "base" if step == 0 else arm,
                    "step": int(step),
                    "probe": "E_ood",
                    "layer": C2_LAYER,
                    "draws": args.draws,
                    "sample_count": len(samples),
                    "sample_ids_sha256": s3.sha256_json(ids),
                    "bundle_path": str(bundle_path),
                    "metric": "centered covariance normalized effective rank",
                },
            )
            del bundle, values
            gc.collect()
            torch.cuda.empty_cache()


def load_npz(path: Path) -> tuple[np.ndarray, float]:
    with np.load(path) as payload:
        return payload["draws"].astype(np.float64), float(payload["point"][0])


def finalize_c2(args: argparse.Namespace) -> None:
    base_draws, base_point = load_npz(c2_draw_path("opd", 0, args.draws))
    rows = []
    structural = []
    for arm in s3.ARMS:
        by_step = {}
        for step in C2_STEPS:
            if step == 0:
                current, point = base_draws, base_point
            else:
                current, point = load_npz(c2_draw_path(arm, step, args.draws))
            delta = current - base_draws
            mean, low, high = interval(delta)
            by_step[step] = delta
            rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "probe": "E_ood",
                    "layer": C2_LAYER,
                    "metric": "normalized_raw_er_current_minus_base",
                    "point_current": point,
                    "point_base": base_point,
                    "point_delta": point - base_point,
                    "bootstrap_mean": mean,
                    "ci95_lo": low,
                    "ci95_hi": high,
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                    "draws": args.draws,
                    "bootstrap_unit": "sample; windows nested",
                }
            )
        stacked = np.stack([by_step[step] for step in s3.TRANSIENT_STEPS])
        maxima = stacked.max(axis=0)
        peak_indices = stacked.argmax(axis=0)
        mean, low, high = interval(maxima)
        counts = {
            str(step): int((peak_indices == index).sum())
            for index, step in enumerate(s3.TRANSIENT_STEPS)
        }
        structural.append(
            {
                "arm": arm,
                "probe": "E_ood",
                "layer": C2_LAYER,
                "metric": "max_transient_normalized_raw_er_delta",
                "bootstrap_mean": mean,
                "ci95_lo": low,
                "ci95_hi": high,
                "probability_max_gt_zero": float((maxima > 0).mean()),
                "peak_step_draw_counts_json": json.dumps(counts, sort_keys=True),
                "draws": args.draws,
            }
        )
    output = s3.MINI / "C2_raw_er_bootstrap.csv"
    structure_output = s3.MINI / "C2_raw_er_transient_structure.csv"
    s3.atomic_csv(output, rows)
    s3.atomic_csv(structure_output, structural)

    audit_rows = []
    for name in ("R5_raw_er_fixed_ckpt.csv", "R5_raw_er_fixed.csv"):
        path = s3.MINI / name
        frame = pd.read_csv(path)
        arm_column = "arm" if "arm" in frame else "weight_arm"
        audit_rows.append(
            {
                "file": name,
                "rows": len(frame),
                "columns_json": json.dumps(list(frame.columns)),
                "arms_or_weight_arms_json": json.dumps(
                    sorted(map(str, frame[arm_column].unique()))
                ),
                "protocol_role": (
                    "legacy_generated_text_cross_grid"
                    if "weight_arm" in frame
                    else "fixed_probe_base_and_offkd_grid"
                ),
                "directly_mergeable_with_C2": False,
            }
        )
    audit_output = s3.MINI / "C2_raw_er_protocol_audit.csv"
    s3.atomic_csv(audit_output, audit_rows)
    manifest = s3.MINI / "C2_raw_er_manifest.json"
    s3.atomic_json(
        manifest,
        {
            "schema_version": 1,
            "status": "complete",
            "contract": s3.artifact(s3.CONTRACT),
            "probe": "E_ood",
            "layer": C2_LAYER,
            "arms": list(s3.ARMS),
            "steps": list(C2_STEPS),
            "draws": args.draws,
            "indices_shared_across_arms_and_steps": True,
            "outputs": [
                s3.artifact(path)
                for path in (output, structure_output, audit_output)
            ],
        },
    )
    print(f"[C2 finalized] rows={len(rows)} structural={len(structural)}", flush=True)


def c3_tasks() -> dict[str, Path]:
    tasks = {
        "legacy_S_math": c4.RUN_ROOT / "corpora/fixed/legacy_S_math.jsonl",
        "E_ood": c4.RUN_ROOT / "corpora/fixed/E_ood.jsonl",
        "E_general": c4.RUN_ROOT / "corpora/fixed/E_general.jsonl",
        "E_math_hard": c4.RUN_ROOT / "corpora/fixed/E_math_hard.jsonl",
    }
    for seed in SBOS_SEEDS:
        tasks[f"S_bos__g{seed}"] = c4.generated_corpus_path(
            "S", "bos", seed, run_root=c4.RUN_ROOT
        )
    return tasks


def c3_existing_bundle(step: int, task: str) -> Path | None:
    path = (
        c4.RUN_ROOT
        / "scratch/bootstrap_factors/opd"
        / s3.step_label(step)
        / f"{task}.pt"
    )
    if not path.is_file():
        return None
    try:
        bundle = load_bundle(path)
        valid = bool(bundle.get("sample_factors")) and all(
            C3_LAYER in item and item[C3_LAYER]
            for item in bundle["sample_factors"]
        )
        del bundle
        return path if valid else None
    except (OSError, ValueError, RuntimeError, KeyError):
        return None


def c3_cache_path(step: int, task: str, draws: int) -> Path:
    return C3_ROOT / "draws" / s3.step_label(step) / f"{task}__L18_d{draws}.npz"


def scaling_for_draw(
    sample_factors: list[dict[int, dict[str, torch.Tensor]]],
    indices: np.ndarray,
    device: str,
) -> dict[str, torch.Tensor]:
    result = {}
    for group in c4.GROUP_TO_MODULES:
        pieces = [
            sample_factors[int(index)][C3_LAYER][group] for index in indices
        ]
        matrix = torch.cat(pieces, dim=0).to(device=device, dtype=torch.float32)
        matrix.mul_(1.0 / math.sqrt(len(indices)))
        gram = matrix.T @ matrix
        result[group] = _gram_to_svdllm_scaling_diag_matrix(
            gram,
            cholesky_jitter=1e-5,
            singular_floor=0.0,
        ).to(device=device, dtype=torch.float32)
        del pieces, matrix, gram
    return result


@torch.inference_mode()
def r_epsilon_draws(
    model,
    bundle: dict[str, Any],
    indices_by_draw: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    factors = bundle["sample_factors"]
    weights = {
        module: campaign.module_at(
            model, C3_LAYER, module
        ).weight.detach().float()
        for module in c4.MODULES
    }

    def evaluate(indices: np.ndarray) -> np.ndarray:
        scalings = scaling_for_draw(factors, indices, device)
        values = np.empty(len(c4.MODULES), dtype=np.float64)
        for module_index, module in enumerate(c4.MODULES):
            spectrum = torch.linalg.svdvals(
                weights[module] @ scalings[c4.MODULE_TO_GROUP[module]]
            )
            values[module_index] = c4.functional_rank(
                spectrum.cpu().numpy(), 0.05
            )
        scalings.clear()
        torch.cuda.empty_cache()
        return values

    point = evaluate(np.arange(len(factors), dtype=np.int64))
    output = np.empty(
        (len(indices_by_draw), len(c4.MODULES)), dtype=np.float64
    )
    for draw, indices in enumerate(indices_by_draw):
        output[draw] = evaluate(indices)
        if (draw + 1) % 32 == 0 or draw + 1 == len(output):
            print(f"[C3 rank bootstrap] {draw + 1}/{len(output)}", flush=True)
    weights.clear()
    return output, point


def profile_factor_bundle(
    model,
    samples: list[c4.PreparedSample],
    task: str,
    device: str,
) -> dict[str, Any]:
    profile = campaign.collect_profile(
        model,
        samples,
        [C3_LAYER],
        device,
        keep_factors=True,
        keep_residual_samples=False,
        factor_layers=(C3_LAYER,),
    )
    return {
        "schema_version": 1,
        "task": task,
        "layer": C3_LAYER,
        "sample_ids": sample_ids(samples),
        "sample_factors": profile["sample_factors"],
    }


def run_c3_cells(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        str(s3.BASE_MODEL), trust_remote_code=True
    )
    prepared = {
        task: c4.prepare_samples(
            path,
            tokenizer,
            corpus_id=task,
            window_seed=c4.WINDOW_SEED,
            max_context_tokens=c4.MAX_CONTEXT_TOKENS,
        )
        for task, path in c3_tasks().items()
    }
    for step in args.c3_steps:
        pending = [
            task
            for task in prepared
            if not c3_cache_path(step, task, args.draws).is_file()
        ]
        if not pending:
            print(f"[C3 cached step] {step}", flush=True)
            continue
        path = s3.require_model("opd", step)
        print(f"[C3 model] step={step} pending={pending}", flush=True)
        model = campaign.load_model(path, args.device)
        try:
            for task in pending:
                target = c3_cache_path(step, task, args.draws)
                with lock(target.with_suffix(".lock")):
                    if target.is_file():
                        continue
                    existing = c3_existing_bundle(step, task)
                    bundle = (
                        load_bundle(existing)
                        if existing is not None
                        else profile_factor_bundle(
                            model, prepared[task], task, args.device
                        )
                    )
                    n = len(bundle["sample_factors"])
                    indices = bootstrap_indices(n, args.draws, "C3", task)
                    values, point = r_epsilon_draws(
                        model, bundle, indices, args.device
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(".npz.tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(
                            handle, draws=values, point=point
                        )
                    os.replace(temporary, target)
                    s3.atomic_json(
                        target.with_suffix(".json"),
                        {
                            "schema_version": 1,
                            "task": "C3",
                            "arm": "opd",
                            "step": step,
                            "probe_task": task,
                            "layer": C3_LAYER,
                            "draws": args.draws,
                            "sample_count": n,
                            "sample_ids_sha256": s3.sha256_json(
                                sample_ids(prepared[task])
                            ),
                            "factor_source": (
                                str(existing)
                                if existing
                                else "ephemeral_profile"
                            ),
                        },
                    )
                    del bundle, values
                    gc.collect()
                    torch.cuda.empty_cache()
        finally:
            campaign.unload_model(model)
            gc.collect()
            torch.cuda.empty_cache()


def load_c3(
    step: int, task: str, draws: int
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(c3_cache_path(step, task, draws)) as payload:
        return (
            payload["draws"].astype(np.float64),
            payload["point"].astype(np.float64),
        )


def c3_family(
    step: int, family: str, draws: int
) -> tuple[np.ndarray, np.ndarray]:
    if family != "S_bos":
        return load_c3(step, family, draws)
    values = [
        load_c3(step, f"S_bos__g{seed}", draws) for seed in SBOS_SEEDS
    ]
    return (
        np.mean([item[0] for item in values], axis=0),
        np.mean([item[1] for item in values], axis=0),
    )


def finalize_c3(args: argparse.Namespace) -> None:
    families = (
        "legacy_S_math",
        "E_ood",
        "E_general",
        "E_math_hard",
        "S_bos",
    )
    rows = []
    for family in families:
        arrays = {
            step: c3_family(step, family, args.draws) for step in C3_STEPS
        }
        contrasts = {
            "overcompression_depth_r0_minus_r40": (
                arrays[0][0] - arrays[40][0],
                arrays[0][1] - arrays[40][1],
            ),
            "rebound_r160_minus_r40": (
                arrays[160][0] - arrays[40][0],
                arrays[160][1] - arrays[40][1],
            ),
            "net_r160_minus_r0": (
                arrays[160][0] - arrays[0][0],
                arrays[160][1] - arrays[0][1],
            ),
        }
        for contrast, (draw_values, point_values) in contrasts.items():
            for module_index, module in enumerate(
                (*c4.MODULES, "mean_fixed_7_modules")
            ):
                values = (
                    draw_values.mean(axis=1)
                    if module == "mean_fixed_7_modules"
                    else draw_values[:, module_index]
                )
                point = (
                    float(point_values.mean())
                    if module == "mean_fixed_7_modules"
                    else float(point_values[module_index])
                )
                mean, low, high = interval(values)
                rows.append(
                    {
                        "arm": "opd",
                        "probe_family": family,
                        "layer": C3_LAYER,
                        "track": "per_checkpoint",
                        "epsilon": 0.05,
                        "module": module,
                        "contrast": contrast,
                        "point_estimate": point,
                        "bootstrap_mean": mean,
                        "ci95_lo": low,
                        "ci95_hi": high,
                        "ci_excludes_zero": bool(
                            low > 0 or high < 0
                        ),
                        "draws": args.draws,
                        "bootstrap_unit": "sample; windows nested",
                    }
                )
    output = s3.MINI / "C3_opd_overcompression_rebound_ci.csv"
    s3.atomic_csv(output, rows)
    manifest = (
        s3.MINI / "C3_opd_overcompression_rebound_manifest.json"
    )
    s3.atomic_json(
        manifest,
        {
            "schema_version": 1,
            "status": "complete",
            "contract": s3.artifact(s3.CONTRACT),
            "arm": "opd",
            "steps": list(C3_STEPS),
            "layer": C3_LAYER,
            "track": "per_checkpoint",
            "epsilon": 0.05,
            "draws": args.draws,
            "probes": list(families),
            "S_bos_generation_seeds": list(SBOS_SEEDS),
            "output": s3.artifact(output),
        },
    )
    print(f"[C3 finalized] rows={len(rows)}", flush=True)


def synthetic_smoke() -> None:
    torch.manual_seed(7)
    residuals = []
    means = []
    for _ in range(6):
        factor = torch.randn(5, 8, dtype=torch.float16) / math.sqrt(5)
        residuals.append({C2_LAYER: factor})
        means.append({C2_LAYER: torch.randn(8) * 0.01})
    bundle = {
        "residual_samples": residuals,
        "residual_sample_means": means,
    }
    indices = bootstrap_indices(6, 8, "synthetic")
    values, point = raw_er_draws(
        bundle, indices, C2_LAYER, "cpu"
    )
    if (
        values.shape != (8,)
        or not np.isfinite(values).all()
        or not math.isfinite(point)
    ):
        raise RuntimeError("C2 synthetic smoke failed")
    mean, low, high = interval(values)
    if not low <= mean <= high:
        raise RuntimeError("interval smoke failed")
    print(
        json.dumps(
            {"status": "ok", "raw_point": point, "draws": values.tolist()}
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("c2", "c3", "both"), default="both"
    )
    parser.add_argument(
        "--phase", choices=("cells", "finalize", "all"), default="all"
    )
    parser.add_argument("--arms", default="all")
    parser.add_argument("--c3-steps", default="0,40,160")
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    s3.assert_contract()
    if args.smoke:
        synthetic_smoke()
        return
    args.arms = s3.parse_names(args.arms, s3.ARMS)
    args.c3_steps = s3.parse_ints(args.c3_steps, C3_STEPS)
    if (
        args.task in ("c2", "both")
        and args.phase in ("cells", "all")
    ):
        run_c2_cells(args)
    if (
        args.task in ("c3", "both")
        and args.phase in ("cells", "all")
    ):
        run_c3_cells(args)
    if (
        args.task in ("c2", "both")
        and args.phase in ("finalize", "all")
    ):
        finalize_c2(args)
    if (
        args.task in ("c3", "both")
        and args.phase in ("finalize", "all")
    ):
        finalize_c3(args)


if __name__ == "__main__":
    main()
