#!/usr/bin/env python3
"""S1-6: full principal-angle spectra and rotating base-direction audit.

The handoff expected saved fp64 UV artifacts, but those bases were never persisted.
This script recomputes the same frozen-base E_ood/L18 bases with fp64 SVD and fp64
QR, checkpoints per model cell, and emits readings only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import tempfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import cycle09_r4_campaign as camp
import cycle09_r4_common as c4


TASK = "E_ood"
LAYER = 18
EPSILON = 0.05
STEPS = (5, 20, 40, 160, 624)
ARMS = ("opd", "sft", "offkd")
OFFKD = Path("/root/autodl-tmp/cycle09_offkd/_merged_models")
REFERENCE = Path("/root/autodl-tmp/cycle09_r4/scratch/references/E_ood.pt")
RUN_ROOT = Path("/root/autodl-tmp/cycle09_s1")
MINI = Path(
    "/root/LLM-output-density/mypaper/local_experiment_results/"
    "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
PROTOCOL_VERSION = "s1-6-eood-l18-fp64-svd-qr-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(tmp, path)


def model_path(arm: str, step: int) -> Path:
    if arm == "offkd":
        path = OFFKD / c4.step_label(step)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(path)
        return path
    return c4.model_path(arm, step)


def reference_scales(device: str):
    if not REFERENCE.is_file():
        raise FileNotFoundError(REFERENCE)
    payload = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    profile = {"grams": {int(key): value for key, value in payload["grams"].items()}}
    return camp.scaling_by_group(profile, [LAYER], device)


@torch.no_grad()
def orthogonal_bases(model, scales, modules: tuple[str, ...], device: str):
    bases = {}
    for module in modules:
        group = c4.MODULE_TO_GROUP[module]
        weight = camp.module_at(model, LAYER, module).weight.detach().to(
            device=device, dtype=torch.float64
        )
        matrix = weight @ scales[LAYER][group].double()
        u, sigma, vh = torch.linalg.svd(matrix, full_matrices=False)
        # Full-basis QR is equivalent for every leading truncated subspace and lets
        # the base factorization be reused across all checkpoints.
        qu = torch.linalg.qr(u, mode="reduced")[0]
        qv = torch.linalg.qr(vh.T, mode="reduced")[0]
        bases[module] = {
            "u": qu,
            "v": qv,
            "sigma": sigma.detach().cpu().numpy().astype(np.float64),
        }
        del weight, matrix, u, sigma, vh
        torch.cuda.empty_cache()
    return bases


def angle_payload(q0: torch.Tensor, qt_raw: torch.Tensor, rank: int) -> dict:
    q0r = q0[:, :rank]
    qt = torch.linalg.qr(qt_raw[:, :rank].double(), mode="reduced")[0]
    cross = q0r.T @ qt
    cosine = torch.linalg.svdvals(cross).clamp(0.0, 1.0)
    canonical = torch.rad2deg(torch.arccos(cosine)).detach().cpu().numpy()

    # For the sigma-rank question, retain the identity of each base singular
    # direction: angle from base direction j to the current r-dimensional subspace.
    projection_norm = torch.linalg.vector_norm(qt.T @ q0r, dim=0).clamp(0.0, 1.0)
    base_direction = (
        torch.rad2deg(torch.arccos(projection_norm)).detach().cpu().numpy()
    )
    del q0r, qt, cross, cosine, projection_norm
    return {
        "canonical_angles_deg": canonical.astype(np.float64).tolist(),
        "base_direction_angles_deg": base_direction.astype(np.float64).tolist(),
    }


@torch.no_grad()
def compute_cell(
    *,
    arm: str,
    step: int,
    base_bases: dict,
    scales,
    modules: tuple[str, ...],
    device: str,
    protocol_id: str,
) -> dict:
    path = model_path(arm, step)
    model = camp.load_model(path, device)
    module_payload = {}
    try:
        for module in modules:
            group = c4.MODULE_TO_GROUP[module]
            weight = camp.module_at(model, LAYER, module).weight.detach().to(
                device=device, dtype=torch.float64
            )
            matrix = weight @ scales[LAYER][group].double()
            ut, sigma, vht = torch.linalg.svd(matrix, full_matrices=False)
            sigma_np = sigma.detach().cpu().numpy().astype(np.float64)
            rank = c4.functional_rank(sigma_np, EPSILON)
            rank = max(
                1,
                min(
                    rank,
                    base_bases[module]["u"].shape[1],
                    base_bases[module]["v"].shape[1],
                    ut.shape[1],
                    vht.shape[0],
                ),
            )
            u_payload = angle_payload(base_bases[module]["u"], ut, rank)
            v_payload = angle_payload(base_bases[module]["v"], vht.T, rank)
            module_payload[module] = {
                "rank": rank,
                "sigma": sigma_np.tolist(),
                "u": u_payload,
                "v": v_payload,
            }
            del weight, matrix, ut, sigma, vht
            torch.cuda.empty_cache()
            print(
                f"[S1-6] {arm}/{c4.step_label(step)} {module} rank={rank}",
                flush=True,
            )
    finally:
        camp.unload_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "arm": arm,
        "step": step,
        "model_path": str(path),
        "task": TASK,
        "layer": LAYER,
        "epsilon": EPSILON,
        "modules": module_payload,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def cache_path(root: Path, arm: str, step: int) -> Path:
    return root / arm / f"{c4.step_label(step)}.json"


def load_or_compute(
    *,
    root: Path,
    arm: str,
    step: int,
    base_bases: dict,
    scales,
    modules: tuple[str, ...],
    device: str,
    protocol_id: str,
) -> dict:
    target = cache_path(root, arm, step)
    if target.is_file():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_id") == protocol_id
            and set(payload.get("modules", {})) == set(modules)
        ):
            print(f"[S1-6 cached] {arm}/{c4.step_label(step)}", flush=True)
            return payload
        raise RuntimeError(f"incompatible S1-6 cache: {target}")
    payload = compute_cell(
        arm=arm,
        step=step,
        base_bases=base_bases,
        scales=scales,
        modules=modules,
        device=device,
        protocol_id=protocol_id,
    )
    atomic_json(payload, target)
    return payload


def interval_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("invalid angle array")
    return {
        "rank_used": len(array),
        "theta_max_deg": float(array.max()),
        "theta_mean_deg": float(array.mean()),
        "theta_median_deg": float(np.median(array)),
        "n_gt_1deg": int((array > 1.0).sum()),
        "fraction_gt_1deg": float((array > 1.0).mean()),
        "n_gt_5deg": int((array > 5.0).sum()),
        "fraction_gt_5deg": float((array > 5.0).mean()),
    }


def build_tables(cells: dict[tuple[str, int], dict], modules: tuple[str, ...]):
    analysis_rows = []
    principal_rows = []
    rank_rows = []
    direction_sets = {}
    for (arm, step), cell in cells.items():
        for module in modules:
            payload = cell["modules"][module]
            rank = int(payload["rank"])
            for space in ("u", "v"):
                canonical = payload[space]["canonical_angles_deg"]
                base_angles = payload[space]["base_direction_angles_deg"]
                if len(canonical) != rank or len(base_angles) != rank:
                    raise RuntimeError(f"angle length mismatch: {arm}/{step}/{module}/{space}")
                summary = interval_summary(canonical)
                analysis_rows.append(
                    {
                        "arm": arm,
                        "step": step,
                        "task_id": TASK,
                        "layer": LAYER,
                        "module": module,
                        "space": space.upper(),
                        "epsilon": EPSILON,
                        **summary,
                        "base_direction_n_gt_5deg": int(
                            (np.asarray(base_angles) > 5.0).sum()
                        ),
                        "base_direction_fraction_gt_5deg": float(
                            (np.asarray(base_angles) > 5.0).mean()
                        ),
                    }
                )
                for index, angle in enumerate(canonical, start=1):
                    principal_rows.append(
                        {
                            "arm": arm,
                            "step": step,
                            "task_id": TASK,
                            "layer": LAYER,
                            "module": module,
                            "space": space.upper(),
                            "epsilon": EPSILON,
                            "principal_index_cosine_desc": index,
                            "principal_angle_deg": angle,
                        }
                    )
                order = np.argsort(np.asarray(base_angles))[::-1][:10]
                for order_index, base_index in enumerate(order, start=1):
                    rank_rows.append(
                        {
                            "arm": arm,
                            "step": step,
                            "task_id": TASK,
                            "layer": LAYER,
                            "module": module,
                            "space": space.upper(),
                            "epsilon": EPSILON,
                            "rotation_order_desc": order_index,
                            "base_sigma_rank": int(base_index) + 1,
                            "base_direction_angle_deg": float(base_angles[base_index]),
                            "base_sigma_value": float(
                                base_sigma_for(module, cells)[int(base_index)]
                            ),
                        }
                    )
                direction_sets[(arm, step, module, space)] = {
                    index + 1
                    for index, angle in enumerate(base_angles)
                    if angle > 5.0
                }

    overlap_rows = []
    for step in STEPS:
        if not all((arm, step) in cells for arm in ARMS):
            continue
        for module in modules:
            for space in ("u", "v"):
                for arm_a, arm_b in combinations(ARMS, 2):
                    left = direction_sets[(arm_a, step, module, space)]
                    right = direction_sets[(arm_b, step, module, space)]
                    intersection = len(left & right)
                    union = len(left | right)
                    minimum = min(len(left), len(right))
                    overlap_rows.append(
                        {
                            "step": step,
                            "task_id": TASK,
                            "layer": LAYER,
                            "module": module,
                            "space": space.upper(),
                            "epsilon": EPSILON,
                            "arm_a": arm_a,
                            "arm_b": arm_b,
                            "n_a_gt_5deg": len(left),
                            "n_b_gt_5deg": len(right),
                            "n_intersection": intersection,
                            "n_union": union,
                            "jaccard": intersection / union if union else np.nan,
                            "overlap_coefficient": (
                                intersection / minimum if minimum else np.nan
                            ),
                            "both_sets_empty": bool(not left and not right),
                        }
                    )
    return (
        pd.DataFrame(analysis_rows),
        pd.DataFrame(principal_rows),
        pd.DataFrame(rank_rows),
        pd.DataFrame(overlap_rows),
    )


_BASE_SIGMA: dict[str, np.ndarray] = {}


def base_sigma_for(module: str, cells: dict) -> np.ndarray:
    del cells
    return _BASE_SIGMA[module]


def parity_check(analysis: pd.DataFrame, smoke: bool) -> list[dict]:
    if smoke:
        return []
    path = MINI / "R5_theta_reps.csv"
    existing = pd.read_csv(path)
    existing = existing[
        (existing["probe"] == TASK)
        & (existing["track"] == "frozen_base")
        & (existing["layer"] == LAYER)
        & (existing["epsilon"] == EPSILON)
    ]
    checks = []
    for row in analysis.itertuples(index=False):
        match = existing[
            (existing["arm"] == row.arm)
            & (existing["step"] == row.step)
            & (existing["module"] == row.module)
        ]
        if len(match) != 1:
            raise RuntimeError(
                f"R5 theta parity row count={len(match)}: "
                f"{row.arm}/{row.step}/{row.module}"
            )
        reference = match.iloc[0]
        if row.space == "U":
            max_reference = float(reference["theta_u_max_deg"])
            mean_reference = float(reference["theta_u_mean_deg"])
        else:
            max_reference = float(reference["theta_v_max_deg"])
            mean_reference = float(reference["theta_v_mean_deg"])
        max_diff = abs(row.theta_max_deg - max_reference)
        mean_diff = abs(row.theta_mean_deg - mean_reference)
        checks.append(
            {
                "arm": row.arm,
                "step": int(row.step),
                "module": row.module,
                "space": row.space,
                "max_abs_diff_deg": max_diff,
                "mean_abs_diff_deg": mean_diff,
            }
        )
        if max(max_diff, mean_diff) > 0.02:
            raise RuntimeError(
                f"R5 theta parity failed: {row.arm}/{row.step}/{row.module}/"
                f"{row.space}, max_diff={max_diff}, mean_diff={mean_diff}"
            )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    arms = ("opd",) if args.smoke else ARMS
    steps = (5,) if args.smoke else STEPS
    modules = (c4.MODULES[0],) if args.smoke else c4.MODULES
    protocol = {
        "version": PROTOCOL_VERSION,
        "mode": "smoke" if args.smoke else "formal",
        "task": TASK,
        "layer": LAYER,
        "epsilon": EPSILON,
        "arms": list(arms),
        "steps": list(steps),
        "modules": list(modules),
        "reference_path": str(REFERENCE),
        "reference_sha256": sha256_file(REFERENCE),
        "weight_load_dtype": "float16, matching R5-A2",
        "svd_dtype": "float64",
        "reorthogonalization": "float64 reduced QR",
        "principal_angles": "Bjorck-Golub arccos(svdvals(Q0.T @ Qt))",
        "base_direction_angle": "arccos(||Qt.T @ q0_j||_2)",
        "large_rotation_threshold_deg": 5.0,
    }
    protocol_id = sha256_json(protocol)
    cache_root = RUN_ROOT / ("s1_6_cache_smoke" if args.smoke else "s1_6_cache")

    scales = reference_scales(args.device)
    base_model = camp.load_model(c4.BASE_MODEL, args.device)
    try:
        base_bases = orthogonal_bases(base_model, scales, modules, args.device)
        _BASE_SIGMA.clear()
        _BASE_SIGMA.update(
            {
                module: base_bases[module]["sigma"].copy()
                for module in modules
            }
        )
        cells = {}
        for arm in arms:
            for step in steps:
                cells[(arm, step)] = load_or_compute(
                    root=cache_root,
                    arm=arm,
                    step=step,
                    base_bases=base_bases,
                    scales=scales,
                    modules=modules,
                    device=args.device,
                    protocol_id=protocol_id,
                )
    finally:
        camp.unload_model(base_model)
        scales.clear()
        if "base_bases" in locals():
            base_bases.clear()
        gc.collect()
        torch.cuda.empty_cache()

    analysis, principal, ranks, overlap = build_tables(cells, modules)
    parity = parity_check(analysis, args.smoke)
    if args.smoke:
        print(analysis.to_string(index=False))
        print(ranks.to_string(index=False))
        return

    expected_analysis = len(ARMS) * len(STEPS) * len(c4.MODULES) * 2
    expected_overlap = len(STEPS) * len(c4.MODULES) * 2 * 3
    if len(analysis) != expected_analysis or len(overlap) != expected_overlap:
        raise RuntimeError(
            f"incomplete S1-6 grid analysis={len(analysis)}/{expected_analysis}, "
            f"overlap={len(overlap)}/{expected_overlap}"
        )
    atomic_csv(analysis, MINI / "S1_direction_analysis.csv")
    atomic_csv(principal, MINI / "S1_direction_principal_angles.csv")
    atomic_csv(ranks, MINI / "S1_direction_rank_distribution.csv")
    atomic_csv(overlap, MINI / "S1_direction_overlap.csv")
    atomic_json(
        {
            "schema_version": 1,
            "task": "S1-6",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "deviation": (
                "The fp64 UV bases expected by the handoff were not persisted; "
                "the frozen R4 E_ood reference and stored checkpoint weights were "
                "used to recompute the R5-A2 fp64 SVD+QR quantities."
            ),
            "protocol": protocol,
            "protocol_id": protocol_id,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "base_model": str(c4.BASE_MODEL),
            "cache_root": str(cache_root),
            "n_analysis_rows": len(analysis),
            "n_principal_angle_rows": len(principal),
            "n_rank_rows": len(ranks),
            "n_overlap_rows": len(overlap),
            "r5_theta_parity": parity,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        MINI / "S1_direction_analysis_manifest.json",
    )
    print(
        f"[S1-6] analysis={len(analysis)} principal={len(principal)} "
        f"rank={len(ranks)} overlap={len(overlap)}"
    )
    print(analysis.to_string(index=False))


if __name__ == "__main__":
    main()
