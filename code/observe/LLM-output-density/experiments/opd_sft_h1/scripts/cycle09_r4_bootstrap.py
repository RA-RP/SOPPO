#!/usr/bin/env python3
"""Exact sample bootstrap for Round 4 L18 re-derivation and discriminability."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import cycle09_r4_common as c
import cycle09_r4_campaign as campaign

from utils.profiling_utils import _gram_to_svdllm_scaling_diag_matrix

LAYER = 18
TRANSITIONS = {"opd": (0, 5, 10), "sft": (10, 20, 40)}
TASKS = ("legacy_S_math", "E_ood")


def bundle_path(run_root: Path, arm: str, step: int, task: str) -> Path:
    return (
        run_root
        / "scratch/bootstrap_factors"
        / arm
        / c.step_label(step)
        / f"{task}.pt"
    )


def cache_path(run_root: Path, task: str, arm: str, step: int) -> Path:
    return (
        run_root
        / "scratch/bootstrap_cache"
        / task
        / arm
        / f"{c.step_label(step)}.npz"
    )


def load_bundle(run_root: Path, arm: str, step: int, task: str) -> dict[str, Any]:
    path = bundle_path(run_root, arm, step, task)
    if not path.exists():
        raise FileNotFoundError(f"missing Round-4 factor bundle: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"sample_factors", "residual_samples", "residual_sample_means"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    return payload


def draw_scalings(
    bundle: dict[str, Any], indices: np.ndarray, device: str
) -> dict[str, torch.Tensor]:
    result = {}
    sample_factors = bundle["sample_factors"]
    for group in c.GROUP_TO_MODULES:
        pieces = [
            sample_factors[int(index)][LAYER][group]
            for index in indices.tolist()
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


def raw_er(bundle: dict[str, Any], indices: np.ndarray, device: str) -> float:
    factors = [
        bundle["residual_samples"][int(index)][LAYER]
        for index in indices.tolist()
    ]
    matrix = torch.cat(factors, dim=0).to(device=device, dtype=torch.float32)
    matrix.mul_(1.0 / math.sqrt(len(indices)))
    second = matrix.T @ matrix
    means = torch.stack(
        [
            bundle["residual_sample_means"][int(index)][LAYER]
            for index in indices.tolist()
        ]
    ).to(device=device, dtype=torch.float32)
    mean = means.mean(dim=0)
    covariance = second - torch.outer(mean, mean)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    value = c.effective_rank(eigenvalues.cpu().numpy()[::-1])
    del factors, matrix, second, means, mean, covariance, eigenvalues
    return value


def subspace_overlap(
    base: tuple[torch.Tensor, torch.Tensor],
    current_u: torch.Tensor,
    current_v: torch.Tensor,
) -> tuple[float, float]:
    base_u, base_v = base
    current_u = current_u.cpu().float()
    current_v = current_v.cpu().float()
    u = torch.linalg.svdvals(base_u.float().T @ current_u).square().mean()
    v = torch.linalg.svdvals(base_v.float().T @ current_v).square().mean()
    return float(u), float(v)


@torch.no_grad()
def compute_model_draws(
    *,
    run_root: Path,
    task: str,
    arm: str,
    step: int,
    indices_by_draw: np.ndarray,
    device: str,
    theta_rank: int,
    base_bases: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] | None,
    collect_bases: bool,
) -> tuple[dict[str, np.ndarray], list[dict[str, tuple[torch.Tensor, torch.Tensor]]] | None]:
    bundle = load_bundle(run_root, arm, step, task)
    model = campaign.load_model(c.model_path(arm, step), device)
    n_draws = len(indices_by_draw)
    er = np.zeros((n_draws, len(c.MODULES)), dtype=np.float64)
    raw = np.zeros(n_draws, dtype=np.float64)
    theta_u = np.full_like(er, np.nan)
    theta_v = np.full_like(er, np.nan)
    bases: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] | None = (
        [] if collect_bases else None
    )
    try:
        weights = {
            module: campaign.module_at(model, LAYER, module).weight.detach().float()
            for module in c.MODULES
        }
        for draw_index, indices in enumerate(indices_by_draw):
            scalings = draw_scalings(bundle, indices, device)
            draw_bases: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for module_index, module in enumerate(c.MODULES):
                scaling = scalings[c.MODULE_TO_GROUP[module]]
                matrix = weights[module] @ scaling
                if collect_bases or base_bases is not None:
                    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
                    er[draw_index, module_index] = c.effective_rank(
                        singular.cpu().numpy()
                    )
                    rank = min(theta_rank, u.shape[1], vh.shape[0])
                    current_u = u[:, :rank]
                    current_v = vh[:rank, :].T
                    if collect_bases:
                        draw_bases[module] = (
                            current_u.cpu().half().contiguous(),
                            current_v.cpu().half().contiguous(),
                        )
                    else:
                        value_u, value_v = subspace_overlap(
                            base_bases[draw_index][module],
                            current_u,
                            current_v,
                        )
                        theta_u[draw_index, module_index] = value_u
                        theta_v[draw_index, module_index] = value_v
                    del u, singular, vh, current_u, current_v
                else:
                    singular = torch.linalg.svdvals(matrix)
                    er[draw_index, module_index] = c.effective_rank(
                        singular.cpu().numpy()
                    )
                    del singular
                del matrix
            raw[draw_index] = raw_er(bundle, indices, device)
            if collect_bases:
                bases.append(draw_bases)
            scalings.clear()
            torch.cuda.empty_cache()
            if (draw_index + 1) % 8 == 0 or draw_index + 1 == n_draws:
                print(
                    f"[R4 bootstrap] {task} {arm}/{c.step_label(step)} "
                    f"{draw_index + 1}/{n_draws}",
                    flush=True,
                )
    finally:
        campaign.unload_model(model)
        del bundle
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "er": er,
        "raw": raw,
        "theta_u": theta_u,
        "theta_v": theta_v,
    }, bases


def save_cache(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(tmp, path)


def load_cache(path: Path, draws: int) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path) as data:
        values = {key: data[key] for key in data.files}
    if values.get("er", np.empty((0,))).shape[0] != draws:
        return None
    return values


def task_draws(
    args: argparse.Namespace,
    task: str,
    indices_by_draw: np.ndarray,
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    outputs: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    need_theta = task == "legacy_S_math"
    base_values, base_bases = compute_model_draws(
        run_root=args.run_root,
        task=task,
        arm="opd",
        step=0,
        indices_by_draw=indices_by_draw,
        device=args.device,
        theta_rank=args.theta_rank,
        base_bases=None,
        collect_bases=need_theta,
    )
    save_cache(cache_path(args.run_root, task, "opd", 0), base_values)
    outputs[("opd", 0)] = base_values
    outputs[("sft", 0)] = base_values

    for arm in c.ARMS:
        for step in c.STEPS:
            if step == 0:
                continue
            path = cache_path(args.run_root, task, arm, step)
            values = load_cache(path, args.draws)
            theta_missing = need_theta and (
                values is None or np.isnan(values["theta_u"]).all()
            )
            if values is None or theta_missing:
                values, _ = compute_model_draws(
                    run_root=args.run_root,
                    task=task,
                    arm=arm,
                    step=step,
                    indices_by_draw=indices_by_draw,
                    device=args.device,
                    theta_rank=args.theta_rank,
                    base_bases=base_bases if need_theta else None,
                    collect_bases=False,
                )
                save_cache(path, values)
            else:
                print(f"[R4 bootstrap cache] {task} {arm}/{c.step_label(step)}", flush=True)
            outputs[(arm, step)] = values
    del base_bases
    gc.collect()
    return outputs


def point_values(
    run_root: Path, arm: str, step: int, task: str
) -> tuple[np.ndarray, float]:
    path = (
        run_root
        / "measurements"
        / arm
        / c.step_label(step)
        / f"{task}.json"
    )
    payload = c.read_json(path)
    if not payload:
        raise FileNotFoundError(path)
    spectra = payload["spectra"]["per_checkpoint"][f"layer_{LAYER}"]
    er = np.asarray(
        [c.effective_rank(spectra[module]) for module in c.MODULES],
        dtype=np.float64,
    )
    raw_rows = [
        row
        for row in payload["raw_residual"]
        if int(row["layer"]) == LAYER and row["position_bin"] == "all"
    ]
    if len(raw_rows) != 1:
        raise ValueError(f"{path}: expected one L18 all-position raw row")
    return er, float(raw_rows[0]["raw_residual_er"])


def interval(values: np.ndarray) -> tuple[float, float, float]:
    return (
        float(values.mean()),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def uptick_rows(
    args: argparse.Namespace,
    draws: dict[tuple[str, int], dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = []
    saved = {}
    for arm, (previous, dip, following) in TRANSITIONS.items():
        point = {
            step: point_values(args.run_root, arm, step, "legacy_S_math")[0]
            for step in (previous, dip, following)
        }
        draw_uptick = draws[(arm, dip)]["er"] - 0.5 * (
            draws[(arm, previous)]["er"] + draws[(arm, following)]["er"]
        )
        for module_index, module in enumerate(c.MODULES):
            values = draw_uptick[:, module_index]
            mean, lo, hi = interval(values)
            rows.append(
                {
                    "arm": arm,
                    "layer": LAYER,
                    "module": module,
                    "previous_step": previous,
                    "dip_step": dip,
                    "next_step": following,
                    "n_probe_samples": c.N_GENERATED,
                    "bootstrap_draws": args.draws,
                    "bootstrap_unit": "sample; windows nested",
                    "point_uptick_er": float(
                        point[dip][module_index]
                        - 0.5 * (
                            point[previous][module_index]
                            + point[following][module_index]
                        )
                    ),
                    "bootstrap_mean_uptick_er": mean,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "ci_excludes_zero": lo > 0 or hi < 0,
                }
            )
        aggregate = draw_uptick.mean(axis=1)
        mean, lo, hi = interval(aggregate)
        point_aggregate = point[dip].mean() - 0.5 * (
            point[previous].mean() + point[following].mean()
        )
        rows.append(
            {
                "arm": arm,
                "layer": LAYER,
                "module": "mean_fixed_7_modules",
                "previous_step": previous,
                "dip_step": dip,
                "next_step": following,
                "n_probe_samples": c.N_GENERATED,
                "bootstrap_draws": args.draws,
                "bootstrap_unit": "sample; windows nested",
                "point_uptick_er": float(point_aggregate),
                "bootstrap_mean_uptick_er": mean,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "ci_excludes_zero": lo > 0 or hi < 0,
            }
        )
        saved[f"uptick_{arm}"] = aggregate
    return rows, saved


def pooled_sd(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        math.sqrt(
            0.5 * (
                float(np.var(left, ddof=1))
                + float(np.var(right, ddof=1))
            )
        )
    )


def weight_theta_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = c.MINI_ROOT / "R3_theta_w.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    selected = [
        row
        for row in source
        if int(row["layer"]) == LAYER and int(row["rank"]) == args.theta_rank
    ]
    identity = {
        (row["arm"], row["module"], side): float(row[f"{side}_mean_angle_deg"])
        for row in selected
        if int(row["step"]) == 0
        for side in ("left", "right")
    }
    rng = np.random.default_rng(c.stable_seed(args.seed, "theta_w_modules"))
    rows = []
    for arm in c.ARMS:
        for step in c.STEPS:
            members = [row for row in selected if row["arm"] == arm and int(row["step"]) == step]
            for side in ("left", "right"):
                raw = np.asarray(
                    [float(row[f"{side}_mean_angle_deg"]) for row in members],
                    dtype=np.float64,
                )
                corrected = np.asarray(
                    [
                        math.sqrt(
                            max(
                                float(row[f"{side}_mean_angle_deg"]) ** 2
                                - identity[(arm, row["module"], side)] ** 2,
                                0.0,
                            )
                        )
                        for row in members
                    ],
                    dtype=np.float64,
                )
                if not len(raw):
                    continue
                indices = rng.integers(0, len(raw), size=(args.module_draws, len(raw)))
                boot = corrected[indices].mean(axis=1)
                rows.append(
                    {
                        "row_type": "theta_w_module_bootstrap",
                        "task_id": "weight_space_window_independent",
                        "arm": arm,
                        "step": step,
                        "layer": LAYER,
                        "module": "mean_fixed_7_modules",
                        "side": side,
                        "rank": args.theta_rank,
                        "point_raw_mean_angle_deg": float(raw.mean()),
                        "point_identity_floor_corrected_mean_angle_deg": float(
                            corrected.mean()
                        ),
                        "bootstrap_mean": float(boot.mean()),
                        "ci95_lo": float(np.percentile(boot, 2.5)),
                        "ci95_hi": float(np.percentile(boot, 97.5)),
                        "bootstrap_draws": args.module_draws,
                        "bootstrap_unit": "module",
                        "identity_floor_rule": "sqrt(max(theta^2-theta_identity^2,0)); per module",
                        "source_path": str(path),
                    }
                )
    return rows


def discriminability_rows(
    args: argparse.Namespace,
    all_draws: dict[str, dict[tuple[str, int], dict[str, np.ndarray]]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    saved: dict[str, np.ndarray] = {}
    for task, draws in all_draws.items():
        for step in c.STEPS:
            opd = draws[("opd", step)]
            sft = draws[("sft", step)]
            opd_w = opd["er"].mean(axis=1)
            sft_w = sft["er"].mean(axis=1)
            sd_w = pooled_sd(opd_w, sft_w)
            sd_r = pooled_sd(opd["raw"], sft["raw"])
            point_opd_w, point_opd_r = point_values(args.run_root, "opd", step, task)
            point_sft_w, point_sft_r = point_values(args.run_root, "sft", step, task)
            d_w = abs(float(point_opd_w.mean() - point_sft_w.mean())) / max(sd_w, 1e-30)
            d_r = abs(point_opd_r - point_sft_r) / max(sd_r, 1e-30)
            gap_draws = (
                np.abs(opd_w - sft_w) / max(sd_w, 1e-30)
                - np.abs(opd["raw"] - sft["raw"]) / max(sd_r, 1e-30)
            )
            mean, lo, hi = interval(gap_draws)
            rows.append(
                {
                    "row_type": "space_discriminability",
                    "task_id": task,
                    "arm": "opd_minus_sft",
                    "step": step,
                    "layer": LAYER,
                    "module": "mean_fixed_7_modules_vs_residual_stream",
                    "d_whitened": d_w,
                    "d_raw": d_r,
                    "point_d_whitened_minus_d_raw": d_w - d_r,
                    "bootstrap_mean": mean,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "ci_excludes_zero": lo > 0 or hi < 0,
                    "pooled_bootstrap_sd_whitened": sd_w,
                    "pooled_bootstrap_sd_raw": sd_r,
                    "bootstrap_draws": args.draws,
                    "bootstrap_unit": "sample; windows nested; paired spaces",
                    "studentization": "full-bootstrap pooled SD fixed inside paired outer draws",
                }
            )
            saved[f"discriminability_{task}_{step}"] = gap_draws

    legacy = all_draws["legacy_S_math"]
    for arm in c.ARMS:
        for step in c.STEPS:
            values = legacy[(arm, step)]
            for side in ("theta_u", "theta_v"):
                for module_index, module in enumerate(c.MODULES):
                    draws = values[side][:, module_index]
                    draws = draws[np.isfinite(draws)]
                    if not len(draws):
                        continue
                    mean, lo, hi = interval(draws)
                    rows.append(
                        {
                            "row_type": "theta_r_sample_bootstrap",
                            "task_id": "legacy_S_math",
                            "arm": arm,
                            "step": step,
                            "layer": LAYER,
                            "module": module,
                            "side": side[-1].upper(),
                            "rank": args.theta_rank,
                            "point_overlap_squared_mean": mean,
                            "bootstrap_mean": mean,
                            "ci95_lo": lo,
                            "ci95_hi": hi,
                            "bootstrap_draws": args.draws,
                            "bootstrap_unit": "sample; windows nested",
                            "metric": "mean_squared_canonical_cosine_vs_base",
                        }
                    )
    rows.extend(weight_theta_rows(args))
    return rows, saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run-root", type=Path, default=c.RUN_ROOT)
    parser.add_argument("--mini-root", type=Path, default=c.MINI_ROOT)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--module-draws", type=int, default=4096)
    parser.add_argument("--theta-rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not (args.all or args.smoke):
        parser.print_help()
        return
    if args.smoke:
        args.run_root = args.run_root / "smoke"
        args.mini_root = args.mini_root / "smoke_r4"
        task = "S_math__g3"
        bundle = load_bundle(args.run_root, "opd", 0, task)
        indices = np.arange(len(bundle["sample_factors"]), dtype=np.int64)[None, :]
        del bundle
        values, bases = compute_model_draws(
            run_root=args.run_root,
            task=task,
            arm="opd",
            step=0,
            indices_by_draw=indices,
            device=args.device,
            theta_rank=args.theta_rank,
            base_bases=None,
            collect_bases=True,
        )
        if not np.isfinite(values["er"]).all() or not np.isfinite(values["raw"]).all():
            raise RuntimeError("bootstrap smoke produced non-finite values")
        c.write_json_atomic(
            args.mini_root / "R4_bootstrap_smoke.json",
            {
                "status": "ok",
                "task": task,
                "er": values["er"].tolist(),
                "raw": values["raw"].tolist(),
                "basis_draws": len(bases or []),
            },
        )
        print("[R4 bootstrap smoke] ok", flush=True)
        return

    rng = np.random.default_rng(args.seed)
    bundles = {
        task: load_bundle(args.run_root, "opd", 0, task)
        for task in TASKS
    }
    indices = {
        task: rng.integers(
            0,
            len(bundle["sample_factors"]),
            size=(args.draws, len(bundle["sample_factors"])),
        )
        for task, bundle in bundles.items()
    }
    del bundles

    all_draws = {
        task: task_draws(args, task, indices[task])
        for task in TASKS
    }
    uptick, saved_uptick = uptick_rows(args, all_draws["legacy_S_math"])
    c.write_csv_atomic(
        args.mini_root / "R4_l18_rederivation.csv",
        uptick,
        list(uptick[0]) if uptick else [],
    )
    discriminability, saved_d = discriminability_rows(args, all_draws)
    fields = sorted({key for row in discriminability for key in row})
    c.write_csv_atomic(
        args.mini_root / "R4_discriminability.csv",
        discriminability,
        fields,
    )
    draw_path = args.mini_root / "R4_bootstrap_draws.npz"
    with open(draw_path.with_suffix(".npz.tmp"), "wb") as handle:
        np.savez_compressed(handle, **saved_uptick, **saved_d)
    os.replace(draw_path.with_suffix(".npz.tmp"), draw_path)
    c.write_json_atomic(
        args.mini_root / "R4_bootstrap_manifest.json",
        {
            "schema_version": 1,
            "draws": args.draws,
            "seed": args.seed,
            "layer": LAYER,
            "tasks": list(TASKS),
            "bootstrap_unit": "sample; windows nested",
            "indices_shared_across_arms_steps_and_spaces_within_task": True,
            "theta_rank": args.theta_rank,
            "module_bootstrap_draws_theta_w": args.module_draws,
        },
    )
    print(
        f"[R4 bootstrap complete] uptick={len(uptick)} "
        f"discriminability={len(discriminability)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
