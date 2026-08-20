#!/usr/bin/env python3
"""Cycle 09 Round 5 — A4: sample-level bootstrap CIs for the load-bearing claims.

Quantities (per draw, layer 18, the layer with saved per-sample factors):
  r_eps        functional rank              -> two-arm difference
  gamma_r_eps  sigma_r - sigma_{r+1}        -> two-arm difference (compression quality)
  theta_r_eps  principal angles vs base     -> two-arm difference (compression direction)
  M2           ||dW X||_F / ||W0 X||_F      -> "equal movement" equivalence check

Bootstrap unit = sample (windows nested), draws = 256, indices shared across arms
and steps within a task so the two-arm differences are paired.

Scope (runtime-bounded, recorded in the manifest): steps {0 (base), 5 (OPD dip),
20 (SFT dip), 624 (endpoint)} x arms x tasks {legacy_S_math, E_ood}.
Unlike R4, every cell is checkpointed to npz as soon as it finishes.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import cycle09_r4_common as c4
import cycle09_r4_campaign as camp
import cycle09_r5_common as c5

from utils.profiling_utils import _gram_to_svdllm_scaling_diag_matrix

LAYER = 18
TASKS = ("legacy_S_math", "E_ood")
STEPS = (0, 5, 20, 624)
BASE_RANK_CACHE = 512  # U0/V0 columns cached per draw (r_eps stays well below)


def bundle_path(arm: str, step: int, task: str) -> Path:
    return (
        c5.R4_ROOT / "scratch/bootstrap_factors" / arm / c4.step_label(step) / f"{task}.pt"
    )


def load_bundle(arm: str, step: int, task: str) -> dict[str, Any]:
    path = bundle_path(arm, step, task)
    if not path.exists():
        raise FileNotFoundError(f"missing R4 factor bundle: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def draw_grams(bundle: dict[str, Any], indices: np.ndarray, device: str):
    """Raw (unwhitened) gram per module group for one bootstrap draw."""
    grams = {}
    sample_factors = bundle["sample_factors"]
    for group in c4.GROUP_TO_MODULES:
        pieces = [sample_factors[int(i)][LAYER][group] for i in indices.tolist()]
        matrix = torch.cat(pieces, dim=0).to(device=device, dtype=torch.float32)
        matrix.mul_(1.0 / math.sqrt(len(indices)))
        grams[group] = matrix.T @ matrix
        del pieces, matrix
    return grams


def scaling_from_gram(gram: torch.Tensor, device: str) -> torch.Tensor:
    return _gram_to_svdllm_scaling_diag_matrix(
        gram, cholesky_jitter=1e-5, singular_floor=0.0
    ).to(device=device, dtype=torch.float32)


@torch.no_grad()
def base_cache(task: str, indices_by_draw: np.ndarray, device: str):
    """U0/V0 (truncated) + base sigma + base gram-derived scaling, per draw."""
    bundle = load_bundle("opd", 0, task)  # step 0 == base for both arms
    model = camp.load_model(c4.BASE_MODEL, device)
    cache: list[dict[str, Any]] = []
    try:
        weights = {
            module: camp.module_at(model, LAYER, module).weight.detach().float()
            for module in c4.MODULES
        }
        for draw_index, indices in enumerate(indices_by_draw):
            grams = draw_grams(bundle, indices, device)
            entry: dict[str, Any] = {}
            for module in c4.MODULES:
                group = c4.MODULE_TO_GROUP[module]
                scaling = scaling_from_gram(grams[group], device)
                m0 = weights[module] @ scaling
                u0, s0, vh0 = torch.linalg.svd(m0, full_matrices=False)
                keep = min(BASE_RANK_CACHE, u0.shape[1])
                entry[module] = {
                    # float32 (not half): the angle floor is already set by the
                    # fp32 SVD's ~1e-3 orthonormality error; do not add to it.
                    "u0": u0[:, :keep].cpu().float().contiguous(),
                    "v0": vh0[:keep, :].T.cpu().float().contiguous(),
                    # Only the scalar energy is needed downstream. Caching the d x d
                    # gram here cost 26 MB x 7 modules x 256 draws = 47 GB of RAM and
                    # was never read (it OOM-killed the first A4 run).
                    "w0_gram_energy": float(
                        torch.einsum(
                            "ij,ij->",
                            weights[module] @ grams[group],
                            weights[module],
                        )
                    ),
                }
                del scaling, m0, u0, s0, vh0
            cache.append(entry)
            grams.clear()
            torch.cuda.empty_cache()
            if (draw_index + 1) % 32 == 0 or draw_index + 1 == len(indices_by_draw):
                print(f"[A4 base] {task} {draw_index + 1}/{len(indices_by_draw)}", flush=True)
    finally:
        camp.unload_model(model)
        del bundle
        gc.collect()
        torch.cuda.empty_cache()
    return cache


@torch.no_grad()
def cell_draws(
    *,
    task: str,
    arm: str,
    step: int,
    indices_by_draw: np.ndarray,
    base: list[dict[str, Any]],
    device: str,
) -> dict[str, np.ndarray]:
    bundle = load_bundle(arm, step, task)
    model = camp.load_model(c4.model_path(arm, step), device)
    base_model = camp.load_model(c4.BASE_MODEL, device) if step != 0 else None
    adapter_state = adapter_scale = None
    if arm == "sft" and step != 0:
        adapter_state, adapter_scale = camp.load_adapter_state(int(step))

    n_draws, n_modules = len(indices_by_draw), len(c4.MODULES)
    out = {
        key: np.full((n_draws, n_modules), np.nan, dtype=np.float64)
        for key in (
            "r_eps", "gamma_r_eps", "theta_u_max", "theta_u_mean", "theta_v_max",
            "m2", "er", "theta_rank_used", "theta_rank_capped",
        )
    }
    try:
        weights = {
            module: camp.module_at(model, LAYER, module).weight.detach().float()
            for module in c4.MODULES
        }
        updates = {}
        for module in c4.MODULES:
            if step == 0:
                updates[module] = torch.zeros_like(weights[module])
            else:
                update, _ = camp.update_matrix(
                    arm, int(step), LAYER, module, model, base_model,
                    adapter_state, adapter_scale, device,
                )
                updates[module] = update.float()

        for draw_index, indices in enumerate(indices_by_draw):
            grams = draw_grams(bundle, indices, device)
            for module_index, module in enumerate(c4.MODULES):
                group = c4.MODULE_TO_GROUP[module]
                gram = grams[group]
                scaling = scaling_from_gram(gram, device)
                matrix = weights[module] @ scaling
                u, sigma, vh = torch.linalg.svd(matrix, full_matrices=False)
                sigma_list = sigma.cpu().numpy().astype(np.float64)

                rank_full = c4.functional_rank(sigma_list, 0.05)
                rank = max(1, min(rank_full, u.shape[1], base[draw_index][module]["u0"].shape[1]))
                out["r_eps"][draw_index, module_index] = rank_full
                out["theta_rank_used"][draw_index, module_index] = rank
                out["theta_rank_capped"][draw_index, module_index] = float(rank < rank_full)
                out["er"][draw_index, module_index] = c4.effective_rank(sigma_list)
                out["gamma_r_eps"][draw_index, module_index] = c5.spectral_gap(
                    sigma_list, rank_full
                )

                u0 = base[draw_index][module]["u0"][:, :rank].to(device).float()
                v0 = base[draw_index][module]["v0"][:, :rank].to(device).float()
                theta_u_max, theta_u_mean = c5.principal_angles(u0, u[:, :rank])
                theta_v_max, _ = c5.principal_angles(v0, vh[:rank, :].T)
                out["theta_u_max"][draw_index, module_index] = theta_u_max
                out["theta_u_mean"][draw_index, module_index] = theta_u_mean
                out["theta_v_max"][draw_index, module_index] = theta_v_max

                # M2 on raw inputs: sqrt(tr(dW^T dW Sigma_X) / tr(W0^T W0 Sigma_X))
                delta = updates[module]
                num = float(torch.einsum("ij,ij->", delta @ gram, delta))
                den = base[draw_index][module]["w0_gram_energy"]
                out["m2"][draw_index, module_index] = math.sqrt(
                    max(num, 0.0) / max(den, 1e-30)
                )
                del u, sigma, vh, matrix, scaling, u0, v0
            grams.clear()
            torch.cuda.empty_cache()
            if (draw_index + 1) % 32 == 0 or draw_index + 1 == n_draws:
                print(
                    f"[A4] {task} {arm}/{c4.step_label(step)} {draw_index + 1}/{n_draws}",
                    flush=True,
                )
    finally:
        camp.unload_model(model)
        if base_model is not None:
            camp.unload_model(base_model)
        if adapter_state is not None:
            adapter_state.clear()
        del bundle
        gc.collect()
        torch.cuda.empty_cache()
    return out


def interval(values: np.ndarray) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(finite)),
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mini-root", type=Path, default=c5.MINI_ROOT)
    parser.add_argument("--cache-root", type=Path, default=c5.RUN_ROOT / "scratch/a4_cache")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--steps", default=",".join(map(str, STEPS)))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tasks = tuple(t for t in args.tasks.split(",") if t)
    steps = tuple(int(s) for s in args.steps.split(",") if s)
    if args.smoke:
        args.draws = 4
        args.mini_root = args.mini_root / "smoke_r5"
        args.cache_root = args.cache_root.parent / "a4_cache_smoke"
        tasks = ("E_ood",)
        steps = (0, 624)
    args.mini_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        n_samples = len(load_bundle("opd", 0, task)["sample_ids"])
        rng = np.random.default_rng(c4.stable_seed(args.seed, task))
        indices_by_draw = rng.integers(0, n_samples, size=(args.draws, n_samples))

        base = base_cache(task, indices_by_draw, args.device)

        per_cell: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        for arm in c4.ARMS:
            for step in steps:
                if step == 0 and arm == "sft":
                    per_cell[("sft", 0)] = per_cell[("opd", 0)]
                    continue
                cache_file = args.cache_root / f"{task}__{arm}__{c4.step_label(step)}.npz"
                if cache_file.exists():
                    payload = np.load(cache_file)
                    if payload["r_eps"].shape[0] == args.draws:
                        per_cell[(arm, step)] = {k: payload[k] for k in payload.files}
                        print(f"[A4 cached] {task} {arm}/{c4.step_label(step)}", flush=True)
                        continue
                values = cell_draws(
                    task=task,
                    arm=arm,
                    step=step,
                    indices_by_draw=indices_by_draw,
                    base=base,
                    device=args.device,
                )
                np.savez_compressed(cache_file, **values)  # per-cell checkpoint
                per_cell[(arm, step)] = values

        for step in steps:
            if step == 0:
                continue
            for module_index, module in enumerate(c4.MODULES + ("mean_fixed_7_modules",)):
                def series(arm: str, key: str) -> np.ndarray:
                    values = per_cell[(arm, step)][key]
                    return (
                        np.nanmean(values, axis=1)
                        if module == "mean_fixed_7_modules"
                        else values[:, module_index]
                    )

                for key in ("r_eps", "gamma_r_eps", "theta_u_max", "theta_v_max", "m2", "er"):
                    diff = series("opd", key) - series("sft", key)
                    mean, lo, hi = interval(diff)
                    opd_mean, opd_lo, opd_hi = interval(series("opd", key))
                    sft_mean, sft_lo, sft_hi = interval(series("sft", key))
                    rows.append(
                        {
                            "task_id": task,
                            "step": int(step),
                            "layer": LAYER,
                            "module": module,
                            "metric": key,
                            "bootstrap_unit": "sample; windows nested",
                            "bootstrap_draws": args.draws,
                            "opd_mean": opd_mean,
                            "opd_ci95_lo": opd_lo,
                            "opd_ci95_hi": opd_hi,
                            "sft_mean": sft_mean,
                            "sft_ci95_lo": sft_lo,
                            "sft_ci95_hi": sft_hi,
                            "opd_minus_sft_mean": mean,
                            "opd_minus_sft_ci95_lo": lo,
                            "opd_minus_sft_ci95_hi": hi,
                            "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
                        }
                    )

    c4.write_csv_atomic(
        args.mini_root / "R5_bootstrap_ci.csv", rows, list(rows[0].keys())
    )
    c4.write_json_atomic(
        args.mini_root / "R5_bootstrap_manifest.json",
        {
            "schema_version": 1,
            "draws": args.draws,
            "seed": args.seed,
            "layer": LAYER,
            "tasks": list(tasks),
            "steps": list(steps),
            "bootstrap_unit": "sample; windows nested",
            "indices_shared_across_arms_and_steps_within_task": True,
            "paired_two_arm_differences": True,
            "scope_note": (
                "steps limited to base / OPD dip (5) / SFT dip (20) / endpoint (624) "
                "for runtime; layer 18 only (the layer with saved per-sample factors)"
            ),
            "m2_definition": "sqrt(tr(dW^T dW Sigma_X) / tr(W0^T W0 Sigma_X)), raw inputs",
            "theta_numerics": (
                "fp32 SVD + float64 QR re-orthonormalization; angle resolution floor "
                "~0.2 deg (validated against an fp64 SVD). Angles far above the floor "
                "(e.g. endpoint ~70-80 deg) are reliable; sub-degree readings are not."
            ),
            "theta_rank_rule": (
                f"per-draw r_eps(0.05), capped at {BASE_RANK_CACHE} cached base columns; "
                "theta_rank_capped flags the affected cells (gamma/r_eps are uncapped)"
            ),
        },
    )
    print(f"[A4] rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
