#!/usr/bin/env python3
"""Cycle 09 Round 5 — line A (results line): direction, quality, diagnosis.

A1  gamma_{r_eps} = sigma_{r_eps} - sigma_{r_eps+1}, per cell, using each cell's
    own r_eps (never k=1, never a global mean); k=64 reported as control.
A2  theta_{r_eps}: Bjorck-Golub principal angles between the base whitened basis
    U0/V0 and the checkpoint's, truncated at that cell's r_eps (frozen-base
    geometry, the track where the M3 family lives); k=64 control.
A3  e_keep saturation diagnosis: ||dW||_F against the Davis-Kahan bound
    sin(Theta) <= ||dW|| / gamma, i.e. e_keep ~ 1 is constructive.
A5  spectral_gap / xs_log_spectrum_gap columns restored on the v2 spectra.
A6  raw ER construct fix: normalized erank per 2605.30524 + top-5 eigen share.

All of A1/A5/A6(base) read the *saved* R4 spectra and grams — no re-profiling.
A2/A3 need the weight matrices, so they stream the checkpoints once.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import cycle09_r4_common as c4
import cycle09_r4_campaign as camp
import cycle09_r5_common as c5

# Probes carried into A2/A3 (frozen-base geometry needs a saved base reference).
A2_PROBES = ("E_ood", "E_general", "E_math_hard", "legacy_S_math", "S_math__g3")
TRACKS = ("per_checkpoint", "frozen_base")


def r4_measurements() -> list[Path]:
    return sorted((c5.R4_ROOT / "measurements").glob("*/*/*.json"))


# --------------------------------------------------------- A1 + A5 (+ lookup)
def gamma_and_gaps(args) -> None:
    gamma_rows: list[dict[str, Any]] = []
    lookup_rows: list[dict[str, Any]] = []
    sigma_index: dict[tuple, list[float]] = {}

    for path in r4_measurements():
        payload = json.loads(path.read_text(encoding="utf-8"))
        arm, step = payload["arm"], int(payload["step"])
        task = payload["task"]
        task_id = task["task_id"]
        for track in TRACKS:
            for layer_key, modules in payload["spectra"][track].items():
                layer = int(layer_key.split("_")[1])
                for module, sigma in modules.items():
                    sigma_index[(arm, step, task_id, track, layer, module)] = sigma
                    row: dict[str, Any] = {
                        "arm": arm,
                        "step": step,
                        "task_id": task_id,
                        "probe_type": task["probe_type"],
                        "domain": task["domain"],
                        "generation_seed": task.get("generation_seed"),
                        "track": track,
                        "layer": layer,
                        "module": module,
                        "sigma_dim": len(sigma),
                        "gamma_k1_legacy": c5.spectral_gap(sigma, 1),
                        f"gamma_k{c5.FIXED_RANK_CONTROL}": c5.spectral_gap(
                            sigma, c5.FIXED_RANK_CONTROL
                        ),
                    }
                    for epsilon in c5.EPSILONS:
                        rank = c4.functional_rank(sigma, epsilon)
                        tag = f"{epsilon:.2f}".split(".")[1]
                        row[f"r_eps_{tag}"] = rank
                        row[f"gamma_r_eps_{tag}"] = c5.spectral_gap(sigma, rank)
                        row[f"sigma_at_r_eps_{tag}"] = float(sigma[rank - 1]) if 0 < rank <= len(sigma) else float("nan")
                        lookup_rows.append(
                            {
                                "arm": arm,
                                "step": step,
                                "task_id": task_id,
                                "track": track,
                                "layer": layer,
                                "module": module,
                                "epsilon": epsilon,
                                "r_eps": rank,
                            }
                        )
                    gamma_rows.append(row)

    c4.write_csv_atomic(
        args.mini_root / "R5_gamma_reps.csv", gamma_rows, list(gamma_rows[0].keys())
    )
    c4.write_csv_atomic(
        args.mini_root / "R5_reps_lookup.csv", lookup_rows, list(lookup_rows[0].keys())
    )

    # A5: xs_log_spectrum_gap — X (training signal) spectrum vs S (target) spectrum.
    xs_rows = []
    for (arm, step, task_id, track, layer, module), sigma_x in sigma_index.items():
        if arm == "opd" and task_id.startswith("X_opd_math__"):
            seed = task_id.split("__g")[-1]
            s_task = f"S_math__g{seed}"
        elif arm == "sft" and task_id == "legacy_S_math":
            seed = "3"
            s_task = "S_math__g3"
        else:
            continue
        sigma_s = sigma_index.get((arm, step, s_task, track, layer, module))
        if sigma_s is None:
            continue
        xs_rows.append(
            {
                "arm": arm,
                "step": step,
                "x_task_id": task_id,
                "s_task_id": s_task,
                "track": track,
                "layer": layer,
                "module": module,
                "xs_log_spectrum_gap": c5.xs_log_spectrum_gap(sigma_x, sigma_s),
                "spectral_gap_k1_x": c5.spectral_gap(sigma_x, 1),
                "spectral_gap_k1_s": c5.spectral_gap(sigma_s, 1),
            }
        )
    c4.write_csv_atomic(
        args.mini_root / "R5_v2_spectra_plus.csv", xs_rows, list(xs_rows[0].keys())
    )
    print(
        f"[A1/A5] gamma={len(gamma_rows)} lookup={len(lookup_rows)} xs={len(xs_rows)}",
        flush=True,
    )


# ------------------------------------------------------------------ A2 + A3
def _reference_scales(task_id: str, layers: list[int], device: str):
    path = c5.R4_ROOT / "scratch/references" / f"{task_id}.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    profile = {"grams": {int(k): v for k, v in payload["grams"].items()}}
    return camp.scaling_by_group(profile, layers, device)


@torch.no_grad()
def theta_and_diagnosis(args) -> None:
    layers = list(args.layers)
    device = args.device
    base_model = camp.load_model(c4.BASE_MODEL, device)

    scales_cache: dict[str, Any] = {}
    for probe in A2_PROBES:
        scales = _reference_scales(probe, layers, device)
        if scales is None:
            print(f"[A2] missing base reference for {probe}; skipped", flush=True)
            continue
        scales_cache[probe] = scales

    theta_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []

    # Base bases (U0/V0) per probe/layer/module — computed once.
    base_bases: dict[tuple, tuple[torch.Tensor, torch.Tensor, list[float]]] = {}
    for probe, scales in scales_cache.items():
        for layer in layers:
            for module in c4.MODULES:
                group = c4.MODULE_TO_GROUP[module]
                w0 = camp.module_at(base_model, layer, module).weight.detach().to(
                    device=device, dtype=torch.float64
                )
                # float64: an fp32 SVD's bases are orthonormal only to ~1e-3, which
                # is the same order as the sub-degree angles A2 must resolve.
                m0 = w0 @ scales[layer][group].double()
                u0, s0, vh0 = torch.linalg.svd(m0, full_matrices=False)
                base_bases[(probe, layer, module)] = (u0, vh0.T, s0.cpu().tolist())
                del w0, m0, s0

    for arm in args.arms:
        for step in args.steps:
            if int(step) == 0:
                continue
            model = camp.load_model(c4.model_path(arm, step), device)
            adapter_state = adapter_scale = None
            if arm == "sft":
                adapter_state, adapter_scale = camp.load_adapter_state(int(step))
            print(f"[A2/A3] {arm}/{c4.step_label(step)}", flush=True)
            try:
                for probe, scales in scales_cache.items():
                    for layer in layers:
                        for module in c4.MODULES:
                            group = c4.MODULE_TO_GROUP[module]
                            u0, v0, sigma0 = base_bases[(probe, layer, module)]
                            wt = camp.module_at(model, layer, module).weight.detach().to(
                                device=device, dtype=torch.float64
                            )
                            mt = wt @ scales[layer][group].double()
                            ut, st, vht = torch.linalg.svd(mt, full_matrices=False)
                            vt = vht.T
                            sigma_t = st.cpu().tolist()

                            update, source_kind = camp.update_matrix(
                                arm,
                                int(step),
                                layer,
                                module,
                                model,
                                base_model,
                                adapter_state,
                                adapter_scale,
                                device,
                            )
                            delta_norm = float(torch.linalg.matrix_norm(update, "fro"))

                            for epsilon in c5.EPSILONS:
                                rank = c4.functional_rank(sigma_t, epsilon)
                                tag = f"{epsilon:.2f}".split(".")[1]
                                rank = max(1, min(rank, u0.shape[1], ut.shape[1]))
                                theta_u_max, theta_u_mean = c5.principal_angles(
                                    u0[:, :rank], ut[:, :rank]
                                )
                                theta_v_max, theta_v_mean = c5.principal_angles(
                                    v0[:, :rank], vt[:, :rank]
                                )
                                theta_rows.append(
                                    {
                                        "arm": arm,
                                        "step": int(step),
                                        "probe": probe,
                                        "track": "frozen_base",
                                        "layer": layer,
                                        "module": module,
                                        "epsilon": epsilon,
                                        "rank_used": rank,
                                        "rank_rule": f"per-cell r_eps ({tag})",
                                        "theta_u_max_deg": theta_u_max,
                                        "theta_u_mean_deg": theta_u_mean,
                                        "theta_v_max_deg": theta_v_max,
                                        "theta_v_mean_deg": theta_v_mean,
                                        "source_kind": source_kind,
                                    }
                                )
                            k = min(c5.FIXED_RANK_CONTROL, u0.shape[1], ut.shape[1])
                            theta_u_max, theta_u_mean = c5.principal_angles(
                                u0[:, :k], ut[:, :k]
                            )
                            theta_v_max, theta_v_mean = c5.principal_angles(
                                v0[:, :k], vt[:, :k]
                            )
                            theta_rows.append(
                                {
                                    "arm": arm,
                                    "step": int(step),
                                    "probe": probe,
                                    "track": "frozen_base",
                                    "layer": layer,
                                    "module": module,
                                    "epsilon": float("nan"),
                                    "rank_used": k,
                                    "rank_rule": f"fixed k={c5.FIXED_RANK_CONTROL} control",
                                    "theta_u_max_deg": theta_u_max,
                                    "theta_u_mean_deg": theta_u_mean,
                                    "theta_v_max_deg": theta_v_max,
                                    "theta_v_mean_deg": theta_v_mean,
                                    "source_kind": source_kind,
                                }
                            )

                            # A3: Davis-Kahan sin(Theta) <= ||dW * S0|| / gamma_r
                            rank05 = max(1, c4.functional_rank(sigma_t, 0.05))
                            gamma_r = c5.spectral_gap(sigma0, rank05)
                            delta_whitened = float(
                                torch.linalg.matrix_norm(update @ scales[layer][group], "fro")
                            )
                            base_whitened_norm = float(
                                np.linalg.norm(np.asarray(sigma0, dtype=np.float64))
                            )
                            bound = (
                                delta_whitened / gamma_r
                                if gamma_r and gamma_r > 0
                                else float("nan")
                            )
                            diag_rows.append(
                                {
                                    "arm": arm,
                                    "step": int(step),
                                    "probe": probe,
                                    "layer": layer,
                                    "module": module,
                                    "rank_r_eps_05": rank05,
                                    "delta_w_fro": delta_norm,
                                    "delta_w_whitened_fro": delta_whitened,
                                    "base_whitened_fro": base_whitened_norm,
                                    "relative_update_norm": delta_whitened
                                    / max(base_whitened_norm, 1e-30),
                                    "gamma_r_eps_05_base": gamma_r,
                                    "davis_kahan_sin_theta_bound": bound,
                                    "source_kind": source_kind,
                                }
                            )
                            del wt, mt, ut, st, vht, vt, update
                            torch.cuda.empty_cache()
            finally:
                camp.unload_model(model)
                if adapter_state is not None:
                    adapter_state.clear()
                gc.collect()
                torch.cuda.empty_cache()

    camp.unload_model(base_model)
    c4.write_csv_atomic(
        args.mini_root / "R5_theta_reps.csv", theta_rows, list(theta_rows[0].keys())
    )
    c4.write_csv_atomic(
        args.mini_root / "R5_ekeep_diagnosis.csv", diag_rows, list(diag_rows[0].keys())
    )
    print(f"[A2/A3] theta={len(theta_rows)} diagnosis={len(diag_rows)}", flush=True)


# ---------------------------------------------------------------------- A6
def raw_er_base(args) -> None:
    rows = []
    for path in sorted((c5.R4_ROOT / "scratch/references").glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        task_id = path.stem
        for layer, second in payload["residual_second"].items():
            mean = payload["residual_mean"][layer]
            covariance = second.double() - torch.outer(mean.double(), mean.double())
            eigenvalues = (
                torch.linalg.eigvalsh(covariance).clamp_min(0).numpy()[::-1]
            )
            rows.append(
                {
                    "arm": "base",
                    "step": 0,
                    "task_id": task_id,
                    "layer": int(layer),
                    "raw_er_unnormalized": c4.effective_rank(eigenvalues),
                    "raw_er_normalized": c5.normalized_effective_rank(eigenvalues),
                    "raw_top5_eigen_share": c5.top_eigen_share(eigenvalues, 5),
                    "raw_dim": int(eigenvalues.size),
                    "raw_trace": float(eigenvalues.sum()),
                    "protocol": "centered covariance; eps=1e-12; normalized erank (2605.30524 §3.1)",
                    "protocol_note": "口径自定，未核原文",
                }
            )
    c4.write_csv_atomic(
        args.mini_root / "R5_raw_er_fixed.csv", rows, list(rows[0].keys())
    )
    print(f"[A6] base raw rows={len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", action="store_true")
    parser.add_argument("--theta", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mini-root", type=Path, default=c5.MINI_ROOT)
    parser.add_argument("--arms", default=",".join(c5.ARMS))
    parser.add_argument("--steps", default=",".join(map(str, c5.STEPS)))
    parser.add_argument("--layers", default=",".join(map(str, c5.LAYERS)))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.arms = tuple(a for a in args.arms.split(",") if a)
    args.steps = tuple(int(s) for s in args.steps.split(",") if s)
    args.layers = tuple(int(s) for s in args.layers.split(",") if s)

    if args.smoke:
        args.mini_root = args.mini_root / "smoke_r5"
        args.steps = (0, 5)
        args.layers = (18,)

    args.mini_root.mkdir(parents=True, exist_ok=True)
    if args.all:
        args.gamma = args.theta = args.raw = True

    if args.gamma:
        gamma_and_gaps(args)
    if args.raw:
        raw_er_base(args)
    if args.theta:
        theta_and_diagnosis(args)


if __name__ == "__main__":
    main()
