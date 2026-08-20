#!/usr/bin/env python3
"""Cycle 09 D11 optional enhancements E5--E7.

E5: layer robustness on landmark steps.
E6: TPNT alpha sensitivity on headline layers.
E7: spectrum-matched random-subspace null on headline landmark cells.

The script appends optional D11 artifacts and never overwrites the completed
E0--E4 core tables.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cycle09_d11_pk_tpnt as d11  # noqa: E402


OUT = d11.OUT_BASE / "formal/final"
MINI = d11.MINI
LANDMARK_STEPS = (20, 160, 320)
MAIN_ALPHAS = (0.01, 0.10)
EXTRA_ALPHAS = (0.05, 0.20)
TPNT_KS = d11.TPNT_KS
ANGLE_KS = d11.ANGLE_KS
LAYERS = {
    "llama": (7, 14, 21),
    "qwen": (9, 18, 27),
}


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def seed_int(*items: Any) -> int:
    return int(d11.sha256_text("|".join(map(str, items)))[:8], 16)


def overlap_lift_from_masks(update_mask: torch.Tensor, principal_mask: torch.Tensor) -> dict[str, float]:
    total = int(update_mask.numel())
    update_count = int(update_mask.sum().item())
    principal_count = int(principal_mask.sum().item())
    overlap_count = int((update_mask & principal_mask).sum().item())
    expected = update_count * principal_count / total if total else float("nan")
    return {
        "total_entries": total,
        "update_count": update_count,
        "principal_top_count": principal_count,
        "overlap_count": overlap_count,
        "update_density": update_count / total if total else float("nan"),
        "coverage": overlap_count / update_count if update_count else float("nan"),
        "overlap_lift": overlap_count / expected if expected and expected > 0 else float("nan"),
    }


def spectrum_matched_null_mask(
    shape: tuple[int, int],
    base_bf16_cpu: torch.Tensor,
    singular_values: torch.Tensor,
    seed: int,
    device: str,
) -> torch.Tensor:
    out_dim, in_dim = shape
    r = int(singular_values.numel())
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    gu = torch.randn((out_dim, r), generator=gen, device=device, dtype=torch.float32)
    gv = torch.randn((in_dim, r), generator=gen, device=device, dtype=torch.float32)
    qu, _ = torch.linalg.qr(gu, mode="reduced")
    qv, _ = torch.linalg.qr(gv, mode="reduced")
    null_delta = (qu * singular_values.to(device=device, dtype=torch.float32).unsqueeze(0)) @ qv.T
    deployed = base_bf16_cpu.to(device=device) + null_delta.to(dtype=torch.bfloat16)
    mask = deployed.to(dtype=torch.float32).ne(base_bf16_cpu.to(device=device, dtype=torch.float32))
    del gu, gv, qu, qv, null_delta, deployed
    return mask


def compute_family(args: argparse.Namespace) -> dict[str, Any]:
    spec = d11.FAMILIES[args.family]
    device = args.device
    started = time.time()
    layers = LAYERS[args.family]
    e5_rows: list[dict[str, Any]] = []
    e6_rows: list[dict[str, Any]] = []
    e7_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    max_k = max(max(TPNT_KS), max(ANGLE_KS))

    for layer in layers:
        local_spec = d11.FamilySpec(spec.family, spec.model_label, spec.base, layer)
        is_headline = layer == spec.layer
        for module in d11.MODULES:
            key = d11.weight_key(layer, module)
            base_bf16_cpu = d11.read_weight(spec.base, key).to(dtype=torch.bfloat16)
            base_hash = d11.tensor_sha256(base_bf16_cpu)
            u0, s0, vh0 = d11.full_svd_top(base_bf16_cpu.to(dtype=torch.float32), max_k=max_k, device=device)
            source_spectrum = d11.normalized_spectrum(s0, max(ANGLE_KS))
            alpha_set = MAIN_ALPHAS + (EXTRA_ALPHAS if is_headline else tuple())
            masks = {
                (k_src, alpha): d11.principal_mask(u0, s0, vh0, k_src, alpha)
                for k_src in TPNT_KS
                for alpha in alpha_set
            }
            for arm in d11.ARMS:
                for step in LANDMARK_STEPS:
                    delta, wt, prov = d11.load_deployed_delta(local_spec, arm, step, module, device, base_bf16_cpu)
                    try:
                        update_mask = delta.ne(0)
                        ut, st, vht = d11.full_svd_top(wt, max_k=max(ANGLE_KS), device=device)
                        current_spectrum = d11.normalized_spectrum(st, max(ANGLE_KS))
                        src = np.array(source_spectrum[: len(current_spectrum)], dtype=np.float64)
                        cur = np.array(current_spectrum, dtype=np.float64)
                        nss_l1 = float(np.abs(cur - src).sum())
                        nss_l2 = float(np.sqrt(np.square(cur - src).sum()))
                        angle_summary = d11.principal_angle_rows(u0, vh0, ut, vht)
                        pabs_by_k = {row["angle_k"]: row for row in angle_summary}

                        for k_src in TPNT_KS:
                            for alpha in MAIN_ALPHAS:
                                metric = overlap_lift_from_masks(update_mask, masks[(k_src, alpha)])
                                e5_rows.append(
                                    {
                                        "model": spec.model_label,
                                        "family": "llama3_2_3b" if spec.family == "llama" else "qwen3_4b",
                                        "arm": arm,
                                        "checkpoint": step,
                                        "step": step,
                                        "layer": layer,
                                        "headline_layer": spec.layer,
                                        "module": module,
                                        "source_rank_k": k_src,
                                        "mask_density_alpha": alpha,
                                        **metric,
                                        "pabs_joint_mean_cos_k32": pabs_by_k[min(32, max(pabs_by_k))]["pabs_joint_mean_cos"],
                                        "theta_u_mean_deg_k32": pabs_by_k[min(32, max(pabs_by_k))]["theta_u_mean_deg"],
                                        "theta_v_mean_deg_k32": pabs_by_k[min(32, max(pabs_by_k))]["theta_v_mean_deg"],
                                        "nss_l1_top32": nss_l1,
                                        "nss_l2_top32": nss_l2,
                                        "delta_construction": prov["delta_construction"],
                                        "native_source": prov["native_source"],
                                        "checkpoint_materialization": prov["checkpoint_materialization"],
                                        "base_tensor_sha256": base_hash,
                                        "checkpoint_tensor_sha256": prov["checkpoint_tensor_sha256"],
                                    }
                                )

                        if is_headline:
                            for k_src in TPNT_KS:
                                for alpha in EXTRA_ALPHAS:
                                    metric = overlap_lift_from_masks(update_mask, masks[(k_src, alpha)])
                                    e6_rows.append(
                                        {
                                            "model": spec.model_label,
                                            "family": "llama3_2_3b" if spec.family == "llama" else "qwen3_4b",
                                            "arm": arm,
                                            "checkpoint": step,
                                            "step": step,
                                            "layer": layer,
                                            "module": module,
                                            "source_rank_k": k_src,
                                            "mask_density_alpha": alpha,
                                            **metric,
                                            "delta_construction": prov["delta_construction"],
                                            "native_source": prov["native_source"],
                                            "checkpoint_materialization": prov["checkpoint_materialization"],
                                            "base_tensor_sha256": base_hash,
                                            "checkpoint_tensor_sha256": prov["checkpoint_tensor_sha256"],
                                        }
                                    )

                            s_delta = torch.linalg.svdvals(delta.to(dtype=torch.float32))
                            s_delta = s_delta[s_delta > 0]
                            if int(args.e7_max_rank) > 0:
                                s_delta = s_delta[: min(int(args.e7_max_rank), s_delta.numel())]
                            for seed_idx in range(int(args.e7_seeds)):
                                nmask = spectrum_matched_null_mask(
                                    tuple(delta.shape),
                                    base_bf16_cpu,
                                    s_delta,
                                    seed_int(spec.family, arm, step, layer, module, "e7", seed_idx),
                                    device,
                                )
                                try:
                                    for k_src in TPNT_KS:
                                        for alpha in MAIN_ALPHAS:
                                            real = overlap_lift_from_masks(update_mask, masks[(k_src, alpha)])
                                            null = overlap_lift_from_masks(nmask, masks[(k_src, alpha)])
                                            e7_rows.append(
                                                {
                                                    "model": spec.model_label,
                                                    "family": "llama3_2_3b" if spec.family == "llama" else "qwen3_4b",
                                                    "arm": arm,
                                                    "checkpoint": step,
                                                    "step": step,
                                                    "layer": layer,
                                                    "module": module,
                                                    "source_rank_k": k_src,
                                                    "mask_density_alpha": alpha,
                                                    "seed": seed_idx,
                                                    "real_overlap_lift": real["overlap_lift"],
                                                    "null_overlap_lift": null["overlap_lift"],
                                                    "null_update_count": null["update_count"],
                                                    "delta_singular_values_used": int(s_delta.numel()),
                                                    "delta_singular_value_policy": "all_positive_svdvals" if int(args.e7_max_rank) <= 0 else f"top_{int(args.e7_max_rank)}_positive_svdvals",
                                                    "delta_construction": prov["delta_construction"],
                                                    "native_source": prov["native_source"],
                                                    "checkpoint_materialization": prov["checkpoint_materialization"],
                                                    "base_tensor_sha256": base_hash,
                                                    "checkpoint_tensor_sha256": prov["checkpoint_tensor_sha256"],
                                                }
                                            )
                                finally:
                                    del nmask
                            del s_delta
                        status_rows.append(
                            {
                                "task": "D11_E5_E7_family_cell",
                                "model": spec.model_label,
                                "arm": arm,
                                "checkpoint": step,
                                "layer": layer,
                                "module": module,
                                "status": "COMPLETE",
                            }
                        )
                        del ut, st, vht
                    finally:
                        del delta, wt
                        torch.cuda.empty_cache()
            for mask in masks.values():
                del mask
            del u0, s0, vh0, base_bf16_cpu
            torch.cuda.empty_cache()

    e5 = pd.DataFrame(e5_rows)
    e6 = pd.DataFrame(e6_rows)
    e7_seed = pd.DataFrame(e7_rows)
    if not e7_seed.empty:
        keys = ["model", "family", "arm", "checkpoint", "step", "layer", "module", "source_rank_k", "mask_density_alpha"]
        e7 = (
            e7_seed.groupby(keys, as_index=False)
            .agg(
                real_overlap_lift=("real_overlap_lift", "first"),
                null_overlap_lift_mean=("null_overlap_lift", "mean"),
                null_overlap_lift_std=("null_overlap_lift", "std"),
                null_update_count_mean=("null_update_count", "mean"),
                delta_singular_values_used=("delta_singular_values_used", "first"),
            )
        )
        e7["z_tpnt"] = (e7["real_overlap_lift"] - e7["null_overlap_lift_mean"]) / e7["null_overlap_lift_std"].replace(0, np.nan)
        e7["spectrum_null_seeds"] = int(args.e7_seeds)
    else:
        e7 = pd.DataFrame()

    atomic_csv(OUT / f"d11_e5_layer_robustness_{spec.family}.csv", e5)
    atomic_csv(OUT / f"d11_e6_alpha_sensitivity_extra_{spec.family}.csv", e6)
    atomic_csv(OUT / f"d11_e7_spectrum_matched_null_seed_rows_{spec.family}.csv", e7_seed)
    atomic_csv(OUT / f"d11_e7_spectrum_matched_null_{spec.family}.csv", e7)
    atomic_csv(OUT / f"d11_e5_e7_task_status_{spec.family}.csv", pd.DataFrame(status_rows))
    payload = {
        "schema_version": "cycle09_d11_e5_e7_family_v1",
        "status": "COMPLETE",
        "family": spec.family,
        "device": device,
        "layers": list(layers),
        "landmark_steps": list(LANDMARK_STEPS),
        "e5_rows": len(e5),
        "e6_rows": len(e6),
        "e7_seed_rows": len(e7_seed),
        "e7_rows": len(e7),
        "e7_seeds": int(args.e7_seeds),
        "e7_max_rank": int(args.e7_max_rank),
        "seconds": round(time.time() - started, 3),
        "created_utc": d11.utc_now(),
    }
    atomic_json(OUT / f"d11_e5_e7_family_{spec.family}_manifest.json", payload)
    return payload


def summarize(_: argparse.Namespace) -> dict[str, Any]:
    e5_parts = [pd.read_csv(p) for p in [OUT / "d11_e5_layer_robustness_llama.csv", OUT / "d11_e5_layer_robustness_qwen.csv"] if p.exists()]
    e6_parts = [pd.read_csv(p) for p in [OUT / "d11_e6_alpha_sensitivity_extra_llama.csv", OUT / "d11_e6_alpha_sensitivity_extra_qwen.csv"] if p.exists()]
    e7_parts = [pd.read_csv(p) for p in [OUT / "d11_e7_spectrum_matched_null_llama.csv", OUT / "d11_e7_spectrum_matched_null_qwen.csv"] if p.exists()]
    if not e5_parts or not e7_parts:
        raise RuntimeError("missing E5/E7 family outputs")
    e5 = pd.concat(e5_parts, ignore_index=True)
    e6_extra = pd.concat(e6_parts, ignore_index=True) if e6_parts else pd.DataFrame()
    e7 = pd.concat(e7_parts, ignore_index=True)

    atomic_csv(OUT / "d11_e5_layer_robustness.csv", e5)
    atomic_csv(OUT / "d11_e6_alpha_sensitivity_extra.csv", e6_extra)
    atomic_csv(OUT / "d11_e7_spectrum_matched_null.csv", e7)

    e5_mean = (
        e5.groupby(["model", "layer", "arm", "checkpoint", "source_rank_k", "mask_density_alpha"], as_index=False)
        .agg(
            mean_overlap_lift=("overlap_lift", "mean"),
            mean_pabs_joint_cos=("pabs_joint_mean_cos_k32", "mean"),
            mean_nss_l1_top32=("nss_l1_top32", "mean"),
        )
    )
    headline = e5_mean[e5_mean["layer"].isin([14, 18])].rename(
        columns={
            "layer": "headline_layer_join",
            "mean_overlap_lift": "headline_mean_overlap_lift",
            "mean_pabs_joint_cos": "headline_mean_pabs_joint_cos",
            "mean_nss_l1_top32": "headline_mean_nss_l1_top32",
        }
    )
    headline["headline_layer_join"] = headline["model"].map({"llama": 14, "qwen": 18})
    merged = e5_mean.merge(
        headline[["model", "arm", "checkpoint", "source_rank_k", "mask_density_alpha", "headline_mean_overlap_lift", "headline_mean_pabs_joint_cos", "headline_mean_nss_l1_top32"]],
        on=["model", "arm", "checkpoint", "source_rank_k", "mask_density_alpha"],
        how="left",
    )
    merged["overlap_lift_minus_headline"] = merged["mean_overlap_lift"] - merged["headline_mean_overlap_lift"]
    merged["pabs_cos_minus_headline"] = merged["mean_pabs_joint_cos"] - merged["headline_mean_pabs_joint_cos"]
    merged["nss_l1_minus_headline"] = merged["mean_nss_l1_top32"] - merged["headline_mean_nss_l1_top32"]
    rank_rows = []
    for keys, group in merged.groupby(["model", "layer", "checkpoint", "source_rank_k", "mask_density_alpha"]):
        g = group.copy()
        g["overlap_rank_desc"] = g["mean_overlap_lift"].rank(ascending=False, method="min")
        for _, row in g.iterrows():
            rank_rows.append(row.to_dict())
    e5_summary = pd.DataFrame(rank_rows)
    atomic_csv(OUT / "d11_e5_layer_robustness_summary.csv", e5_summary)

    main = pd.read_csv(OUT / "d11_tpnt_principal_mask.csv")
    main = main[
        main["checkpoint"].isin(LANDMARK_STEPS)
        & (((main["model"].eq("llama")) & (main["layer"].eq(14))) | ((main["model"].eq("qwen")) & (main["layer"].eq(18))))
    ].copy()
    main = main[main["mask_density_alpha"].isin(MAIN_ALPHAS)].copy()
    e6_all = pd.concat([main, e6_extra], ignore_index=True, sort=False)
    atomic_csv(OUT / "d11_e6_alpha_sensitivity.csv", e6_all)
    e6_summary = (
        e6_all.groupby(["model", "arm", "checkpoint", "source_rank_k", "mask_density_alpha"], as_index=False)
        .agg(mean_overlap_lift=("overlap_lift", "mean"), mean_coverage=("coverage", "mean"))
    )
    atomic_csv(OUT / "d11_e6_alpha_sensitivity_summary.csv", e6_summary)

    e7_summary = (
        e7.groupby(["model", "arm", "checkpoint", "source_rank_k", "mask_density_alpha"], as_index=False)
        .agg(
            mean_real_overlap_lift=("real_overlap_lift", "mean"),
            mean_null_overlap_lift=("null_overlap_lift_mean", "mean"),
            mean_z_tpnt=("z_tpnt", "mean"),
            median_z_tpnt=("z_tpnt", "median"),
        )
    )
    atomic_csv(OUT / "d11_e7_spectrum_matched_null_summary.csv", e7_summary)

    handoff = write_handoff(e5, e5_summary, e6_all, e6_summary, e7, e7_summary)
    manifest = write_manifest(handoff)
    mirror()
    update_core_manifest(manifest)
    return manifest


def write_handoff(e5: pd.DataFrame, e5_summary: pd.DataFrame, e6_all: pd.DataFrame, e6_summary: pd.DataFrame, e7: pd.DataFrame, e7_summary: pd.DataFrame) -> Path:
    lines = [
        "# D11 E5-E7 optional enhancement handoff",
        "",
        "## Status",
        "",
        "- status: `COMPLETE_D11_OPTIONAL_E5_E7`",
        f"- created_utc: `{d11.utc_now()}`",
        "- no training, no behavior eval, no c_epsilon/r_epsilon recomputation",
        "",
        "## Coverage",
        "",
        "| artifact | rows |",
        "|---|---:|",
        f"| d11_e5_layer_robustness.csv | {len(e5)} |",
        f"| d11_e5_layer_robustness_summary.csv | {len(e5_summary)} |",
        f"| d11_e6_alpha_sensitivity.csv | {len(e6_all)} |",
        f"| d11_e6_alpha_sensitivity_summary.csv | {len(e6_summary)} |",
        f"| d11_e7_spectrum_matched_null.csv | {len(e7)} |",
        f"| d11_e7_spectrum_matched_null_summary.csv | {len(e7_summary)} |",
        "",
        "## E7 Summary Head",
        "",
        d11.frame_to_markdown(e7_summary.head(20)),
        "",
        "## Boundaries",
        "",
        "- E7 uses 10 fixed spectrum-matched random-subspace seeds per landmark cell.",
        "- Spectrum-matched null preserves the selected positive singular values of the deployed BF16 merged-minus-base update.",
        "- Layer x module rows are reported as mechanical cells, not independent statistical seeds.",
    ]
    path = OUT / "d11_e5_e7_optional_handoff.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_manifest(handoff: Path) -> dict[str, Any]:
    names = [
        "d11_e5_layer_robustness.csv",
        "d11_e5_layer_robustness_summary.csv",
        "d11_e6_alpha_sensitivity.csv",
        "d11_e6_alpha_sensitivity_summary.csv",
        "d11_e7_spectrum_matched_null.csv",
        "d11_e7_spectrum_matched_null_summary.csv",
        "d11_e5_e7_optional_handoff.md",
    ]
    payload = {
        "schema_version": "cycle09_d11_e5_e7_manifest_v1",
        "status": "COMPLETE_D11_OPTIONAL_E5_E7",
        "created_utc": d11.utc_now(),
        "handoff": str(handoff),
        "outputs": [d11.artifact(OUT / name) for name in names],
    }
    atomic_json(OUT / "d11_e5_e7_manifest.json", payload)
    return payload


def update_core_manifest(optional_manifest: dict[str, Any]) -> None:
    core_path = OUT / "d11_pk_tpnt_manifest.json"
    if not core_path.exists():
        return
    data = json.loads(core_path.read_text(encoding="utf-8"))
    data["optional_e5_e7"] = {
        "status": optional_manifest["status"],
        "manifest": str(OUT / "d11_e5_e7_manifest.json"),
        "handoff": str(OUT / "d11_e5_e7_optional_handoff.md"),
    }
    data["status"] = "COMPLETE_D11_CORE_PLUS_OPTIONAL_E5_E7"
    atomic_json(core_path, data)


def mirror() -> None:
    MINI.mkdir(parents=True, exist_ok=True)
    for path in OUT.glob("d11_e5*"):
        if path.is_file():
            shutil.copy2(path, MINI / path.name)
    for path in OUT.glob("d11_e6*"):
        if path.is_file():
            shutil.copy2(path, MINI / path.name)
    for path in OUT.glob("d11_e7*"):
        if path.is_file():
            shutil.copy2(path, MINI / path.name)
    for name in ["d11_pk_tpnt_manifest.json"]:
        path = OUT / name
        if path.exists():
            shutil.copy2(path, MINI / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("family", "summarize"), required=True)
    parser.add_argument("--family", choices=("llama", "qwen"), default="llama")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--e7-seeds", type=int, default=10)
    parser.add_argument("--e7-max-rank", type=int, default=0, help="0 means all positive singular values")
    args = parser.parse_args()
    result = compute_family(args) if args.phase == "family" else summarize(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
