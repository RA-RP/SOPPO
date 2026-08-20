#!/usr/bin/env python3
"""Cycle 09 D11: deployed-weight p_k and TPNT minimal comparison.

This script implements the frozen D11_PK_TPNT protocol from
``mypaper/theory/stage_plan_handoff.md``.  It keeps the old adapter-BA Llama
T-PK table as an audit input, writes a new deployed BF16 merged-minus-base
Llama p_k track, reuses the existing Qwen deployed-effective p_k track, and
builds the minimal TPNT principal-mask / angle / NSS comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cycle09_stage3_followup_common as common  # noqa: E402
import cycle09_stage3_tpk as old_tpk  # noqa: E402


AUTODL = Path("/root/autodl-tmp")
REPO = Path("/root/LLM-output-density")
OUT_BASE = AUTODL / "cycle09_relative_functional_contraction/d11_pk_tpnt"
MINI = REPO / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"

LLAMA_BASE = AUTODL / "model/Meta/modelscope/Llama-3.2-3B"
QWEN_BASE = AUTODL / "model/Qwen/Qwen3-4B-Base"
LLAMA_MERGED = AUTODL / "cycle09_block3/llama_models/merged"

D10_FINAL = AUTODL / "cycle09_relative_functional_contraction/d10_llama_numeric_parity/formal/final"
STAGE3 = AUTODL / "cycle09_stage3_followup"

ARMS = ("opd", "sft", "offkd", "seqkd")
STEPS = (5, 20, 40, 80, 160, 320)
MODULES = tuple(common.MODULES)
PK_KS = (4, 8, 16, 32)
TPNT_KS = (16, 32, 50)
ANGLE_KS = (4, 8, 16, 32)
ALPHAS = (0.01, 0.10)
PROBES = ("E_general", "E_ood", "E_if", "E_math")


@dataclass(frozen=True)
class FamilySpec:
    family: str
    model_label: str
    base: Path
    layer: int


FAMILIES = {
    "llama": FamilySpec("llama", "llama", LLAMA_BASE, 14),
    "qwen": FamilySpec("qwen", "qwen", QWEN_BASE, 18),
}


def utc_now() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def out_dir(tag: str) -> Path:
    return ensure_dir(OUT_BASE / tag / "final")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]] | pd.DataFrame) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(tmp, index=False)
    else:
        pd.DataFrame(list(rows)).to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path), "exists": True, "bytes": path.stat().st_size, "sha256": h.hexdigest()}


_INDEX_CACHE: dict[Path, dict[str, str]] = {}


def weight_key(layer: int, module: str) -> str:
    return f"model.layers.{layer}.{module}.weight"


def read_weight(root: Path, key: str) -> torch.Tensor:
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        single = root / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError(f"missing safetensors index or single file in {root}")
        shard = single
    else:
        if root not in _INDEX_CACHE:
            _INDEX_CACHE[root] = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shard_name = _INDEX_CACHE[root].get(key)
        if shard_name is None:
            raise KeyError(f"{key} not found in {index_path}")
        shard = root / shard_name
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def llama_merged_root(arm: str, step: int) -> Path:
    root = LLAMA_MERGED / arm / f"step_{step:03d}"
    if not (root / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"missing Llama merged checkpoint: {root}")
    return root


def qwen_source(arm: str, step: int) -> dict[str, Any]:
    source = old_tpk.delta_source("qwen3_4b", arm, int(step), "bf16_merged_minus_base")
    if not source.get("complete"):
        raise FileNotFoundError(f"missing Qwen deployed-effective source: {source}")
    return source


def load_deployed_delta(
    spec: FamilySpec,
    arm: str,
    step: int,
    module: str,
    device: str,
    base_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return (delta_fp32, deployed_weight_fp32, provenance)."""
    key = weight_key(spec.layer, module)
    if spec.family == "llama":
        root = llama_merged_root(arm, step)
        wt_bf16 = read_weight(root, key).to(dtype=torch.bfloat16)
        delta = wt_bf16.to(device=device, dtype=torch.float32) - base_bf16.to(device=device, dtype=torch.float32)
        prov = {
            "delta_construction": "bf16_merged_minus_base",
            "native_source": "saved_merged_bf16",
            "checkpoint_materialization": "serialized_bf16_deployed_merged_checkpoint",
            "checkpoint_root": str(root),
            "checkpoint_tensor_sha256": tensor_sha256(wt_bf16),
        }
        return delta, wt_bf16.to(device=device, dtype=torch.float32), prov

    source = qwen_source(arm, step)
    delta = old_tpk.load_delta(source, spec.layer, module, device, base_bf16.to(device=device))
    if delta is None:
        raise RuntimeError(f"unexpected zero delta for {spec.family} {arm} step {step}")
    wt = base_bf16.to(device=device, dtype=torch.float32) + delta
    prov = {
        "delta_construction": "bf16_merged_minus_base",
        "native_source": source.get("native_source", "unknown"),
        "checkpoint_materialization": "saved_merged_bf16"
        if source.get("kind") == "merged_bf16"
        else "adapter_merge_quantized_to_bf16_in_memory",
        "checkpoint_root": str(source.get("path", "")),
        "checkpoint_tensor_sha256": tensor_sha256(wt.to(dtype=torch.bfloat16)),
    }
    return delta, wt, prov


def full_svd_top(weight: torch.Tensor, max_k: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = weight.to(device=device, dtype=torch.float32)
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    k = min(max_k, s.numel())
    return u[:, :k].contiguous(), s[:k].contiguous(), vh[:k, :].contiguous()


def normalized_spectrum(s: torch.Tensor, k: int = 50) -> list[float]:
    vals = s[: min(k, s.numel())].detach().float().cpu()
    denom = float(vals.sum().item())
    if denom <= 0:
        return [0.0 for _ in range(vals.numel())]
    return [float(x / denom) for x in vals.tolist()]


def p_k_rows(
    spec: FamilySpec,
    arm: str,
    step: int,
    module: str,
    u0: torch.Tensor,
    vh0: torch.Tensor,
    delta: torch.Tensor,
    prov: dict[str, Any],
    base_hash: str,
) -> list[dict[str, Any]]:
    denom = torch.sum(delta.square()).clamp_min(1e-30)
    rows = []
    for k in PK_KS:
        kk = min(k, u0.shape[1], vh0.shape[0])
        core = u0[:, :kk].T @ delta @ vh0[:kk, :].T
        value = float(torch.sum(core.square()) / denom)
        if not math.isfinite(value) or value < -1e-8 or value > 1 + 1e-8:
            raise RuntimeError(f"invalid p_k={value} for {spec.family} {arm} {step} {module} k={k}")
        rows.append(
            {
                "family": "llama3_2_3b" if spec.family == "llama" else "qwen3_4b",
                "model": spec.model_label,
                "arm": arm,
                "step": step,
                "checkpoint": step,
                "layer": spec.layer,
                "module": module,
                "rank_fraction": float(kk / min(delta.shape)),
                "k": kk,
                "rank_spec_kind": "fixed_k",
                "p_k": max(0.0, min(1.0, value)),
                "delta_construction": prov["delta_construction"],
                "native_source": prov["native_source"],
                "checkpoint_materialization": prov["checkpoint_materialization"],
                "source_svd_dtype": "torch.float32",
                "projection_dtype": "torch.float32",
                "base_tensor_sha256": base_hash,
                "checkpoint_tensor_sha256": prov["checkpoint_tensor_sha256"],
                "checkpoint_root": prov["checkpoint_root"],
            }
        )
    return rows


def principal_mask(u0: torch.Tensor, s0: torch.Tensor, vh0: torch.Tensor, rank_k: int, alpha: float) -> torch.Tensor:
    k = min(rank_k, s0.numel(), u0.shape[1], vh0.shape[0])
    recon = (u0[:, :k] * s0[:k].unsqueeze(0)) @ vh0[:k, :]
    flat = recon.abs().flatten()
    count = max(1, min(flat.numel(), int(round(float(alpha) * flat.numel()))))
    _, idx = torch.topk(flat, count, largest=True, sorted=False)
    mask = torch.zeros(flat.numel(), dtype=torch.bool, device=flat.device)
    mask[idx] = True
    return mask.reshape(recon.shape)


def lowrank_random_mask(
    shape: tuple[int, int],
    base_bf16: torch.Tensor,
    target_fro: float,
    rank: int,
    seed: int,
    device: str,
) -> torch.Tensor:
    if target_fro <= 0:
        return torch.zeros(shape, dtype=torch.bool, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    out_dim, in_dim = shape
    a = torch.randn((rank, in_dim), generator=gen, device=device, dtype=torch.float32)
    b = torch.randn((out_dim, rank), generator=gen, device=device, dtype=torch.float32)
    delta = b @ a
    delta *= target_fro / float(torch.linalg.vector_norm(delta).clamp_min(1e-30))
    deployed = base_bf16.to(device=device) + delta.to(dtype=torch.bfloat16)
    mask = deployed.to(dtype=torch.float32).ne(base_bf16.to(device=device, dtype=torch.float32))
    del a, b, delta, deployed
    return mask


def principal_angle_rows(
    u0: torch.Tensor,
    vh0: torch.Tensor,
    ut: torch.Tensor,
    vht: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in ANGLE_KS:
        kk = min(k, u0.shape[1], ut.shape[1], vh0.shape[0], vht.shape[0])
        cu = torch.linalg.svdvals((u0[:, :kk].T @ ut[:, :kk]).to(dtype=torch.float64)).clamp(0, 1)
        cv = torch.linalg.svdvals((vh0[:kk, :] @ vht[:kk, :].T).to(dtype=torch.float64)).clamp(0, 1)
        au = torch.rad2deg(torch.acos(cu.clamp(-1, 1)))
        av = torch.rad2deg(torch.acos(cv.clamp(-1, 1)))
        rows.append(
            {
                "angle_k": kk,
                "theta_u_mean_deg": float(au.mean().item()),
                "theta_u_max_deg": float(au.max().item()),
                "theta_v_mean_deg": float(av.mean().item()),
                "theta_v_max_deg": float(av.max().item()),
                "pabs_mean_cos_u": float(cu.mean().item()),
                "pabs_mean_cos_v": float(cv.mean().item()),
                "pabs_joint_mean_cos": float(((cu.mean() + cv.mean()) / 2).item()),
                "theta_u_all_deg_json": json.dumps([float(x) for x in au.cpu().tolist()]),
                "theta_v_all_deg_json": json.dumps([float(x) for x in av.cpu().tolist()]),
            }
        )
    return rows


def run_family(args: argparse.Namespace) -> dict[str, Any]:
    tag = args.tag
    spec = FAMILIES[args.family]
    arms = tuple(x.strip() for x in args.arms.split(",") if x.strip())
    steps = tuple(int(x) for x in args.steps.split(",") if x.strip())
    modules = tuple(x.strip() for x in args.modules.split(",") if x.strip())
    out = out_dir(tag)
    device = args.device
    max_k = max(max(PK_KS), max(TPNT_KS), max(ANGLE_KS))

    pk_rows_all: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    angle_rows_all: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    started = time.time()

    for module in modules:
        key = weight_key(spec.layer, module)
        base_bf16_cpu = read_weight(spec.base, key).to(dtype=torch.bfloat16)
        base_hash = tensor_sha256(base_bf16_cpu)
        u0, s0, vh0 = full_svd_top(base_bf16_cpu.to(dtype=torch.float32), max_k=max_k, device=device)
        source_spectrum = normalized_spectrum(s0, max_k)
        masks: dict[tuple[int, float], torch.Tensor] = {}
        for k_src in TPNT_KS:
            for alpha in ALPHAS:
                masks[(k_src, alpha)] = principal_mask(u0, s0, vh0, k_src, alpha)

        for arm in arms:
            if arm not in ARMS:
                raise ValueError(f"unknown arm {arm}")
            for step in steps:
                delta, wt, prov = load_deployed_delta(spec, arm, step, module, device, base_bf16_cpu)
                try:
                    if spec.family == "llama":
                        pk_rows_all.extend(p_k_rows(spec, arm, step, module, u0, vh0, delta, prov, base_hash))

                    update_mask = delta.ne(0)
                    update_count = int(update_mask.sum().item())
                    total = int(update_mask.numel())
                    delta_fro = float(torch.linalg.vector_norm(delta).item())
                    null_masks = [
                        lowrank_random_mask(
                            tuple(delta.shape),
                            base_bf16_cpu,
                            delta_fro,
                            rank=int(args.null_rank),
                            seed=int(sha256_text(f"{spec.family}-{arm}-{step}-{module}-{seed}")[:8], 16),
                            device=device,
                        )
                        for seed in range(int(args.null_seeds))
                    ]

                    for (k_src, alpha), pmask in masks.items():
                        principal_count = int(pmask.sum().item())
                        overlap_count = int((update_mask & pmask).sum().item())
                        expected = update_count * principal_count / total if total else float("nan")
                        lift = overlap_count / expected if expected and expected > 0 else float("nan")
                        null_lifts = []
                        for nmask in null_masks:
                            n_update = int(nmask.sum().item())
                            n_overlap = int((nmask & pmask).sum().item())
                            n_expected = n_update * principal_count / total if total else float("nan")
                            null_lifts.append(n_overlap / n_expected if n_expected and n_expected > 0 else float("nan"))
                        null_mean = float(np.nanmean(null_lifts)) if null_lifts else float("nan")
                        null_std = float(np.nanstd(null_lifts, ddof=0)) if null_lifts else float("nan")
                        mask_rows.append(
                            {
                                "model": spec.model_label,
                                "family": "llama3_2_3b" if spec.family == "llama" else "qwen3_4b",
                                "arm": arm,
                                "checkpoint": step,
                                "step": step,
                                "layer": spec.layer,
                                "module": module,
                                "source_rank_k": k_src,
                                "mask_density_alpha": alpha,
                                "principal_top_count": principal_count,
                                "update_count": update_count,
                                "total_entries": total,
                                "update_density": update_count / total if total else float("nan"),
                                "overlap_count": overlap_count,
                                "coverage": overlap_count / update_count if update_count else float("nan"),
                                "overlap_lift": lift,
                                "random_null_type": "rank32_norm_matched_lowrank_bf16_mask",
                                "random_null_rank": int(args.null_rank),
                                "random_null_seeds": int(args.null_seeds),
                                "random_null_overlap_lift_mean": null_mean,
                                "random_null_overlap_lift_std": null_std,
                                "overlap_lift_minus_random_null": lift - null_mean if math.isfinite(lift) and math.isfinite(null_mean) else float("nan"),
                                "delta_construction": prov["delta_construction"],
                                "native_source": prov["native_source"],
                                "checkpoint_materialization": prov["checkpoint_materialization"],
                                "base_tensor_sha256": base_hash,
                                "checkpoint_tensor_sha256": prov["checkpoint_tensor_sha256"],
                            }
                        )
                    for nmask in null_masks:
                        del nmask

                    ut, st, vht = full_svd_top(wt, max_k=max(ANGLE_KS), device=device)
                    current_spectrum = normalized_spectrum(st, max(ANGLE_KS))
                    src = np.array(source_spectrum[: len(current_spectrum)], dtype=np.float64)
                    cur = np.array(current_spectrum, dtype=np.float64)
                    nss_l1 = float(np.abs(cur - src).sum())
                    nss_l2 = float(np.sqrt(np.square(cur - src).sum()))
                    for angle_row in principal_angle_rows(u0, vh0, ut, vht):
                        angle_rows_all.append(
                            {
                                "model": spec.model_label,
                                "family": "llama3_2_3b" if spec.family == "llama" else "qwen3_4b",
                                "arm": arm,
                                "checkpoint": step,
                                "step": step,
                                "layer": spec.layer,
                                "module": module,
                                **angle_row,
                                "nss_l1_top32": nss_l1,
                                "nss_l2_top32": nss_l2,
                                "source_normalized_spectrum_top32_json": json.dumps(source_spectrum[:32]),
                                "checkpoint_normalized_spectrum_top32_json": json.dumps(current_spectrum[:32]),
                                "nss_formula": "L1/L2 distance between top-32 sum-normalized singular spectra of BF16 deployed checkpoint and BF16 base.",
                                "delta_construction": prov["delta_construction"],
                                "native_source": prov["native_source"],
                                "checkpoint_materialization": prov["checkpoint_materialization"],
                                "base_tensor_sha256": base_hash,
                                "checkpoint_tensor_sha256": prov["checkpoint_tensor_sha256"],
                            }
                        )
                    del ut, st, vht
                    status_rows.append(
                        {
                            "task": f"family_{spec.family}_module_cell",
                            "model": spec.model_label,
                            "arm": arm,
                            "checkpoint": step,
                            "layer": spec.layer,
                            "module": module,
                            "status": "COMPLETE",
                            "delta_construction": prov["delta_construction"],
                            "native_source": prov["native_source"],
                            "checkpoint_materialization": prov["checkpoint_materialization"],
                        }
                    )
                finally:
                    del delta, wt
                    torch.cuda.empty_cache()
        for value in masks.values():
            del value
        del u0, s0, vh0, base_bf16_cpu
        torch.cuda.empty_cache()

    if spec.family == "llama":
        atomic_csv(out / "d11_llama_merged_pk.csv", pk_rows_all)
        write_llama_pk_audit(out)
    atomic_csv(out / f"d11_tpnt_principal_mask_{spec.family}.csv", mask_rows)
    atomic_csv(out / f"d11_tpnt_angles_pabs_nss_{spec.family}.csv", angle_rows_all)
    atomic_csv(out / f"d11_task_status_{spec.family}.csv", status_rows)
    payload = {
        "schema_version": "cycle09_d11_family_v1",
        "status": "COMPLETE",
        "tag": tag,
        "family": spec.family,
        "device": device,
        "arms": list(arms),
        "steps": list(steps),
        "layer": spec.layer,
        "modules": list(modules),
        "pk_rows": len(pk_rows_all),
        "tpnt_mask_rows": len(mask_rows),
        "tpnt_angle_rows": len(angle_rows_all),
        "null_rank": int(args.null_rank),
        "null_seeds": int(args.null_seeds),
        "seconds": round(time.time() - started, 3),
        "created_utc": utc_now(),
    }
    atomic_json(out / f"d11_family_{spec.family}_manifest.json", payload)
    return payload


def write_llama_pk_audit(out: Path) -> None:
    new_path = out / "d11_llama_merged_pk.csv"
    old_path = STAGE3 / "H2_tpk/T_PK_llama3_2_3b.csv"
    if not new_path.exists() or not old_path.exists():
        return
    new = pd.read_csv(new_path)
    old = pd.read_csv(old_path)
    key = ["arm", "step", "layer", "module", "k"]
    merged = old.merge(new, on=key, suffixes=("_adapter_ba", "_merged_bf16"))
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                **{k: row[k] for k in key},
                "p_k_adapter_ba": row["p_k_adapter_ba"],
                "p_k_merged_bf16": row["p_k_merged_bf16"],
                "abs_diff": abs(row["p_k_merged_bf16"] - row["p_k_adapter_ba"]),
                "signed_diff_merged_minus_adapter": row["p_k_merged_bf16"] - row["p_k_adapter_ba"],
                "adapter_delta_construction": row["delta_construction_adapter_ba"],
                "merged_delta_construction": row["delta_construction_merged_bf16"],
            }
        )
    atomic_csv(out / "d11_llama_pk_numeric_audit.csv", rows)


def add_d11_pk_to_feature_matrix(out: Path) -> pd.DataFrame:
    feature_path = D10_FINAL / "d10_5_a4_feature_matrix.csv"
    features = pd.read_csv(feature_path)
    features = features[(features["epsilon"].round(6) == 0.05) & (features["checkpoint"].isin(STEPS))].copy()
    features = features[features["probe_name"].isin(PROBES)].copy()

    new_pk = pd.read_csv(out / "d11_llama_merged_pk.csv")
    new_wide = (
        new_pk.pivot_table(index=["arm", "step", "layer"], columns="k", values="p_k", aggfunc="mean")
        .reset_index()
        .rename(columns={4: "p_k4", 8: "p_k8", 16: "p_k16", 32: "p_k32", "step": "checkpoint"})
    )
    for col in ["p_k4", "p_k8", "p_k16", "p_k32"]:
        if col not in new_wide.columns:
            raise RuntimeError(f"missing {col} from D11 Llama p_k")
    mask = features["model"].eq("llama")
    lhs = features.loc[mask, ["arm", "checkpoint", "layer"]].merge(
        new_wide, on=["arm", "checkpoint", "layer"], how="left"
    )
    if lhs[["p_k4", "p_k8", "p_k16", "p_k32"]].isna().any().any():
        raise RuntimeError("D11 Llama p_k merge left missing cells")
    features.loc[mask, ["p_k4", "p_k8", "p_k16", "p_k32"]] = lhs[["p_k4", "p_k8", "p_k16", "p_k32"]].to_numpy()
    features["pk_track"] = np.where(features["model"].eq("llama"), "llama_d11_bf16_merged_minus_base", "qwen_existing_deployed_effective")
    atomic_csv(out / "d11_a4_feature_matrix_replaced_pk.csv", features)
    return features


def standardize(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xtr = train[cols].to_numpy(dtype=np.float64)
    xte = test[cols].to_numpy(dtype=np.float64)
    mu = np.nanmean(xtr, axis=0)
    sd = np.nanstd(xtr, axis=0)
    sd[sd < 1e-12] = 1.0
    return np.nan_to_num((xtr - mu) / sd), np.nan_to_num((xte - mu) / sd)


def ridge_fit_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, alpha: float = 1e-2) -> np.ndarray:
    xtr1 = np.column_stack([np.ones(len(xtr)), xtr])
    xte1 = np.column_stack([np.ones(len(xte)), xte])
    eye = np.eye(xtr1.shape[1])
    eye[0, 0] = 0.0
    beta = np.linalg.pinv(xtr1.T @ xtr1 + alpha * eye) @ xtr1.T @ ytr
    return xte1 @ beta


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.square(y - y.mean()).sum())
    if denom <= 1e-12:
        return float("nan")
    return 1.0 - float(np.square(y - pred).sum()) / denom


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    vals = [(p > n) + 0.5 * (p == n) for p in pos for n in neg]
    return float(np.mean(vals))


def balanced_acc(y: np.ndarray, score: np.ndarray) -> float:
    pred = (score >= 0.5).astype(int)
    vals = []
    for label in (0, 1):
        mask = y == label
        if mask.any():
            vals.append(float(np.mean(pred[mask] == label)))
    return float(np.mean(vals)) if vals else float("nan")


def cv_regression(df: pd.DataFrame, feature_sets: dict[str, list[str]], targets: list[str]) -> pd.DataFrame:
    rows = []
    groups = sorted(df["checkpoint"].unique())
    for model_scope, part in [("pooled", df)] + [(str(m), df[df["model"].eq(m)]) for m in sorted(df["model"].unique())]:
        if part.empty:
            continue
        for target in targets:
            for name, cols in feature_sets.items():
                preds = np.full(len(part), np.nan, dtype=np.float64)
                y = part[target].to_numpy(dtype=np.float64)
                for group in groups:
                    te_mask = part["checkpoint"].eq(group).to_numpy()
                    tr_mask = ~te_mask
                    if te_mask.sum() == 0 or tr_mask.sum() <= len(cols) + 1:
                        continue
                    train = part.iloc[np.where(tr_mask)[0]]
                    test = part.iloc[np.where(te_mask)[0]]
                    xtr, xte = standardize(train, test, cols)
                    preds[te_mask] = ridge_fit_predict(xtr, train[target].to_numpy(dtype=np.float64), xte)
                valid = np.isfinite(preds) & np.isfinite(y)
                rows.append(
                    {
                        "analysis": "checkpoint_grouped_regression",
                        "model_scope": model_scope,
                        "target": target,
                        "feature_set": name,
                        "features": ",".join(cols),
                        "n": int(valid.sum()),
                        "heldout_r2": r2_score(y[valid], preds[valid]) if valid.sum() else float("nan"),
                        "mae": float(np.mean(np.abs(y[valid] - preds[valid]))) if valid.sum() else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def cv_binary(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    macro = df.groupby(["model", "arm", "checkpoint"], as_index=False).mean(numeric_only=True)
    macro["is_opd"] = macro["arm"].eq("opd").astype(int)
    for model_scope, part in [("pooled", macro)] + [(str(m), macro[macro["model"].eq(m)]) for m in sorted(macro["model"].unique())]:
        if part.empty:
            continue
        y = part["is_opd"].to_numpy(dtype=int)
        for name, cols in feature_sets.items():
            scores = np.full(len(part), np.nan)
            for group in sorted(part["checkpoint"].unique()):
                te = part["checkpoint"].eq(group).to_numpy()
                tr = ~te
                if te.sum() == 0 or len(np.unique(y[tr])) < 2:
                    continue
                train = part.iloc[np.where(tr)[0]]
                test = part.iloc[np.where(te)[0]]
                xtr, xte = standardize(train, test, cols)
                raw = ridge_fit_predict(xtr, train["is_opd"].to_numpy(dtype=np.float64), xte)
                scores[te] = 1.0 / (1.0 + np.exp(-raw))
            valid = np.isfinite(scores)
            rows.append(
                {
                    "analysis": "checkpoint_grouped_opd_vs_nonopd_macro",
                    "model_scope": model_scope,
                    "target": "is_opd",
                    "feature_set": name,
                    "features": ",".join(cols),
                    "n": int(valid.sum()),
                    "auc": auc_score(y[valid], scores[valid]) if valid.sum() else float("nan"),
                    "balanced_accuracy": balanced_acc(y[valid], scores[valid]) if valid.sum() else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def postprocess(args: argparse.Namespace) -> dict[str, Any]:
    out = out_dir(args.tag)
    features = add_d11_pk_to_feature_matrix(out)

    mask_parts = [pd.read_csv(p) for p in [out / "d11_tpnt_principal_mask_llama.csv", out / "d11_tpnt_principal_mask_qwen.csv"] if p.exists()]
    angle_parts = [pd.read_csv(p) for p in [out / "d11_tpnt_angles_pabs_nss_llama.csv", out / "d11_tpnt_angles_pabs_nss_qwen.csv"] if p.exists()]
    if mask_parts:
        mask = pd.concat(mask_parts, ignore_index=True)
        atomic_csv(out / "d11_tpnt_principal_mask.csv", mask)
        mask_cell = (
            mask.groupby(["model", "arm", "checkpoint", "layer"], as_index=False)
            .agg(
                tpnt_overlap_lift=("overlap_lift", "mean"),
                tpnt_lift_minus_null=("overlap_lift_minus_random_null", "mean"),
                tpnt_coverage=("coverage", "mean"),
            )
        )
        features = features.merge(mask_cell, on=["model", "arm", "checkpoint", "layer"], how="left")
    else:
        features["tpnt_overlap_lift"] = np.nan
        features["tpnt_lift_minus_null"] = np.nan
        features["tpnt_coverage"] = np.nan
    if angle_parts:
        angles = pd.concat(angle_parts, ignore_index=True)
        atomic_csv(out / "d11_tpnt_angles_pabs_nss.csv", angles)
        angle_cell = (
            angles.groupby(["model", "arm", "checkpoint", "layer"], as_index=False)
            .agg(
                pabs_joint_mean_cos=("pabs_joint_mean_cos", "mean"),
                nss_l1_top32=("nss_l1_top32", "mean"),
                nss_l2_top32=("nss_l2_top32", "mean"),
            )
        )
        features = features.merge(angle_cell, on=["model", "arm", "checkpoint", "layer"], how="left")
    else:
        features["pabs_joint_mean_cos"] = np.nan
        features["nss_l1_top32"] = np.nan
        features["nss_l2_top32"] = np.nan

    atomic_csv(out / "d11_same_cell_feature_matrix.csv", features)

    feature_sets = {
        "W": ["raw_update_energy_equal7"],
        "p_k": ["p_k4", "p_k8", "p_k16", "p_k32"],
        "C": ["c_epsilon"],
        "W_plus_C": ["raw_update_energy_equal7", "c_epsilon"],
        "p_k_plus_C": ["p_k4", "p_k8", "p_k16", "p_k32", "c_epsilon"],
        "TPNT": ["tpnt_overlap_lift", "tpnt_lift_minus_null", "pabs_joint_mean_cos", "nss_l1_top32"],
        "TPNT_plus_C": ["tpnt_overlap_lift", "tpnt_lift_minus_null", "pabs_joint_mean_cos", "nss_l1_top32", "c_epsilon"],
        "p_k_TPNT_C": [
            "p_k4",
            "p_k8",
            "p_k16",
            "p_k32",
            "tpnt_overlap_lift",
            "tpnt_lift_minus_null",
            "pabs_joint_mean_cos",
            "nss_l1_top32",
            "c_epsilon",
        ],
    }
    targets = ["cumulative_kl_base_to_current", "absolute_delta_nll_cumulative", "delta_nll_cumulative"]
    reg = cv_regression(features, feature_sets, targets)
    disc = cv_binary(features, feature_sets)
    comparison = pd.concat([reg, disc], ignore_index=True, sort=False)

    # Add simple delta-R2 rows against matched baselines for quick reading.
    r2 = comparison[comparison["analysis"].eq("checkpoint_grouped_regression")].copy()
    extras = []
    for _, group in r2.groupby(["model_scope", "target"]):
        lookup = dict(zip(group["feature_set"], group["heldout_r2"]))
        for base, full in [("W", "W_plus_C"), ("p_k", "p_k_plus_C"), ("TPNT", "TPNT_plus_C"), ("p_k", "p_k_TPNT_C")]:
            if base in lookup and full in lookup:
                extras.append(
                    {
                        "analysis": "delta_r2",
                        "model_scope": group["model_scope"].iloc[0],
                        "target": group["target"].iloc[0],
                        "feature_set": f"{full}_minus_{base}",
                        "heldout_r2": lookup[full] - lookup[base],
                        "n": int(group["n"].max()),
                    }
                )
    if extras:
        comparison = pd.concat([comparison, pd.DataFrame(extras)], ignore_index=True, sort=False)
    atomic_csv(out / "d11_same_cell_incremental_comparison.csv", comparison)

    task_status = []
    for family in ("llama", "qwen"):
        p = out / f"d11_task_status_{family}.csv"
        if p.exists():
            task_status.append(pd.read_csv(p))
    status = pd.concat(task_status, ignore_index=True) if task_status else pd.DataFrame()
    summary_rows = [
        {"task": "E0_llama_merged_pk", "status": "COMPLETE" if (out / "d11_llama_merged_pk.csv").exists() else "MISSING"},
        {"task": "E1_incremental_rebuild", "status": "COMPLETE"},
        {"task": "E2_tpnt_principal_mask", "status": "COMPLETE" if mask_parts else "MISSING"},
        {"task": "E3_angles_pabs_nss", "status": "COMPLETE" if angle_parts else "MISSING"},
        {"task": "E4_fair_comparison", "status": "COMPLETE"},
    ]
    status = pd.concat([status, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)
    atomic_csv(out / "d11_pk_tpnt_task_status.csv", status)
    write_handoff(out, args.tag, comparison, features)
    manifest = write_manifest(out, args.tag)
    mirror_outputs(out)
    return manifest


def write_handoff(out: Path, tag: str, comparison: pd.DataFrame, features: pd.DataFrame) -> None:
    n_pk = len(pd.read_csv(out / "d11_llama_merged_pk.csv")) if (out / "d11_llama_merged_pk.csv").exists() else 0
    n_audit = len(pd.read_csv(out / "d11_llama_pk_numeric_audit.csv")) if (out / "d11_llama_pk_numeric_audit.csv").exists() else 0
    n_mask = len(pd.read_csv(out / "d11_tpnt_principal_mask.csv")) if (out / "d11_tpnt_principal_mask.csv").exists() else 0
    n_angles = len(pd.read_csv(out / "d11_tpnt_angles_pabs_nss.csv")) if (out / "d11_tpnt_angles_pabs_nss.csv").exists() else 0
    n_comp = len(comparison)
    branch = "COMPLETE_D11_CORE"
    lines = [
        "# D11 PK-TPNT handoff",
        "",
        "## Status",
        "",
        f"- status: `{branch}`",
        f"- tag: `{tag}`",
        f"- created_utc: `{utc_now()}`",
        "- protocol: `D11_PK_TPNT` from `mypaper/theory/stage_plan_handoff.md`",
        "- Llama strict p_k official track: `bf16_merged_minus_base`",
        "- Qwen strict p_k official track: reused existing deployed-effective result",
        "- no training, no free-generation behavior eval, no c_epsilon/r_epsilon recomputation",
        "",
        "## Coverage",
        "",
        "| artifact | rows |",
        "|---|---:|",
        f"| d11_llama_merged_pk.csv | {n_pk} |",
        f"| d11_llama_pk_numeric_audit.csv | {n_audit} |",
        f"| d11_tpnt_principal_mask.csv | {n_mask} |",
        f"| d11_tpnt_angles_pabs_nss.csv | {n_angles} |",
        f"| d11_same_cell_feature_matrix.csv | {len(features)} |",
        f"| d11_same_cell_incremental_comparison.csv | {n_comp} |",
        "",
        "## Raw Comparison Head",
        "",
    ]
    head = comparison.head(20).copy()
    lines.extend(frame_to_markdown(head).splitlines())
    lines += [
        "",
        "## Boundaries",
        "",
        "- TPNT random-null column uses a rank-32 Frobenius-norm-matched low-rank BF16 update mask with fixed deterministic seeds.",
        "- E5--E7 optional enhancements are not included in this core handoff.",
        "- This handoff reports raw tables and mechanical status only; it does not adjudicate Theory claims.",
        "",
        "## Output Files",
        "",
    ]
    for name in [
        "d11_pk_tpnt_task_status.csv",
        "d11_llama_merged_pk.csv",
        "d11_llama_pk_numeric_audit.csv",
        "d11_tpnt_principal_mask.csv",
        "d11_tpnt_angles_pabs_nss.csv",
        "d11_same_cell_incremental_comparison.csv",
        "d11_modelwise_and_crossmodel_summary.md",
        "d11_pk_tpnt_manifest.json",
    ]:
        lines.append(f"- `{name}`: `{out / name}`")
    (out / "d11_modelwise_and_crossmodel_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append("" if not math.isfinite(val) else f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_manifest(out: Path, tag: str) -> dict[str, Any]:
    names = [
        "d11_pk_tpnt_task_status.csv",
        "d11_llama_merged_pk.csv",
        "d11_llama_pk_numeric_audit.csv",
        "d11_tpnt_principal_mask.csv",
        "d11_tpnt_angles_pabs_nss.csv",
        "d11_same_cell_feature_matrix.csv",
        "d11_same_cell_incremental_comparison.csv",
        "d11_modelwise_and_crossmodel_summary.md",
    ]
    payload = {
        "schema_version": "cycle09_d11_pk_tpnt_manifest_v1",
        "status": "COMPLETE_D11_CORE",
        "tag": tag,
        "created_utc": utc_now(),
        "protocol": "D11_PK_TPNT",
        "outputs": [artifact(out / name) for name in names if (out / name).exists()],
    }
    atomic_json(out / "d11_pk_tpnt_manifest.json", payload)
    return payload


def mirror_outputs(out: Path) -> None:
    ensure_dir(MINI)
    for path in out.glob("d11_*"):
        if path.is_file():
            shutil.copy2(path, MINI / path.name)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    tag = args.tag
    rows = []
    for fam_name, spec in FAMILIES.items():
        for module in MODULES:
            key = weight_key(spec.layer, module)
            base_ok = spec.base.exists()
            try:
                _ = read_weight(spec.base, key)
                base_tensor = True
            except Exception as exc:
                base_tensor = False
                rows.append({"family": fam_name, "module": module, "status": "MISSING_BASE_TENSOR", "error": str(exc)})
                continue
            for arm in ARMS:
                for step in STEPS:
                    try:
                        if fam_name == "llama":
                            root = llama_merged_root(arm, step)
                            _ = read_weight(root, key)
                            source_status = "saved_merged_bf16"
                        else:
                            source = qwen_source(arm, step)
                            source_status = source.get("native_source", source.get("kind", "unknown"))
                        status = "COMPLETE"
                        error = ""
                    except Exception as exc:
                        status = "MISSING"
                        source_status = ""
                        error = str(exc)
                    rows.append(
                        {
                            "family": fam_name,
                            "base_exists": base_ok,
                            "base_tensor": base_tensor,
                            "arm": arm,
                            "checkpoint": step,
                            "layer": spec.layer,
                            "module": module,
                            "status": status,
                            "source_status": source_status,
                            "error": error,
                        }
                    )
    out = out_dir(tag)
    atomic_csv(out / "d11_preflight_cells.csv", rows)
    complete = all(row["status"] == "COMPLETE" for row in rows)
    payload = {"schema_version": "cycle09_d11_preflight_v1", "status": "COMPLETE" if complete else "BLOCKED", "tag": tag, "rows": len(rows), "created_utc": utc_now(), "output": str(out / "d11_preflight_cells.csv")}
    atomic_json(out / "d11_preflight_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "family", "postprocess", "all"), required=True)
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--family", choices=("llama", "qwen"), default="llama")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--steps", default=",".join(str(x) for x in STEPS))
    parser.add_argument("--modules", default=",".join(MODULES))
    parser.add_argument("--null-rank", type=int, default=32)
    parser.add_argument("--null-seeds", type=int, default=3)
    args = parser.parse_args()

    if args.phase == "preflight":
        result = preflight(args)
    elif args.phase == "family":
        result = run_family(args)
    elif args.phase == "postprocess":
        result = postprocess(args)
    else:
        result = preflight(args)
        if result["status"] != "COMPLETE":
            raise RuntimeError(f"D11 preflight failed: {result}")
        run_family(args)
        result = postprocess(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
