#!/usr/bin/env python3
"""Cycle 09 Round 3 weight-side principal-angle measurement (R3-3).

SFT uses the clean fp32 LoRA B@A update. OPD uses a top-32 approximation of
the merged-minus-base update because the original OPD adapter was pruned.
The output records both tracks and never treats the OPD approximation as a
clean adapter measurement.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

REPO = Path("/root/LLM-output-density")
SIDE = REPO / "experiments/opd_sft_h1"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SIDE) not in sys.path:
    sys.path.insert(0, str(SIDE))

from scripts.export_weights import MODULES, export_model_weights  # noqa: E402

MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
BASE_MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen3-4B-Base")
OPD_MERGED = Path("/root/autodl-tmp/cycle08_opd_trajectory/_merged_models")
SFT_ADAPTERS = Path("/root/autodl-tmp/cycle07_base_sft_trajectory/checkpoints")
DEFAULT_CACHE = Path("/root/autodl-tmp/cycle09_r3/theta_w")

STEPS = (0, 5, 10, 20, 40, 160, 624)
DEFAULT_LAYERS = (9, 18, 27)
DEFAULT_RANKS = (8, 16, 32, 64, 128)
MODULE_NAMES = tuple(name for name, _ in MODULES)


def step_label(step: int) -> str:
    return f"step_{int(step):03d}"


def matrix_filename(layer: int, module: str) -> str:
    return f"model_layers_{layer}_{module}_weight.npy"


def parse_ints(value: str, default: tuple[int, ...]) -> list[int]:
    if not value:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_strings(value: str, default: tuple[str, ...]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def adapter_scaling(adapter_dir: Path) -> float:
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    rank = float(config["r"])
    alpha = float(config["lora_alpha"])
    return alpha / math.sqrt(rank) if config.get("use_rslora", False) else alpha / rank


def load_sft_delta(adapter_dir: Path, layer: int, module: str, device: torch.device) -> torch.Tensor:
    weights = load_file(adapter_dir / "adapter_model.safetensors")
    prefix = f"base_model.model.model.layers.{layer}.{module}"
    left = weights[f"{prefix}.lora_B.weight"].to(device=device, dtype=torch.float32)
    right = weights[f"{prefix}.lora_A.weight"].to(device=device, dtype=torch.float32)
    return adapter_scaling(adapter_dir) * (left @ right)


def top32_approx(delta: torch.Tensor) -> torch.Tensor:
    q = min(40, min(delta.shape))
    if q <= 32:
        left, singular, right = torch.linalg.svd(delta, full_matrices=False)
        return (left[:, :32] * singular[:32]) @ right[:32, :]
    # Randomized SVD only approximates an update that is already an OPD proxy.
    left, singular, right = torch.svd_lowrank(delta, q=q, niter=4)
    order = torch.argsort(singular, descending=True)
    left = left[:, order]
    singular = singular[order]
    right = right[:, order]
    return (left[:, :32] * singular[:32]) @ right[:, :32].T


def top_subspaces(
    matrix: torch.Tensor,
    rank_max: int,
    solver: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank_max = min(rank_max, min(matrix.shape))
    if solver == "exact":
        left, _singular, right = torch.linalg.svd(matrix, full_matrices=False)
        return left[:, :rank_max].contiguous(), right[:rank_max, :].T.contiguous()

    q = min(rank_max + 16, min(matrix.shape))
    left, singular, right = torch.svd_lowrank(matrix, q=q, niter=4)
    order = torch.argsort(singular, descending=True)
    return (
        left[:, order[:rank_max]].contiguous(),
        right[:, order[:rank_max]].contiguous(),
    )


def angle_summary(base: torch.Tensor, current: torch.Tensor, rank: int) -> tuple[float, float, float]:
    cosine = torch.linalg.svdvals(base[:, :rank].T @ current[:, :rank]).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosine))
    return (
        float(angles.max().item()),
        float(angles.mean().item()),
        float(cosine.min().item()),
    )


def ensure_exports(
    cache: Path,
    layers: list[int],
    modules: list[tuple[str, list[str]]],
) -> Path:
    base_dir = cache / "exports" / "base"
    export_model_weights(str(BASE_MODEL), base_dir, layers=layers, modules=modules)
    return base_dir


def export_opd_step(
    cache: Path,
    step: int,
    layers: list[int],
    modules: list[tuple[str, list[str]]],
) -> Path:
    output = cache / "exports" / "opd" / step_label(step)
    model = OPD_MERGED / step_label(step)
    if not (model / "config.json").exists():
        raise FileNotFoundError(f"missing OPD merged model: {model}")
    export_model_weights(str(model), output, layers=layers, modules=modules)
    return output


def base_weight(base_dir: Path, layer: int, module: str, device: torch.device) -> torch.Tensor:
    path = base_dir / matrix_filename(layer, module)
    if not path.exists():
        raise FileNotFoundError(f"missing base export: {path}")
    return torch.from_numpy(np.load(path)).to(device=device, dtype=torch.float32)


def opd_weight(
    base: torch.Tensor,
    export_dir: Path,
    layer: int,
    module: str,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    path = export_dir / matrix_filename(layer, module)
    if not path.exists():
        raise FileNotFoundError(f"missing OPD export: {path}")
    merged = torch.from_numpy(np.load(path)).to(device=device, dtype=torch.float32)
    update = top32_approx(merged - base)
    update_norm = float(torch.linalg.vector_norm(update).item())
    return base + update, update_norm


def row_for_rank(
    *,
    arm: str,
    step: int,
    layer: int,
    module: str,
    rank: int,
    source_kind: str,
    base_u: torch.Tensor,
    base_v: torch.Tensor,
    cur_u: torch.Tensor,
    cur_v: torch.Tensor,
    update_norm: float,
    solver: str,
) -> dict[str, Any]:
    u_max, u_mean, u_min_cos = angle_summary(base_u, cur_u, rank)
    v_max, v_mean, v_min_cos = angle_summary(base_v, cur_v, rank)
    return {
        "arm": arm,
        "step": step,
        "layer": layer,
        "module": module,
        "rank": rank,
        "source_kind": source_kind,
        "solver": solver,
        "left_max_angle_deg": f"{u_max:.8f}",
        "left_mean_angle_deg": f"{u_mean:.8f}",
        "left_min_cosine": f"{u_min_cos:.8f}",
        "right_max_angle_deg": f"{v_max:.8f}",
        "right_mean_angle_deg": f"{v_mean:.8f}",
        "right_min_cosine": f"{v_min_cos:.8f}",
        "update_frobenius": f"{update_norm:.8f}",
    }


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if int(row["rank"]) == 32:
            grouped.setdefault((row["arm"], int(row["step"])), []).append(row)

    lines = [
        "# R3-3 — Weight-Side Principal Angles",
        "",
        "Readings only. SFT uses clean fp32 B@A. OPD is labelled opd_top32_approx because "
        "the original OPD LoRA adapter is unavailable.",
        "",
        "| arm | step | source | mean left max angle deg over layer×module | mean right max angle deg over layer×module |",
        "|---|---:|---|---:|---:|",
    ]
    for (arm, step), items in sorted(grouped.items()):
        left = np.mean([float(item["left_max_angle_deg"]) for item in items])
        right = np.mean([float(item["right_max_angle_deg"]) for item in items])
        sources = sorted({item["source_kind"] for item in items})
        lines.append(f"| {arm} | {step} | {', '.join(sources)} | {left:.8f} | {right:.8f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini-root", type=Path, default=MINI)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--arms", default="opd,sft")
    parser.add_argument("--steps", default=",".join(map(str, STEPS)))
    parser.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    parser.add_argument("--modules", default=",".join(MODULE_NAMES))
    parser.add_argument("--ranks", default=",".join(map(str, DEFAULT_RANKS)))
    parser.add_argument("--rank-max", type=int, default=128)
    parser.add_argument("--solver", choices=("exact", "lowrank"), default="exact")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--keep-opd-exports", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.mini_root = args.mini_root / "smoke_theta_w"
        args.cache_root = args.cache_root / "smoke"
        args.arms = "opd,sft"
        args.steps = "0,5"
        args.layers = "18"
        args.modules = "self_attn.q_proj"
        args.ranks = "8,16"
        args.rank_max = 16
        args.solver = "lowrank"

    arms = parse_strings(args.arms, ("opd", "sft"))
    steps = parse_ints(args.steps, STEPS)
    layers = parse_ints(args.layers, DEFAULT_LAYERS)
    module_names = parse_strings(args.modules, MODULE_NAMES)
    modules = [(name, attrs) for name, attrs in MODULES if name in module_names]
    ranks = [rank for rank in parse_ints(args.ranks, DEFAULT_RANKS) if rank <= args.rank_max]
    if not modules:
        raise ValueError("no modules selected")
    if not ranks:
        raise ValueError("no ranks <= rank-max selected")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")

    print(
        "[R3-3 plan] "
        f"arms={arms} steps={steps} layers={layers} modules={[name for name, _ in modules]} "
        f"ranks={ranks} solver={args.solver} device={device}",
        flush=True,
    )
    if args.dry_run:
        return

    args.mini_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    base_dir = ensure_exports(args.cache_root, layers, modules)
    rows: list[dict[str, Any]] = []
    base_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    try:
        for layer in layers:
            for module, _attrs in modules:
                base = base_weight(base_dir, layer, module, device)
                base_u, base_v = top_subspaces(base, args.rank_max, args.solver)
                base_cache[(layer, module)] = (base, base_u, base_v)

        for arm in arms:
            for step in steps:
                opd_exports = None
                if arm == "opd" and step != 0:
                    opd_exports = export_opd_step(args.cache_root, step, layers, modules)
                adapter_dir = SFT_ADAPTERS / step_label(step)

                for layer in layers:
                    for module, _attrs in modules:
                        base, base_u, base_v = base_cache[(layer, module)]
                        if step == 0:
                            current = base
                            source_kind = "base_identity"
                            update_norm = 0.0
                        elif arm == "sft":
                            if not (adapter_dir / "adapter_model.safetensors").exists():
                                raise FileNotFoundError(f"missing SFT adapter: {adapter_dir}")
                            delta = load_sft_delta(adapter_dir, layer, module, device)
                            current = base + delta
                            source_kind = "sft_clean_fp32_ba"
                            update_norm = float(torch.linalg.vector_norm(delta).item())
                            del delta
                        elif arm == "opd":
                            if opd_exports is None:
                                raise RuntimeError("missing OPD export")
                            current, update_norm = opd_weight(base, opd_exports, layer, module, device)
                            source_kind = "opd_top32_approx"
                        else:
                            raise ValueError(f"unknown arm: {arm}")

                        cur_u, cur_v = top_subspaces(current, args.rank_max, args.solver)
                        for rank in ranks:
                            rows.append(
                                row_for_rank(
                                    arm=arm,
                                    step=step,
                                    layer=layer,
                                    module=module,
                                    rank=rank,
                                    source_kind=source_kind,
                                    base_u=base_u,
                                    base_v=base_v,
                                    cur_u=cur_u,
                                    cur_v=cur_v,
                                    update_norm=update_norm,
                                    solver=args.solver,
                                )
                            )
                        del current, cur_u, cur_v
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                if opd_exports is not None and not args.keep_opd_exports:
                    shutil.rmtree(opd_exports, ignore_errors=True)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[R3-3] completed {arm}/{step_label(step)}", flush=True)
    finally:
        base_cache.clear()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = args.mini_root / "R3_theta_w.csv"
    write_csv(csv_path, rows)
    write_markdown(args.mini_root / "R3_theta_w.md", rows)
    print(f"[R3-3] wrote {csv_path} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()

