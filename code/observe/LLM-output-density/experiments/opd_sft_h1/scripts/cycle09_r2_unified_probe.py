#!/usr/bin/env python3
"""Cycle 09 mini Round 2 compressed GetSlice/UV runner.

Default plan, per the 2026-07-09 discussion:
  * key checkpoints: 0,5,10,20,40,160,624
  * full 36-layer spectra: S + X_math
  * landmark spectra: X_math, X_ood_knowledge, X_general, X_math_hard, X_bos
    on layers 9,18,27
  * UV/theta_r: S-side top-r on layers 9,18,27

The landmark path is intentionally implemented with one model load per
arm/checkpoint, then multiple probes/layers inside that load. Running one
GetSlice process per probe/layer would spend most of the budget on reloads.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/root/LLM-output-density")
SIDE = REPO / "experiments/opd_sft_h1"
GETSLICE = REPO / "GetSlice"

DEFAULT_RUN = Path("/root/autodl-tmp/cycle09_r2")
DEFAULT_MINI = (
    REPO
    / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"
)

BASE_MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen3-4B-Base")
OPD_MERGED = Path("/root/autodl-tmp/cycle08_opd_trajectory/_merged_models")
SFT_CKPT = Path("/root/autodl-tmp/cycle07_base_sft_trajectory/checkpoints")
SFT_EXISTING_MERGED = Path("/root/autodl-tmp/cycle07_base_sft_trajectory/_merged_tmp")
S_ROOT = Path("/root/autodl-tmp/cycle07_base_sft_trajectory/getslice/inputs/S")
X_MATH = Path("/root/autodl-tmp/cycle07_base_sft_trajectory/getslice/inputs/X_base/x_probe.jsonl")
X_BOS_CANDIDATES = [
    Path("/root/autodl-tmp/exp0609/opd_minimal_03_v2/getslice/inputs/X_bos/x_probe.jsonl"),
    Path("/root/autodl-tmp/cycle04_opd_stability_gain/getslice/inputs/X_bos/x_probe.jsonl"),
    Path("/root/autodl-tmp/cycle04_smoke/getslice/inputs/X_bos/x_probe.jsonl"),
]

KEY_STEPS = [0, 5, 10, 20, 40, 160, 624]
ALL_LAYERS = list(range(36))
LANDMARK_LAYERS = [9, 18, 27]
FULL_X_PROBES = ["X_math"]
LANDMARK_X_PROBES = ["X_math", "X_ood_knowledge", "X_general", "X_math_hard", "X_bos"]
MODULES = [
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "self_attn.q_proj",
    "self_attn.v_proj",
]
DEFAULT_N = 32
DEFAULT_SEQLEN = 512
SEED = 3

RUN = DEFAULT_RUN
GS = RUN / "getslice"
MINI = DEFAULT_MINI

for path_item in (REPO, SIDE, GETSLICE):
    if str(path_item) not in sys.path:
        sys.path.insert(0, str(path_item))

from opd_sft_h1.geometry_metrics import (  # noqa: E402
    effective_rank,
    log_spectrum_drift,
    spectral_gap,
    xs_log_spectrum_gap,
)


def configure_roots(run_root: Path | None, mini_root: Path | None) -> None:
    global RUN, GS, MINI
    RUN = run_root or DEFAULT_RUN
    GS = RUN / "getslice"
    MINI = mini_root or DEFAULT_MINI


def step_label(step: int) -> str:
    return f"step_{int(step):03d}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [x.strip() for x in value.split(",") if x.strip()]


def battery_input(name: str) -> Path:
    local = GS / "inputs" / name / "x_probe.jsonl"
    if local.exists():
        return local
    return DEFAULT_RUN / "getslice" / "inputs" / name / "x_probe.jsonl"


def probe_paths() -> dict[str, Path]:
    paths = {
        "X_math": X_MATH,
        "X_ood_knowledge": battery_input("X_ood_knowledge"),
        "X_general": battery_input("X_general"),
        "X_math_hard": battery_input("X_math_hard"),
    }
    for candidate in X_BOS_CANDIDATES:
        if candidate.exists():
            paths["X_bos"] = candidate
            break
    return {name: path for name, path in paths.items() if path.exists()}


def select_probes(names: list[str]) -> dict[str, Path]:
    available = probe_paths()
    selected = {name: available[name] for name in names if name in available}
    missing = [name for name in names if name not in available]
    if missing:
        print(f"[Warn] requested probes not found and skipped: {missing}", flush=True)
    return selected


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def validate_x_jsonl(path: Path) -> None:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj.get("output"), dict) or "text" not in obj["output"]:
                raise ValueError(f"{path}:{line_no} expected output.text")


def validate_s_jsonl(path: Path) -> None:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if "question" not in obj or "answer" not in obj:
                raise ValueError(f"{path}:{line_no} expected question and answer")


def model_path_for(arm: str, step: int) -> Path:
    if step == 0:
        return BASE_MODEL
    if arm == "opd":
        return OPD_MERGED / step_label(step)
    if arm == "sft":
        existing = SFT_EXISTING_MERGED / step_label(step)
        if (existing / "config.json").exists():
            return existing
        merged = RUN / "sft_merged" / step_label(step)
        if (merged / "config.json").exists():
            return merged
        from scripts.run_opd_minimal_closure import merge_lora_adapter

        merge_lora_adapter(BASE_MODEL, SFT_CKPT / step_label(step), merged)
        return merged
    raise ValueError(f"unknown arm: {arm}")


def maybe_drop_sft(step: int, drop: bool) -> None:
    if not drop or step == 0:
        return
    merged = RUN / "sft_merged" / step_label(step)
    if merged.exists():
        shutil.rmtree(merged, ignore_errors=True)
        print(f"[Clean] removed {merged}", flush=True)


def full_s_json(arm: str, step: int) -> Path:
    return (
        GS
        / "outputs"
        / arm
        / step_label(step)
        / "spectra"
        / "S"
        / "math_cot_probe"
        / "sMat_math_cot_probe.json"
    )


def full_x_json(arm: str, step: int, probe: str) -> Path:
    return GS / "outputs" / arm / step_label(step) / "spectra" / probe / "X" / "xMat_X.json"


def landmark_x_json(arm: str, step: int, layer: int, probe: str) -> Path:
    return (
        GS
        / "outputs"
        / arm
        / step_label(step)
        / "landmark"
        / f"layer_{layer}"
        / probe
        / "X"
        / f"layer_{layer}"
        / "xMat_X.json"
    )


def uv_s_json(arm: str, step: int, layer: int) -> Path:
    return (
        GS
        / "outputs"
        / arm
        / step_label(step)
        / "uv"
        / f"layer_{layer}"
        / "S"
        / "math_cot_probe"
        / f"layer_{layer}"
        / "sMat_math_cot_probe.json"
    )


def uv_s_pt(arm: str, step: int, layer: int) -> Path:
    return (
        GS
        / "outputs"
        / arm
        / step_label(step)
        / "uv"
        / f"layer_{layer}"
        / "S"
        / "math_cot_probe"
        / f"layer_{layer}"
        / "sUV_math_cot_probe.pt"
    )


def cache_path(kind: str, name: str, n_samples: int, seqlen: int) -> Path:
    return GS / "cache" / kind / f"{name}__n{n_samples}__seq{seqlen}__seed{SEED}.pt"


def common_cfg(
    model: Path,
    save_path: Path,
    n_samples: int,
    seqlen: int,
    target_layer: int | None,
    uv: bool,
) -> dict[str, Any]:
    return {
        "tasks": ["math_cot_probe"],
        "DEV": "cuda",
        "layer_gpu_chunk_size": 12,
        "single_layer_task_group_size": 1,
        "epsilon": 0.001,
        "svd_singular_floor": 0.0,
        "cholesky_jitter": 0.00001,
        "activation_cache_device": "cuda",
        "uv_dtype": "float32",
        "cleanup_intermediate": True,
        "skip_existing_outputs": True,
        "model_dtype": "float16",
        "trust_remote_code": True,
        "s_batch_size": 1,
        "x_batch_size": 1,
        "save_s_json_path": "sMat_{task}.json",
        "save_x_json_path": "xMat_X.json",
        "save_s_pt_path": None,
        "save_x_pt_path": None,
        "save_s_uv_path": "sUV_{task}.pt" if uv else None,
        "save_x_uv_path": None,
        "save_metrics_pt_path": None,
        "save_metrics_json_path": None,
        "seed": SEED,
        "model_seq_len": seqlen,
        "target_layer": target_layer,
        "model": str(model),
        "save_path": str(save_path),
        "s_nsamples": n_samples,
        "x_nsamples": n_samples,
    }


def run_slice(cfg_path: Path) -> None:
    env = dict(os.environ)
    env.update(
        {
            "TMPDIR": "/root/autodl-tmp/pip-tmp",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    cmd = [sys.executable, str(GETSLICE / "slice.py"), "--config", str(cfg_path)]
    print(f"[SLICE] {cfg_path}", flush=True)
    result = subprocess.run(cmd, cwd=str(GETSLICE), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"GetSlice failed rc={result.returncode}: {cfg_path}")


def run_core(args: argparse.Namespace) -> None:
    probes = select_probes(parse_str_list(args.full_probes, FULL_X_PROBES))
    jobs: list[tuple[str, int, str, str | None, Path | None]] = []
    for arm in args.arms:
        for step in args.steps:
            jobs.append((arm, step, "S", None, None))
            for probe, path in probes.items():
                jobs.append((arm, step, "X", probe, path))

    print(f"[Plan] core full36 jobs={len(jobs)} probes={list(probes)}", flush=True)
    if args.dry_run:
        return

    current_sft: int | None = None
    for arm, step, side, probe, probe_path in jobs:
        if arm == "sft" and current_sft not in (None, step) and args.drop_sft_merged:
            maybe_drop_sft(current_sft, True)
        if arm == "sft":
            current_sft = step

        out_json = full_s_json(arm, step) if side == "S" else full_x_json(arm, step, str(probe))
        if out_json.exists() and out_json.stat().st_size > 0:
            print(f"[Skip core] {arm} {step_label(step)} {side} {probe or 'math_cot_probe'}", flush=True)
            continue

        model = model_path_for(arm, step)
        if not (model / "config.json").exists():
            raise FileNotFoundError(f"model missing: {model}")

        save = GS / "outputs" / arm / step_label(step) / "spectra" / ("S" if side == "S" else str(probe))
        cfg = common_cfg(model, save, args.n_samples, args.seqlen, target_layer=None, uv=False)
        if side == "S":
            cfg.update(
                {
                    "mode": "s_only_svd",
                    "s_jsonl_path": str(S_ROOT),
                    "s_jsonl_file": "gamma_s.jsonl",
                }
            )
            cfg_path = GS / "configs" / arm / f"{step_label(step)}__core_full36__S.json"
        else:
            cfg.update({"mode": "x_only_svd", "x_jsonl_path": str(probe_path)})
            cfg_path = GS / "configs" / arm / f"{step_label(step)}__core_full36__{probe}.json"
        write_json(cfg_path, cfg)
        run_slice(cfg_path)

    if args.drop_sft_merged and current_sft is not None:
        maybe_drop_sft(current_sft, True)


def load_model_for_custom(model_path: Path, seqlen: int):
    from utils.model_utils import get_model_from_huggingface

    model, tokenizer = get_model_from_huggingface(
        model_id=str(model_path),
        torch_dtype="float16",
        trust_remote_code=True,
        cache_dir=None,
    )
    model = model.eval()
    model.seqlen = int(seqlen)
    return model, tokenizer


def save_sigma_json(path: Path, sigma_dict: dict[str, Any]) -> None:
    write_json(path, sigma_dict)
    print(f"[Save] {path}", flush=True)


def top_rank_uv(uv_dict: dict[str, Any], rank: int) -> dict[str, Any]:
    trimmed = {}
    for layer_key, layer_data in uv_dict.items():
        trimmed[layer_key] = {}
        for module, item in layer_data.items():
            trimmed[layer_key][module] = {
                "U": item["U"][:, :rank].contiguous(),
                "S": item["S"][:rank].contiguous(),
                "VT": item["VT"][:rank, :].contiguous(),
            }
    return trimmed


def custom_env_offline() -> None:
    os.environ.setdefault("TMPDIR", "/root/autodl-tmp/pip-tmp")
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def landmark_output_available(arm: str, step: int, layer: int, probe: str) -> bool:
    if probe in FULL_X_PROBES and full_x_json(arm, step, probe).exists():
        return True
    path = landmark_x_json(arm, step, layer, probe)
    return path.exists() and path.stat().st_size > 0


def run_landmark(args: argparse.Namespace) -> None:
    custom_env_offline()
    import torch
    from utils.data_utils import get_token_data_from_jsonl
    from utils.profiling_utils import profile_svdllm_single_layer_group, whitening

    probe_names = parse_str_list(args.probes, LANDMARK_X_PROBES)
    probes = select_probes(probe_names)
    layers = args.uv_layers
    print(
        f"[Plan] landmark X jobs={len(args.arms) * len(args.steps) * len(layers)} "
        f"model-layer groups; probes={list(probes)} layers={layers}",
        flush=True,
    )
    if args.dry_run:
        return

    current_sft: int | None = None
    for arm in args.arms:
        for step in args.steps:
            if arm == "sft" and current_sft not in (None, step) and args.drop_sft_merged:
                maybe_drop_sft(current_sft, True)
            if arm == "sft":
                current_sft = step

            model_path = model_path_for(arm, step)
            if not (model_path / "config.json").exists():
                raise FileNotFoundError(f"model missing: {model_path}")

            pending_by_layer = {}
            for layer in layers:
                pending = {
                    probe: path
                    for probe, path in probes.items()
                    if not landmark_output_available(arm, step, layer, probe)
                }
                if pending:
                    pending_by_layer[layer] = pending
            if not pending_by_layer:
                print(f"[Skip landmark] {arm} {step_label(step)} all requested cells exist", flush=True)
                continue

            print(f"[Model] landmark X {arm} {step_label(step)} -> {model_path}", flush=True)
            model, tokenizer = load_model_for_custom(model_path, args.seqlen)
            try:
                for layer, pending in pending_by_layer.items():
                    loaders = {}
                    for probe, path in pending.items():
                        loaders[probe] = get_token_data_from_jsonl(
                            jsonl_path=str(path),
                            tokenizer=tokenizer,
                            nsamples=args.n_samples,
                            seqlen=args.seqlen,
                            seed=SEED,
                            batch_size=1,
                            cache_file=str(cache_path("x", probe, args.n_samples, args.seqlen)),
                            mode="x",
                        )
                    print(
                        f"[Landmark X] {arm} {step_label(step)} L{layer} probes={list(loaders)}",
                        flush=True,
                    )
                    profiling = profile_svdllm_single_layer_group(
                        model_name=str(model_path),
                        model=model,
                        calib_loaders_by_task=loaders,
                        dev="cuda",
                        target_layer=layer,
                        layer_gpu_chunk_size=args.layer_gpu_chunk_size,
                        singular_floor=0.0,
                        activation_cache_device=args.activation_cache_device,
                        cholesky_jitter=0.00001,
                    )
                    for probe, profile in profiling.items():
                        sigma, _uv = whitening(
                            model_name=str(model_path),
                            model=model,
                            profiling_mat=profile,
                            dev="cuda",
                            uv_dtype="float32",
                            return_uv=False,
                        )
                        save_sigma_json(landmark_x_json(arm, step, layer, probe), sigma)
                        del _uv
                    del profiling, loaders
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            finally:
                del model, tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if args.drop_sft_merged and current_sft is not None:
        maybe_drop_sft(current_sft, True)


def run_uv(args: argparse.Namespace) -> None:
    custom_env_offline()
    import torch
    from utils.data_utils import get_token_data_from_jsonl
    from utils.profiling_utils import profile_svdllm_single_layer_group, whitening

    layers = args.uv_layers
    print(
        f"[Plan] UV S-side model-layer groups={len(args.arms) * len(args.steps) * len(layers)} "
        f"layers={layers} top-{args.uv_rank}",
        flush=True,
    )
    if args.dry_run:
        return

    current_sft: int | None = None
    for arm in args.arms:
        for step in args.steps:
            if arm == "sft" and current_sft not in (None, step) and args.drop_sft_merged:
                maybe_drop_sft(current_sft, True)
            if arm == "sft":
                current_sft = step

            model_path = model_path_for(arm, step)
            pending_layers = [
                layer
                for layer in layers
                if not (uv_s_json(arm, step, layer).exists() and uv_s_pt(arm, step, layer).exists())
            ]
            if not pending_layers:
                print(f"[Skip UV] {arm} {step_label(step)} all requested layers exist", flush=True)
                continue

            print(f"[Model] UV S {arm} {step_label(step)} -> {model_path}", flush=True)
            model, tokenizer = load_model_for_custom(model_path, args.seqlen)
            try:
                loader = get_token_data_from_jsonl(
                    jsonl_path=str(S_ROOT / "math_cot_probe/gamma_s.jsonl"),
                    tokenizer=tokenizer,
                    nsamples=args.n_samples,
                    seqlen=args.seqlen,
                    seed=SEED,
                    batch_size=1,
                    cache_file=str(cache_path("s", "math_cot_probe", args.n_samples, args.seqlen)),
                    mode="s",
                )
                for layer in pending_layers:
                    print(f"[UV S] {arm} {step_label(step)} L{layer}", flush=True)
                    profiling = profile_svdllm_single_layer_group(
                        model_name=str(model_path),
                        model=model,
                        calib_loaders_by_task={"math_cot_probe": loader},
                        dev="cuda",
                        target_layer=layer,
                        layer_gpu_chunk_size=args.layer_gpu_chunk_size,
                        singular_floor=0.0,
                        activation_cache_device=args.activation_cache_device,
                        cholesky_jitter=0.00001,
                    )
                    sigma, uv = whitening(
                        model_name=str(model_path),
                        model=model,
                        profiling_mat=profiling["math_cot_probe"],
                        dev="cuda",
                        uv_dtype="float32",
                    )
                    save_sigma_json(uv_s_json(arm, step, layer), sigma)
                    ensure_dir(uv_s_pt(arm, step, layer).parent)
                    torch.save(top_rank_uv(uv, args.uv_rank), uv_s_pt(arm, step, layer))
                    print(f"[Save] {uv_s_pt(arm, step, layer)}", flush=True)
                    del profiling, sigma, uv
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                del loader
            finally:
                del model, tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if args.drop_sft_merged and current_sft is not None:
        maybe_drop_sft(current_sft, True)


def layer_dict(path: Path, layer: int) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    data = read_json(path)
    return data.get(f"layer_{layer}", {})


def x_layer_dict(arm: str, step: int, layer: int, probe: str) -> tuple[dict[str, list[float]], str]:
    full = full_x_json(arm, step, probe)
    if full.exists():
        vals = layer_dict(full, layer)
        if vals:
            return vals, str(full)
    landmark = landmark_x_json(arm, step, layer, probe)
    vals = layer_dict(landmark, layer)
    return vals, str(landmark) if vals else ""


def summarize_t5(args: argparse.Namespace) -> None:
    probe_names = parse_str_list(args.probes, LANDMARK_X_PROBES)
    probes = select_probes(probe_names)
    required_pairs = [("X_math", ALL_LAYERS)]
    for probe in probes:
        if probe != "X_math":
            required_pairs.append((probe, args.uv_layers))

    s_sigma: dict[tuple[str, int, int, str], list[float]] = {}
    base_s: dict[tuple[str, int, str], list[float]] = {}
    missing: list[dict[str, Any]] = []

    for arm in args.arms:
        for step in args.steps:
            s_path = full_s_json(arm, step)
            if not s_path.exists():
                missing.append({"arm": arm, "step": step, "side": "S", "probe": "math_cot_probe", "path": str(s_path)})
                continue
            for layer in ALL_LAYERS:
                for module, sigma in layer_dict(s_path, layer).items():
                    s_sigma[(arm, step, layer, module)] = sigma
                    if step == 0:
                        base_s[(arm, layer, module)] = sigma

    rows: list[dict[str, Any]] = []
    for arm in args.arms:
        for step in args.steps:
            for probe, layers in required_pairs:
                if probe not in probes:
                    continue
                for layer in layers:
                    x_vals, x_path = x_layer_dict(arm, step, layer, probe)
                    if not x_vals:
                        missing.append({"arm": arm, "step": step, "side": "X", "probe": probe, "layer": layer, "path": x_path})
                    for module in MODULES:
                        sigma_s = s_sigma.get((arm, step, layer, module))
                        if sigma_s is None:
                            continue
                        sigma_x = x_vals.get(module)
                        er = effective_rank(sigma_s)
                        gap = spectral_gap(sigma_s, 1)
                        drift = log_spectrum_drift(sigma_s, base_s.get((arm, layer, module), sigma_s))
                        rows.append(
                            {
                                "arm": arm,
                                "step": step,
                                "layer": layer,
                                "module": module,
                                "probe": probe,
                                "resolution": "full36_core" if probe == "X_math" else "landmark3",
                                "effective_rank": f"{er:.8f}",
                                "spectral_gap": "" if gap is None else f"{gap:.8f}",
                                "drift_from_base": "" if drift is None else f"{drift:.8f}",
                                "xs_log_spectrum_gap": ""
                                if sigma_x is None
                                else f"{xs_log_spectrum_gap(sigma_x, sigma_s):.8f}",
                                "s_json": str(full_s_json(arm, step)),
                                "x_json": x_path,
                            }
                        )

    write_csv(MINI / "T5_full_layer_profile.csv", rows)
    write_csv(MINI / "cycle09_r2_missing_getslice.csv", missing)
    write_manifest(args, missing)
    print(f"[T5] rows={len(rows)} missing={len(missing)}", flush=True)


def bootstrap_ci(values: list[float], draws: int) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(42)
    means = [float(arr[rng.integers(0, arr.size, arr.size)].mean()) for _ in range(draws)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def summarize_t6(draws: int) -> None:
    t5 = MINI / "T5_full_layer_profile.csv"
    if not t5.exists() or t5.stat().st_size == 0:
        raise FileNotFoundError(f"run --summarize with T5 inputs first: {t5}")

    by_cell: dict[tuple[int, str, int, str], dict[str, float]] = defaultdict(dict)
    with open(t5, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            layer = int(row["layer"])
            if layer not in LANDMARK_LAYERS:
                continue
            key = (layer, row["probe"], int(row["step"]), row["module"])
            by_cell[key][row["arm"]] = float(row["effective_rank"])

    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for (layer, probe, _step, _module), arms in by_cell.items():
        if "opd" in arms and "sft" in arms:
            grouped[(layer, probe)].append(arms["opd"] - arms["sft"])

    rows = []
    for (layer, probe), diffs in sorted(grouped.items()):
        mean, lo, hi = bootstrap_ci(diffs, draws)
        rows.append(
            {
                "layer": layer,
                "probe": probe,
                "unit": "aligned_step_module_pairs",
                "n_pairs": len(diffs),
                "draws": draws,
                "mean_opd_minus_sft_er": "" if math.isnan(mean) else f"{mean:.8f}",
                "ci95_lo": "" if math.isnan(lo) else f"{lo:.8f}",
                "ci95_hi": "" if math.isnan(hi) else f"{hi:.8f}",
                "excludes_0": "" if math.isnan(lo) else ("yes" if lo > 0 or hi < 0 else "no"),
            }
        )
    write_csv(MINI / "T6_er_error_bands.csv", rows)
    print(f"[T6] rows={len(rows)}", flush=True)


def top_basis(path: Path, layer: int, module: str, rank: int):
    import torch

    uv = torch.load(path, map_location="cpu")
    item = uv[f"layer_{layer}"][module]
    return item["U"][:, :rank].float(), item["VT"][:rank, :].T.float()


def overlap(path_a: Path, path_b: Path, layer: int, module: str, rank: int) -> tuple[float, float]:
    import torch

    ua, va = top_basis(path_a, layer, module, rank)
    ub, vb = top_basis(path_b, layer, module, rank)
    return (
        float(torch.linalg.svdvals(ua.T @ ub).pow(2).mean()),
        float(torch.linalg.svdvals(va.T @ vb).pow(2).mean()),
    )


def summarize_t7(args: argparse.Namespace, ranks: list[int]) -> None:
    rows = []
    for arm in args.arms:
        for layer in args.uv_layers:
            base = uv_s_pt(arm, 0, layer)
            for step in args.steps:
                cur = uv_s_pt(arm, step, layer)
                if step != 0 and base.exists() and cur.exists():
                    for module in MODULES:
                        for rank in ranks:
                            try:
                                theta_u, theta_v = overlap(base, cur, layer, module, rank)
                            except Exception:
                                theta_u, theta_v = math.nan, math.nan
                            rows.append(
                                {
                                    "arm": arm,
                                    "comparison": "vs_base",
                                    "step_a": 0,
                                    "step_b": step,
                                    "layer": layer,
                                    "module": module,
                                    "r": rank,
                                    "theta_u": "" if math.isnan(theta_u) else f"{theta_u:.8f}",
                                    "theta_v": "" if math.isnan(theta_v) else f"{theta_v:.8f}",
                                    "uv_a": str(base),
                                    "uv_b": str(cur),
                                }
                            )
            for prev, step in zip(args.steps, args.steps[1:]):
                path_a = uv_s_pt(arm, prev, layer)
                path_b = uv_s_pt(arm, step, layer)
                if not (path_a.exists() and path_b.exists()):
                    continue
                for module in MODULES:
                    for rank in ranks:
                        try:
                            theta_u, theta_v = overlap(path_a, path_b, layer, module, rank)
                        except Exception:
                            theta_u, theta_v = math.nan, math.nan
                        rows.append(
                            {
                                "arm": arm,
                                "comparison": "adjacent",
                                "step_a": prev,
                                "step_b": step,
                                "layer": layer,
                                "module": module,
                                "r": rank,
                                "theta_u": "" if math.isnan(theta_u) else f"{theta_u:.8f}",
                                "theta_v": "" if math.isnan(theta_v) else f"{theta_v:.8f}",
                                "uv_a": str(path_a),
                                "uv_b": str(path_b),
                            }
                        )
    write_csv(MINI / "T7_theta_r.csv", rows)
    print(f"[T7] rows={len(rows)}", flush=True)


def write_manifest(args: argparse.Namespace, missing: list[dict[str, Any]]) -> None:
    write_json(
        MINI / "cycle09_r2_probe_manifest.json",
        {
            "cycle": "cycle09_round2_compressed",
            "key_steps": args.steps,
            "full36": {"S": True, "X": parse_str_list(args.full_probes, FULL_X_PROBES)},
            "landmark_layers": args.uv_layers,
            "landmark_x_probes": parse_str_list(args.probes, LANDMARK_X_PROBES),
            "uv": {"side": "S", "layers": args.uv_layers, "rank": args.uv_rank},
            "n_samples": args.n_samples,
            "seqlen": args.seqlen,
            "x_probes": {k: str(v) for k, v in sorted(probe_paths().items())},
            "s_probe": str(S_ROOT / "math_cot_probe/gamma_s.jsonl"),
            "x_teacher": "deferred",
            "run_root": str(RUN),
            "getslice_root": str(GS),
            "mini_root": str(MINI),
            "missing_getslice_cells": len(missing),
        },
    )


def status(args: argparse.Namespace) -> None:
    probes = probe_paths()
    print("[Status] roots:")
    print(f"  RUN={RUN}")
    print(f"  GS={GS}")
    print(f"  MINI={MINI}")
    print("[Status] probes:")
    for name, path in sorted(probes.items()):
        print(f"  {name}: n={count_jsonl(path)} path={path}")
    print(f"  S: n={count_jsonl(S_ROOT / 'math_cot_probe/gamma_s.jsonl')} path={S_ROOT / 'math_cot_probe/gamma_s.jsonl'}")

    full_expected = len(args.arms) * len(args.steps) * (1 + len(parse_str_list(args.full_probes, FULL_X_PROBES)))
    full_done = 0
    for arm in args.arms:
        for step in args.steps:
            full_done += int(full_s_json(arm, step).exists())
            for probe in parse_str_list(args.full_probes, FULL_X_PROBES):
                full_done += int(full_x_json(arm, step, probe).exists())

    landmark_names = parse_str_list(args.probes, LANDMARK_X_PROBES)
    landmark_expected = len(args.arms) * len(args.steps) * len(args.uv_layers) * len(landmark_names)
    landmark_done = 0
    for arm in args.arms:
        for step in args.steps:
            for layer in args.uv_layers:
                for probe in landmark_names:
                    landmark_done += int(landmark_output_available(arm, step, layer, probe))

    uv_expected = len(args.arms) * len(args.steps) * len(args.uv_layers)
    uv_done = sum(
        int(uv_s_json(arm, step, layer).exists() and uv_s_pt(arm, step, layer).exists())
        for arm in args.arms
        for step in args.steps
        for layer in args.uv_layers
    )
    print(f"[Status] core full36 jobs={full_done}/{full_expected}")
    print(f"[Status] landmark X cells={landmark_done}/{landmark_expected}")
    print(f"[Status] UV S cells={uv_done}/{uv_expected}")
    for name in ["T5_full_layer_profile.csv", "T6_er_error_bands.csv", "T7_theta_r.csv"]:
        print(f"[Status] {name}: {(MINI / name).exists()}")


def check_inputs(args: argparse.Namespace) -> None:
    if not (BASE_MODEL / "config.json").exists():
        raise FileNotFoundError(f"base model missing: {BASE_MODEL}")
    validate_s_jsonl(S_ROOT / "math_cot_probe/gamma_s.jsonl")
    for name, path in select_probes(parse_str_list(args.probes, LANDMARK_X_PROBES)).items():
        validate_x_jsonl(path)
        print(f"[Input OK] {name}: n={count_jsonl(path)} {path}", flush=True)
    for arm in args.arms:
        for step in args.steps:
            if step == 0:
                continue
            if arm == "opd" and not ((OPD_MERGED / step_label(step)) / "config.json").exists():
                raise FileNotFoundError(f"OPD merged missing: {OPD_MERGED / step_label(step)}")
            if arm == "sft" and not ((SFT_CKPT / step_label(step)) / "adapter_config.json").exists():
                raise FileNotFoundError(f"SFT adapter missing: {SFT_CKPT / step_label(step)}")
    print("[Input OK] models/adapters for requested arms and steps are present", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--run-core", action="store_true")
    parser.add_argument("--run-landmark", action="store_true")
    parser.add_argument("--run-uv", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--mini-root", type=Path, default=DEFAULT_MINI)
    parser.add_argument("--arms", default="opd,sft")
    parser.add_argument("--steps", default=",".join(map(str, KEY_STEPS)))
    parser.add_argument("--full-probes", default=",".join(FULL_X_PROBES))
    parser.add_argument("--probes", default=",".join(LANDMARK_X_PROBES))
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N)
    parser.add_argument("--seqlen", type=int, default=DEFAULT_SEQLEN)
    parser.add_argument("--drop-sft-merged", action="store_true")
    parser.add_argument("--uv-layers", default=",".join(map(str, LANDMARK_LAYERS)))
    parser.add_argument("--uv-rank", type=int, default=128)
    parser.add_argument("--t6-draws", type=int, default=1024)
    parser.add_argument("--theta-ranks", default="8,16,32,64,128")
    parser.add_argument("--layer-gpu-chunk-size", type=int, default=12)
    parser.add_argument("--activation-cache-device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    configure_roots(args.run_root, args.mini_root)
    args.arms = parse_str_list(args.arms, ["opd", "sft"])
    args.steps = parse_int_list(args.steps, KEY_STEPS)
    args.uv_layers = parse_int_list(args.uv_layers, LANDMARK_LAYERS)
    ensure_dir(GS)
    ensure_dir(MINI)

    if args.status:
        status(args)
    if args.check_inputs:
        check_inputs(args)
    if args.run_core:
        run_core(args)
    if args.run_landmark:
        run_landmark(args)
    if args.run_uv:
        run_uv(args)
    if args.summarize:
        summarize_t5(args)
        summarize_t6(args.t6_draws)
        summarize_t7(args, parse_int_list(args.theta_ranks, [8, 16, 32, 64, 128]))
    if not any(
        [args.status, args.check_inputs, args.run_core, args.run_landmark, args.run_uv, args.summarize]
    ):
        parser.print_help()


if __name__ == "__main__":
    main()
