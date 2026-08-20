#!/usr/bin/env python3
"""Cycle 09 Round 3 GetSlice v3 collection (R3-4).

This runner keeps the Round 2 legacy S/X taxonomy separate from R3-5's
new S/X/H taxonomy. It streams all tasks while one model is loaded, writes
full-layer aggregate spectra, persists sample factors at landmark layers, and
computes raw residual-stream ER plus centered/uncentered cosine anisotropy.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/root/LLM-output-density")
GETSLICE = REPO / "GetSlice"
SIDE = REPO / "experiments/opd_sft_h1"
for item in (REPO, GETSLICE, SIDE):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import cycle09_r2_unified_probe as r2  # noqa: E402
from opd_sft_h1.geometry_metrics import effective_rank  # noqa: E402

DEFAULT_RUN = Path("/root/autodl-tmp/cycle09_r3")
DEFAULT_MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
ARMS = ("opd", "sft")
STEPS = (0, 5, 10, 20, 40, 160, 624)
ALL_LAYERS = tuple(range(36))
LANDMARK_LAYERS = (9, 18, 27)
X_PROBES = ("X_math", "X_ood_knowledge", "X_general", "X_math_hard", "X_bos")
MODULES = tuple(r2.MODULES)
MODULE_TO_GROUP = {
    "self_attn.q_proj": "attn_qkv_input",
    "self_attn.k_proj": "attn_qkv_input",
    "self_attn.v_proj": "attn_qkv_input",
    "self_attn.o_proj": "attn_o_input",
    "mlp.gate_proj": "mlp_gate_up_input",
    "mlp.up_proj": "mlp_gate_up_input",
    "mlp.down_proj": "mlp_down_input",
}
GROUP_TO_MODULES = {
    "attn_qkv_input": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "attn_o_input": ["self_attn.o_proj"],
    "mlp_gate_up_input": ["mlp.gate_proj", "mlp.up_proj"],
    "mlp_down_input": ["mlp.down_proj"],
}
SEED = 3


def step_label(step: int) -> str:
    return f"step_{int(step):03d}"


def parse_ints(value: str, default: tuple[int, ...]) -> list[int]:
    if not value:
        return list(default)
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_names(value: str, default: tuple[str, ...]) -> list[str]:
    if not value:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def spectrum_path(run_root: Path, arm: str, step: int, task: str) -> Path:
    return run_root / "spectra" / arm / step_label(step) / f"{task}.json"


def raw_path(run_root: Path, arm: str, step: int, task: str) -> Path:
    return run_root / "raw_geometry" / arm / step_label(step) / f"{task}.json"


def factor_path(
    run_root: Path,
    arm: str,
    step: int,
    layer: int,
    task: str,
    sample_idx: int,
) -> Path:
    return (
        run_root
        / "factors"
        / arm
        / step_label(step)
        / f"layer_{int(layer)}"
        / task
        / f"sample_{int(sample_idx):03d}.pt"
    )


def factors_complete(
    run_root: Path,
    arm: str,
    step: int,
    layer: int,
    task: str,
    n_samples: int,
) -> bool:
    return all(
        factor_path(run_root, arm, step, layer, task, sample_idx).exists()
        for sample_idx in range(n_samples)
    )


def spectrum_complete(path: Path, layers: list[int]) -> bool:
    data = read_json(path)
    return all(f"layer_{int(layer)}" in data for layer in layers)


def task_specs(x_names: list[str]) -> dict[str, tuple[Path, str]]:
    available = r2.probe_paths()
    missing = [name for name in x_names if name not in available]
    if missing:
        raise FileNotFoundError(f"Missing required legacy X probes: {missing}")
    s_path = r2.S_ROOT / "math_cot_probe" / "gamma_s.jsonl"
    if not s_path.exists():
        raise FileNotFoundError(f"Missing legacy S probe: {s_path}")
    specs: dict[str, tuple[Path, str]] = {"S": (s_path, "s")}
    specs.update({name: (available[name], "x") for name in x_names})
    return specs


def make_factor_callback(
    run_root: Path,
    arm: str,
    step: int,
    layer: int,
    task: str,
):
    import torch

    def callback(sample_idx: int, factors: dict[str, Any]) -> None:
        target = factor_path(run_root, arm, step, layer, task, sample_idx)
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "arm": arm,
            "step": int(step),
            "layer": int(layer),
            "task": task,
            "sample_idx": int(sample_idx),
            "factor_dtype": str(next(iter(factors.values())).dtype),
            "input_group_modules": GROUP_TO_MODULES,
            "factors": factors,
        }
        tmp = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, target)

    return callback


def build_loaders(tokenizer, specs: dict[str, tuple[Path, str]], args: argparse.Namespace):
    from utils.data_utils import get_token_data_from_jsonl

    loaders = {}
    requested_windows = args.n_samples + args.loader_headroom
    for task, (path, mode) in specs.items():
        cache = (
            args.run_root
            / "cache"
            / (
                f"{task}__n{args.n_samples}__attempts{requested_windows}"
                f"__seq{args.seqlen}__seed{SEED}__{mode}.pt"
            )
        )
        candidate_windows = get_token_data_from_jsonl(
            jsonl_path=str(path),
            tokenizer=tokenizer,
            nsamples=requested_windows,
            seqlen=args.seqlen,
            seed=SEED,
            batch_size=1,
            cache_file=str(cache),
            mode=mode,
        )
        if len(candidate_windows) < args.n_samples:
            raise RuntimeError(
                f"{task} produced {len(candidate_windows)} valid windows after "
                f"{requested_windows} attempts; expected at least {args.n_samples}"
            )
        # The legacy loader may skip an under-length random character window.
        # Keep the first n valid windows under the fixed seed for a true n=32 design.
        loaders[task] = candidate_windows[:args.n_samples]
    return loaders


def make_spectrum_handler(args: argparse.Namespace, arm: str, step: int):
    from utils.profiling_utils import whitening

    def handler(task: str, layer: int, profile: dict[int, dict[str, Any]]) -> None:
        target = spectrum_path(args.run_root, arm, step, task)
        existing = read_json(target)
        key = f"layer_{int(layer)}"
        if key in existing:
            return
        sigma, _ = whitening(
            model_name=str(r2.model_path_for(arm, step)),
            model=handler.model,
            profiling_mat=profile,
            dev=args.device,
            uv_dtype="float32",
            return_uv=False,
        )
        if key not in sigma:
            raise RuntimeError(f"Whitening did not return {key} for {arm}/{step}/{task}")
        existing[key] = sigma[key]
        write_json_atomic(target, existing)

    handler.model = None
    return handler


def pairwise_cosine_from_sum(direction_sum, count: int) -> float | None:
    if count < 2:
        return None
    value = (float(direction_sum.dot(direction_sum)) - float(count)) / (count * (count - 1))
    return value


def compute_raw_geometry(model, loaders, layers: list[int], device: str) -> dict[str, Any]:
    import torch

    dev = torch.device(device)
    model.to(dev).eval()
    model.config.use_cache = False
    hidden_size = int(model.config.hidden_size)
    stats = {
        layer: {
            "n": 0,
            "sum": torch.zeros(hidden_size, dtype=torch.float64),
            "gram": torch.zeros((hidden_size, hidden_size), dtype=torch.float64),
            "uncentered_direction_sum": torch.zeros(hidden_size, dtype=torch.float64),
        }
        for layer in layers
    }

    with torch.no_grad():
        for batch in loaders:
            batch = {key: value.to(dev) for key, value in batch.items()}
            output = model(**batch, output_hidden_states=True, use_cache=False)
            for layer in layers:
                h = output.hidden_states[layer + 1].detach().float().reshape(-1, hidden_size).cpu().double()
                normalized = h / torch.linalg.vector_norm(h, dim=1, keepdim=True).clamp_min(1e-12)
                stats[layer]["n"] += int(h.shape[0])
                stats[layer]["sum"] += h.sum(dim=0)
                stats[layer]["gram"] += h.T @ h
                stats[layer]["uncentered_direction_sum"] += normalized.sum(dim=0)
            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    means = {layer: stats[layer]["sum"] / max(stats[layer]["n"], 1) for layer in layers}
    centered_direction_sums = {
        layer: torch.zeros(hidden_size, dtype=torch.float64) for layer in layers
    }
    with torch.no_grad():
        for batch in loaders:
            batch = {key: value.to(dev) for key, value in batch.items()}
            output = model(**batch, output_hidden_states=True, use_cache=False)
            for layer in layers:
                h = output.hidden_states[layer + 1].detach().float().reshape(-1, hidden_size).cpu().double()
                centered = h - means[layer]
                normalized = centered / torch.linalg.vector_norm(
                    centered, dim=1, keepdim=True
                ).clamp_min(1e-12)
                centered_direction_sums[layer] += normalized.sum(dim=0)
            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result_layers = {}
    eig_device = dev if dev.type == "cuda" else torch.device("cpu")
    for layer in layers:
        count = max(int(stats[layer]["n"]), 1)
        centered_cross = stats[layer]["gram"] - torch.outer(stats[layer]["sum"], stats[layer]["sum"]) / count
        covariance = centered_cross / max(count - 1, 1)
        eigvals = torch.linalg.eigvalsh(covariance.to(eig_device)).detach().cpu().clamp_min(0).numpy()[::-1]
        raw_er = effective_rank(eigvals)
        result_layers[str(layer)] = {
            "token_count": count,
            "raw_effective_rank": float(raw_er),
            "raw_normalized_effective_rank": float(raw_er / hidden_size),
            "uncentered_mean_pairwise_cosine": pairwise_cosine_from_sum(
                stats[layer]["uncentered_direction_sum"], count
            ),
            "centered_mean_pairwise_cosine": pairwise_cosine_from_sum(
                centered_direction_sums[layer], count
            ),
            "covariance_normalization": "centered_sample_covariance_then_divide_by_d_for_construct",
        }
        del covariance, centered_cross
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result_layers


def collect_one_model(
    args: argparse.Namespace,
    arm: str,
    step: int,
    specs: dict[str, tuple[Path, str]],
) -> None:
    from utils.profiling_utils import (
        profile_svdllm_all_layers_group,
        profile_svdllm_single_layer_group,
    )

    profile_needed = any(
        not spectrum_complete(spectrum_path(args.run_root, arm, step, task), args.profile_layers)
        for task in specs
    )
    profile_needed = profile_needed or any(
        not factors_complete(args.run_root, arm, step, layer, task, args.n_samples)
        for layer in args.factor_layers
        for task in specs
    )
    raw_needed = (not args.skip_raw) and any(
        not raw_path(args.run_root, arm, step, task).exists() for task in specs
    )
    if not profile_needed and not raw_needed:
        print(f"[Skip] {arm}/{step_label(step)} complete", flush=True)
        return

    model_path = r2.model_path_for(arm, step)
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    print(f"[Model] {arm}/{step_label(step)} -> {model_path}", flush=True)
    model, tokenizer = r2.load_model_for_custom(model_path, args.seqlen)
    try:
        loaders = build_loaders(tokenizer, specs, args)
        if profile_needed:
            callbacks = {
                task: {
                    layer: make_factor_callback(args.run_root, arm, step, layer, task)
                    for layer in args.factor_layers
                    if not factors_complete(args.run_root, arm, step, layer, task, args.n_samples)
                }
                for task in specs
            }
            handler = make_spectrum_handler(args, arm, step)
            handler.model = model
            if args.profile_layers == list(ALL_LAYERS):
                profile_svdllm_all_layers_group(
                    model_name=str(model_path),
                    model=model,
                    calib_loaders_by_task=loaders,
                    dev=args.device,
                    singular_floor=0.0,
                    activation_cache_device=args.activation_cache_device,
                    cholesky_jitter=1e-5,
                    after_profile=handler,
                    sample_factor_callbacks_by_task_and_layer=callbacks,
                )
            else:
                for layer in args.profile_layers:
                    profiles = profile_svdllm_single_layer_group(
                        model_name=str(model_path),
                        model=model,
                        calib_loaders_by_task=loaders,
                        dev=args.device,
                        target_layer=layer,
                        layer_gpu_chunk_size=args.layer_gpu_chunk_size,
                        singular_floor=0.0,
                        activation_cache_device=args.activation_cache_device,
                        cholesky_jitter=1e-5,
                        sample_factor_callbacks_by_task={
                            task_name: callbacks.get(task_name, {}).get(layer)
                            for task_name in loaders
                        },
                    )
                    for task, profile in profiles.items():
                        handler(task, layer, profile)
                    del profiles
                    gc.collect()
        if raw_needed:
            for task, loader in loaders.items():
                target = raw_path(args.run_root, arm, step, task)
                if target.exists():
                    continue
                print(f"[Raw] {arm}/{step_label(step)}/{task}", flush=True)
                payload = {
                    "schema_version": 1,
                    "arm": arm,
                    "step": int(step),
                    "task": task,
                    "n_samples": args.n_samples,
                    "seqlen": args.seqlen,
                    "layers": compute_raw_geometry(
                        model, loader, args.factor_layers, args.device
                    ),
                }
                write_json_atomic(target, payload)
    finally:
        del model, tokenizer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def write_manifest(args: argparse.Namespace, specs: dict[str, tuple[Path, str]]) -> None:
    manifest = {
        "schema_version": 1,
        "cycle": "cycle09_round3_r3_4",
        "taxonomy": "legacy_round2_S_X_only; distinct from R3-5 S_X_H",
        "arms": args.arms,
        "steps": args.steps,
        "aggregate_spectrum_layers": args.profile_layers,
        "factor_layers": args.factor_layers,
        "tasks": {task: {"path": str(path), "mode": mode} for task, (path, mode) in specs.items()},
        "n_samples": args.n_samples,
        "seqlen": args.seqlen,
        "seed": SEED,
        "loader_valid_window_policy": {
            "attempts_per_task": args.n_samples + args.loader_headroom,
            "kept_per_task": args.n_samples,
            "selection": "first n valid deterministic windows after strict tokenization",
        },
        "factor_storage": {
            "unit": "one sample/window per .pt",
            "dtype": "native model activation dtype; reconstructed in fp32",
            "input_groups": GROUP_TO_MODULES,
            "implementation_correction": "four streams, not the earlier QA's three: qkv, o, gate_up, down",
        },
        "raw_anisotropy": {
            "statistic": "mean pairwise cosine over residual-stream tokens",
            "variants": ["uncentered", "centered"],
            "raw_er": "entropy ER of centered covariance eigenvalues, plus /d normalized value",
        },
    }
    write_json_atomic(args.mini_root / "R3_getslice_manifest.json", manifest)


def summarize(args: argparse.Namespace, specs: dict[str, tuple[Path, str]]) -> None:
    x_rows: list[dict[str, Any]] = []
    anisotropy_rows: list[dict[str, Any]] = []
    for arm in args.arms:
        for step in args.steps:
            for task in specs:
                spectra = read_json(spectrum_path(args.run_root, arm, step, task))
                for layer in args.factor_layers:
                    for module, values in spectra.get(f"layer_{layer}", {}).items():
                        if task != "S":
                            x_rows.append(
                                {
                                    "arm": arm,
                                    "step": step,
                                    "layer": layer,
                                    "probe": task,
                                    "module": module,
                                    "x_whitened_effective_rank": f"{effective_rank(values):.8f}",
                                    "n_singular_values": len(values),
                                    "spectrum_path": str(spectrum_path(args.run_root, arm, step, task)),
                                    "taxonomy": "legacy_X",
                                }
                            )
                raw = read_json(raw_path(args.run_root, arm, step, task))
                for layer, values in raw.get("layers", {}).items():
                    anisotropy_rows.append(
                        {
                            "arm": arm,
                            "step": step,
                            "task": task,
                            "layer": layer,
                            "raw_effective_rank": values.get("raw_effective_rank", ""),
                            "raw_normalized_effective_rank": values.get("raw_normalized_effective_rank", ""),
                            "uncentered_mean_pairwise_cosine": values.get("uncentered_mean_pairwise_cosine", ""),
                            "centered_mean_pairwise_cosine": values.get("centered_mean_pairwise_cosine", ""),
                            "token_count": values.get("token_count", ""),
                            "raw_path": str(raw_path(args.run_root, arm, step, task)),
                            "taxonomy": "legacy_S" if task == "S" else "legacy_X",
                        }
                    )
    write_csv(
        args.mini_root / "R3_xcond_whitened_er.csv",
        x_rows,
        [
            "arm", "step", "layer", "probe", "module", "x_whitened_effective_rank",
            "n_singular_values", "spectrum_path", "taxonomy",
        ],
    )
    write_csv(
        args.mini_root / "R3_anisotropy.csv",
        anisotropy_rows,
        [
            "arm", "step", "task", "layer", "raw_effective_rank",
            "raw_normalized_effective_rank", "uncentered_mean_pairwise_cosine",
            "centered_mean_pairwise_cosine", "token_count", "raw_path", "taxonomy",
        ],
    )
    print(
        f"[Summary] X ER rows={len(x_rows)} anisotropy rows={len(anisotropy_rows)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--mini-root", type=Path, default=DEFAULT_MINI)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--steps", default=",".join(map(str, STEPS)))
    parser.add_argument("--x-probes", default=",".join(X_PROBES))
    parser.add_argument("--profile-layers", default=",".join(map(str, ALL_LAYERS)))
    parser.add_argument("--factor-layers", default=",".join(map(str, LANDMARK_LAYERS)))
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--loader-headroom", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--activation-cache-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--layer-gpu-chunk-size", type=int, default=12)
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.run_root = args.run_root / "smoke_getslice"
        args.mini_root = args.mini_root / "smoke_getslice"
        args.collect = True
        args.summarize = True
        args.arms = "opd"
        args.steps = "0"
        args.x_probes = "X_math"
        args.profile_layers = "18"
        args.factor_layers = "18"
        args.n_samples = 2
        args.seqlen = 64
        args.activation_cache_device = "cpu"

    if args.all:
        args.collect = True
        args.summarize = True
    if not (args.collect or args.summarize):
        parser.print_help()
        return

    args.arms = parse_names(args.arms, ARMS)
    args.steps = parse_ints(args.steps, STEPS)
    args.x_probes = parse_names(args.x_probes, X_PROBES)
    args.profile_layers = parse_ints(args.profile_layers, ALL_LAYERS)
    args.factor_layers = parse_ints(args.factor_layers, LANDMARK_LAYERS)
    if not set(args.factor_layers).issubset(args.profile_layers):
        raise ValueError("--factor-layers must be a subset of --profile-layers")
    if args.n_samples <= 0 or args.seqlen <= 0 or args.loader_headroom < 0:
        raise ValueError("--n-samples and --seqlen must be positive; --loader-headroom must be nonnegative")

    args.run_root.mkdir(parents=True, exist_ok=True)
    args.mini_root.mkdir(parents=True, exist_ok=True)
    r2.configure_roots(args.run_root, args.mini_root)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    specs = task_specs(args.x_probes)
    write_manifest(args, specs)
    print(
        f"[Plan] arms={args.arms} steps={args.steps} tasks={list(specs)} "
        f"profile_layers={args.profile_layers} factor_layers={args.factor_layers} "
        f"n={args.n_samples} seq={args.seqlen}",
        flush=True,
    )
    if args.dry_run:
        return
    if args.collect:
        for arm in args.arms:
            for step in args.steps:
                collect_one_model(args, arm, step, specs)
    if args.summarize:
        summarize(args, specs)


if __name__ == "__main__":
    main()

