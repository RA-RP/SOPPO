#!/usr/bin/env python3
"""Exact probe-sample bootstrap for Cycle 09 R3-4 legacy S-side ER upticks.

Factors are captured by cycle09_r3_getslice.py. Each draw resamples the fixed
probe windows jointly across the three local trajectory checkpoints, rebuilds
pooled input Grams, and recomputes the same whitened spectra used for the
point estimate. Modules are fixed measurement dimensions, not bootstrap units.
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
LAYER = 18
MODULES = tuple(r2.MODULES)
GROUP_TO_MODULES = {
    "attn_qkv_input": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "attn_o_input": ["self_attn.o_proj"],
    "mlp_gate_up_input": ["mlp.gate_proj", "mlp.up_proj"],
    "mlp_down_input": ["mlp.down_proj"],
}
TRANSITIONS = {
    "opd": (0, 5, 10),
    "sft": (10, 20, 40),
}
SEED = 42


def step_label(step: int) -> str:
    return f"step_{int(step):03d}"


def factor_path(
    run_root: Path,
    arm: str,
    step: int,
    layer: int,
    task: str,
    sample_idx: int,
) -> Path:
    return (
        run_root / "factors" / arm / step_label(step) / f"layer_{layer}"
        / task / f"sample_{sample_idx:03d}.pt"
    )


def spectrum_path(run_root: Path, arm: str, step: int, task: str) -> Path:
    return run_root / "spectra" / arm / step_label(step) / f"{task}.json"


def parse_names(value: str, default: tuple[str, ...]) -> list[str]:
    if not value:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else [
        "arm", "layer", "module", "previous_step", "dip_step", "next_step",
        "n_probe_samples", "bootstrap_draws", "point_uptick_er",
        "bootstrap_mean_uptick_er", "ci95_lo", "ci95_hi", "excludes_zero",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_factors(
    run_root: Path,
    arm: str,
    step: int,
    layer: int,
    task: str,
    n_samples: int,
) -> list[dict[str, Any]]:
    import torch

    payloads = []
    for sample_idx in range(n_samples):
        path = factor_path(run_root, arm, step, layer, task, sample_idx)
        if not path.exists():
            raise FileNotFoundError(f"Missing factor: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        missing = set(GROUP_TO_MODULES).difference(payload.get("factors", {}))
        if missing:
            raise ValueError(f"{path} missing factor groups: {sorted(missing)}")
        payloads.append(payload)
    return payloads


def resampled_profile(
    payloads: list[dict[str, Any]],
    indices: np.ndarray,
    layer: int,
    device: str,
):
    import torch
    from utils.profiling_utils import _gram_to_svdllm_scaling_diag_matrix

    counts = np.bincount(indices, minlength=len(payloads))
    profile = {layer: {}}
    for group, modules in GROUP_TO_MODULES.items():
        gram = None
        for sample_idx, count in enumerate(counts):
            if count == 0:
                continue
            factor = payloads[sample_idx]["factors"][group].to(
                device=device, dtype=torch.float32
            )
            contribution = factor.T @ factor
            if gram is None:
                gram = contribution.mul(int(count))
            else:
                gram.add_(contribution, alpha=int(count))
            del factor, contribution
        if gram is None:
            raise RuntimeError("bootstrap draw selected no factors")
        scaling = _gram_to_svdllm_scaling_diag_matrix(
            gram,
            cholesky_jitter=1e-5,
            singular_floor=0.0,
        ).cpu()
        for module in modules:
            profile[layer][module] = scaling
        del gram
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return profile


def bootstrap_model(
    run_root: Path,
    arm: str,
    step: int,
    layer: int,
    task: str,
    indices_by_draw: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    import torch
    from utils.profiling_utils import whitening

    n_samples = int(indices_by_draw.shape[1])
    payloads = load_factors(run_root, arm, step, layer, task, n_samples)
    model_path = r2.model_path_for(arm, step)
    model, tokenizer = r2.load_model_for_custom(model_path, seqlen=512)
    outputs = {module: np.zeros(indices_by_draw.shape[0], dtype=np.float64) for module in MODULES}
    try:
        for draw_idx, indices in enumerate(indices_by_draw):
            profile = resampled_profile(payloads, indices, layer, device)
            sigma, _ = whitening(
                model_name=str(model_path),
                model=model,
                profiling_mat=profile,
                dev=device,
                uv_dtype="float32",
                return_uv=False,
            )
            values = sigma[f"layer_{layer}"]
            for module in MODULES:
                outputs[module][draw_idx] = effective_rank(values[module])
            del profile, sigma, values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (draw_idx + 1) % 8 == 0 or draw_idx + 1 == len(indices_by_draw):
                print(
                    f"[Bootstrap] {arm}/{step_label(step)} draw {draw_idx + 1}/{len(indices_by_draw)}",
                    flush=True,
                )
    finally:
        del model, tokenizer, payloads
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs


def point_ers(run_root: Path, arm: str, step: int, layer: int, task: str) -> dict[str, float]:
    path = spectrum_path(run_root, arm, step, task)
    if not path.exists():
        raise FileNotFoundError(f"Missing point spectrum: {path}")
    spectra = read_json(path).get(f"layer_{layer}", {})
    missing = set(MODULES).difference(spectra)
    if missing:
        raise ValueError(f"{path} missing modules: {sorted(missing)}")
    return {module: effective_rank(spectra[module]) for module in MODULES}


def confidence_interval(values: np.ndarray) -> tuple[float, float, float]:
    return (
        float(values.mean()),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def validate(args: argparse.Namespace) -> None:
    path = factor_path(args.run_root, args.validate_arm, args.validate_step, args.layer, "S", 0)
    payload = load_factors(
        args.run_root, args.validate_arm, args.validate_step, args.layer, "S", 1
    )[0]
    report = {
        "path": str(path),
        "groups": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in payload["factors"].items()
        },
        "expected_groups": sorted(GROUP_TO_MODULES),
    }
    write_json(args.mini_root / "R3_factor_validation.json", report)
    print(f"[Validate] wrote {args.mini_root / 'R3_factor_validation.json'}", flush=True)


def run_bootstrap(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    indices_by_draw = rng.integers(
        0, args.n_samples, size=(args.draws, args.n_samples), endpoint=False
    )
    rows: list[dict[str, Any]] = []
    raw_draw_paths = {}

    for arm in args.arms:
        previous, dip, following = TRANSITIONS[arm]
        draw_ers = {
            step: bootstrap_model(
                args.run_root, arm, step, args.layer, "S", indices_by_draw, args.device
            )
            for step in (previous, dip, following)
        }
        point = {
            step: point_ers(args.run_root, arm, step, args.layer, "S")
            for step in (previous, dip, following)
        }
        raw_draw_paths[arm] = {}
        module_upticks = {}
        for module in MODULES:
            draws = draw_ers[dip][module] - 0.5 * (
                draw_ers[previous][module] + draw_ers[following][module]
            )
            module_upticks[module] = draws
            mean, lo, hi = confidence_interval(draws)
            point_value = point[dip][module] - 0.5 * (
                point[previous][module] + point[following][module]
            )
            rows.append(
                {
                    "arm": arm,
                    "layer": args.layer,
                    "module": module,
                    "previous_step": previous,
                    "dip_step": dip,
                    "next_step": following,
                    "n_probe_samples": args.n_samples,
                    "bootstrap_draws": args.draws,
                    "seed": args.seed,
                    "point_uptick_er": f"{point_value:.8f}",
                    "bootstrap_mean_uptick_er": f"{mean:.8f}",
                    "ci95_lo": f"{lo:.8f}",
                    "ci95_hi": f"{hi:.8f}",
                    "excludes_zero": "yes" if lo > 0 or hi < 0 else "no",
                    "bootstrap_unit": "probe_window; modules fixed",
                }
            )
        aggregate = np.stack([module_upticks[module] for module in MODULES]).mean(axis=0)
        mean, lo, hi = confidence_interval(aggregate)
        point_value = float(np.mean([
            point[dip][module] - 0.5 * (point[previous][module] + point[following][module])
            for module in MODULES
        ]))
        rows.append(
            {
                "arm": arm,
                "layer": args.layer,
                "module": "mean_fixed_7_modules",
                "previous_step": previous,
                "dip_step": dip,
                "next_step": following,
                "n_probe_samples": args.n_samples,
                "bootstrap_draws": args.draws,
                "seed": args.seed,
                "point_uptick_er": f"{point_value:.8f}",
                "bootstrap_mean_uptick_er": f"{mean:.8f}",
                "ci95_lo": f"{lo:.8f}",
                "ci95_hi": f"{hi:.8f}",
                "excludes_zero": "yes" if lo > 0 or hi < 0 else "no",
                "bootstrap_unit": "probe_window; modules fixed",
            }
        )
        raw_draw_paths[arm] = {
            module: values.tolist() for module, values in module_upticks.items()
        }

    write_csv(args.mini_root / "R3_er_sample_bands.csv", rows)
    write_json(
        args.mini_root / "R3_er_sample_bands_draws.json",
        {
            "schema_version": 1,
            "method": "paired nonparametric probe-window bootstrap of pooled-Gram ER",
            "layer": args.layer,
            "task": "legacy_S",
            "draws": args.draws,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "draw_indices_shared_across_checkpoint_triplet": True,
            "module_resamples": False,
            "values": raw_draw_paths,
        },
    )
    print(f"[Bootstrap] wrote {args.mini_root / 'R3_er_sample_bands.csv'} rows={len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--validate-factors", action="store_true")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--mini-root", type=Path, default=DEFAULT_MINI)
    parser.add_argument("--arms", default="opd,sft")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validate-arm", default="opd")
    parser.add_argument("--validate-step", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.arms = parse_names(args.arms, ("opd", "sft"))
    if set(args.arms).difference(TRANSITIONS):
        raise ValueError(f"--arms must be a subset of {sorted(TRANSITIONS)}")
    if args.n_samples <= 0 or args.draws <= 0:
        raise ValueError("--n-samples and --draws must be positive")
    args.mini_root.mkdir(parents=True, exist_ok=True)
    r2.configure_roots(args.run_root, args.mini_root)

    if args.validate_factors:
        validate(args)
    if args.bootstrap:
        print(
            f"[Plan] exact bootstrap arms={args.arms} L{args.layer} n={args.n_samples} "
            f"draws={args.draws} task=legacy_S",
            flush=True,
        )
        if not args.dry_run:
            run_bootstrap(args)
    if not (args.validate_factors or args.bootstrap):
        parser.print_help()


if __name__ == "__main__":
    main()

