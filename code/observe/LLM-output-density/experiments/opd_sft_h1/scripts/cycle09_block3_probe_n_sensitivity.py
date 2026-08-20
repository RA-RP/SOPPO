#!/usr/bin/env python3
"""R1 sample-count sensitivity from audited per-sample Qwen factor bundles."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import cycle09_block3_common as block3
import cycle09_block3_qwen_probe_geometry as probe_geom
import cycle09_r4_campaign as campaign
import cycle09_r4_common as c4
import cycle09_stage3_common as s3
from utils.profiling_utils import _gram_to_svdllm_scaling_diag_matrix


ROOT = block3.RUN_ROOT / "probe_n_sensitivity"
CACHE_ROOT = ROOT / "draws"
OUTPUT = ROOT / "probe_n_sensitivity.csv"
EVENT_OUTPUT = ROOT / "probe_n_event_stability.csv"
MANIFEST = ROOT / "probe_n_sensitivity_manifest.json"
LAYER = 18
EPSILON = 0.05
SEED = 42
DRAWS = 200


def n_grid(probe: str) -> tuple[int, ...]:
    full = probe_geom.ANALYSIS_CAPS[probe]
    if full <= 32:
        return tuple(value for value in (8, 16, 24, full) if value <= full)
    return (16, 32, 64, 128)


def canonical_arm(arm: str, step: int) -> str:
    return "base" if step == 0 else arm


def model_path(arm: str, step: int) -> Path:
    return (
        s3.require_model("opd", 0)
        if step == 0
        else s3.require_model(arm, step)
    )


def cache_path(arm: str, step: int, probe: str) -> Path:
    return CACHE_ROOT / canonical_arm(arm, step) / s3.step_label(step) / f"{probe}.npz"


def load_bundle(arm: str, step: int, probe: str) -> dict[str, Any]:
    canonical = canonical_arm(arm, step)
    path = probe_geom.factor_path(canonical, step, probe)
    meta = block3.read_json(probe_geom.factor_meta_path(canonical, step, probe), {})
    if (
        not path.is_file()
        or meta.get("status") != "complete"
        or int(meta.get("layer", -1)) != LAYER
        or int(meta.get("n_samples", -1)) != probe_geom.ANALYSIS_CAPS[probe]
    ):
        raise FileNotFoundError(
            f"compatible factor bundle missing: {canonical}/{step}/{probe}; "
            f"run {probe_geom.__file__} --phase factor for this cell"
        )
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    required = {"sample_factors", "sample_ids", "sample_ids_sha256"}
    if not required.issubset(bundle):
        raise RuntimeError(f"invalid factor bundle {path}: missing={required - set(bundle)}")
    if len(bundle["sample_factors"]) != probe_geom.ANALYSIS_CAPS[probe]:
        raise RuntimeError(f"factor/sample count drift: {path}")
    return bundle


def draw_indices(probe: str, n: int, full_n: int) -> np.ndarray:
    if n == full_n:
        return np.arange(full_n, dtype=np.int64)[None, :]
    rng = np.random.default_rng(c4.stable_seed(SEED, "probe_n", probe, n))
    return np.stack(
        [rng.choice(full_n, size=n, replace=False) for _ in range(DRAWS)],
        axis=0,
    )


def draw_grams(
    factors: list[dict[int, dict[str, torch.Tensor]]],
    indices: np.ndarray,
    device: str,
) -> dict[str, torch.Tensor]:
    grams = {}
    for group in c4.GROUP_TO_MODULES:
        pieces = [factors[int(index)][LAYER][group] for index in indices.tolist()]
        matrix = torch.cat(pieces, dim=0).to(device=device, dtype=torch.float32)
        matrix.mul_(1.0 / math.sqrt(len(indices)))
        grams[group] = matrix.T @ matrix
        del matrix, pieces
    return grams


def scaling(gram: torch.Tensor, device: str) -> torch.Tensor:
    return _gram_to_svdllm_scaling_diag_matrix(
        gram, cholesky_jitter=1e-5, singular_floor=0.0
    ).to(device=device, dtype=torch.float32)


def rank_and_margin(sigma: torch.Tensor) -> tuple[int, float, float, float]:
    values = sigma.double().cpu().numpy()
    rank = c4.functional_rank(values, EPSILON)
    tail_below = c4.tail_energy(values, rank)
    tail_above = c4.tail_energy(values, max(rank - 1, 0))
    below_margin = EPSILON - tail_below
    above_margin = tail_above - EPSILON
    return rank, below_margin, above_margin, min(below_margin, above_margin)


@torch.inference_mode()
def compute_cell(args: argparse.Namespace) -> dict[str, Any]:
    if args.step not in probe_geom.CORE_STEPS:
        raise ValueError(f"R1 step must be one of {probe_geom.CORE_STEPS}")
    if args.step == 0 and args.arm != "base":
        raise ValueError("R1 step0 is canonical base only")
    if args.step > 0 and args.arm not in s3.ARMS:
        raise ValueError("R1 nonzero cell requires a training arm")
    target = cache_path(args.arm, args.step, args.probe)
    if target.is_file() and not args.force:
        with np.load(target) as cached:
            metadata = json.loads(str(cached["metadata_json"].item()))
        if metadata.get("status") == "complete":
            return metadata

    bundle = load_bundle(args.arm, args.step, args.probe)
    factors = bundle["sample_factors"]
    sample_ids_sha256 = str(bundle["sample_ids_sha256"])
    full_n = len(factors)
    grids = n_grid(args.probe)
    ranks = np.empty((len(grids), DRAWS, len(c4.MODULES)), dtype=np.int32)
    below = np.empty_like(ranks, dtype=np.float32)
    above = np.empty_like(ranks, dtype=np.float32)
    margin = np.empty_like(ranks, dtype=np.float32)
    index_hashes = {}
    model = campaign.load_model(model_path(args.arm, args.step), args.device)
    try:
        weights = {
            module: campaign.module_at(model, LAYER, module).weight.detach().float()
            for module in c4.MODULES
        }
        for n_index, n in enumerate(grids):
            indices = draw_indices(args.probe, n, full_n)
            index_hashes[str(n)] = hashlib.sha256(indices.tobytes()).hexdigest()
            unique_draws = len(indices)
            for draw_index, selected in enumerate(indices):
                grams = draw_grams(factors, selected, args.device)
                scales = {
                    group: scaling(gram, args.device) for group, gram in grams.items()
                }
                for module_index, module in enumerate(c4.MODULES):
                    matrix = weights[module] @ scales[c4.MODULE_TO_GROUP[module]]
                    sigma = torch.linalg.svdvals(matrix)
                    values = rank_and_margin(sigma)
                    ranks[n_index, draw_index, module_index] = values[0]
                    below[n_index, draw_index, module_index] = values[1]
                    above[n_index, draw_index, module_index] = values[2]
                    margin[n_index, draw_index, module_index] = values[3]
                    del matrix, sigma
                grams.clear()
                scales.clear()
                torch.cuda.empty_cache()
                if (draw_index + 1) % 20 == 0 or draw_index + 1 == unique_draws:
                    print(
                        f"[R1] {args.arm}/{args.step}/{args.probe}/n={n} "
                        f"{draw_index + 1}/{unique_draws}",
                        flush=True,
                    )
            if unique_draws == 1:
                ranks[n_index] = np.repeat(ranks[n_index, :1], DRAWS, axis=0)
                below[n_index] = np.repeat(below[n_index, :1], DRAWS, axis=0)
                above[n_index] = np.repeat(above[n_index, :1], DRAWS, axis=0)
                margin[n_index] = np.repeat(margin[n_index, :1], DRAWS, axis=0)
    finally:
        campaign.unload_model(model)
        del bundle, factors
        gc.collect()

    metadata = {
        "schema_version": "cycle09_block3_probe_n_draws_v1",
        "status": "complete",
        "arm": canonical_arm(args.arm, args.step),
        "step": args.step,
        "probe": args.probe,
        "layer": LAYER,
        "epsilon": EPSILON,
        "draws": DRAWS,
        "seed": SEED,
        "sampling": "without replacement; sample unit; windows remain nested",
        "n_grid": list(grids),
        "full_n_unique_subsets": 1,
        "sample_ids_sha256": sample_ids_sha256,
        "index_sha256_by_n": index_hashes,
        "factor_source": str(
            probe_geom.factor_path(canonical_arm(args.arm, args.step), args.step, args.probe)
        ),
        "created_utc": block3.utc_now(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            n_grid=np.asarray(grids, dtype=np.int32),
            r_epsilon=ranks,
            margin_below=below,
            margin_above=above,
            threshold_margin=margin,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    os.replace(temporary, target)
    return metadata


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values.astype(np.float64), q))


def sign_equal(observed: np.ndarray, expected: float) -> np.ndarray:
    if expected > 0:
        return observed > 0
    if expected < 0:
        return observed < 0
    return observed == 0


def load_cache(arm: str, step: int, probe: str) -> dict[str, np.ndarray]:
    target = cache_path(arm, step, probe)
    if not target.is_file():
        raise FileNotFoundError(target)
    with np.load(target) as payload:
        return {key: payload[key].copy() for key in payload.files if key != "metadata_json"}


def finalize() -> dict[str, Any]:
    rows = []
    event_rows = []
    caches: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    for probe in probe_geom.ALL_PROBES:
        caches[("base", 0, probe)] = load_cache("base", 0, probe)
        for arm in s3.ARMS:
            for step in probe_geom.CORE_STEPS[1:]:
                caches[(arm, step, probe)] = load_cache(arm, step, probe)

    for probe in probe_geom.ALL_PROBES:
        grids = n_grid(probe)
        full_index = len(grids) - 1
        base = caches[("base", 0, probe)]
        base_full = base["r_epsilon"][full_index].mean(axis=1)[0]
        for arm in s3.ARMS:
            for step in probe_geom.CORE_STEPS:
                cache = base if step == 0 else caches[(arm, step, probe)]
                for n_index, n in enumerate(grids):
                    rank_draws = cache["r_epsilon"][n_index].mean(axis=1)
                    margin_draws = cache["threshold_margin"][n_index].min(axis=1)
                    full_rank = cache["r_epsilon"][full_index].mean(axis=1)[0]
                    if step == 0:
                        delta_draws = np.zeros(DRAWS, dtype=np.float64)
                        full_delta = 0.0
                    else:
                        base_draws = base["r_epsilon"][n_index].mean(axis=1)
                        delta_draws = rank_draws - base_draws
                        full_delta = full_rank - base_full
                    rows.append(
                        {
                            "arm": arm,
                            "step": step,
                            "probe": probe,
                            "layer": LAYER,
                            "epsilon": EPSILON,
                            "n": n,
                            "draws": DRAWS,
                            "r_epsilon_median": quantile(rank_draws, 0.50),
                            "r_epsilon_q25": quantile(rank_draws, 0.25),
                            "r_epsilon_q75": quantile(rank_draws, 0.75),
                            "r_epsilon_q025": quantile(rank_draws, 0.025),
                            "r_epsilon_q975": quantile(rank_draws, 0.975),
                            "full_sample_r_epsilon": full_rank,
                            "absolute_difference_from_full_median": quantile(
                                np.abs(rank_draws - full_rank), 0.50
                            ),
                            "full_sample_delta_from_base": full_delta,
                            "delta_from_base_median": quantile(delta_draws, 0.50),
                            "sign_retention_rate": float(
                                sign_equal(delta_draws, full_delta).mean()
                            ),
                            "threshold_margin_median": quantile(
                                margin_draws, 0.50
                            ),
                            "threshold_margin_q025": quantile(
                                margin_draws, 0.025
                            ),
                            "shared_base_compute": step == 0,
                        }
                    )

            full_trajectory = []
            for step in probe_geom.CORE_STEPS:
                cache = base if step == 0 else caches[(arm, step, probe)]
                rank = cache["r_epsilon"][full_index].mean(axis=1)[0]
                full_trajectory.append(rank - base_full)
            full_peak = int(
                probe_geom.CORE_STEPS[int(np.argmax(full_trajectory))]
            )
            full_trough = int(
                probe_geom.CORE_STEPS[int(np.argmin(full_trajectory))]
            )
            for n_index, n in enumerate(grids):
                trajectories = []
                base_draws = base["r_epsilon"][n_index].mean(axis=1)
                for step in probe_geom.CORE_STEPS:
                    cache = base if step == 0 else caches[(arm, step, probe)]
                    trajectories.append(
                        cache["r_epsilon"][n_index].mean(axis=1) - base_draws
                    )
                matrix = np.stack(trajectories, axis=1)
                peak_steps = np.asarray(probe_geom.CORE_STEPS)[
                    np.argmax(matrix, axis=1)
                ]
                trough_steps = np.asarray(probe_geom.CORE_STEPS)[
                    np.argmin(matrix, axis=1)
                ]
                event_rows.append(
                    {
                        "arm": arm,
                        "probe": probe,
                        "layer": LAYER,
                        "epsilon": EPSILON,
                        "n": n,
                        "draws": DRAWS,
                        "full_sample_peak_step": full_peak,
                        "peak_step_retention_rate": float(
                            (peak_steps == full_peak).mean()
                        ),
                        "full_sample_trough_step": full_trough,
                        "trough_step_retention_rate": float(
                            (trough_steps == full_trough).mean()
                        ),
                        "tie_rule": "earliest observed checkpoint",
                    }
                )

    expected_rows = (
        len(s3.ARMS)
        * len(probe_geom.CORE_STEPS)
        * sum(len(n_grid(probe)) for probe in probe_geom.ALL_PROBES)
    )
    expected_events = len(s3.ARMS) * sum(
        len(n_grid(probe)) for probe in probe_geom.ALL_PROBES
    )
    if len(rows) != expected_rows or len(event_rows) != expected_events:
        raise RuntimeError(
            f"R1 row-count drift rows={len(rows)}/{expected_rows} "
            f"events={len(event_rows)}/{expected_events}"
        )
    block3.atomic_csv(OUTPUT, rows)
    block3.atomic_csv(EVENT_OUTPUT, event_rows)
    block3.atomic_csv(block3.MINI / OUTPUT.name, rows)
    block3.atomic_csv(block3.MINI / EVENT_OUTPUT.name, event_rows)
    inventory = probe_geom.factor_inventory()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "task": "Cycle09 block3 R1 fixed-probe sample-count sensitivity",
        "seed": SEED,
        "draws": DRAWS,
        "sampling": "without replacement; paired indices across arms/checkpoints",
        "layer": LAYER,
        "modules": list(c4.MODULES),
        "epsilon": EPSILON,
        "n_grid_by_probe": {
            probe: list(n_grid(probe)) for probe in probe_geom.ALL_PROBES
        },
        "factor_inventory": {
            "expected_cells": inventory["expected_cells"],
            "compatible_cells": inventory["compatible_cells"],
        },
        "outputs": [block3.artifact(OUTPUT), block3.artifact(EVENT_OUTPUT)],
        "created_utc": block3.utc_now(),
    }
    block3.atomic_json(MANIFEST, manifest)
    block3.atomic_json(block3.MINI / MANIFEST.name, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cell", "finalize"), default="cell")
    parser.add_argument("--arm", choices=("base", *s3.ARMS), default="base")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--probe", choices=probe_geom.ALL_PROBES, default="E_math")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = compute_cell(arguments) if arguments.phase == "cell" else finalize()
    print(json.dumps(result, indent=2))
