#!/usr/bin/env python3
"""Cycle 09 off-KD T1: paired sample bootstrap for three-arm endpoint deltas.

The frozen scope is layer 18, five static probe families, steps {0, 624},
three arms, 256 paired draws, and the mean over the seven fixed modules.
Generation probe S_bos keeps its three seed batches separate during resampling;
the three batch estimates are averaged only after each paired draw.

The runner is resumable at two boundaries:
  * compact per-sample factor bundles are committed atomically per cell;
  * r_epsilon draws are committed atomically per cell.

Existing Round-5 A4 draws are reused exactly for legacy_S_math and E_ood at
base/OPD624/SFT624. New cells use the same deterministic sample indices and
the same SVD-LLM Cholesky whitening as A4.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

import cycle09_offkd_geometry as offgeom
import cycle09_r4_campaign as camp
import cycle09_r4_common as c4
import cycle09_r5_common as c5

from utils.profiling_utils import _gram_to_svdllm_scaling_diag_matrix


LAYER = 18
STEP = 624
SEED = 42
DRAWS = 256
MODEL_CELLS = (("base", 0), ("opd", STEP), ("sft", STEP), ("offkd", STEP))
ARMS = ("opd", "sft", "offkd")
PROBE_FAMILIES = {
    "legacy_S_math": ("legacy_S_math",),
    "E_ood": ("E_ood",),
    "E_general": ("E_general",),
    "E_math_hard": ("E_math_hard",),
    "S_bos": ("S_bos__g3", "S_bos__g17", "S_bos__g31"),
}
OLD_A4_TASKS = frozenset(("legacy_S_math", "E_ood"))
WORK_ROOT = Path("/root/autodl-tmp/cycle09_t1")
OLD_FACTOR_ROOT = c5.R4_ROOT / "scratch/bootstrap_factors"
OLD_CACHE_ROOT = c5.RUN_ROOT / "scratch/a4_cache"
SHARED_CI = c5.MINI_ROOT / "R5_bootstrap_ci.csv"
HANDIN = c4.REPO / "mypaper/code/code_evolution.md"
HANDIN_START = "<!-- cycle09-offkd-t1-start -->"
HANDIN_END = "<!-- cycle09-offkd-t1-end -->"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def task_catalog() -> dict[str, dict[str, Any]]:
    return {task["task_id"]: task for task in offgeom.probe_tasks()}


def parse_families(value: str | None, *, smoke: bool) -> tuple[str, ...]:
    if value:
        families = tuple(item.strip() for item in value.split(",") if item.strip())
    elif smoke:
        families = ("E_math_hard",)
    else:
        families = tuple(PROBE_FAMILIES)
    unknown = sorted(set(families) - set(PROBE_FAMILIES))
    if unknown:
        raise ValueError(f"unknown probe families: {unknown}")
    return families


def task_ids(families: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(task for family in families for task in PROBE_FAMILIES[family])


def sample_id_sha256(sample_ids: list[str]) -> str:
    payload = "\n".join(map(str, sample_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compact_bundle_path(root: Path, arm: str, step: int, task: str) -> Path:
    return root / "factors" / arm / c4.step_label(step) / f"{task}.pt"


def compact_meta_path(root: Path, arm: str, step: int, task: str) -> Path:
    return root / "factors" / arm / c4.step_label(step) / f"{task}.json"


def draw_cache_path(root: Path, arm: str, step: int, task: str) -> Path:
    return root / "draws" / arm / c4.step_label(step) / f"{task}.npz"


def old_factor_path(arm: str, step: int, task: str) -> Path:
    source_arm = "opd" if int(step) == 0 else arm
    return OLD_FACTOR_ROOT / source_arm / c4.step_label(step) / f"{task}.pt"


def old_cache_path(arm: str, step: int, task: str) -> Path | None:
    if task not in OLD_A4_TASKS or arm == "offkd":
        return None
    source_arm = "opd" if int(step) == 0 else arm
    return OLD_CACHE_ROOT / f"{task}__{source_arm}__{c4.step_label(step)}.npz"


def can_reuse_old_draws(
    *, arm: str, step: int, task: str, draws: int, seed: int, smoke: bool
) -> bool:
    path = old_cache_path(arm, step, task)
    return (
        not smoke
        and draws == DRAWS
        and seed == SEED
        and path is not None
        and path.exists()
    )


def model_path(arm: str, step: int) -> Path:
    if arm == "base":
        return c4.BASE_MODEL
    if arm == "offkd":
        return offgeom.offkd_model_path(step)
    return c4.model_path(arm, step)


def compact_bundle_complete(root: Path, arm: str, step: int, task: str) -> bool:
    bundle = compact_bundle_path(root, arm, step, task)
    meta = c4.read_json(compact_meta_path(root, arm, step, task), {})
    return (
        bundle.exists()
        and bundle.stat().st_size > 0
        and meta.get("status") == "complete"
        and meta.get("task_id") == task
        and int(meta.get("layer", -1)) == LAYER
    )


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_npz(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(tmp, path)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    catalog = task_catalog()
    missing: list[str] = []
    for task in args.task_ids:
        corpus = Path(catalog[task]["corpus"])
        if not corpus.exists():
            missing.append(str(corpus))
    for arm, step in MODEL_CELLS:
        path = model_path(arm, step)
        if not (path / "config.json").exists():
            missing.append(str(path / "config.json"))
    for task in sorted(OLD_A4_TASKS.intersection(args.task_ids)):
        for arm in ("base", "opd", "sft"):
            step = 0 if arm == "base" else STEP
            path = old_cache_path(arm, step, task)
            if not args.smoke and (path is None or not path.exists()):
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("missing T1 inputs:\n" + "\n".join(missing))

    disk = shutil.disk_usage(args.work_root.parent)
    if not args.smoke and disk.free < 30 * 1024**3:
        raise RuntimeError(f"less than 30 GiB free for T1 factors: {disk.free / 1024**3:.1f}")

    compact_cells = []
    reused_cells = []
    for arm, step in MODEL_CELLS:
        for task in args.task_ids:
            if can_reuse_old_draws(
                arm=arm, step=step, task=task, draws=args.draws,
                seed=args.seed, smoke=args.smoke,
            ):
                reused_cells.append(f"{arm}/{c4.step_label(step)}/{task}")
            else:
                compact_cells.append(f"{arm}/{c4.step_label(step)}/{task}")
    payload = {
        "timestamp_utc": utc_now(),
        "families": list(args.families),
        "actual_task_ids": list(args.task_ids),
        "draws": args.draws,
        "seed": args.seed,
        "sample_limit": args.sample_limit,
        "svd_mode": args.svd_mode,
        "old_a4_draw_cells_reused": reused_cells,
        "compact_factor_and_draw_cells": compact_cells,
        "free_gib": disk.free / 1024**3,
    }
    args.work_root.mkdir(parents=True, exist_ok=True)
    c4.write_json_atomic(args.work_root / "preflight.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def prepare_task_samples(
    args: argparse.Namespace, tokenizer, task: dict[str, Any]
) -> list[c4.PreparedSample]:
    samples = c4.prepare_samples(
        Path(task["corpus"]),
        tokenizer,
        corpus_id=task["task_id"],
        window_seed=c4.WINDOW_SEED,
        max_context_tokens=c4.MAX_CONTEXT_TOKENS,
    )
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    if not samples:
        raise RuntimeError(f"no samples for {task['task_id']}")
    return samples


@torch.no_grad()
def collect(args: argparse.Namespace) -> None:
    catalog = task_catalog()
    tokenizer = AutoTokenizer.from_pretrained(
        str(c4.BASE_MODEL), trust_remote_code=True, use_fast=True
    )
    samples_by_task = {
        task: prepare_task_samples(args, tokenizer, catalog[task])
        for task in args.task_ids
    }
    del tokenizer

    for arm, step in MODEL_CELLS:
        pending = []
        for task in args.task_ids:
            if can_reuse_old_draws(
                arm=arm, step=step, task=task, draws=args.draws,
                seed=args.seed, smoke=args.smoke,
            ):
                continue
            if not compact_bundle_complete(args.work_root, arm, step, task):
                pending.append(task)
        if not pending:
            print(f"[T1 collect cached] {arm}/{c4.step_label(step)}", flush=True)
            continue

        model = camp.load_model(model_path(arm, step), args.device)
        print(
            f"[T1 model] {arm}/{c4.step_label(step)} pending={len(pending)}",
            flush=True,
        )
        try:
            for task in pending:
                samples = samples_by_task[task]
                profile = camp.collect_profile(
                    model,
                    samples,
                    [LAYER],
                    args.device,
                    keep_factors=True,
                    keep_residual_samples=False,
                    factor_layers=(LAYER,),
                )
                ids = [sample.sample_id for sample in samples]
                factors = profile["sample_factors"]
                if len(factors) != len(ids):
                    raise RuntimeError(f"factor/sample mismatch for {arm}/{task}")
                payload = {
                    "schema_version": "cycle09_t1_compact_factors_v1",
                    "arm": arm,
                    "step": int(step),
                    "task_id": task,
                    "layer": LAYER,
                    "sample_ids": ids,
                    "sample_id_sha256": sample_id_sha256(ids),
                    "sample_factors": factors,
                }
                target = compact_bundle_path(args.work_root, arm, step, task)
                atomic_torch_save(target, payload)
                c4.write_json_atomic(
                    compact_meta_path(args.work_root, arm, step, task),
                    {
                        "status": "complete",
                        "timestamp_utc": utc_now(),
                        "arm": arm,
                        "step": int(step),
                        "task_id": task,
                        "layer": LAYER,
                        "n_samples": len(ids),
                        "sample_id_sha256": payload["sample_id_sha256"],
                        "bundle_path": str(target),
                        "bundle_bytes": target.stat().st_size,
                        "contents": "sample_ids + layer-18 sample_factors only",
                    },
                )
                print(
                    f"[T1 factor] {arm}/{c4.step_label(step)}/{task} "
                    f"n={len(ids)} bytes={target.stat().st_size}",
                    flush=True,
                )
                del profile, payload, factors
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            camp.unload_model(model)


def load_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    required = {"sample_ids", "sample_factors"}
    if not required.issubset(bundle):
        raise RuntimeError(f"invalid factor bundle {path}: missing {required - set(bundle)}")
    return bundle


def indices_by_draw(n_samples: int, draws: int, seed: int, task: str) -> np.ndarray:
    rng = np.random.default_rng(c4.stable_seed(seed, task))
    return rng.integers(0, n_samples, size=(draws, n_samples))


def draw_grams(
    bundle: dict[str, Any], indices: np.ndarray, device: str
) -> dict[str, torch.Tensor]:
    grams: dict[str, torch.Tensor] = {}
    factors = bundle["sample_factors"]
    for group in c4.GROUP_TO_MODULES:
        pieces = [factors[int(index)][LAYER][group] for index in indices.tolist()]
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
def cell_ranks(
    *,
    model,
    bundle: dict[str, Any],
    indices: np.ndarray,
    device: str,
    svd_mode: str,
    label: str,
) -> np.ndarray:
    weights = {
        module: camp.module_at(model, LAYER, module).weight.detach().float()
        for module in c4.MODULES
    }
    out = np.full((len(indices), len(c4.MODULES)), np.nan, dtype=np.float64)
    for draw_index, draw_indices in enumerate(indices):
        grams = draw_grams(bundle, draw_indices, device)
        scalings = {
            group: scaling_from_gram(gram, device) for group, gram in grams.items()
        }
        for module_index, module in enumerate(c4.MODULES):
            matrix = weights[module] @ scalings[c4.MODULE_TO_GROUP[module]]
            if svd_mode == "values":
                sigma = torch.linalg.svdvals(matrix)
            else:
                _u, sigma, _vh = torch.linalg.svd(matrix, full_matrices=False)
                del _u, _vh
            out[draw_index, module_index] = c4.functional_rank(
                sigma.cpu().numpy().astype(np.float64), 0.05
            )
            del matrix, sigma
        grams.clear()
        scalings.clear()
        torch.cuda.empty_cache()
        if (draw_index + 1) % 16 == 0 or draw_index + 1 == len(indices):
            print(f"[T1 draws] {label} {draw_index + 1}/{len(indices)}", flush=True)
    if not np.isfinite(out).all():
        raise RuntimeError(f"non-finite r_epsilon draws in {label}")
    return out


def own_cache_valid(
    path: Path, *, draws: int, seed: int, task: str, svd_mode: str
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as payload:
            if payload["r_eps"].shape != (draws, len(c4.MODULES)):
                return False
            meta = json.loads(str(payload["metadata_json"].item()))
    except Exception:
        return False
    return (
        int(meta.get("draws", -1)) == draws
        and int(meta.get("seed", -1)) == seed
        and meta.get("task_id") == task
        and meta.get("svd_mode") == svd_mode
    )


def load_old_ranks(arm: str, step: int, task: str) -> np.ndarray:
    path = old_cache_path(arm, step, task)
    if path is None or not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        ranks = payload["r_eps"].astype(np.float64, copy=True)
    if ranks.shape != (DRAWS, len(c4.MODULES)):
        raise RuntimeError(f"unexpected old A4 shape at {path}: {ranks.shape}")
    return ranks


def assert_parity(args: argparse.Namespace) -> None:
    if args.smoke:
        return
    path = args.work_root / f"parity_{args.svd_mode}.json"
    payload = c4.read_json(path, {})
    if payload.get("status") != "passed" or payload.get("svd_mode") != args.svd_mode:
        raise RuntimeError(
            f"formal bootstrap requires a passed parity check: {path}; "
            f"run --mode parity --svd-mode {args.svd_mode}"
        )


@torch.no_grad()
def parity(args: argparse.Namespace) -> None:
    task = "E_ood"
    arm, step = "base", 0
    path = old_factor_path(arm, step, task)
    bundle = load_bundle(path)
    n_samples = len(bundle["sample_ids"])
    indices = indices_by_draw(n_samples, args.parity_draws, args.seed, task)
    model = camp.load_model(c4.BASE_MODEL, args.device)
    try:
        observed = cell_ranks(
            model=model,
            bundle=bundle,
            indices=indices,
            device=args.device,
            svd_mode=args.svd_mode,
            label=f"parity/{task}/base",
        )
    finally:
        camp.unload_model(model)
        del bundle
        gc.collect()
    expected = load_old_ranks(arm, step, task)[: args.parity_draws]
    equal = np.array_equal(observed, expected)
    mismatches = np.argwhere(observed != expected).tolist()
    payload = {
        "status": "passed" if equal else "failed",
        "timestamp_utc": utc_now(),
        "task_id": task,
        "cell": "base/step_000",
        "draws_checked": args.parity_draws,
        "modules_checked": list(c4.MODULES),
        "svd_mode": args.svd_mode,
        "expected_source": str(old_cache_path(arm, step, task)),
        "factor_source": str(path),
        "exact_integer_rank_match": bool(equal),
        "mismatch_indices": mismatches,
    }
    c4.write_json_atomic(args.work_root / f"parity_{args.svd_mode}.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    if not equal:
        raise RuntimeError(f"A4 parity failed: {mismatches[:8]}")


def bootstrap(args: argparse.Namespace) -> None:
    assert_parity(args)
    catalog = task_catalog()
    tokenizer = AutoTokenizer.from_pretrained(
        str(c4.BASE_MODEL), trust_remote_code=True, use_fast=True
    )
    expected_ids_by_task = {}
    for task in args.task_ids:
        samples = prepare_task_samples(args, tokenizer, catalog[task])
        expected_ids_by_task[task] = [sample.sample_id for sample in samples]
        del samples
    del tokenizer

    for arm, step in MODEL_CELLS:
        pending = []
        for task in args.task_ids:
            if can_reuse_old_draws(
                arm=arm, step=step, task=task, draws=args.draws,
                seed=args.seed, smoke=args.smoke,
            ):
                print(
                    f"[T1 old A4] {arm}/{c4.step_label(step)}/{task}",
                    flush=True,
                )
                continue
            target = draw_cache_path(args.work_root, arm, step, task)
            if own_cache_valid(
                target, draws=args.draws, seed=args.seed,
                task=task, svd_mode=args.svd_mode,
            ):
                print(
                    f"[T1 draw cached] {arm}/{c4.step_label(step)}/{task}",
                    flush=True,
                )
                continue
            pending.append(task)
        if not pending:
            continue

        model = camp.load_model(model_path(arm, step), args.device)
        try:
            for task in pending:
                bundle_path = compact_bundle_path(args.work_root, arm, step, task)
                bundle = load_bundle(bundle_path)
                ids = [str(value) for value in bundle["sample_ids"]]
                if ids != expected_ids_by_task[task]:
                    raise RuntimeError(
                        f"sample order mismatch for {arm}/{c4.step_label(step)}/{task}"
                    )
                indices = indices_by_draw(len(ids), args.draws, args.seed, task)
                values = cell_ranks(
                    model=model,
                    bundle=bundle,
                    indices=indices,
                    device=args.device,
                    svd_mode=args.svd_mode,
                    label=f"{arm}/{c4.step_label(step)}/{task}",
                )
                target = draw_cache_path(args.work_root, arm, step, task)
                metadata = {
                    "schema_version": "cycle09_t1_draws_v1",
                    "timestamp_utc": utc_now(),
                    "arm": arm,
                    "step": int(step),
                    "task_id": task,
                    "layer": LAYER,
                    "draws": args.draws,
                    "seed": args.seed,
                    "svd_mode": args.svd_mode,
                    "epsilon": 0.05,
                    "bootstrap_unit": "sample; windows nested",
                    "sample_id_sha256": sample_id_sha256(ids),
                    "factor_source": str(bundle_path),
                }
                atomic_npz(
                    target,
                    r_eps=values,
                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
                print(f"[T1 cache] {target}", flush=True)
                del bundle, values, indices
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            camp.unload_model(model)


def ranks_for(args: argparse.Namespace, arm: str, step: int, task: str) -> np.ndarray:
    if can_reuse_old_draws(
        arm=arm, step=step, task=task, draws=args.draws,
        seed=args.seed, smoke=args.smoke,
    ):
        return load_old_ranks(arm, step, task)
    path = draw_cache_path(args.work_root, arm, step, task)
    if not own_cache_valid(
        path, draws=args.draws, seed=args.seed, task=task, svd_mode=args.svd_mode
    ):
        raise FileNotFoundError(f"missing valid T1 draw cache: {path}")
    with np.load(path) as payload:
        return payload["r_eps"].astype(np.float64, copy=True)


def interval(values: np.ndarray) -> tuple[float, float, float]:
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("CI input must be a finite one-dimensional draw vector")
    return (
        float(np.mean(values)),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def excludes_zero(lo: float, hi: float) -> bool:
    return bool(lo > 0.0 or hi < 0.0)


def family_draws(
    args: argparse.Namespace, family: str, arm: str
) -> tuple[np.ndarray, float | None]:
    task_vectors = []
    point_values = []
    for task in PROBE_FAMILIES[family]:
        base = ranks_for(args, "base", 0, task)
        endpoint = ranks_for(args, arm, STEP, task)
        delta = endpoint - base
        vector = np.mean(delta, axis=1)
        task_vectors.append(vector)
        point_values.append(float(np.mean(vector)))
    draws = np.mean(np.stack(task_vectors, axis=0), axis=0)
    seed_sd = (
        float(np.std(np.asarray(point_values), ddof=1))
        if len(point_values) > 1 else None
    )
    return draws, seed_sd


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for family in args.families:
        vectors: dict[str, np.ndarray] = {}
        seed_sd: dict[str, float | None] = {}
        for arm in ARMS:
            vectors[arm], seed_sd[arm] = family_draws(args, family, arm)

        opd = interval(vectors["opd"])
        sft = interval(vectors["sft"])
        offkd = interval(vectors["offkd"])
        offkd_opd = interval(vectors["offkd"] - vectors["opd"])
        offkd_sft = interval(vectors["offkd"] - vectors["sft"])
        opd_sft = interval(vectors["opd"] - vectors["sft"])
        rows.append(
            {
                "task_id": family,
                "step": STEP,
                "layer": LAYER,
                "module": "mean_fixed_7_modules",
                "metric": "r_eps_delta_624_minus_0",
                "bootstrap_unit": (
                    "sample; windows nested; S_bos seed batches independently resampled"
                ),
                "bootstrap_draws": args.draws,
                "opd_mean": opd[0],
                "opd_ci95_lo": opd[1],
                "opd_ci95_hi": opd[2],
                "sft_mean": sft[0],
                "sft_ci95_lo": sft[1],
                "sft_ci95_hi": sft[2],
                "opd_minus_sft_mean": opd_sft[0],
                "opd_minus_sft_ci95_lo": opd_sft[1],
                "opd_minus_sft_ci95_hi": opd_sft[2],
                "ci_excludes_zero": excludes_zero(opd_sft[1], opd_sft[2]),
                "offkd_mean": offkd[0],
                "offkd_ci95_lo": offkd[1],
                "offkd_ci95_hi": offkd[2],
                "offkd_minus_opd_mean": offkd_opd[0],
                "offkd_minus_opd_ci95_lo": offkd_opd[1],
                "offkd_minus_opd_ci95_hi": offkd_opd[2],
                "offkd_minus_opd_ci_excludes_zero": excludes_zero(
                    offkd_opd[1], offkd_opd[2]
                ),
                "offkd_minus_sft_mean": offkd_sft[0],
                "offkd_minus_sft_ci95_lo": offkd_sft[1],
                "offkd_minus_sft_ci95_hi": offkd_sft[2],
                "offkd_minus_sft_ci_excludes_zero": excludes_zero(
                    offkd_sft[1], offkd_sft[2]
                ),
                "opd_minus_sft_ci_excludes_zero": excludes_zero(
                    opd_sft[1], opd_sft[2]
                ),
                "generation_seed_batches": len(PROBE_FAMILIES[family]),
                "opd_seed_batch_sd": seed_sd["opd"],
                "sft_seed_batch_sd": seed_sd["sft"],
                "offkd_seed_batch_sd": seed_sd["offkd"],
                "quantity_definition": (
                    "mean_7_modules(r_eps(step624)-r_eps(base)); epsilon=0.05"
                ),
                "source_kind": "offkd_geometry_handoff_section_6_T1",
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| probe | OPD d [95% CI] | SFT d [95% CI] | off-KD d [95% CI] | offKD-OPD [95% CI] | offKD-SFT [95% CI] | OPD-SFT [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    specs = (
        ("opd_mean", "opd_ci95_lo", "opd_ci95_hi"),
        ("sft_mean", "sft_ci95_lo", "sft_ci95_hi"),
        ("offkd_mean", "offkd_ci95_lo", "offkd_ci95_hi"),
        ("offkd_minus_opd_mean", "offkd_minus_opd_ci95_lo", "offkd_minus_opd_ci95_hi"),
        ("offkd_minus_sft_mean", "offkd_minus_sft_ci95_lo", "offkd_minus_sft_ci95_hi"),
        ("opd_minus_sft_mean", "opd_minus_sft_ci95_lo", "opd_minus_sft_ci95_hi"),
    )
    for row in rows:
        cells = []
        for mean_key, lo_key, hi_key in specs:
            cells.append(
                f"{float(row[mean_key]):.6f} [{float(row[lo_key]):.6f}, {float(row[hi_key]):.6f}]"
            )
        lines.append(f"| {row['task_id']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def append_shared_ci(rows: list[dict[str, Any]]) -> None:
    existing: list[dict[str, Any]] = []
    fields: list[str] = []
    if SHARED_CI.exists():
        with open(SHARED_CI, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            existing = list(reader)
    families = {row["task_id"] for row in rows}
    existing = [
        row for row in existing
        if not (
            row.get("metric") == "r_eps_delta_624_minus_0"
            and row.get("task_id") in families
            and row.get("module") == "mean_fixed_7_modules"
        )
    ]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    c4.write_csv_atomic(SHARED_CI, [*existing, *rows], fields)


def append_handin(table: str, manifest_path: Path) -> None:
    existing = HANDIN.read_text(encoding="utf-8") if HANDIN.exists() else ""
    block = (
        f"{HANDIN_START}\n\n"
        "## Cycle 09 off-KD control - Stage 3 additional tasks (T1/T2/T3)\n\n"
        "T2 xs_gap and T3 OPD/SFT geometry backfill at steps {80,320,480} are complete. "
        "T1 uses layer 18, endpoints {0,624}, 256 paired sample bootstrap draws, the "
        "seven-module mean, and all three S_bos generation seed batches. Existing A4 "
        "draws are reused for legacy_S_math/E_ood base/OPD/SFT cells; all other factor "
        "and draw cells are atomically checkpointed under /root/autodl-tmp/cycle09_t1.\n\n"
        "Raw T1 readings (no interpretation):\n\n"
        f"{table}\n"
        f"Manifest: `{manifest_path}`.\n\n"
        f"{HANDIN_END}"
    )
    if HANDIN_START in existing and HANDIN_END in existing:
        prefix = existing.split(HANDIN_START, 1)[0]
        suffix = existing.split(HANDIN_END, 1)[1]
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = existing.rstrip() + "\n\n---\n\n" + block + "\n"
    tmp = HANDIN.with_suffix(HANDIN.suffix + ".tmp")
    tmp.write_text(updated, encoding="utf-8")
    os.replace(tmp, HANDIN)


def report(args: argparse.Namespace) -> None:
    rows = build_rows(args)
    output_csv = args.mini_root / "R5_t1_bootstrap_ci.csv"
    output_md = args.mini_root / "R5_t1_bootstrap_ci.md"
    fields = list(rows[0])
    c4.write_csv_atomic(output_csv, rows, fields)
    table = markdown_table(rows)
    output_md.write_text(table, encoding="utf-8")

    factor_sources = {}
    draw_sources = {}
    for arm, step in MODEL_CELLS:
        for task in args.task_ids:
            key = f"{arm}/{c4.step_label(step)}/{task}"
            if can_reuse_old_draws(
                arm=arm, step=step, task=task, draws=args.draws,
                seed=args.seed, smoke=args.smoke,
            ):
                factor_sources[key] = "not_loaded; old A4 draw cache reused"
                draw_sources[key] = str(old_cache_path(arm, step, task))
            else:
                factor_sources[key] = str(
                    compact_bundle_path(args.work_root, arm, step, task)
                )
                draw_sources[key] = str(
                    draw_cache_path(args.work_root, arm, step, task)
                )
    manifest = {
        "schema_version": "cycle09_offkd_T1_v1",
        "status": "complete",
        "timestamp_utc": utc_now(),
        "specification": str(
            c4.REPO / "mypaper/theory/offkd_geometry_handoff.md"
        ) + "#section-6-T1",
        "layer": LAYER,
        "steps": [0, STEP],
        "arms": list(ARMS),
        "probe_families": list(args.families),
        "actual_task_ids": list(args.task_ids),
        "generation_seed_batches": {family: list(PROBE_FAMILIES[family]) for family in args.families},
        "draws": args.draws,
        "seed": args.seed,
        "bootstrap_unit": "sample; windows nested",
        "paired_across_arms_and_steps_within_task": True,
        "S_bos_aggregation": (
            "independent sample resampling within g3/g17/g31; then seed-batch mean "
            "and seven-module mean within each draw"
        ),
        "svd_mode": args.svd_mode,
        "whitening": "SVD-LLM Cholesky; jitter=1e-5; fp32 matrix SVD",
        "old_a4_cache_reuse_rule": (
            "legacy_S_math and E_ood base/OPD624/SFT624 only; exact draws=256 seed=42"
        ),
        "factor_sources": factor_sources,
        "draw_sources": draw_sources,
        "outputs": [str(output_csv), str(output_md), str(SHARED_CI)],
        "interpretation": "none; raw readings only",
    }
    manifest_path = args.mini_root / "T1_bootstrap_manifest.json"
    c4.write_json_atomic(manifest_path, manifest)
    if not args.smoke:
        append_shared_ci(rows)
        append_handin(table, manifest_path)
    print(table, flush=True)
    print(f"[T1 report] rows={len(rows)} manifest={manifest_path}", flush=True)


def configure_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("preflight", "collect", "parity", "bootstrap", "report", "all"),
        default="all",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--families", default=None)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--parity-draws", type=int, default=4)
    parser.add_argument("--svd-mode", choices=("values", "full"), default="values")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--mini-root", type=Path, default=c5.MINI_ROOT)
    args = parser.parse_args()
    if args.smoke:
        args.work_root = args.work_root / "smoke"
        args.mini_root = args.mini_root / "smoke_t1"
        args.draws = 2
        args.sample_limit = 4
    args.families = parse_families(args.families, smoke=args.smoke)
    args.task_ids = task_ids(args.families)
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.mini_root.mkdir(parents=True, exist_ok=True)
    return args


def main() -> None:
    args = configure_args()
    if args.mode in ("preflight", "all"):
        preflight(args)
    if args.mode in ("collect", "all"):
        collect(args)
    if args.mode == "parity":
        preflight(args)
        parity(args)
    if args.mode in ("bootstrap", "all"):
        bootstrap(args)
    if args.mode in ("report", "all"):
        report(args)


if __name__ == "__main__":
    main()
