#!/usr/bin/env python3
"""Measure the off-KD arm with the frozen Cycle-09 R4/R5 geometry protocol.

The implementation deliberately keeps the three numerically distinct tracks apart:

* spectra use the R4 per-checkpoint and frozen-base whitening tracks;
* M2 uses the clean fp32 LoRA B@A update, never merged-minus-base;
* theta uses merged weights but fp64 SVD and fp64 QR, matching R5-A2.

Base reference profiles are loaded directly from the R4 cache, so this script makes
zero base forward calls. Each checkpoint is committed atomically and can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

import cycle09_r4_campaign as camp
import cycle09_r4_common as c4
import cycle09_r5_common as c5

ARM = "offkd"
SCHEMA_VERSION = "cycle09_offkd_geometry_v2_20260716"
OFFKD_ROOT = Path("/root/autodl-tmp/cycle09_offkd")
OFFKD_MERGED = OFFKD_ROOT / "_merged_models"
OFFKD_CKPTS = OFFKD_ROOT / "checkpoints"
OFFKD_BACKFILL = OFFKD_ROOT / "checkpoint_backfill"
REFERENCE_ROOT = c4.RUN_ROOT / "scratch/references"
RUN_ROOT = OFFKD_ROOT / "geometry"
MINI = c4.MINI_ROOT
MAIN_GRID = (0, 5, 10, 20, 40, 160, 624)
EXTENDED_GRID = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
NUMERICAL_BACKFILL_STEPS = {80, 320, 480}
EPSILONS = (0.05, 0.01)
TAIL_RANKS = (32, 64, 128, 256)

FIXED_PROBES = {
    "legacy_S_math": ("legacy_S", "math", "corpora/fixed/legacy_S_math.jsonl", None),
    "E_ood": ("E", "ood", "corpora/fixed/E_ood.jsonl", None),
    "E_general": ("E", "general", "corpora/fixed/E_general.jsonl", None),
    "E_math_hard": ("E", "math_hard", "corpora/fixed/E_math_hard.jsonl", None),
}
SBOS_SEEDS = (3, 17, 31)

SPECTRA_FIELDS = (
    "arm", "step", "task_id", "probe_type", "domain", "generation_seed",
    "n_samples", "track", "layer", "module", "effective_rank",
    "r_eps_005", "r_eps_001", "tail_energy_r32", "sigma_json",
    "measurement_path",
)
M1_FIELDS = (
    "arm", "step", "task_id", "probe_family", "probe_type", "domain",
    "generation_seed", "track", "layer", "module", "epsilon",
    "r_epsilon_base", "r_epsilon_current", "r_epsilon_delta",
    "rank_reduced_vs_base", "drift_core", "core_rank_definition",
    "ec_core_small_threshold", "tail_energy_r32", "tail_energy_r64",
    "tail_energy_r128", "tail_energy_r256",
)
M2_FIELDS = (
    "arm", "step", "task_id", "probe_type", "domain", "generation_seed",
    "n_samples", "layer", "module", "reference", "source_kind",
    "m2_output_drift",
)
THETA_FIELDS = (
    "arm", "step", "probe", "track", "layer", "module", "epsilon",
    "rank_used", "rank_rule", "theta_u_max_deg", "theta_u_mean_deg",
    "theta_v_max_deg", "theta_v_mean_deg", "source_kind",
)


def probe_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for task_id, (probe_type, domain, relative_path, seed) in FIXED_PROBES.items():
        tasks.append(
            {
                "task_id": task_id,
                "probe_type": probe_type,
                "domain": domain,
                "corpus": c4.RUN_ROOT / relative_path,
                "generation_seed": seed,
                "reference": task_id,
            }
        )
    for seed in SBOS_SEEDS:
        task_id = f"S_bos__g{seed}"
        tasks.append(
            {
                "task_id": task_id,
                "probe_type": "S",
                "domain": "bos",
                "corpus": c4.generated_corpus_path(
                    "S", "bos", seed, run_root=c4.RUN_ROOT
                ),
                "generation_seed": seed,
                "reference": task_id,
            }
        )
    return tasks


def offkd_adapter_dir(step: int) -> Path | None:
    if step == 0:
        return None
    direct = OFFKD_CKPTS / f"checkpoint-{step:06d}"
    if (direct / "adapter_model.safetensors").exists():
        return direct
    for parent in sorted(OFFKD_BACKFILL.glob("*/")):
        candidate = parent / f"checkpoint-{step:06d}"
        if (candidate / "adapter_model.safetensors").exists():
            return candidate
    return None


def offkd_model_path(step: int) -> Path:
    if step == 0:
        return c4.BASE_MODEL
    path = OFFKD_MERGED / c4.step_label(step)
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"missing off-KD merged model: {path}")
    return path


def preflight(steps: list[int], tasks: list[dict[str, Any]]) -> dict[int, Path | None]:
    adapters: dict[int, Path | None] = {}
    missing_adapters: list[int] = []
    for step in steps:
        if step == 0:
            adapters[step] = None
            continue
        offkd_model_path(step)
        adapter = offkd_adapter_dir(step)
        adapters[step] = adapter
        if adapter is None:
            missing_adapters.append(step)
    if missing_adapters:
        raise SystemExit(
            "[offkd-geom] STOP: adapter checkpoints missing for steps "
            f"{missing_adapters}; merge-subtract fallback is forbidden"
        )

    missing_inputs = []
    for task in tasks:
        if not task["corpus"].exists():
            missing_inputs.append(str(task["corpus"]))
        reference = REFERENCE_ROOT / f"{task['reference']}.pt"
        if not reference.exists():
            missing_inputs.append(str(reference))
    if missing_inputs:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing_inputs))
    return adapters


def _normalize_layer_keys(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {int(key): item for key, item in value.items()}


def load_reference_profile(
    task: dict[str, Any], expected_samples: int, *, smoke: bool
) -> dict[str, Any]:
    path = REFERENCE_ROOT / f"{task['reference']}.pt"
    profile = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "n_samples", "grams", "residual_samples", "residual_sample_means",
        "residual_second", "residual_mean",
    }
    missing = sorted(required.difference(profile))
    if missing:
        raise ValueError(f"reference profile {path} lacks keys {missing}")
    for key in ("grams", "residual_second", "residual_mean", "position_second", "position_mean"):
        if key in profile:
            profile[key] = _normalize_layer_keys(profile[key])
    for key in ("residual_samples", "residual_sample_means", "sample_factors"):
        if key in profile:
            profile[key] = [_normalize_layer_keys(item) for item in profile[key]]
    if not smoke and int(profile["n_samples"]) != expected_samples:
        raise ValueError(
            f"reference/sample count mismatch for {task['task_id']}: "
            f"cache={profile['n_samples']} current={expected_samples}"
        )
    if len(profile["residual_samples"]) < expected_samples:
        raise ValueError(
            f"reference {path} has only {len(profile['residual_samples'])} residual samples; "
            f"need {expected_samples}"
        )
    return profile


def scales_to_cpu(scales: dict[int, dict[str, torch.Tensor]]) -> dict[int, dict[str, torch.Tensor]]:
    return {
        int(layer): {group: tensor.detach().cpu() for group, tensor in groups.items()}
        for layer, groups in scales.items()
    }


def scales_to_device(
    scales: dict[int, dict[str, torch.Tensor]], device: str
) -> dict[int, dict[str, torch.Tensor]]:
    return {
        int(layer): {
            group: tensor.to(device=device, dtype=torch.float32)
            for group, tensor in groups.items()
        }
        for layer, groups in scales.items()
    }


def protocol_payload(
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
    sample_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "arm": ARM,
        "steps": list(args.steps),
        "tasks": [task["task_id"] for task in tasks],
        "sample_counts": sample_counts,
        "layers": list(c4.LAYERS),
        "modules": list(c4.MODULES),
        "epsilons": list(EPSILONS),
        "fixed_theta_rank": c5.FIXED_RANK_CONTROL,
        "window_seed": c4.WINDOW_SEED,
        "window_tokens": c4.WINDOW_TOKENS,
        "window_k": c4.WINDOW_K,
        "max_context_tokens": c4.MAX_CONTEXT_TOKENS,
        "measurement_n_override": args.measurement_n,
        "smoke": args.smoke,
        "dW": "adapter_BA_fp32_only",
        "theta": "fp64_svd_fp64_qr",
    }


def protocol_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def checkpoint_path(work_root: Path, step: int) -> Path:
    return work_root / "checkpoints" / f"{c4.step_label(step)}.json"


def acquire_checkpoint_lock(work_root: Path, step: int):
    path = work_root / "locks" / f"{c4.step_label(step)}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def write_worker_status(
    args: argparse.Namespace,
    status: str,
    *,
    completed_steps: list[int],
    current_step: int | None = None,
) -> None:
    if not args.worker_only:
        return
    write_json_atomic(
        args.work_root / "workers" / f"{args.worker_id}.json",
        {
            "status": status,
            "worker_id": args.worker_id,
            "pid": os.getpid(),
            "protocol_steps": list(args.steps),
            "worker_steps": list(args.worker_steps),
            "completed_steps": completed_steps,
            "current_step": current_step,
            "shared_outputs_written": False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def expected_step_counts(task_count: int, step: int) -> dict[str, int]:
    cell_count = task_count * len(c4.LAYERS) * len(c4.MODULES)
    return {
        "spectra": cell_count * 2,
        "m2": task_count * len(c4.LAYERS) * (len(c4.MODULES) * 2 + 1),
        "theta": 0 if step == 0 else cell_count * (len(EPSILONS) + 1),
    }


def validate_step_payload(
    payload: dict[str, Any], fingerprint: str, task_count: int, step: int
) -> None:
    if payload.get("protocol_fingerprint") != fingerprint:
        raise ValueError(
            f"stale/incompatible checkpoint cache for step {step}; refusing silent reuse"
        )
    if int(payload.get("step", -1)) != step or payload.get("status") != "complete":
        raise ValueError(f"incomplete checkpoint cache for step {step}")
    expected = expected_step_counts(task_count, step)
    actual = {key: len(payload.get(key, [])) for key in expected}
    if actual != expected:
        raise ValueError(f"row-count mismatch at step {step}: expected={expected}, actual={actual}")


def load_completed_steps(
    work_root: Path,
    steps: list[int],
    accepted_fingerprints: set[str],
    task_count: int,
) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    for step in steps:
        path = checkpoint_path(work_root, step)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        cache_fingerprint = payload.get("protocol_fingerprint")
        if cache_fingerprint not in accepted_fingerprints:
            raise ValueError(
                f"stale/incompatible checkpoint cache for step {step}; "
                "only the main-grid to extended-grid transition is accepted"
            )
        validate_step_payload(payload, cache_fingerprint, task_count, step)
        completed[step] = payload
    return completed


def _assert_finite(rows: list[dict[str, Any]], label: str) -> None:
    for row_index, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite {label}[{row_index}].{key}: {value}")


def spectrum_row(
    *,
    task: dict[str, Any],
    step: int,
    sample_count: int,
    track: str,
    layer: int,
    module: str,
    sigma: list[float],
    measurement_path: Path,
) -> dict[str, Any]:
    return {
        "arm": ARM,
        "step": step,
        "task_id": task["task_id"],
        "probe_type": task["probe_type"],
        "domain": task["domain"],
        "generation_seed": task["generation_seed"],
        "n_samples": sample_count,
        "track": track,
        "layer": layer,
        "module": module,
        "effective_rank": c4.effective_rank(sigma),
        "r_eps_005": c4.functional_rank(sigma, 0.05),
        "r_eps_001": c4.functional_rank(sigma, 0.01),
        "tail_energy_r32": c4.tail_energy(sigma, 32),
        "sigma_json": json.dumps(sigma, separators=(",", ":")),
        "measurement_path": str(measurement_path),
    }


def theta_rows_for_cell(
    *,
    step: int,
    task_id: str,
    layer: int,
    module: str,
    source_kind: str,
    u0_cpu: torch.Tensor,
    v0_cpu: torch.Tensor,
    ut: torch.Tensor,
    vt: torch.Tensor,
    sigma_t: list[float],
    device: str,
) -> list[dict[str, Any]]:
    rank_specs: list[tuple[float | None, int, str]] = []
    for epsilon in EPSILONS:
        rank = c4.functional_rank(sigma_t, epsilon)
        rank = max(1, min(rank, u0_cpu.shape[1], ut.shape[1]))
        tag = f"{epsilon:.2f}".split(".")[1]
        rank_specs.append((epsilon, rank, f"per-cell r_eps ({tag})"))
    fixed_rank = min(c5.FIXED_RANK_CONTROL, u0_cpu.shape[1], ut.shape[1])
    rank_specs.append((None, fixed_rank, f"fixed k={c5.FIXED_RANK_CONTROL} control"))

    max_rank = max(rank for _, rank, _ in rank_specs)
    u0 = u0_cpu[:, :max_rank].to(device=device, dtype=torch.float64)
    v0 = v0_cpu[:, :max_rank].to(device=device, dtype=torch.float64)
    rows: list[dict[str, Any]] = []
    for epsilon, rank, rank_rule in rank_specs:
        theta_u_max, theta_u_mean = c5.principal_angles(u0[:, :rank], ut[:, :rank])
        theta_v_max, theta_v_mean = c5.principal_angles(v0[:, :rank], vt[:, :rank])
        rows.append(
            {
                "arm": ARM,
                "step": step,
                "probe": task_id,
                "track": "frozen_base",
                "layer": layer,
                "module": module,
                "epsilon": epsilon,
                "rank_used": rank,
                "rank_rule": rank_rule,
                "theta_u_max_deg": theta_u_max,
                "theta_u_mean_deg": theta_u_mean,
                "theta_v_max_deg": theta_v_max,
                "theta_v_mean_deg": theta_v_mean,
                "source_kind": source_kind,
            }
        )
    del u0, v0
    return rows


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = args.device
    layers = list(c4.LAYERS)
    tasks = probe_tasks()
    if args.smoke:
        tasks = tasks[:1]
        args.steps = (0, 5)
        if args.worker_only:
            args.worker_steps = (5,)
        args.measurement_n = 4
        args.mini_root = MINI / f"smoke_{ARM}"
        args.work_root = RUN_ROOT / "smoke"
    steps = list(args.steps)
    adapters = preflight(steps, tasks)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(c4.BASE_MODEL))
    samples: dict[str, list[Any]] = {}
    for task in tasks:
        prepared = c4.prepare_samples(
            task["corpus"],
            tokenizer,
            corpus_id=task["task_id"],
            window_seed=c4.WINDOW_SEED,
            max_context_tokens=c4.MAX_CONTEXT_TOKENS,
        )
        if args.measurement_n:
            prepared = prepared[: args.measurement_n]
        if not prepared:
            raise ValueError(f"no prepared samples for {task['task_id']}")
        samples[task["task_id"]] = prepared

    sample_counts = {task_id: len(rows) for task_id, rows in samples.items()}
    protocol = protocol_payload(args, tasks, sample_counts)
    fingerprint = protocol_fingerprint(protocol)
    accepted_fingerprints = {fingerprint}
    if not args.smoke and set(MAIN_GRID).issubset(steps):
        main_protocol = dict(protocol)
        main_protocol["steps"] = list(MAIN_GRID)
        accepted_fingerprints.add(protocol_fingerprint(main_protocol))
    completed = load_completed_steps(
        args.work_root, steps, accepted_fingerprints, len(tasks)
    )
    work_steps = list(args.worker_steps) if args.worker_only else steps
    pending = [step for step in work_steps if step not in completed]
    worker_completed = [step for step in work_steps if step in completed]
    write_worker_status(
        args,
        "running",
        completed_steps=worker_completed,
        current_step=pending[0] if pending else None,
    )
    if not pending:
        if args.worker_only:
            write_worker_status(args, "complete", completed_steps=worker_completed)
            print("[offkd-geom worker] every assigned step already complete", flush=True)
        else:
            print("[offkd-geom] every requested step already complete; rebuilding outputs", flush=True)
            write_outputs(args, completed, tasks, protocol, fingerprint, adapters)
        return

    print(
        f"[offkd-geom] pending={pending}; base forward calls=0; "
        f"checkpoint outer / probe inner",
        flush=True,
    )
    base_model = camp.load_model(c4.BASE_MODEL, device)
    base_scales_cpu: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    base_bases: dict[tuple[str, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    need_theta = any(step != 0 for step in pending)

    # Keep the seven large whitening collections on CPU between probes/steps.
    for task_index, task in enumerate(tasks, start=1):
        task_id = task["task_id"]
        print(f"[offkd-geom] base cache {task_index}/{len(tasks)}: {task_id}", flush=True)
        reference = load_reference_profile(
            task, sample_counts[task_id], smoke=args.smoke
        )
        scale_gpu = camp.scaling_by_group(reference, layers, device)
        if need_theta:
            for layer in layers:
                for module in c4.MODULES:
                    group = c4.MODULE_TO_GROUP[module]
                    w0 = camp.module_at(base_model, layer, module).weight.detach().to(
                        device=device, dtype=torch.float64
                    )
                    m0 = w0 @ scale_gpu[layer][group].double()
                    u0, _s0, vh0 = torch.linalg.svd(m0, full_matrices=False)
                    base_bases[(task_id, layer, module)] = (
                        u0.cpu().contiguous(), vh0.T.cpu().contiguous()
                    )
                    del w0, m0, u0, _s0, vh0
        base_scales_cpu[task_id] = scales_to_cpu(scale_gpu)
        del reference, scale_gpu
        gc.collect()
        torch.cuda.empty_cache()

    for step in pending:
        lock_handle = acquire_checkpoint_lock(args.work_root, step)
        refreshed = load_completed_steps(
            args.work_root, [step], accepted_fingerprints, len(tasks)
        )
        if step in refreshed:
            completed[step] = refreshed[step]
            worker_completed.append(step)
            write_worker_status(
                args,
                "running",
                completed_steps=worker_completed,
                current_step=None,
            )
            lock_handle.close()
            print(f"[offkd-geom] cached after lock step {step}", flush=True)
            continue
        write_worker_status(
            args,
            "running",
            completed_steps=worker_completed,
            current_step=step,
        )
        current_model = base_model if step == 0 else camp.load_model(offkd_model_path(step), device)
        adapter_state = None
        adapter_scale = 0.0
        if step != 0:
            adapter_state = load_file(adapters[step] / "adapter_model.safetensors")
            adapter_scale = camp.adapter_scaling(adapters[step])
        source_kind = "base_identity" if step == 0 else "offkd_clean_fp32_ba"
        spectra_rows: list[dict[str, Any]] = []
        m2_rows: list[dict[str, Any]] = []
        theta_rows: list[dict[str, Any]] = []
        cache_path = checkpoint_path(args.work_root, step)
        print(f"[offkd-geom] step {step}: {len(tasks)} probes", flush=True)
        try:
            for task_index, task in enumerate(tasks, start=1):
                task_id = task["task_id"]
                print(
                    f"[offkd-geom] step {step} probe {task_index}/{len(tasks)}: {task_id}",
                    flush=True,
                )
                reference = load_reference_profile(
                    task, sample_counts[task_id], smoke=args.smoke
                )
                base_scale = scales_to_device(base_scales_cpu[task_id], device)
                if step == 0:
                    current_profile = reference
                    current_scale = base_scale
                else:
                    current_profile = camp.collect_profile(
                        current_model,
                        samples[task_id],
                        layers,
                        device,
                        keep_factors=False,
                        keep_residual_samples=True,
                    )
                    current_scale = camp.scaling_by_group(current_profile, layers, device)

                m2b_by_layer = {
                    row["layer"]: row["m2b_representation_drift"]
                    for row in camp.representation_drift_rows(
                        current_profile, reference, layers
                    )
                }
                for layer in layers:
                    for module in c4.MODULES:
                        group = c4.MODULE_TO_GROUP[module]
                        wt = camp.module_at(current_model, layer, module).weight.detach()
                        x0_scale = base_scale[layer][group]
                        xt_scale = current_scale[layer][group]

                        primary = torch.linalg.svdvals(wt.float() @ xt_scale).cpu().tolist()
                        if step == 0:
                            frozen = primary
                        else:
                            frozen = torch.linalg.svdvals(wt.float() @ x0_scale).cpu().tolist()
                        for track, sigma in (
                            ("per_checkpoint", primary),
                            ("frozen_base", frozen),
                        ):
                            spectra_rows.append(
                                spectrum_row(
                                    task=task,
                                    step=step,
                                    sample_count=sample_counts[task_id],
                                    track=track,
                                    layer=layer,
                                    module=module,
                                    sigma=sigma,
                                    measurement_path=cache_path,
                                )
                            )

                        if step == 0:
                            update = torch.zeros_like(wt, dtype=torch.float32)
                        else:
                            update = camp.sft_delta(
                                adapter_state,
                                adapter_scale,
                                layer,
                                module,
                                device,
                            )
                        w0 = camp.module_at(base_model, layer, module).weight.detach().float()
                        for reference_name, scale in (
                            ("X0_primary", x0_scale),
                            ("Xt_secondary", xt_scale),
                        ):
                            numerator = float(torch.linalg.matrix_norm(update @ scale, "fro"))
                            denominator = float(torch.linalg.matrix_norm(w0 @ scale, "fro"))
                            m2_rows.append(
                                {
                                    "arm": ARM,
                                    "step": step,
                                    "task_id": task_id,
                                    "probe_type": task["probe_type"],
                                    "domain": task["domain"],
                                    "generation_seed": task["generation_seed"],
                                    "n_samples": sample_counts[task_id],
                                    "layer": layer,
                                    "module": module,
                                    "reference": reference_name,
                                    "source_kind": source_kind,
                                    "m2_output_drift": numerator / max(denominator, 1e-30),
                                }
                            )

                        if step != 0:
                            mt = wt.to(device=device, dtype=torch.float64) @ x0_scale.double()
                            ut, st, vht = torch.linalg.svd(mt, full_matrices=False)
                            u0_cpu, v0_cpu = base_bases[(task_id, layer, module)]
                            theta_rows.extend(
                                theta_rows_for_cell(
                                    step=step,
                                    task_id=task_id,
                                    layer=layer,
                                    module=module,
                                    source_kind=source_kind,
                                    u0_cpu=u0_cpu,
                                    v0_cpu=v0_cpu,
                                    ut=ut,
                                    vt=vht.T,
                                    sigma_t=st.cpu().tolist(),
                                    device=device,
                                )
                            )
                            del mt, ut, st, vht
                        del update, w0

                    m2_rows.append(
                        {
                            "arm": ARM,
                            "step": step,
                            "task_id": task_id,
                            "probe_type": task["probe_type"],
                            "domain": task["domain"],
                            "generation_seed": task["generation_seed"],
                            "n_samples": sample_counts[task_id],
                            "layer": layer,
                            "module": "__representation__",
                            "reference": "paired_hidden_states",
                            "source_kind": "same_forward_text",
                            "m2_output_drift": m2b_by_layer[layer],
                        }
                    )

                del current_profile, current_scale, reference, base_scale
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            if step != 0:
                camp.unload_model(current_model)
            if adapter_state is not None:
                adapter_state.clear()
            gc.collect()
            torch.cuda.empty_cache()

        _assert_finite(spectra_rows, "spectra")
        _assert_finite(m2_rows, "m2")
        _assert_finite(theta_rows, "theta")
        step_payload = {
            "status": "complete",
            "step": step,
            "protocol_fingerprint": fingerprint,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "spectra": spectra_rows,
            "m2": m2_rows,
            "theta": theta_rows,
        }
        validate_step_payload(step_payload, fingerprint, len(tasks), step)
        write_json_atomic(cache_path, step_payload)
        completed[step] = step_payload
        worker_completed.append(step)
        lock_handle.close()
        write_worker_status(
            args,
            "running",
            completed_steps=worker_completed,
            current_step=None,
        )
        print(f"[offkd-geom] committed step {step}: {cache_path}", flush=True)

    camp.unload_model(base_model)
    if args.worker_only:
        write_worker_status(args, "complete", completed_steps=worker_completed)
        print(f"[offkd-geom worker] complete steps={worker_completed}", flush=True)
    else:
        write_outputs(args, completed, tasks, protocol, fingerprint, adapters)


def append_arm_rows(path: Path, new_rows: list[dict[str, Any]], fields: tuple[str, ...]) -> int:
    old_rows: list[dict[str, Any]] = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != fields:
                raise ValueError(
                    f"schema mismatch for {path}: expected={fields}, got={reader.fieldnames}"
                )
            old_rows = [row for row in reader if row.get("arm") != ARM]
    for row in new_rows:
        if set(row) != set(fields):
            raise ValueError(
                f"new-row schema mismatch for {path}: missing={set(fields)-set(row)}, "
                f"extra={set(row)-set(fields)}"
            )
    c4.write_csv_atomic(path, [*old_rows, *new_rows], list(fields))
    return len(new_rows)


def write_outputs(
    args: argparse.Namespace,
    completed: dict[int, dict[str, Any]],
    tasks: list[dict[str, Any]],
    protocol: dict[str, Any],
    fingerprint: str,
    adapters: dict[int, Path | None],
) -> None:
    mini = args.mini_root
    mini.mkdir(parents=True, exist_ok=True)
    spectra_rows: list[dict[str, Any]] = []
    m2_rows: list[dict[str, Any]] = []
    theta_rows: list[dict[str, Any]] = []
    for step in args.steps:
        payload = completed[int(step)]
        spectra_rows.extend(payload["spectra"])
        m2_rows.extend(payload["m2"])
        theta_rows.extend(payload["theta"])

    expected = {
        "spectra": sum(expected_step_counts(len(tasks), int(step))["spectra"] for step in args.steps),
        "m2": sum(expected_step_counts(len(tasks), int(step))["m2"] for step in args.steps),
        "theta": sum(expected_step_counts(len(tasks), int(step))["theta"] for step in args.steps),
    }
    actual = {
        "spectra": len(spectra_rows),
        "m2": len(m2_rows),
        "theta": len(theta_rows),
    }
    if actual != expected:
        raise ValueError(f"final row-count mismatch: expected={expected}, actual={actual}")

    c4.write_csv_atomic(
        mini / f"R4_v2_spectra_{ARM}.csv", spectra_rows, list(SPECTRA_FIELDS)
    )

    from cycle09_r4_postprocess import drift_core, family

    base_sigma: dict[tuple[str, str, int, str], list[float]] = {}
    for row in spectra_rows:
        if int(row["step"]) == 0:
            key = (
                family(row["task_id"]),
                row["track"],
                int(row["layer"]),
                row["module"],
            )
            base_sigma[key] = json.loads(row["sigma_json"])
    m1_rows: list[dict[str, Any]] = []
    for row in spectra_rows:
        key = (
            family(row["task_id"]),
            row["track"],
            int(row["layer"]),
            row["module"],
        )
        if key not in base_sigma:
            raise ValueError(f"missing step-0 spectrum for {key}")
        sigma = json.loads(row["sigma_json"])
        baseline = base_sigma[key]
        for epsilon in EPSILONS:
            base_rank = c4.functional_rank(baseline, epsilon)
            current_rank = c4.functional_rank(sigma, epsilon)
            m1_rows.append(
                {
                    "arm": ARM,
                    "step": row["step"],
                    "task_id": row["task_id"],
                    "probe_family": family(row["task_id"], drop_seed=True),
                    "probe_type": row["probe_type"],
                    "domain": row["domain"],
                    "generation_seed": row["generation_seed"],
                    "track": row["track"],
                    "layer": row["layer"],
                    "module": row["module"],
                    "epsilon": epsilon,
                    "r_epsilon_base": base_rank,
                    "r_epsilon_current": current_rank,
                    "r_epsilon_delta": current_rank - base_rank,
                    "rank_reduced_vs_base": current_rank < base_rank,
                    "drift_core": drift_core(sigma, baseline, base_rank),
                    "core_rank_definition": "base_r_epsilon",
                    "ec_core_small_threshold": "not_numerically_preregistered",
                    **{
                        f"tail_energy_r{tail_rank}": c4.tail_energy(sigma, tail_rank)
                        for tail_rank in TAIL_RANKS
                    },
                }
            )

    expected_m1 = expected["spectra"] * len(EPSILONS)
    if len(m1_rows) != expected_m1:
        raise ValueError(f"M1 row-count mismatch: expected={expected_m1}, got={len(m1_rows)}")

    n_m1 = append_arm_rows(mini / "R4_m1_tail_ec.csv", m1_rows, M1_FIELDS)
    n_m2 = append_arm_rows(mini / "R4_m2_output_drift.csv", m2_rows, M2_FIELDS)
    n_theta = append_arm_rows(mini / "R5_theta_reps.csv", theta_rows, THETA_FIELDS)

    manifest = {
        "status": "complete",
        "arm": ARM,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "protocol_fingerprint": fingerprint,
        "steps": list(args.steps),
        "probes": [task["task_id"] for task in tasks],
        "generation_seeds": {
            task["task_id"]: task["generation_seed"] for task in tasks
        },
        "window_seed": c4.WINDOW_SEED,
        "base_forward_calls": 0,
        "base_reference_profiles": {
            task["task_id"]: str(REFERENCE_ROOT / f"{task['reference']}.pt")
            for task in tasks
        },
        "model_iteration_order": "checkpoint_outer_probe_inner",
        "dW_track": "adapter B@A fp32 only; no merged-minus-base",
        "base_whitening": "R4 scratch/references profiles reused directly",
        "theta_numerics": "fp64 SVD plus fp64 QR re-orthonormalization",
        "theta_rank_rules": ["per-cell r_eps (05)", "per-cell r_eps (01)", "fixed k=64 control"],
        "windowing": "R4 window-v2: 512 tokens, k=3, hierarchical sample-equal normalization",
        "rows_expected": {**expected, "m1": expected_m1},
        "rows_written": {
            f"spectra_{ARM}": len(spectra_rows),
            "m1": n_m1,
            "m2": n_m2,
            "theta": n_theta,
        },
        "step_caches": {
            str(step): str(checkpoint_path(args.work_root, int(step)))
            for step in args.steps
        },
        "adapters": {
            str(step): (str(adapters[int(step)]) if int(step) != 0 else "base")
            for step in args.steps
        },
        "checkpoint_provenance": {
            str(step): (
                "numerical_backfill_from_landmark"
                if int(step) in NUMERICAL_BACKFILL_STEPS
                else "formal_landmark"
            )
            for step in args.steps
        },
    }
    write_json_atomic(mini / f"{ARM}_geometry_manifest.json", manifest)
    print(
        f"[offkd-geom] wrote spectra={len(spectra_rows)} m1={n_m1} "
        f"m2={n_m2} theta={n_theta}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", default=",".join(map(str, MAIN_GRID)))
    parser.add_argument("--measurement-n", type=int, default=0)
    parser.add_argument("--mini-root", type=Path, default=MINI)
    parser.add_argument("--work-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker-only", action="store_true")
    parser.add_argument("--worker-steps", default="")
    parser.add_argument("--worker-id", default="gpu_worker")
    args = parser.parse_args()
    args.steps = tuple(int(value) for value in args.steps.split(",") if value)
    args.worker_steps = tuple(
        int(value) for value in args.worker_steps.split(",") if value
    )
    unknown = sorted(set(args.worker_steps).difference(args.steps))
    if unknown:
        parser.error(f"worker steps outside protocol grid: {unknown}")
    if args.worker_only and not args.worker_steps:
        parser.error("--worker-only requires --worker-steps")
    run(args)


if __name__ == "__main__":
    main()
