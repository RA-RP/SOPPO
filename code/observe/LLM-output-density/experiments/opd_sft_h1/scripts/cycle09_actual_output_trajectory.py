#!/usr/bin/env python3
"""Cycle 09 O0--O5: actual checkpoint-update and checkpoint-output trajectory.

This is intentionally separate from the superseded B3/B4 Fisher/rollback code.  It
implements only the current handoff: serialized forward-state effective updates,
fixed token IDs, current/fixed activation whitening, and exact selected-token
full-vocabulary KL/NLL.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

import sys

REPO = Path("/root/LLM-output-density")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cycle09_b_lora_proxy as b0  # noqa: E402
import cycle09_block3_common as b3  # noqa: E402
import cycle09_llama_geometry as lgeom  # noqa: E402
import cycle09_llama_model_export as lexport  # noqa: E402
import cycle09_r4_campaign as campaign  # noqa: E402
import cycle09_r4_common as c4  # noqa: E402
import cycle09_stage3_common as qstage  # noqa: E402


ROOT = b3.AUTODL / "cycle09_actual_output_trajectory"
PROFILES = ROOT / "profiles"
BASE_LOGITS = ROOT / "base_logits"
GEOMETRY = ROOT / "geometry_cells"
OUTPUT = ROOT / "output_cells"
AUDIT = ROOT / "audit"
FINAL = ROOT / "final"
MINI = REPO / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"

EPSILONS = (0.01, 0.025, 0.05, 0.10)
PROBES = ("E_general", "E_math", "E_ood", "E_if")
MODULES = tuple(b3.MODULES)
LLAMA_O4_STEPS = (0, 20, 160, 320)
ARM_ORDER = ("opd", "offkd", "sft", "seqkd")
SCHEMA = "cycle09_actual_checkpoint_output_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


@contextmanager
def lock(path: Path):
    import fcntl

    target = path.with_suffix(path.suffix + ".lock")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def label(step: int) -> str:
    return f"step_{int(step):03d}"


def split_csv(value: str, allowed: Iterable[str]) -> tuple[str, ...]:
    known = tuple(allowed)
    if value == "all":
        return known
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result or set(result) - set(known):
        raise ValueError(f"invalid value={value}; allowed={known}")
    return result


def split_ints(value: str, allowed: Iterable[int]) -> tuple[int, ...]:
    known = tuple(int(item) for item in allowed)
    if value == "all":
        return known
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or set(result) - set(known):
        raise ValueError(f"invalid steps={result}; allowed={known}")
    return result


def model_path(model: str, arm: str, step: int) -> Path:
    if model == "llama":
        return lexport.merged_target(arm, step)
    if model == "qwen":
        return qstage.model_path(arm, step)
    raise ValueError(model)


def model_integrity(model: str, path: Path) -> bool:
    return bool(b3.model_check(path)["complete"] if model == "llama" else qstage.model_integrity(path)["complete"])


def group_for(module: str) -> str:
    if module in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
        return "attn_qkv_input"
    if module == "self_attn.o_proj":
        return "attn_o_input"
    if module in ("mlp.gate_proj", "mlp.up_proj"):
        return "mlp_gate_up_input"
    if module == "mlp.down_proj":
        return "mlp_down_input"
    raise ValueError(module)


def layer_for(model: str) -> int:
    return 14 if model == "llama" else 18


def samples_for(model: str, probe: str, n: int) -> list[Any]:
    if model == "llama":
        samples = lgeom.prepare_samples(b3.load_llama_tokenizer(), probe, 0)
    else:
        from transformers import AutoTokenizer
        import cycle09_block3_qwen_probe_geometry as qprobe

        tokenizer = AutoTokenizer.from_pretrained(str(c4.BASE_MODEL), local_files_only=True, trust_remote_code=True)
        samples = qprobe.samples_for(probe, tokenizer, factor_only=False)
    return samples[:n] if n else samples


def profile_path(model: str, arm: str, step: int, probe: str, n: int) -> Path:
    return PROFILES / model / arm / label(step) / f"{probe}.n{n or 'all'}.pt"


def profile_meta(model: str, arm: str, step: int, probe: str, n: int) -> Path:
    return profile_path(model, arm, step, probe, n).with_suffix(".json")


def profile_complete(model: str, arm: str, step: int, probe: str, n: int) -> bool:
    return read_json(profile_meta(model, arm, step, probe, n), {}).get("status") == "complete"


def ensure_profile(model: str, arm: str, step: int, probe: str, n: int, device: str) -> dict[str, Any]:
    report_arm = "base" if step == 0 else arm
    target = profile_path(model, report_arm, step, probe, n)
    metadata = profile_meta(model, report_arm, step, probe, n)
    with lock(target):
        cached = read_json(metadata, {})
        if cached.get("status") == "complete" and target.is_file():
            return torch.load(target, map_location="cpu", weights_only=True)
        fixed_samples = samples_for(model, probe, n)
        model_obj = campaign.load_model(model_path(model, report_arm, step), device)
        try:
            measured = campaign.collect_profile(
                model_obj,
                fixed_samples,
                [layer_for(model)],
                device,
                keep_factors=False,
                keep_residual_samples=False,
                keep_input_sample_means=False,
                factor_layers=(),
                forward_batch_size=8,
                max_batch_tokens=32768,
                early_stop=True,
            )
        finally:
            campaign.unload_model(model_obj)
        sample_ids = [sample.sample_id for sample in fixed_samples]
        value = {
            "schema_version": SCHEMA,
            "status": "complete",
            "model": model,
            "arm": report_arm,
            "checkpoint": step,
            "probe": probe,
            "measurement_n": n,
            "sample_ids": sample_ids,
            "sample_ids_sha256": sha256_json(sample_ids),
            "token_manifest": "fixed external token IDs/sample order/positions/mask from v2 prepared samples",
            "window_protocol": "v2_three_level_equal_sample",
            "grams": measured["grams"],
            "created_utc": utc_now(),
        }
        atomic_torch(target, value)
        atomic_json(metadata, {
            key: value[key] for key in (
                "schema_version", "status", "model", "arm", "checkpoint", "probe", "measurement_n",
                "sample_ids_sha256", "token_manifest", "window_protocol", "created_utc"
            )
        } | {"profile": str(target), "bytes": target.stat().st_size, "sha256": sha256_file(target)})
        return value


def sqrt_psd(gram: torch.Tensor, device: str) -> torch.Tensor:
    vals, vectors = torch.linalg.eigh(((gram + gram.T) / 2).to(device=device, dtype=torch.float64))
    return ((vectors * vals.clamp_min(0).sqrt()) @ vectors.T).to(dtype=torch.float32)


def spectrum(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(matrix.to(dtype=torch.float32))


def rank_from_spectrum(values: torch.Tensor, epsilon: float) -> int:
    energy = values.square()
    total = energy.sum()
    if float(total) <= 0:
        return 0
    return int(torch.searchsorted(energy.cumsum(0), (1.0 - epsilon) * total).item() + 1)


def effective_rank(values: torch.Tensor) -> float:
    energy = values.square()
    total = energy.sum()
    if float(total) <= 0:
        return 0.0
    p = energy / total
    return float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum()))


def tail_share(values: torch.Tensor, k: int = 32) -> float:
    energy = values.square()
    total = energy.sum()
    return float(energy[k:].sum() / total) if float(total) > 0 else 0.0


def direct_ba_delta(arm: str, step: int, layer: int, module: str, device: str) -> torch.Tensor:
    """The audit-only direct LoRA object; never used as the formal output main track."""
    from safetensors.torch import load_file

    adapter = lexport.adapter_target(arm, step)
    info = lexport.validate_adapter(adapter, arm, step)
    tensors = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
    suffix_a = f"layers.{layer}.{module}.lora_A.weight"
    suffix_b = f"layers.{layer}.{module}.lora_B.weight"
    keys_a = [key for key in tensors if key.endswith(suffix_a)]
    keys_b = [key for key in tensors if key.endswith(suffix_b)]
    if len(keys_a) != 1 or len(keys_b) != 1:
        raise RuntimeError(f"adapter tensor unavailable or ambiguous: {arm}/{step}/{module}")
    return (tensors[keys_b[0]].float() @ tensors[keys_a[0]].float()).mul_(
        float(info["alpha"]) / float(info["rank"])
    ).to(device)


def base_logits_path(model: str, probe: str, n: int, selected_cap: int) -> Path:
    suffix = f"n{n or 'all'}.cap{selected_cap or 'all'}"
    return BASE_LOGITS / model / f"{probe}.{suffix}.pt"


def selected_positions(sample: Any, cap: int) -> torch.Tensor:
    positions = torch.nonzero(sample.token_weights > 0, as_tuple=False).flatten()
    positions = positions[positions > 0]
    if cap:
        positions = positions[:cap]
    if not len(positions):
        raise RuntimeError(f"sample has no eligible target positions: {sample.sample_id}")
    return positions.to(dtype=torch.long)


@torch.no_grad()
def selected_logits(model: Any, sample: Any, positions: torch.Tensor, device: str) -> torch.Tensor:
    """Exact full-vocabulary logits only at already-fixed causal target positions."""
    ids = sample.input_ids.to(device)
    attention = sample.attention_mask.to(device)
    # Logit at position t-1 predicts token t. `logits_to_keep` avoids materializing L x V.
    keep = (positions - 1).to(device)
    outputs = model(input_ids=ids, attention_mask=attention, use_cache=False, logits_to_keep=keep)
    logits = outputs.logits[0].detach().to("cpu", dtype=torch.bfloat16).contiguous()
    if logits.shape[0] != len(positions):
        raise RuntimeError(f"selected-logit shape drift: {tuple(logits.shape)} vs positions={len(positions)}")
    return logits


def ensure_base_logits(model: str, probe: str, n: int, selected_cap: int, device: str) -> dict[str, Any]:
    path = base_logits_path(model, probe, n, selected_cap)
    metadata = path.with_suffix(".json")
    with lock(path):
        cached = read_json(metadata, {})
        if cached.get("status") == "complete" and path.is_file():
            return torch.load(path, map_location="cpu", weights_only=True)
        fixed_samples = samples_for(model, probe, n)
        base = campaign.load_model(model_path(model, "base", 0), device)
        try:
            records = []
            for sample in fixed_samples:
                positions = selected_positions(sample, selected_cap)
                records.append({
                    "sample_id": sample.sample_id,
                    "positions": positions.cpu(),
                    "token_weights": sample.token_weights[positions].float().cpu(),
                    "target_ids": sample.input_ids[0, positions].long().cpu(),
                    "logits": selected_logits(base, sample, positions, device),
                })
        finally:
            campaign.unload_model(base)
        value = {
            "schema_version": SCHEMA,
            "status": "complete",
            "model": model,
            "arm": "base",
            "checkpoint": 0,
            "probe": probe,
            "measurement_n": n,
            "selected_token_cap": selected_cap,
            "records": records,
            "sample_ids_sha256": sha256_json([record["sample_id"] for record in records]),
            "logit_storage_dtype": "bf16_exact_forward_logits",
            "log_softmax_dtype": "fp32_at_metric_time",
            "created_utc": utc_now(),
        }
        atomic_torch(path, value)
        atomic_json(metadata, {
            key: value[key] for key in (
                "schema_version", "status", "model", "arm", "checkpoint", "probe", "measurement_n",
                "selected_token_cap", "sample_ids_sha256", "logit_storage_dtype", "log_softmax_dtype", "created_utc"
            )
        } | {"artifact": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        return value


def b2_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.model != "llama" or args.step == 0:
        raise ValueError("B2 smoke currently freezes Llama nonzero adapter landmarks only")
    target = AUDIT / f"b2_smoke_{args.model}_{args.arm}_{label(args.step)}_{args.probe}.json"
    with lock(target):
        cached = read_json(target, {})
        if cached.get("status") == "complete":
            return cached
        base_profile = ensure_profile(args.model, "base", 0, args.probe, args.measurement_n, args.device)
        base = campaign.load_model(model_path(args.model, "base", 0), args.device)
        current = campaign.load_model(model_path(args.model, args.arm, args.step), args.device)
        rows = []
        try:
            for module in ("self_attn.o_proj", "mlp.down_proj"):
                g = group_for(module)
                s0 = sqrt_psd(base_profile["grams"][layer_for(args.model)][g], args.device)
                merged = (
                    campaign.module_at(current, layer_for(args.model), module).weight.detach().to(args.device, torch.float32)
                    - campaign.module_at(base, layer_for(args.model), module).weight.detach().to(args.device, torch.float32)
                )
                direct = direct_ba_delta(args.arm, args.step, layer_for(args.model), module, args.device)
                direct_spectrum, merged_spectrum = spectrum(direct), spectrum(merged)
                for name, matrix, values in (
                    ("direct_BA_from_bf16_factors_fp32_matmul", direct, direct_spectrum),
                    ("serialized_merged_bf16_effective_difference", merged, merged_spectrum),
                ):
                    row = {
                        "model": args.model, "arm": args.arm, "checkpoint": args.step, "probe_name": args.probe,
                        "layer": layer_for(args.model), "module": module, "weight_object": name,
                        "weight_energy": float(matrix.square().sum()),
                        "weight_norm_fro": float(torch.linalg.norm(matrix)),
                        "delta_w_s_energy_fixed_base": float((matrix @ s0).square().sum()),
                        "effective_rank": effective_rank(values), "tail_share_k32": tail_share(values, 32),
                        "numerical_rank": int(torch.linalg.matrix_rank(matrix.float()).item()),
                        "svd_dtype": "fp32_gpu_for_smoke",
                    }
                    for epsilon in EPSILONS:
                        row[f"r_epsilon_{epsilon:g}"] = rank_from_spectrum(values, epsilon)
                    rows.append(row)
                denominator = torch.linalg.norm(direct).clamp_min(1e-30)
                rows.append({
                    "model": args.model, "arm": args.arm, "checkpoint": args.step, "probe_name": args.probe,
                    "layer": layer_for(args.model), "module": module, "weight_object": "dual_track_discrepancy",
                    "e_rel": float(torch.linalg.norm(direct - merged) / denominator),
                    "q_norm": float(torch.linalg.norm(merged) / denominator),
                    "fixed_k": "4;8;16;32",
                    "selection_rule": "formal forward uses serialized merged checkpoint; select merged effective difference independent of readings",
                })
                del direct, merged, s0
                torch.cuda.empty_cache()
        finally:
            campaign.unload_model(current)
            campaign.unload_model(base)
        payload = {
            "schema_version": SCHEMA,
            "status": "complete",
            "scope": "retained B2 direct-BA / serialized-merged and Delta-W-S smoke; no G/readout-gradient",
            "rows": rows,
            "formal_forward_representation": "serialized merged checkpoints",
            "selected_effective_weight_object": "serialized_merged_bf16_effective_difference",
            "selection_reason": "O1 pre-registered source-of-forward rule, not correlation with outputs",
            "created_utc": utc_now(),
        }
        atomic_json(target, payload)
        atomic_json(AUDIT / "o1_effective_weight_selection.json", {
            "schema_version": SCHEMA,
            "status": "complete",
            "model": args.model,
            "selected_effective_weight_object": payload["selected_effective_weight_object"],
            "formal_forward_representation": payload["formal_forward_representation"],
            "selection_rule": payload["selection_reason"],
            "b2_smoke": str(target),
            "created_utc": utc_now(),
        })
        return payload


def previous_step(step: int, checkpoints: tuple[int, ...]) -> int | None:
    ordered = tuple(sorted(checkpoints))
    index = ordered.index(step)
    return ordered[index - 1] if index else None


def geometry_path(model: str, arm: str, step: int, probe: str, n: int) -> Path:
    return GEOMETRY / model / arm / label(step) / f"{probe}.n{n or 'all'}.json"


def geometry_cell(args: argparse.Namespace) -> dict[str, Any]:
    report_arm = "base" if args.step == 0 else args.arm
    target = geometry_path(args.model, report_arm, args.step, args.probe, args.measurement_n)
    with lock(target):
        cached = read_json(target, {})
        if cached.get("status") == "complete":
            return cached
        selected = read_json(AUDIT / "o1_effective_weight_selection.json", {})
        if selected.get("selected_effective_weight_object") != "serialized_merged_bf16_effective_difference":
            raise RuntimeError("O1 selection missing: run B2 smoke before O2/O3")
        checkpoints = LLAMA_O4_STEPS if args.model == "llama" else tuple(qstage.STEPS)
        base_profile = ensure_profile(args.model, "base", 0, args.probe, args.measurement_n, args.device)
        current_profile = ensure_profile(args.model, report_arm, args.step, args.probe, args.measurement_n, args.device)
        if base_profile["sample_ids"] != current_profile["sample_ids"]:
            raise RuntimeError("fixed token manifest mismatch between base and current profile")
        previous = previous_step(args.step, checkpoints)
        previous_profile = None
        if previous is not None:
            previous_arm = "base" if previous == 0 else args.arm
            previous_profile = ensure_profile(args.model, previous_arm, previous, args.probe, args.measurement_n, args.device)
            if previous_profile["sample_ids"] != current_profile["sample_ids"]:
                raise RuntimeError("fixed token manifest mismatch between adjacent profiles")
        base = campaign.load_model(model_path(args.model, "base", 0), args.device)
        current = base if args.step == 0 else campaign.load_model(model_path(args.model, args.arm, args.step), args.device)
        previous_model = None
        if previous not in (None, 0) and previous != args.step:
            previous_model = campaign.load_model(model_path(args.model, args.arm, previous), args.device)
        rows: list[dict[str, Any]] = []
        try:
            for module in MODULES:
                g = group_for(module)
                s_current = sqrt_psd(current_profile["grams"][layer_for(args.model)][g], args.device)
                s_base = sqrt_psd(base_profile["grams"][layer_for(args.model)][g], args.device)
                base_weight = campaign.module_at(base, layer_for(args.model), module).weight.detach().to(args.device, torch.float32)
                current_weight = campaign.module_at(current, layer_for(args.model), module).weight.detach().to(args.device, torch.float32)
                delta = current_weight - base_weight
                cumulative_current = delta @ s_current
                cumulative_fixed = delta @ s_base
                state = current_weight @ s_current
                current_values, fixed_values, state_values = spectrum(cumulative_current), spectrum(cumulative_fixed), spectrum(state)
                common = {
                    "model": args.model, "arm": report_arm, "checkpoint": args.step, "source_checkpoint": 0,
                    "probe_name": args.probe, "layer": layer_for(args.model), "module": module,
                    "weight_object": "serialized_merged_bf16_effective_difference",
                    "factor_storage_dtype": "N/A_forward_state", "merged_storage_dtype": "BF16",
                    "subtraction_dtype": "fp32", "svd_dtype": "fp32_gpu", "sample_count": len(current_profile["sample_ids"]),
                    "sample_ids_sha256": current_profile["sample_ids_sha256"],
                    "window_protocol": "v2_three_level_equal_sample",
                    "raw_weight_energy": float(delta.square().sum()),
                    "raw_weight_norm_fro": float(torch.linalg.norm(delta)),
                    "whitened_update_energy_current": float(cumulative_current.square().sum()),
                    "whitened_update_energy_fixed": float(cumulative_fixed.square().sum()),
                    "activation_exposure_ratio": float(cumulative_current.square().sum() / cumulative_fixed.square().sum().clamp_min(1e-30)),
                    "whitened_update_tail_share_current_k32": tail_share(current_values),
                    "whitened_update_tail_share_fixed_k32": tail_share(fixed_values),
                    "state_tail_share_k32": tail_share(state_values),
                    "state_effective_rank": effective_rank(state_values),
                    "kind": "cumulative",
                }
                for epsilon in EPSILONS:
                    rows.append({
                        **common, "epsilon": epsilon,
                        "whitened_update_rank_current": rank_from_spectrum(current_values, epsilon),
                        "whitened_update_rank_fixed": rank_from_spectrum(fixed_values, epsilon),
                        "state_rank": rank_from_spectrum(state_values, epsilon),
                    })
                if previous is not None:
                    source_model = base if previous == 0 else previous_model
                    assert source_model is not None and previous_profile is not None
                    source_weight = campaign.module_at(source_model, layer_for(args.model), module).weight.detach().to(args.device, torch.float32)
                    step_delta = current_weight - source_weight
                    source_s = sqrt_psd(previous_profile["grams"][layer_for(args.model)][g], args.device)
                    step_matrix = step_delta @ source_s
                    step_values = spectrum(step_matrix)
                    matrix_cosine = float((step_delta * delta).sum() / (torch.linalg.norm(step_delta) * torch.linalg.norm(delta)).clamp_min(1e-30))
                    step_common = {
                        "model": args.model, "arm": report_arm, "checkpoint": args.step, "source_checkpoint": previous,
                        "probe_name": args.probe, "layer": layer_for(args.model), "module": module,
                        "weight_object": "serialized_merged_bf16_effective_difference",
                        "merged_storage_dtype": "BF16", "subtraction_dtype": "fp32", "svd_dtype": "fp32_gpu",
                        "sample_count": len(current_profile["sample_ids"]), "sample_ids_sha256": current_profile["sample_ids_sha256"],
                        "window_protocol": "v2_three_level_equal_sample", "kind": "stepwise",
                        "raw_weight_energy": float(step_delta.square().sum()),
                        "raw_weight_norm_fro": float(torch.linalg.norm(step_delta)),
                        "whitened_update_energy_current": float(step_matrix.square().sum()),
                        "whitened_update_tail_share_current_k32": tail_share(step_values),
                        "matrix_cosine_to_cumulative": matrix_cosine,
                    }
                    for epsilon in EPSILONS:
                        rows.append({
                            **step_common, "epsilon": epsilon,
                            "whitened_update_rank_current": rank_from_spectrum(step_values, epsilon),
                        })
                    del source_s, step_delta, step_matrix
                del s_current, s_base, delta, cumulative_current, cumulative_fixed, state
                torch.cuda.empty_cache()
        finally:
            if previous_model is not None:
                campaign.unload_model(previous_model)
            if current is not base:
                campaign.unload_model(current)
            campaign.unload_model(base)
        payload = {"schema_version": SCHEMA, "status": "complete", "rows": rows, "created_utc": utc_now()}
        atomic_json(target, payload)
        return payload


def output_path(model: str, arm: str, step: int, probe: str, n: int, selected_cap: int) -> Path:
    return OUTPUT / model / arm / label(step) / f"{probe}.n{n or 'all'}.cap{selected_cap or 'all'}.parquet"


def output_meta(model: str, arm: str, step: int, probe: str, n: int, selected_cap: int) -> Path:
    return output_path(model, arm, step, probe, n, selected_cap).with_suffix(".json")


def output_cell(args: argparse.Namespace) -> dict[str, Any]:
    report_arm = "base" if args.step == 0 else args.arm
    target = output_path(args.model, report_arm, args.step, args.probe, args.measurement_n, args.selected_token_cap)
    metadata = output_meta(args.model, report_arm, args.step, args.probe, args.measurement_n, args.selected_token_cap)
    with lock(target):
        cached = read_json(metadata, {})
        if cached.get("status") == "complete" and target.is_file():
            return cached
        checkpoints = LLAMA_O4_STEPS if args.model == "llama" else tuple(qstage.STEPS)
        previous = previous_step(args.step, checkpoints)
        baseline = ensure_base_logits(args.model, args.probe, args.measurement_n, args.selected_token_cap, args.device)
        samples = samples_for(args.model, args.probe, args.measurement_n)
        sample_map = {sample.sample_id: sample for sample in samples}
        if baseline["sample_ids_sha256"] != sha256_json([sample.sample_id for sample in samples]):
            raise RuntimeError("base logits token manifest mismatches current sample list")
        current = campaign.load_model(model_path(args.model, report_arm, args.step), args.device)
        previous_model = None
        if previous not in (None, 0) and previous != args.step:
            previous_model = campaign.load_model(model_path(args.model, args.arm, previous), args.device)
        rows: list[dict[str, Any]] = []
        try:
            for record in baseline["records"]:
                sample = sample_map[record["sample_id"]]
                positions = record["positions"].long()
                base_log = record["logits"].float()
                current_log = selected_logits(current, sample, positions, args.device).float()
                previous_log = base_log if previous in (None, 0) else selected_logits(previous_model, sample, positions, args.device).float()
                logp0 = torch.log_softmax(base_log, dim=-1)
                logpt = torch.log_softmax(current_log, dim=-1)
                logpp = torch.log_softmax(previous_log, dim=-1)
                p0 = logp0.exp()
                pp = logpp.exp()
                cumulative_kl = (p0 * (logp0 - logpt)).sum(dim=-1)
                stepwise_kl = (pp * (logpp - logpt)).sum(dim=-1)
                target_ids = record["target_ids"].long()
                nll0 = -logp0.gather(1, target_ids[:, None]).squeeze(1)
                nllt = -logpt.gather(1, target_ids[:, None]).squeeze(1)
                nllp = -logpp.gather(1, target_ids[:, None]).squeeze(1)
                for index, position in enumerate(positions.tolist()):
                    rows.append({
                        "schema_version": SCHEMA, "model": args.model, "arm": report_arm, "checkpoint": args.step,
                        "source_checkpoint": previous if previous is not None else 0, "probe_name": args.probe,
                        "sample_id": record["sample_id"], "token_position": int(position),
                        "target_token_id": int(target_ids[index]), "token_weight": float(record["token_weights"][index]),
                        "cumulative_kl_base_to_current": float(cumulative_kl[index]),
                        "stepwise_kl_source_to_current": float(stepwise_kl[index]),
                        "nll_base": float(nll0[index]), "nll_current": float(nllt[index]), "nll_source": float(nllp[index]),
                        "delta_nll_cumulative": float(nllt[index] - nll0[index]),
                        "delta_nll_stepwise": float(nllt[index] - nllp[index]),
                        "full_vocabulary": True, "log_softmax_dtype": "fp32",
                        "base_logit_cache": str(base_logits_path(args.model, args.probe, args.measurement_n, args.selected_token_cap)),
                    })
                del current_log, previous_log, logp0, logpt, logpp, p0, pp
        finally:
            if previous_model is not None:
                campaign.unload_model(previous_model)
            campaign.unload_model(current)
        frame = pd.DataFrame(rows)
        atomic_parquet(target, frame)
        payload = {
            "schema_version": SCHEMA, "status": "complete", "rows": len(rows), "samples": len(baseline["records"]),
            "artifact": str(target), "sha256": sha256_file(target), "bytes": target.stat().st_size,
            "base_logit_cache": str(base_logits_path(args.model, args.probe, args.measurement_n, args.selected_token_cap)),
            "full_vocabulary": True, "log_softmax_dtype": "fp32", "created_utc": utc_now(),
        }
        atomic_json(metadata, payload)
        return payload


def o4_cell(args: argparse.Namespace) -> dict[str, Any]:
    geometry = geometry_cell(args)
    output = output_cell(args)
    if not geometry["rows"] or output["rows"] <= 0:
        raise RuntimeError("O4 cell has no geometry or output rows")
    return {"status": "complete", "geometry_rows": len(geometry["rows"]), "output_rows": output["rows"]}


def audit(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for model, arms, checkpoints in (
        ("llama", ARM_ORDER, b3.MEASURED_CHECKPOINTS),
        ("qwen", ARM_ORDER, qstage.STEPS),
    ):
        for arm in arms:
            for step in checkpoints:
                if step == 0:
                    continue
                path = model_path(model, arm, step)
                rows.append({
                    "model": model, "arm": arm, "checkpoint": step, "forward_path": str(path),
                    "forward_representation": "serialized_merged_checkpoint",
                    "forward_complete": model_integrity(model, path),
                    "direct_adapter_available": (lexport.adapter_target(arm, step) / "adapter_model.safetensors").is_file()
                    if model == "llama" else (bool(b0.qwen_adapter_path(arm, step)) if arm in ("opd", "offkd") else False),
                })
    atomic_csv(AUDIT / "o0_forward_artifact_inventory.csv", rows)
    payload = {
        "schema_version": SCHEMA, "status": "complete", "rows": len(rows),
        "current_instruction": "O0: do not launch superseded B3/B4 Fisher, rollback, compression, or new behavior eval",
        "retained_b2": "direct-BA / merged-BF16 and Delta-W-S smoke only; no G/readout-gradient",
        "inventory": str(AUDIT / "o0_forward_artifact_inventory.csv"), "created_utc": utc_now(),
    }
    atomic_json(AUDIT / "o0_audit_manifest.json", payload)
    return payload


def weighted_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "arm", "checkpoint", "source_checkpoint", "probe_name"]
    rows = []
    for values, group in frame.groupby(keys, dropna=False, sort=True):
        weights = group["token_weight"].to_numpy(dtype=np.float64)
        total = float(weights.sum())
        if total <= 0:
            continue
        item = dict(zip(keys, values, strict=True))
        item.update({
            "sample_count": int(group["sample_id"].nunique()), "token_count": int(len(group)),
            "cumulative_kl_base_to_current": float(np.average(group["cumulative_kl_base_to_current"], weights=weights)),
            "stepwise_kl_source_to_current": float(np.average(group["stepwise_kl_source_to_current"], weights=weights)),
            "delta_nll_cumulative": float(np.average(group["delta_nll_cumulative"], weights=weights)),
            "delta_nll_stepwise": float(np.average(group["delta_nll_stepwise"], weights=weights)),
            "nll_base": float(np.average(group["nll_base"], weights=weights)),
            "nll_current": float(np.average(group["nll_current"], weights=weights)),
        })
        rows.append(item)
    return pd.DataFrame(rows)


def finalize(_: argparse.Namespace) -> dict[str, Any]:
    geometry_rows = []
    for path in sorted(GEOMETRY.rglob("*.nall.json")):
        payload = read_json(path, {})
        if payload.get("status") == "complete":
            geometry_rows.extend(payload.get("rows", []))
    cumulative = [row for row in geometry_rows if row.get("kind") == "cumulative"]
    stepwise = [row for row in geometry_rows if row.get("kind") == "stepwise"]
    atomic_csv(FINAL / "actual_update_cumulative_geometry.csv", cumulative)
    atomic_csv(FINAL / "actual_update_stepwise_geometry.csv", stepwise)
    exposure = [{key: row.get(key) for key in (
        "model", "arm", "checkpoint", "probe_name", "layer", "module", "epsilon",
        "whitened_update_energy_current", "whitened_update_energy_fixed", "activation_exposure_ratio"
    )} for row in cumulative]
    atomic_csv(FINAL / "actual_update_current_fixed_exposure.csv", exposure)
    parquet_paths = sorted(OUTPUT.rglob("*.nall.capall.parquet"))
    token_frame = pd.concat([pd.read_parquet(path) for path in parquet_paths], ignore_index=True) if parquet_paths else pd.DataFrame()
    if not token_frame.empty:
        atomic_parquet(FINAL / "actual_checkpoint_token_kl_all.parquet", token_frame)
        cumulative_columns = [column for column in token_frame.columns if column not in (
            "stepwise_kl_source_to_current", "delta_nll_stepwise")]
        stepwise_columns = [column for column in token_frame.columns if column not in (
            "cumulative_kl_base_to_current", "delta_nll_cumulative")]
        atomic_parquet(FINAL / "actual_checkpoint_token_kl_cumulative.parquet", token_frame[cumulative_columns])
        atomic_parquet(FINAL / "actual_checkpoint_token_kl_stepwise.parquet", token_frame[stepwise_columns])
        aggregate = weighted_aggregate(token_frame)
        atomic_csv(FINAL / "actual_checkpoint_nll_trajectory.csv", aggregate.to_dict("records"))
    else:
        aggregate = pd.DataFrame()
        atomic_csv(FINAL / "actual_checkpoint_nll_trajectory.csv", [])
    manifest = {
        "schema_version": SCHEMA, "status": "complete", "geometry_rows": len(geometry_rows),
        "cumulative_geometry_rows": len(cumulative), "stepwise_geometry_rows": len(stepwise),
        "token_rows": int(len(token_frame)), "aggregate_output_rows": int(len(aggregate)),
        "outputs": {
            "cumulative_geometry": str(FINAL / "actual_update_cumulative_geometry.csv"),
            "stepwise_geometry": str(FINAL / "actual_update_stepwise_geometry.csv"),
            "exposure": str(FINAL / "actual_update_current_fixed_exposure.csv"),
            "token_kl": str(FINAL / "actual_checkpoint_token_kl_all.parquet"),
            "token_kl_cumulative": str(FINAL / "actual_checkpoint_token_kl_cumulative.parquet"),
            "token_kl_stepwise": str(FINAL / "actual_checkpoint_token_kl_stepwise.parquet"),
            "nll": str(FINAL / "actual_checkpoint_nll_trajectory.csv"),
        }, "created_utc": utc_now(),
    }
    atomic_json(FINAL / "actual_update_geometry_manifest.json", manifest)
    atomic_json(FINAL / "actual_checkpoint_output_manifest.json", manifest)
    return manifest


def o5(_: argparse.Namespace) -> dict[str, Any]:
    """Run the O5 analysis with formal-only, target-matched geometry/output pairs."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cumulative_path = FINAL / "actual_update_cumulative_geometry.csv"
    stepwise_path = FINAL / "actual_update_stepwise_geometry.csv"
    output_path_ = FINAL / "actual_checkpoint_nll_trajectory.csv"
    if not cumulative_path.is_file() or not stepwise_path.is_file() or not output_path_.is_file():
        raise FileNotFoundError("run finalize before O5")

    merge_base = ["model", "arm", "checkpoint", "probe_name"]
    merge_stepwise = ["model", "arm", "checkpoint", "source_checkpoint", "probe_name"]
    outputs = pd.read_csv(output_path_)
    cumulative = pd.read_csv(cumulative_path)
    cumulative = cumulative[cumulative["epsilon"] == 0.05].copy()
    cumulative["state_rank_current"] = cumulative["state_rank"]
    stepwise = pd.read_csv(stepwise_path)
    stepwise = stepwise[stepwise["epsilon"] == 0.05].copy()
    cumulative_joined = cumulative.merge(outputs, on=merge_base, how="inner", suffixes=("_geometry", "_output"))
    state_keys = merge_base + ["module"]
    state = cumulative[state_keys + ["state_rank"]].rename(columns={"state_rank": "state_rank_current"})
    stepwise = stepwise.merge(state, on=state_keys, how="left")
    stepwise_joined = stepwise.merge(outputs, on=merge_stepwise, how="inner", suffixes=("_geometry", "_output"))

    model_specs = {
        "Model-W": ["raw_weight_energy", "raw_weight_norm_fro"],
        "Model-WS": ["raw_weight_energy", "raw_weight_norm_fro", "whitened_update_energy_current"],
        "Model-WS-fixed-sensitivity": ["raw_weight_energy", "raw_weight_norm_fro", "whitened_update_energy_fixed"],
        "Model-WSR": [
            "raw_weight_energy", "raw_weight_norm_fro", "whitened_update_energy_current",
            "whitened_update_rank_current", "whitened_update_tail_share_current_k32",
        ],
        "Model-State": [
            "raw_weight_energy", "raw_weight_norm_fro", "whitened_update_energy_current",
            "whitened_update_rank_current", "whitened_update_tail_share_current_k32", "state_rank_current",
        ],
    }
    target_frames = (
        ("cumulative", "cumulative_kl_base_to_current", cumulative_joined),
        ("cumulative", "delta_nll_cumulative", cumulative_joined),
        ("stepwise", "stepwise_kl_source_to_current", stepwise_joined),
        ("stepwise", "delta_nll_stepwise", stepwise_joined),
    )
    result_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    monotonic_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260725)

    def fit_oof(local: pd.DataFrame, fields: list[str], target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        complete = np.isfinite(local[target].to_numpy(dtype=float))
        complete &= np.isfinite(local[fields].to_numpy(dtype=float)).all(axis=1)
        if not complete.any():
            return None
        subset = local.loc[complete].copy()
        x = subset[fields].to_numpy(dtype=float)
        y = subset[target].to_numpy(dtype=float)
        groups = subset["checkpoint"].to_numpy()
        if len(subset) < 16 or len(np.unique(groups)) < 3:
            return None
        pred = np.full(len(y), np.nan, dtype=float)
        splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        for train, test in splitter.split(x, y, groups):
            reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            reg.fit(x[train], y[train])
            pred[test] = reg.predict(x[test])
        return y, pred, groups, subset.index.to_numpy()

    def paired_delta(base: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], alt: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> dict[str, float]:
        y0, p0, g0, i0 = base
        y1, p1, g1, i1 = alt
        table0 = pd.DataFrame({"index": i0, "base_error": np.abs(y0 - p0), "group": g0})
        table1 = pd.DataFrame({"index": i1, "alt_error": np.abs(y1 - p1)})
        paired = table0.merge(table1, on="index", how="inner")
        if paired.empty:
            return {}
        group_means = paired.groupby("group", sort=True).apply(lambda x: float((x["base_error"] - x["alt_error"]).mean()), include_groups=False)
        values = group_means.to_numpy(dtype=float)
        draws = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(256)])
        signs = rng.choice(np.array([-1.0, 1.0]), size=(2048, len(values)))
        observed = float(values.mean())
        permutation_p = float((np.abs((signs * values).mean(axis=1)) >= abs(observed)).mean())
        return {
            "delta_vs_model_w_mae": observed,
            "delta_vs_model_w_group_bootstrap_ci_low": float(np.quantile(draws, 0.025)),
            "delta_vs_model_w_group_bootstrap_ci_high": float(np.quantile(draws, 0.975)),
            "delta_vs_model_w_group_sign_permutation_p": permutation_p,
        }

    for geometry_kind, target, joined in target_frames:
        for model in sorted(joined["model"].unique()):
            local = joined[joined["model"] == model].copy()
            oof: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
            row_lookup: dict[str, dict[str, Any]] = {}
            for feature_name, fields in model_specs.items():
                missing = [field for field in fields if field not in local]
                if missing:
                    row = {
                        "model": model, "geometry_kind": geometry_kind, "target": target,
                        "feature_set": feature_name, "status": "NOT_AVAILABLE_REQUIRED_FEATURE_MISSING",
                        "missing_features": ";".join(missing), "rows": int(len(local)),
                        "checkpoint_groups": int(local["checkpoint"].nunique()),
                    }
                    result_rows.append(row)
                    row_lookup[feature_name] = row
                    continue
                fitted = fit_oof(local, fields, target)
                if fitted is None:
                    row = {
                        "model": model, "geometry_kind": geometry_kind, "target": target,
                        "feature_set": feature_name, "status": "DEFERRED_INSUFFICIENT_GROUPED_O4_CELLS",
                        "rows": int(len(local)), "checkpoint_groups": int(local["checkpoint"].nunique()),
                    }
                    result_rows.append(row)
                    row_lookup[feature_name] = row
                    continue
                y, pred, groups, indices = fitted
                valid = np.isfinite(pred)
                row = {
                    "model": model, "geometry_kind": geometry_kind, "target": target,
                    "feature_set": feature_name, "status": "complete", "rows": int(valid.sum()),
                    "checkpoint_groups": int(len(np.unique(groups))),
                    "heldout_mae": float(mean_absolute_error(y[valid], pred[valid])),
                    "heldout_r2": float(r2_score(y[valid], pred[valid])) if valid.sum() > 2 else None,
                    "heldout_log_abs_mae": float(np.mean(np.abs(np.log1p(np.abs(y[valid])) - np.log1p(np.abs(pred[valid]))))),
                    "heldout_spearman_actual_pred": float(pd.Series(y[valid]).corr(pd.Series(pred[valid]), method="spearman")),
                }
                result_rows.append(row)
                row_lookup[feature_name] = row
                oof[feature_name] = fitted
                subset = local.loc[indices]
                for pos, (_, item) in enumerate(subset.iterrows()):
                    prediction_rows.append({
                        "evaluation_protocol": "checkpoint_grouped_cv", "model": model,
                        "geometry_kind": geometry_kind, "target": target, "feature_set": feature_name,
                        "arm": item["arm"], "checkpoint": int(item["checkpoint"]),
                        "source_checkpoint": int(item.get("source_checkpoint", item.get("source_checkpoint_geometry", item.get("source_checkpoint_output", 0)))), "probe_name": item["probe_name"],
                        "module": item["module"], "actual": float(y[pos]), "predicted": float(pred[pos]),
                    })
            if "Model-W" in oof:
                for feature_name, fitted in oof.items():
                    if feature_name != "Model-W":
                        row_lookup[feature_name].update(paired_delta(oof["Model-W"], fitted))

            # O4 has only OPD/off-KD. Preserve the specified offline-arm -> OPD test when feasible.
            for held_out_arm in sorted(a for a in local["arm"].unique() if a != "base"):
                train = local[(local["arm"] != held_out_arm) & (local["arm"] != "base")].copy()
                test = local[local["arm"] == held_out_arm].copy()
                for feature_name, fields in model_specs.items():
                    if any(field not in local for field in fields):
                        continue
                    train_ok = np.isfinite(train[target].to_numpy(dtype=float)) & np.isfinite(train[fields].to_numpy(dtype=float)).all(axis=1)
                    test_ok = np.isfinite(test[target].to_numpy(dtype=float)) & np.isfinite(test[fields].to_numpy(dtype=float)).all(axis=1)
                    if train_ok.sum() < 8 or test_ok.sum() < 1:
                        continue
                    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                    reg.fit(train.loc[train_ok, fields].to_numpy(float), train.loc[train_ok, target].to_numpy(float))
                    estimates = reg.predict(test.loc[test_ok, fields].to_numpy(float))
                    for (_, item), estimate in zip(test.loc[test_ok].iterrows(), estimates, strict=True):
                        prediction_rows.append({
                            "evaluation_protocol": "leave_one_arm_out", "held_out_arm": held_out_arm,
                            "train_arms": ";".join(sorted(train["arm"].unique())), "model": model,
                            "geometry_kind": geometry_kind, "target": target, "feature_set": feature_name,
                            "arm": item["arm"], "checkpoint": int(item["checkpoint"]),
                            "source_checkpoint": int(item.get("source_checkpoint", item.get("source_checkpoint_geometry", item.get("source_checkpoint_output", 0)))), "probe_name": item["probe_name"],
                            "module": item["module"], "actual": float(item[target]), "predicted": float(estimate),
                        })

            for key, group in local[local["arm"] != "base"].groupby(["arm", "probe_name"], sort=True):
                monotonic_rows.append({
                    "model": model, "geometry_kind": geometry_kind, "target": target,
                    "arm": key[0], "probe_name": key[1], "rows": int(len(group)),
                    "checkpoint_groups": int(group["checkpoint"].nunique()), "status": "raw_point_pairs_available",
                    "spearman_energy_target": float(group["whitened_update_energy_current"].corr(group[target], method="spearman")),
                })

    atomic_csv(FINAL / "actual_output_incremental_models.csv", result_rows)
    atomic_csv(FINAL / "actual_output_leave_arm_out_predictions.csv", prediction_rows)
    atomic_csv(FINAL / "actual_output_monotonicity_calibration.csv", monotonic_rows)
    payload = {
        "schema_version": SCHEMA, "status": "complete", "formal_only": True,
        "cumulative_joined_rows": int(len(cumulative_joined)), "stepwise_joined_rows": int(len(stepwise_joined)),
        "output_coverage": "Llama base+OPD+offKD only; O6 all-arm/all-model extension requires Theory GO",
        "outputs": [str(FINAL / name) for name in (
            "actual_output_incremental_models.csv", "actual_output_leave_arm_out_predictions.csv",
            "actual_output_monotonicity_calibration.csv")],
        "created_utc": utc_now(),
    }
    atomic_json(FINAL / "actual_output_incremental_manifest.json", payload)
    return payload

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("audit", "b2-smoke", "o4-cell", "finalize", "o5"))
    parser.add_argument("--model", choices=("llama", "qwen"), default="llama")
    parser.add_argument("--arm", choices=("base", *ARM_ORDER), default="opd")
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--probe", choices=PROBES, default="E_math")
    parser.add_argument("--measurement-n", type=int, default=0)
    parser.add_argument("--selected-token-cap", type=int, default=0,
                        help="Smoke-only cap when nonzero; formal runs must keep it at 0.")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    action = {"audit": audit, "b2-smoke": b2_smoke, "o4-cell": o4_cell, "finalize": finalize, "o5": o5}[args.phase]
    value = action(args)
    print(json.dumps({"phase": args.phase, "status": value.get("status"),
                      "rows": value.get("rows", value.get("geometry_rows")), "output": value.get("artifact")}, indent=2))


if __name__ == "__main__":
    main()
