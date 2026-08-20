#!/usr/bin/env python3
"""Resumable A0-A7 state/displacement/readout cells for Cycle 09."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import torch

import cycle09_block3_common as b3
import cycle09_block3_qwen_probe_geometry as qprobe
import cycle09_llama_geometry as lgeom
import cycle09_llama_model_export as lexport
import cycle09_r4_campaign as campaign
import cycle09_r4_common as c4
import cycle09_stage3_common as qstage

ROOT = b3.AUTODL / "cycle09_stage4_state_displacement"
PROFILES = ROOT / "profiles"
CELLS = ROOT / "cells"
OUT = ROOT / "outputs"
AUDIT = ROOT / "audit"
EPS = (0.01, 0.025, 0.05, 0.10)
PROBES = ("E_general", "E_math", "E_ood", "E_if")
SPECS = {
    "qwen": {"layer": 18, "layers": (9, 18, 27), "arms": qstage.ARMS,
             "steps": qstage.STEPS, "modules": qstage.MODULES},
    "llama": {"layer": 14, "layers": (7, 14, 21), "arms": b3.ARMS,
              "steps": b3.MEASURED_CHECKPOINTS, "modules": b3.MODULES},
}


def now() -> str:
    return b3.utc_now()


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as h:
        tmp = Path(h.name)
        json.dump(value, h, ensure_ascii=False, indent=2, sort_keys=True)
        h.write("\n")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}) if rows else []
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as h:
        tmp = Path(h.name)
        writer = csv.DictWriter(h, fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def label(step: int) -> str:
    return f"step_{step:03d}"


def split_names(value: str, allowed: Iterable[str]) -> tuple[str, ...]:
    known = tuple(allowed)
    if value.lower() == "all":
        return known
    found = tuple(x.strip() for x in value.split(",") if x.strip())
    if not found or set(found) - set(known):
        raise ValueError(f"invalid names={found}; allowed={known}")
    return found


def split_steps(value: str, allowed: Iterable[int]) -> tuple[int, ...]:
    known = tuple(int(x) for x in allowed)
    if value.lower() == "all":
        return known
    found = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not found or set(found) - set(known):
        raise ValueError(f"invalid steps={found}; allowed={known}")
    return found


@contextmanager
def lock(path: Path):
    import fcntl
    p = path.with_suffix(path.suffix + ".lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def model_path(model: str, arm: str, step: int, *, materialize: bool = True) -> Path:
    if step == 0:
        return qstage.BASE_MODEL if model == "qwen" else b3.LLAMA_STUDENT_RUNTIME
    if model == "qwen":
        return qstage.require_model(arm, step)
    path = lexport.merged_target(arm, step)
    if b3.model_check(path)["complete"]:
        return path
    if not materialize:
        raise FileNotFoundError(f"missing merged Llama model: {path}")
    # The adapter is the source of truth.  Export once, atomically, then reuse the merged runtime model.
    adapter = lexport.adapter_target(arm, step)
    lexport.validate_adapter(adapter, arm, step)
    merged = lexport.merge(arm, step, adapter)
    if not b3.model_check(merged)["complete"]:
        raise RuntimeError(f"Llama adapter merge did not produce a complete model: {merged}")
    return merged


def samples(model: str, probe: str, n: int) -> list[Any]:
    if model == "qwen":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(c4.BASE_MODEL), local_files_only=True, trust_remote_code=True, use_fast=True
        )
        rows = qprobe.samples_for(probe, tokenizer, factor_only=False)
    else:
        rows = lgeom.prepare_samples(b3.load_llama_tokenizer(), probe, 0)
    return rows[:n] if n else rows


def profile_file(model: str, arm: str, step: int, probe: str, tag: str) -> Path:
    return PROFILES / model / arm / label(step) / f"{probe}.{tag}.pt"


def profile_meta(model: str, arm: str, step: int, probe: str, tag: str) -> Path:
    return profile_file(model, arm, step, probe, tag).with_suffix(".json")


def cell_file(model: str, arm: str, step: int, probe: str, layer: int, tag: str) -> Path:
    return CELLS / model / arm / label(step) / f"{probe}.L{layer}.{tag}.json"


def direction_file(model: str, arm: str, step: int, probe: str, layer: int, tag: str) -> Path:
    return cell_file(model, arm, step, probe, layer, tag).with_suffix(".direction.pt")


def group(module: str) -> str:
    if module in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
        return "attn_qkv_input"
    if module == "self_attn.o_proj":
        return "attn_o_input"
    if module in ("mlp.gate_proj", "mlp.up_proj"):
        return "mlp_gate_up_input"
    if module == "mlp.down_proj":
        return "mlp_down_input"
    raise ValueError(module)


def square_root(gram: torch.Tensor, device: str) -> torch.Tensor:
    vals, vecs = torch.linalg.eigh(((gram + gram.T) / 2).to(device, torch.float64))
    return ((vecs * vals.clamp_min(0).sqrt()) @ vecs.T).float()


def rank(matrix: torch.Tensor, epsilon: float) -> int:
    s = torch.linalg.svdvals(matrix.float())
    energy = s.square()
    if float(energy.sum()) == 0:
        return 0
    return int(torch.searchsorted(energy.cumsum(0), (1 - epsilon) * energy.sum()).item() + 1)


def effective_rank(matrix: torch.Tensor) -> float:
    e = torch.linalg.svdvals(matrix.float()).square()
    if float(e.sum()) == 0:
        return 0.0
    p = e / e.sum()
    return float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum()))


def profile(args: argparse.Namespace) -> dict[str, Any]:
    arm = "base" if args.step == 0 else args.arm
    target = profile_file(args.model, arm, args.step, args.probe, args.tag)
    meta = profile_meta(args.model, arm, args.step, args.probe, args.tag)
    with lock(target):
        cached = read_json(meta, {})
        if cached.get("status") == "complete" and target.is_file():
            return cached
        layers = split_steps(args.layers, SPECS[args.model]["layers"])
        free_bytes = shutil.disk_usage(b3.AUTODL).free
        required_free = 60 << 30 if args.retain_samples else 30 << 30
        if free_bytes < required_free:
            raise RuntimeError(
                f"disk guard: free={free_bytes} bytes, required>={required_free}; "
                "do not start another profile until space is reclaimed"
            )
        batch = samples(args.model, args.probe, args.measurement_n)
        model = campaign.load_model(
            model_path(args.model, arm, args.step, materialize=True), args.device
        )
        try:
            measured = campaign.collect_profile(
                model, batch, list(layers), args.device,
                keep_factors=args.retain_samples, keep_residual_samples=args.retain_samples,
                keep_input_sample_means=True, factor_layers=layers,
                forward_batch_size=args.forward_batch_size, max_batch_tokens=args.max_batch_tokens,
                early_stop=True,
            )
        finally:
            campaign.unload_model(model)
        ids = [x.sample_id for x in batch]
        payload = {
            "schema_version": "cycle09_stage4_profile_v1", "status": "complete",
            "model": args.model, "arm": arm, "step": args.step, "probe": args.probe,
            "layers": list(layers), "n_samples": len(batch), "sample_ids": ids,
            "sample_ids_sha256": b3.sha256_json(ids), "grams": measured["grams"],
            "input_sample_means": measured["input_sample_means"],
            "residual_second": measured["residual_second"], "residual_mean": measured["residual_mean"],
            "forward_execution": measured["forward_execution"], "retains_per_sample": args.retain_samples,
            "created_utc": now(),
        }
        if args.retain_samples:
            payload["sample_factors"] = measured["sample_factors"]
            payload["residual_samples"] = measured["residual_samples"]
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, target)
        result = {k: v for k, v in payload.items() if k not in {
            "grams", "input_sample_means", "residual_second", "residual_mean",
            "sample_factors", "residual_samples"}}
        result.update({"profile": str(target), "bytes": target.stat().st_size, "sha256": digest(target)})
        atomic_json(meta, result)
        return result


def load_profile(model: str, arm: str, step: int, probe: str, tag: str) -> dict[str, Any]:
    path = profile_file(model, arm, step, probe, tag)
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _adapter_ba_delta(arm: str, step: int, layer: int, module: str, device: str) -> torch.Tensor:
    """Load the PEFT update directly, avoiding a merged-minus-base artifact."""
    if step == 0:
        raise ValueError("base checkpoint has no nonzero adapter delta")
    from safetensors.torch import load_file

    adapter = lexport.adapter_target(arm, step)
    info = lexport.validate_adapter(adapter, arm, step)
    weights = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
    suffix_a = f"layers.{layer}.{module}.lora_A.weight"
    suffix_b = f"layers.{layer}.{module}.lora_B.weight"
    keys_a = [key for key in weights if key.endswith(suffix_a)]
    keys_b = [key for key in weights if key.endswith(suffix_b)]
    if len(keys_a) != 1 or len(keys_b) != 1:
        raise RuntimeError(
            f"adapter key mismatch arm={arm} step={step} layer={layer} module={module}: "
            f"A={keys_a}, B={keys_b}"
        )
    scale = float(info["alpha"]) / float(info["rank"])
    return (weights[keys_b[0]].float() @ weights[keys_a[0]].float()).mul_(scale).to(device)


def effective_delta(
    args: argparse.Namespace, current: Any, base: Any, layer: int, module: str
) -> tuple[torch.Tensor, str]:
    """Use adapter BA when available; make the historic Qwen fallback explicit."""
    if args.step == 0:
        return torch.zeros_like(
            campaign.module_at(base, layer, module).weight.detach(),
            dtype=torch.float32,
            device=args.device,
        ), "base_zero_update"
    if args.model == "llama":
        return _adapter_ba_delta(args.arm, args.step, layer, module, args.device), "adapter_BA_fp32"
    if not args.allow_effective_weight_diff:
        raise RuntimeError(
            "Qwen adapter BA is unavailable in this historical trajectory. "
            "Pass --allow-effective-weight-diff only for the documented legacy exception."
        )
    return (
        campaign.module_at(current, layer, module).weight.detach().to(args.device, torch.float32)
        - campaign.module_at(base, layer, module).weight.detach().to(args.device, torch.float32),
        "effective_weight_difference_fp32_authorized_legacy_exception",
    )


def metric(args: argparse.Namespace, centered: bool = False) -> dict[str, Any]:
    arm = "base" if args.step == 0 else args.arm
    suffix = args.tag + (".centered" if centered else "")
    target = cell_file(args.model, arm, args.step, args.probe, args.layer, suffix)
    with lock(target):
        cached = read_json(target, {})
        if cached.get("status") == "complete":
            return cached
        curp = load_profile(args.model, arm, args.step, args.probe, args.tag)
        basep = load_profile(args.model, "base", 0, args.probe, args.tag)
        current = campaign.load_model(model_path(args.model, arm, args.step), args.device)
        base = campaign.load_model(model_path(args.model, "base", 0), args.device)
        rows, dirs = [], {}
        try:
            for module in SPECS[args.model]["modules"]:
                g = group(module)
                gcur, gbase = curp["grams"][args.layer][g], basep["grams"][args.layer][g]
                if centered:
                    means0 = torch.stack([x[args.layer][g].float() for x in basep["input_sample_means"]]).mean(0)
                    meanst = torch.stack([x[args.layer][g].float() for x in curp["input_sample_means"]]).mean(0)
                    gbase = gbase - torch.outer(means0, means0)
                    gcur = gcur - torch.outer(meanst, meanst)
                s0, st = square_root(gbase, args.device), square_root(gcur, args.device)
                w0 = campaign.module_at(base, args.layer, module).weight.detach().to(args.device, torch.float32)
                dw, delta_source = effective_delta(args, current, base, args.layer, module)
                wt = (
                    w0 + dw if delta_source == "adapter_BA_fp32" else
                    campaign.module_at(current, args.layer, module).weight.detach().to(args.device, torch.float32)
                )
                state, disp, denom = wt @ st, dw @ s0, torch.linalg.norm(w0 @ s0).clamp_min(1e-30)
                u, sing, vh = torch.linalg.svd(disp, full_matrices=False)
                keep = min(128, sing.numel())
                dirs[module] = {"u": u[:, :keep].cpu(), "v": vh[:keep].T.cpu(), "singular": sing[:keep].cpu()}
                for eps in EPS:
                    rows.append({
                        "model": args.model, "arm": arm, "checkpoint": args.step, "probe_name": args.probe,
                        "layer": args.layer, "module": module, "epsilon": eps, "centered": centered,
                        "support_ruler": "fixed_base", "sample_count": curp["n_samples"],
                        "second_moment_type": "centered_covariance" if centered else "uncentered_second_moment",
                        "square_root_method": "fp64_eigh_clamp_nonnegative_then_fp32",
                        "weight_arithmetic": "fp32",
                        "state_rank": rank(state, eps), "displacement_rank": rank(disp, eps) if args.step else None,
                        "displacement_rank_normalized": rank(disp, eps) / min(disp.shape) if args.step else None,
                        "displacement_norm_raw": float(torch.linalg.norm(disp)),
                        "displacement_norm_denominator": float(denom),
                        "displacement_norm_normalized": float(torch.linalg.norm(disp) / denom),
                        "weight_norm_fro": float(torch.linalg.norm(dw)),
                        "weight_effective_rank": effective_rank(dw),
                        "delta_w_source": delta_source,
                    })
                del s0, st, wt, w0, dw, state, disp
                torch.cuda.empty_cache()
        finally:
            campaign.unload_model(current)
            campaign.unload_model(base)
        dp = direction_file(args.model, arm, args.step, args.probe, args.layer, suffix)
        torch.save(dirs, dp)
        result = {"schema_version": "cycle09_stage4_metric_v1", "status": "complete", "rows": rows,
                  "direction": str(dp), "created_utc": now()}
        atomic_json(target, result)
        return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for model in split_names(args.models, SPECS):
        for arm in ("base", *SPECS[model]["arms"]):
            for step in SPECS[model]["steps"]:
                if (step == 0) != (arm == "base"):
                    continue
                try:
                    path = model_path(model, arm, step, materialize=False)
                    ok = (qstage.model_integrity(path) if model == "qwen" else b3.model_check(path))["complete"]
                    status = "DONE_EXISTING" if ok else "NOT_FOUND"
                except Exception:
                    if model == "llama" and arm != "base" and step:
                        try:
                            lexport.validate_adapter(lexport.adapter_target(arm, step), arm, step)
                            path, ok = lexport.merged_target(arm, step), True
                            status = "MATERIALIZABLE_FROM_VALIDATED_ADAPTER"
                        except Exception:
                            path, ok, status = Path(""), False, "NOT_FOUND"
                    else:
                        path, ok, status = Path(""), False, "NOT_FOUND"
                for probe in PROBES:
                    rows.append({
                        "model": model, "arm": arm, "checkpoint": step, "probe_name": probe,
                        "model_status": status,
                        "new_profile_status": "NOT_FOUND",
                        "existing_base_reference": "DONE_EXISTING" if (
                            (model == "qwen" and qprobe.reference_path(probe).is_file()) or
                            (model == "llama" and lgeom.reference_path(probe, False).is_file())
                        ) else "NOT_FOUND", "model_path": str(path),
                    })
    output = AUDIT / "state_displacement_missing_inventory.csv"
    atomic_csv(output, rows)
    factor_inventory = []
    for model in split_names(args.models, SPECS):
        for probe in PROBES:
            existing = [
                str(path) for path in PROFILES.glob(f"{model}/**/{probe}.main.json")
                if read_json(path, {}).get("status") == "complete"
            ]
            factor_inventory.append({
                "model": model,
                "probe_name": probe,
                "current_state_profile_status": "DONE_EXISTING" if existing else "NOT_FOUND",
                "centered_per_sample_means_status": "DONE_EXISTING" if existing else "NOT_FOUND",
                "required_action": "verify_existing_profile" if existing else "generation_free_forward_required",
                "profile_meta_paths": existing,
            })
    atomic_json(AUDIT / "state_displacement_missing_inventory.json", {
        "schema_version": "cycle09_stage4_factor_inventory_v1",
        "status": "complete",
        "meaning": "A missing factor is a current-domain hidden-input second moment or per-sample mean; old spectra cannot reconstruct it.",
        "inventory": factor_inventory,
        "created_utc": now(),
    })
    payload = {
        "schema_version": "cycle09_stage4_a0_v1", "status": "complete",
        "task": "A0 artifact audit", "rows": len(rows), "output": str(output),
        "legacy_factor_roots": [str(b3.AUTODL / x) for x in (
            "cycle09_block3/qwen_alpha05/geometry/factors", "cycle09_t1/factors",
            "cycle09_block3/llama_geometry/scratch/references")],
        "created_utc": now(),
    }
    atomic_json(AUDIT / "state_displacement_audit_manifest.json", payload)
    return payload


def pairs(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for left, right in (("opd", "offkd"), ("offkd", "seqkd"), ("opd", "sft")):
        for step in split_steps(args.steps, SPECS[args.model]["steps"]):
            if step == 0:
                continue
            for probe in PROBES:
                lp = direction_file(args.model, left, step, probe, args.layer, args.tag)
                rp = direction_file(args.model, right, step, probe, args.layer, args.tag)
                if not lp.is_file() or not rp.is_file():
                    continue
                a, b = torch.load(lp, map_location="cpu", weights_only=True), torch.load(rp, map_location="cpu", weights_only=True)
                for module in SPECS[args.model]["modules"]:
                    ku = min(a[module]["u"].shape[1], b[module]["u"].shape[1])
                    kv = min(a[module]["v"].shape[1], b[module]["v"].shape[1])
                    su = torch.linalg.svdvals(a[module]["u"][:, :ku].T @ b[module]["u"][:, :ku]).clamp(0, 1)
                    sv = torch.linalg.svdvals(a[module]["v"][:, :kv].T @ b[module]["v"][:, :kv]).clamp(0, 1)
                    rows.append({"model": args.model, "left_arm": left, "right_arm": right,
                                 "checkpoint": step, "probe_name": probe, "layer": args.layer, "module": module,
                                 "theta_u_deg": float(torch.rad2deg(torch.acos(su)).mean()),
                                 "theta_v_deg": float(torch.rad2deg(torch.acos(sv)).mean()),
                                 "overlap_u": float(su.square().mean()), "overlap_v": float(sv.square().mean())})
    output = OUT / f"{args.model}_displacement_direction_full_cells.csv"
    atomic_csv(output, rows)
    return {"status": "complete", "output": str(output), "rows": len(rows)}


def protocol_audit(_: argparse.Namespace) -> dict[str, Any]:
    paths = [b3.REPO / "experiments/opd_sft_h1/scripts/run_cycle09_frozen_self.sh",
             b3.REPO / "experiments/opd_sft_h1/scripts/cycle09_llama_behavior.py"]
    out = OUT / "trainer_arm_implementation_audit.json"
    atomic_json(out, {"status": "partial", "files": [
        {"path": str(p), "exists": p.is_file(), "sha256": digest(p) if p.is_file() else None} for p in paths
    ], "created_utc": now()})
    return {"status": "complete", "output": str(out)}


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for p in CELLS.rglob(f"*.{args.tag}.json"):
        payload = read_json(p, {})
        if payload.get("status") == "complete":
            rows.extend(payload.get("rows", []))
    for name in ("state_rank_full_cells.csv", "fixed_support_displacement_full_cells.csv"):
        atomic_csv(OUT / name, rows)
    result = {"status": "complete", "rows": len(rows), "output": str(OUT / "state_rank_full_cells.csv")}
    atomic_json(OUT / "state_displacement_manifest.json", result)
    return result


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=("audit", "profile", "metric", "centered", "pairs", "protocol-audit", "finalize"))
    p.add_argument("--model", choices=tuple(SPECS), default="qwen")
    p.add_argument("--models", default="all")
    p.add_argument("--arm", default="opd")
    p.add_argument("--step", type=int, default=20)
    p.add_argument("--steps", default="all")
    p.add_argument("--probe", choices=PROBES, default="E_general")
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--layers", default="18")
    p.add_argument("--tag", default="main")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--measurement-n", type=int, default=0)
    p.add_argument("--retain-samples", action="store_true")
    p.add_argument("--forward-batch-size", type=int, default=4)
    p.add_argument("--max-batch-tokens", type=int, default=16384)
    p.add_argument("--allow-effective-weight-diff", action="store_true",
                   help="documented legacy Qwen fallback when adapter BA artifacts no longer exist")
    return p.parse_args()


def main() -> None:
    args = arguments()
    if args.phase == "audit":
        value = audit(args)
    elif args.phase == "profile":
        value = profile(args)
    elif args.phase == "metric":
        value = metric(args)
    elif args.phase == "centered":
        value = metric(args, True)
    elif args.phase == "pairs":
        value = pairs(args)
    elif args.phase == "protocol-audit":
        value = protocol_audit(args)
    else:
        value = finalize(args)
    print(json.dumps({"phase": args.phase, "status": value.get("status"), "output": value.get("output")}, indent=2))


if __name__ == "__main__":
    main()
