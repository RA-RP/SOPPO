#!/usr/bin/env python3
"""A5 local-output and A6 finite-zeroing cells on fixed token manifests."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

import cycle09_block3_common as b3
import cycle09_r4_campaign as campaign
import cycle09_stage4_state_displacement as s4


ROOT = s4.ROOT / "readout"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as h:
        tmp = Path(h.name)
        json.dump(value, h, ensure_ascii=False, indent=2, sort_keys=True)
        h.write("\n")
    os.replace(tmp, path)


def target(kind: str, args: argparse.Namespace) -> Path:
    return ROOT / kind / args.model / args.arm / s4.label(args.step) / f"{args.probe}.L{args.layer}.json"


def local_output(args: argparse.Namespace) -> dict[str, Any]:
    if args.step == 0:
        raise ValueError("A5 requires nonzero checkpoint")
    out = target("local_output", args)
    with s4.lock(out):
        cached = s4.read_json(out, {})
        if cached.get("status") == "complete":
            return cached
        basep = s4.load_profile(args.model, "base", 0, args.probe, args.tag)
        curp = s4.load_profile(args.model, args.arm, args.step, args.probe, args.tag)
        if not (basep.get("retains_per_sample") and curp.get("retains_per_sample")):
            raise RuntimeError("A5 needs --retain-samples profiles")
        if basep["sample_ids"] != curp["sample_ids"]:
            raise RuntimeError("fixed token/sample manifest mismatch")
        base = campaign.load_model(s4.model_path(args.model, "base", 0), args.device)
        current = campaign.load_model(s4.model_path(args.model, args.arm, args.step), args.device)
        rows = []
        try:
            for module in s4.SPECS[args.model]["modules"]:
                g = s4.group(module)
                w0 = campaign.module_at(base, args.layer, module).weight.detach().to(args.device, torch.float32)
                wt = campaign.module_at(current, args.layer, module).weight.detach().to(args.device, torch.float32)
                total = {"y0": 0.0, "dy": 0.0, "dw": 0.0, "dh": 0.0, "cross": 0.0}
                for f0, ft in zip(basep["sample_factors"], curp["sample_factors"], strict=True):
                    h0 = f0[args.layer][g].to(args.device, torch.float32)
                    ht = ft[args.layer][g].to(args.device, torch.float32)
                    y0 = w0 @ h0.T
                    dy = wt @ ht.T - y0
                    dw = (wt - w0) @ ((h0 + ht) / 2).T
                    dh = ((wt + w0) / 2) @ (ht - h0).T
                    total["y0"] += float(y0.square().sum())
                    total["dy"] += float(dy.square().sum())
                    total["dw"] += float(dw.square().sum())
                    total["dh"] += float(dh.square().sum())
                    total["cross"] += float((dw * dh).sum())
                denom = max(math.sqrt(total["y0"]), 1e-30)
                rows.append({
                    "model": args.model, "arm": args.arm, "checkpoint": args.step,
                    "probe_name": args.probe, "layer": args.layer, "module": module,
                    "m_Y": math.sqrt(total["dy"]) / denom,
                    "weight_component_norm": math.sqrt(total["dw"]) / denom,
                    "activation_component_norm": math.sqrt(total["dh"]) / denom,
                    "component_cosine": total["cross"] / max(math.sqrt(total["dw"] * total["dh"]), 1e-30),
                    "identity_residual_relative": abs(total["dy"] - total["dw"] - total["dh"] - 2 * total["cross"]) / max(total["dy"], 1e-30),
                    "token_manifest_sha256": b3.sha256_json(basep["sample_ids"]),
                    "delta_w_source": "effective_weight_difference_fp32",
                })
        finally:
            campaign.unload_model(current)
            campaign.unload_model(base)
        payload = {"schema_version": "cycle09_stage4_a5_v1", "status": "complete", "rows": rows, "created_utc": b3.utc_now()}
        atomic_json(out, payload)
        return payload


def zeroing(args: argparse.Namespace) -> dict[str, Any]:
    if args.step == 0:
        raise ValueError("A6 requires nonzero checkpoint")
    out = target("zeroing", args)
    with s4.lock(out):
        cached = s4.read_json(out, {})
        if cached.get("status") == "complete":
            return cached
        samples = s4.samples(args.model, args.probe, args.measurement_n or 2)
        base = campaign.load_model(s4.model_path(args.model, "base", 0), args.device)
        current = campaign.load_model(s4.model_path(args.model, args.arm, args.step), args.device)
        rows = []
        try:
            for module in s4.SPECS[args.model]["modules"]:
                linear = campaign.module_at(current, args.layer, module)
                original = linear.weight.detach().clone()
                base_weight = campaign.module_at(base, args.layer, module).weight.detach().to(args.device, linear.weight.dtype)
                kl = js = margin = eos_delta = 0.0
                try:
                    for sample in samples:
                        ids = sample.input_ids.to(args.device)
                        mask = sample.attention_mask.to(args.device)
                        with torch.no_grad():
                            reference = current(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1].float()
                            linear.weight.copy_(base_weight)
                            edited = current(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1].float()
                            linear.weight.copy_(original)
                        p, q = torch.softmax(reference, -1).clamp_min(1e-30), torch.softmax(edited, -1).clamp_min(1e-30)
                        mid = (p + q) / 2
                        kl += float((p * (p.log() - q.log())).sum())
                        js += float(((p * (p.log() - mid.log())).sum() + (q * (q.log() - mid.log())).sum()) / 2)
                        margin += float((reference.max(-1).values - edited.max(-1).values).abs().mean())
                        eos = int(getattr(current.config, "eos_token_id", -1))
                        if eos >= 0:
                            eos_delta += float((p[:, eos] - q[:, eos]).abs().mean())
                finally:
                    with torch.no_grad():
                        linear.weight.copy_(original)
                rows.append({
                    "model": args.model, "arm": args.arm, "checkpoint": args.step,
                    "probe_name": args.probe, "layer": args.layer, "module": module,
                    "n_samples": len(samples), "kl_full_vocab": kl / len(samples),
                    "js_full_vocab": js / len(samples), "answer_margin_abs_delta": margin / len(samples),
                    "eos_probability_abs_delta": eos_delta / len(samples),
                    "intervention": "replace current module effective weight with base weight",
                })
        finally:
            campaign.unload_model(current)
            campaign.unload_model(base)
        payload = {"schema_version": "cycle09_stage4_a6_v1", "status": "complete", "rows": rows, "created_utc": b3.utc_now()}
        atomic_json(out, payload)
        return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=("local-output", "zeroing"))
    p.add_argument("--model", required=True, choices=tuple(s4.SPECS))
    p.add_argument("--arm", required=True)
    p.add_argument("--step", required=True, type=int)
    p.add_argument("--probe", default="E_general", choices=s4.PROBES)
    p.add_argument("--layer", required=True, type=int)
    p.add_argument("--tag", default="sensitivity")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--measurement-n", type=int, default=0)
    args = p.parse_args()
    value = local_output(args) if args.phase == "local-output" else zeroing(args)
    print(json.dumps({"phase": args.phase, "status": value["status"], "rows": len(value["rows"])}, indent=2))


if __name__ == "__main__":
    main()
