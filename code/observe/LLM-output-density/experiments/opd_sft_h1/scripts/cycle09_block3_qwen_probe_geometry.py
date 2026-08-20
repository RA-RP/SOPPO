#!/usr/bin/env python3
"""Prepare and measure the new domain-matched Qwen fixed probes.

Each GPU invocation owns exactly one arm/checkpoint/probe cell. Step zero is
computed once under the canonical base arm and expanded only in the final
table, where shared_base_compute makes that provenance explicit.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

import cycle09_block3_common as block3
import cycle09_r4_campaign as campaign
import cycle09_r4_common as c4
import cycle09_stage3_common as s3


ROOT = block3.RUN_ROOT / "qwen_domain_probes"
CORPUS_ROOT = ROOT / "corpora"
CELL_ROOT = ROOT / "cells"
REFERENCE_ROOT = ROOT / "references"
FACTOR_ROOT = ROOT / "factors"
PROBE_MANIFEST = CORPUS_ROOT / "probe_manifest.json"
R2_OUTPUT = ROOT / "qwen_emath_emathhardv2_r_epsilon.csv"
R2_MANIFEST = ROOT / "qwen_domain_probe_geometry_manifest.json"

LAYER = 18
EPSILONS = (0.01, 0.025, 0.05, 0.10)
R2_PROBES = ("E_math", "E_math_hard_v2")
ALL_PROBES = (
    "S_math",
    "E_math",
    "E_math_hard_v2",
    "E_ood",
    "E_if",
    "E_general",
)
CORE_STEPS = (0, 20, 40, 624)
ANALYSIS_CAPS = {
    "S_math": 32,
    "E_math": 32,
    "E_math_hard_v2": 30,
    "E_ood": 128,
    "E_if": 128,
    "E_general": 32,
}
SOURCE_PATHS = {
    "S_math": c4.RUN_ROOT / "corpora/fixed/legacy_S_math.jsonl",
    "E_math": block3.RUN_ROOT / "llama_geometry/corpora/E_math.jsonl",
    "E_math_hard_v2": block3.RUN_ROOT / "llama_geometry/corpora/E_math_hard_v2.jsonl",
    "E_ood": c4.RUN_ROOT / "corpora/fixed/E_ood.jsonl",
    "E_if": s3.RUN_ROOT / "c5_eif/corpus/E_if.jsonl",
    "E_general": c4.RUN_ROOT / "corpora/fixed/E_general.jsonl",
}
EXPECTED_ROWS = {
    "S_math": 32,
    "E_math": 32,
    "E_math_hard_v2": 30,
    "E_ood": 128,
    "E_if": 541,
    "E_general": 128,
}


def text_from(row: dict[str, Any]) -> str:
    for key in ("generation_text", "text", "question", "problem", "prompt"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    output = row.get("output")
    if isinstance(output, dict) and isinstance(output.get("text"), str):
        return str(output["text"]).strip()
    raise KeyError("probe row has no supported text field")


def corpus_path(probe: str) -> Path:
    return CORPUS_ROOT / f"{probe}.jsonl"


def load_qwen_tokenizer() -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        str(c4.BASE_MODEL), local_files_only=True, trust_remote_code=True, use_fast=True
    )


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_corpora() -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        c4.BASE_MODEL, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    outputs = []
    details = {}
    for probe in ALL_PROBES:
        source = SOURCE_PATHS[probe]
        if not source.is_file():
            raise FileNotFoundError(
                f"{probe} source missing: {source}; E_math/AIME25 are prepared by "
                "cycle09_llama_probe_prepare.py --phase fixed"
            )
        source_rows = c4.read_jsonl(source)
        expected = EXPECTED_ROWS[probe]
        if len(source_rows) < expected:
            raise RuntimeError(f"{probe} source rows={len(source_rows)} expected>={expected}")
        source_rows = source_rows[:expected]
        rows = []
        for index, source_row in enumerate(source_rows):
            text = text_from(source_row)
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not token_ids:
                raise RuntimeError(f"empty tokenized probe row: {probe}/{index}")
            rows.append(
                {
                    "sample_id": f"{probe}_{index:03d}",
                    "probe_type": "S" if probe == "S_math" else "E",
                    "domain": probe,
                    "source_kind": f"qwen_retokenized:{source}",
                    "prompt_text": "",
                    "generation_text": text,
                    "prompt_token_ids": [],
                    "generation_token_ids": list(map(int, token_ids)),
                    "full_token_ids": list(map(int, token_ids)),
                    "eligible_start": 0,
                    "eligible_end": len(token_ids),
                    "text_sha256": text_sha256(text),
                }
            )
        target = corpus_path(probe)
        block3.atomic_jsonl(target, rows)
        outputs.append(block3.artifact(target))
        details[probe] = {
            "source": block3.artifact(source),
            "rows": len(rows),
            "sample_ids_sha256": block3.sha256_json(
                [row["sample_id"] for row in rows]
            ),
            "text_sha256": block3.sha256_json(
                [row["text_sha256"] for row in rows]
            ),
            "analysis_cap": ANALYSIS_CAPS[probe],
        }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "task": "Cycle09 block3 Qwen domain-matched fixed probe preparation",
        "tokenizer": str(c4.BASE_MODEL),
        "window_protocol": {
            "window_seed": c4.WINDOW_SEED,
            "window_tokens": c4.WINDOW_TOKENS,
            "window_k": c4.WINDOW_K,
            "normalization": "window token mean -> sample window mean -> sample equal mean",
        },
        "probe_details": details,
        "outputs": outputs,
        "created_utc": block3.utc_now(),
    }
    block3.atomic_json(PROBE_MANIFEST, manifest)
    return manifest


def samples_for(probe: str, tokenizer: Any, *, factor_only: bool) -> list[c4.PreparedSample]:
    samples = c4.prepare_samples(
        corpus_path(probe),
        tokenizer,
        corpus_id=f"qwen_block3:{probe}",
        window_seed=c4.WINDOW_SEED,
        max_context_tokens=c4.MAX_CONTEXT_TOKENS,
    )
    if factor_only:
        samples = samples[: ANALYSIS_CAPS[probe]]
    if not samples:
        raise RuntimeError(f"no prepared samples for {probe}")
    return samples


def profile(
    model: Any,
    samples: list[c4.PreparedSample],
    device: str,
    keep: bool,
    *,
    forward_batch_size: int,
    max_batch_tokens: int,
):
    return campaign.collect_profile(
        model,
        samples,
        [LAYER],
        device,
        keep_factors=keep,
        keep_residual_samples=False,
        factor_layers=(LAYER,),
        forward_batch_size=forward_batch_size,
        max_batch_tokens=max_batch_tokens,
        early_stop=True,
    )


def spectra_for(model: Any, measured: dict[str, Any], device: str) -> dict[str, list[float]]:
    scales = campaign.scaling_by_group(measured, [LAYER], device)
    spectra = {}
    try:
        for module in c4.MODULES:
            group = c4.MODULE_TO_GROUP[module]
            weight = campaign.module_at(model, LAYER, module).weight.detach().to(
                device=device, dtype=torch.float32
            )
            sigma = torch.linalg.svdvals(weight @ scales[LAYER][group])
            spectra[module] = sigma.double().cpu().tolist()
            del weight, sigma
            torch.cuda.empty_cache()
    finally:
        scales.clear()
        gc.collect()
        torch.cuda.empty_cache()
    return spectra


def metric_rows(
    arm: str,
    step: int,
    probe: str,
    spectra: dict[str, list[float]],
    base: dict[str, list[float]],
    n_samples: int,
) -> list[dict[str, Any]]:
    rows = []
    for module in c4.MODULES:
        for epsilon in EPSILONS:
            rank = c4.functional_rank(spectra[module], epsilon)
            base_rank = c4.functional_rank(base[module], epsilon)
            rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "probe": probe,
                    "track": "per_checkpoint",
                    "layer": LAYER,
                    "module": module,
                    "epsilon": epsilon,
                    "r_epsilon": rank,
                    "base_r_epsilon": base_rank,
                    "r_epsilon_delta": rank - base_rank,
                    "n_samples": n_samples,
                    "shared_base_compute": step == 0,
                }
            )
    return rows


def reference_path(probe: str) -> Path:
    return REFERENCE_ROOT / f"{probe}_spectra.json"


def cell_path(arm: str, step: int, probe: str) -> Path:
    return CELL_ROOT / arm / s3.step_label(step) / f"{probe}.json"


def factor_path(arm: str, step: int, probe: str) -> Path:
    return FACTOR_ROOT / arm / s3.step_label(step) / f"{probe}.pt"


def factor_meta_path(arm: str, step: int, probe: str) -> Path:
    return factor_path(arm, step, probe).with_suffix(".json")


def lock(path: Path):
    target = path.with_suffix(path.suffix + ".lock")
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = target.open("w", encoding="utf-8")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def save_factor(
    measured: dict[str, Any],
    samples: list[c4.PreparedSample],
    arm: str,
    step: int,
    probe: str,
) -> dict[str, Any]:
    target = factor_path(arm, step, probe)
    sample_ids = [sample.sample_id for sample in samples]
    metadata = {
        "schema_version": "cycle09_block3_qwen_factor_v1",
        "status": "complete",
        "arm": arm,
        "step": step,
        "probe": probe,
        "layer": LAYER,
        "n_samples": len(samples),
        "sample_ids": sample_ids,
        "sample_ids_sha256": block3.sha256_json(sample_ids),
        "corpus": block3.artifact(corpus_path(probe)),
        "window_seed": c4.WINDOW_SEED,
        "window_tokens": c4.WINDOW_TOKENS,
        "window_k": c4.WINDOW_K,
        "normalization": "window token mean -> sample window mean -> sample equal mean",
        "forward_execution": measured["forward_execution"],
        "created_utc": block3.utc_now(),
    }
    payload = {**metadata, "sample_factors": measured["sample_factors"]}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    metadata["bundle"] = block3.artifact(target)
    block3.atomic_json(factor_meta_path(arm, step, probe), metadata)
    return metadata


def load_reference(probe: str) -> dict[str, list[float]]:
    path = reference_path(probe)
    if not path.is_file():
        raise FileNotFoundError(f"base reference missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_reference(args: argparse.Namespace) -> dict[str, Any]:
    probe = args.probe
    target = cell_path("base", 0, probe)
    with lock(target):
        cached = block3.read_json(target, {})
        if cached.get("status") == "complete":
            return cached
        tokenizer = load_qwen_tokenizer()
        samples = samples_for(probe, tokenizer, factor_only=False)
        model = campaign.load_model(c4.BASE_MODEL, args.device)
        try:
            measured = profile(
                model,
                samples,
                args.device,
                keep=True,
                forward_batch_size=args.forward_batch_size,
                max_batch_tokens=args.max_batch_tokens,
            )
            spectra = spectra_for(model, measured, args.device)
            block3.atomic_json(reference_path(probe), spectra)
            if len(samples) > ANALYSIS_CAPS[probe]:
                del measured
                measured = profile(
                    model,
                    samples[: ANALYSIS_CAPS[probe]],
                    args.device,
                    keep=True,
                    forward_batch_size=args.forward_batch_size,
                    max_batch_tokens=args.max_batch_tokens,
                )
            factor = save_factor(
                measured, samples[: ANALYSIS_CAPS[probe]], "base", 0, probe
            )
            rows = metric_rows("base", 0, probe, spectra, spectra, len(samples))
        finally:
            campaign.unload_model(model)
        payload = {
            "schema_version": 1,
            "status": "complete",
            "arm": "base",
            "step": 0,
            "probe": probe,
            "rows": rows,
            "spectra": block3.artifact(reference_path(probe)),
            "factor": factor,
            "forward_execution": measured["forward_execution"],
            "created_utc": block3.utc_now(),
        }
        block3.atomic_json(target, payload)
        return payload


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in s3.ARMS or args.step <= 0:
        raise ValueError("non-base cell requires --arm and step > 0")
    target = cell_path(args.arm, args.step, args.probe)
    with lock(target):
        cached = block3.read_json(target, {})
        if cached.get("status") == "complete":
            return cached
        base = load_reference(args.probe)
        tokenizer = load_qwen_tokenizer()
        samples = samples_for(args.probe, tokenizer, factor_only=False)
        model_path = s3.require_model(args.arm, args.step)
        model = campaign.load_model(model_path, args.device)
        keep = args.step in CORE_STEPS and not args.no_retain_factor
        try:
            measured = profile(
                model,
                samples,
                args.device,
                keep=keep,
                forward_batch_size=args.forward_batch_size,
                max_batch_tokens=args.max_batch_tokens,
            )
            spectra = spectra_for(model, measured, args.device)
            factor = None
            if keep:
                if len(samples) > ANALYSIS_CAPS[args.probe]:
                    del measured
                    measured = profile(
                        model,
                        samples[: ANALYSIS_CAPS[args.probe]],
                        args.device,
                        keep=True,
                        forward_batch_size=args.forward_batch_size,
                        max_batch_tokens=args.max_batch_tokens,
                    )
                factor = save_factor(
                    measured,
                    samples[: ANALYSIS_CAPS[args.probe]],
                    args.arm,
                    args.step,
                    args.probe,
                )
            rows = metric_rows(
                args.arm, args.step, args.probe, spectra, base, len(samples)
            )
            spectra_path = (
                ROOT
                / "spectra"
                / args.arm
                / s3.step_label(args.step)
                / f"{args.probe}.json"
            )
            block3.atomic_json(spectra_path, spectra)
        finally:
            campaign.unload_model(model)
        payload = {
            "schema_version": 1,
            "status": "complete",
            "arm": args.arm,
            "step": args.step,
            "probe": args.probe,
            "model": s3.model_integrity(model_path),
            "rows": rows,
            "spectra": block3.artifact(spectra_path),
            "factor": factor,
            "forward_execution": measured["forward_execution"],
            "created_utc": block3.utc_now(),
        }
        block3.atomic_json(target, payload)
        return payload


def collect_factor_only(args: argparse.Namespace) -> dict[str, Any]:
    arm = "base" if args.step == 0 else args.arm
    if args.step == 0 and args.arm != "base":
        raise ValueError("step0 factor is canonical base only")
    target = factor_meta_path(arm, args.step, args.probe)
    cached = block3.read_json(target, {})
    if cached.get("status") == "complete":
        return cached
    with lock(target):
        tokenizer = load_qwen_tokenizer()
        samples = samples_for(args.probe, tokenizer, factor_only=True)
        model_path = (
            s3.require_model("opd", 0)
            if args.step == 0
            else s3.require_model(args.arm, args.step)
        )
        model = campaign.load_model(model_path, args.device)
        try:
            measured = profile(
                model,
                samples,
                args.device,
                keep=True,
                forward_batch_size=args.forward_batch_size,
                max_batch_tokens=args.max_batch_tokens,
            )
            return save_factor(measured, samples, arm, args.step, args.probe)
        finally:
            campaign.unload_model(model)


def finalize_r2() -> dict[str, Any]:
    rows = []
    cells = []
    for probe in R2_PROBES:
        base = block3.read_json(cell_path("base", 0, probe), {})
        if base.get("status") != "complete":
            raise RuntimeError(f"incomplete Qwen probe base cell: {probe}")
        for arm in s3.ARMS:
            for row in base["rows"]:
                rows.append({**row, "arm": arm, "shared_base_compute": True})
        cells.append(block3.artifact(cell_path("base", 0, probe)))
        for arm in s3.ARMS:
            for step in s3.STEPS:
                if step == 0:
                    continue
                payload = block3.read_json(cell_path(arm, step, probe), {})
                if payload.get("status") != "complete":
                    raise RuntimeError(f"incomplete Qwen probe cell: {arm}/{step}/{probe}")
                rows.extend(payload["rows"])
                cells.append(block3.artifact(cell_path(arm, step, probe)))
    expected = (
        len(s3.ARMS)
        * len(s3.STEPS)
        * len(R2_PROBES)
        * len(c4.MODULES)
        * len(EPSILONS)
    )
    if len(rows) != expected:
        raise RuntimeError(f"Qwen R2 row-count drift: {len(rows)} != {expected}")
    block3.atomic_csv(R2_OUTPUT, rows)
    block3.atomic_csv(block3.MINI / R2_OUTPUT.name, rows)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "task": "Cycle09 block3 Qwen E_math/E_math_hard_v2 domain probe supplement",
        "arms": list(s3.ARMS),
        "steps": list(s3.STEPS),
        "probes": list(R2_PROBES),
        "layer": LAYER,
        "modules": list(c4.MODULES),
        "epsilons": list(EPSILONS),
        "track": "per_checkpoint_only",
        "shared_base_forward_cells": len(R2_PROBES),
        "nonzero_cells": len(s3.ARMS) * (len(s3.STEPS) - 1) * len(R2_PROBES),
        "cells": cells,
        "probe_manifest": block3.artifact(PROBE_MANIFEST),
        "output": block3.artifact(R2_OUTPUT),
        "created_utc": block3.utc_now(),
    }
    block3.atomic_json(R2_MANIFEST, manifest)
    block3.atomic_json(block3.MINI / R2_MANIFEST.name, manifest)
    return manifest


def factor_inventory() -> dict[str, Any]:
    rows = []
    for probe in ALL_PROBES:
        for arm in ("base", *s3.ARMS):
            for step in CORE_STEPS:
                if (step == 0) != (arm == "base"):
                    continue
                path = factor_path(arm, step, probe)
                meta = block3.read_json(factor_meta_path(arm, step, probe), {})
                rows.append(
                    {
                        "arm": arm,
                        "step": step,
                        "probe": probe,
                        "expected_n": ANALYSIS_CAPS[probe],
                        "factor_path": str(path),
                        "status": (
                            "compatible"
                            if path.is_file()
                            and meta.get("status") == "complete"
                            and int(meta.get("layer", -1)) == LAYER
                            and int(meta.get("n_samples", -1)) == ANALYSIS_CAPS[probe]
                            else "missing"
                        ),
                        "bytes": path.stat().st_size if path.is_file() else 0,
                    }
                )
    expected = len(ALL_PROBES) * (
        1 + len(s3.ARMS) * (len(CORE_STEPS) - 1)
    )
    if len(rows) != expected:
        raise RuntimeError("factor inventory row-count drift")
    output = ROOT / "probe_factor_inventory.json"
    payload = {
        "schema_version": 1,
        "status": "complete_inventory",
        "expected_cells": expected,
        "compatible_cells": sum(row["status"] == "compatible" for row in rows),
        "missing_cells": sum(row["status"] == "missing" for row in rows),
        "rows": rows,
        "legacy_factor_policy": (
            "not silently reused; legacy bundles lack the new corpus/sample-id "
            "fingerprints and require explicit deep validation"
        ),
        "created_utc": block3.utc_now(),
    }
    block3.atomic_json(output, payload)
    block3.atomic_json(block3.MINI / output.name, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("prepare", "reference", "cell", "factor", "finalize-r2", "inventory"),
        required=True,
    )
    parser.add_argument("--arm", choices=("base", *s3.ARMS), default="base")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--probe", choices=ALL_PROBES, default="E_math")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--forward-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=16384)
    parser.add_argument("--no-retain-factor", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "prepare":
        result = prepare_corpora()
    elif args.phase == "reference":
        result = run_reference(args)
    elif args.phase == "cell":
        result = run_cell(args)
    elif args.phase == "factor":
        result = collect_factor_only(args)
    elif args.phase == "finalize-r2":
        result = finalize_r2()
    else:
        result = factor_inventory()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
