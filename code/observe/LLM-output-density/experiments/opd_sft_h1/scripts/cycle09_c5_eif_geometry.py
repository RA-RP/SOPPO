#!/usr/bin/env python3
"""C5: register IFEval prompts as fixed E_if and run the R4 point geometry grid."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

import cycle09_r4_campaign as campaign
import cycle09_r4_common as c4
import cycle09_stage3_common as s3


TASK = "E_if"
ROOT = s3.RUN_ROOT / "c5_eif"
CORPUS = ROOT / "corpus/E_if.jsonl"
REFERENCE = ROOT / "reference/E_if.pt"
MEASUREMENTS = ROOT / "measurements"
EPSILONS = (0.05, 0.01)
TAIL_RANKS = (32, 64, 128, 256)


@contextmanager
def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def prepare_corpus() -> dict[str, Any]:
    source_rows = s3.read_jsonl(s3.IFEVAL_INPUT)
    if len(source_rows) != 541:
        raise RuntimeError(f"expected 541 IFEval rows, found {len(source_rows)}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(s3.BASE_MODEL), trust_remote_code=True
    )
    rows = []
    for index, source in enumerate(source_rows):
        text = str(source["prompt"]).strip()
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not tokens:
            raise ValueError(f"empty IFEval prompt at row {index}")
        rows.append(
            {
                "sample_id": f"ifeval_{int(source['key'])}",
                "probe_type": "E",
                "domain": "instruction_following",
                "source_kind": "fixed_ifeval_prompt_text",
                "prompt_text": "",
                "generation_text": text,
                "prompt_token_ids": [],
                "generation_token_ids": [int(token) for token in tokens],
                "full_token_ids": [int(token) for token in tokens],
                "eligible_start": 0,
                "eligible_end": len(tokens),
                "ifeval_key": int(source["key"]),
                "instruction_ids": list(source["instruction_id_list"]),
            }
        )
    expected = {
        "source_sha256": s3.sha256_file(s3.IFEVAL_INPUT),
        "rows": len(rows),
        "sample_ids_sha256": s3.sha256_json(
            [row["sample_id"] for row in rows]
        ),
    }
    metadata = CORPUS.with_suffix(".manifest.json")
    if CORPUS.is_file() and metadata.is_file():
        old = json.loads(metadata.read_text(encoding="utf-8"))
        if all(old.get(key) == value for key, value in expected.items()):
            return old
        raise RuntimeError(f"incompatible existing E_if corpus: {CORPUS}")
    s3.atomic_jsonl(CORPUS, rows)
    payload = {
        "schema_version": 1,
        "task": TASK,
        **expected,
        "corpus_path": str(CORPUS),
        "corpus_sha256": s3.sha256_file(CORPUS),
        "source_path": str(s3.IFEVAL_INPUT),
        "text_region": "entire prompt text; no generated response",
        "window_seed": c4.WINDOW_SEED,
        "window_tokens": c4.WINDOW_TOKENS,
        "window_k": c4.WINDOW_K,
        "token_count_min": min(len(row["full_token_ids"]) for row in rows),
        "token_count_max": max(len(row["full_token_ids"]) for row in rows),
    }
    s3.atomic_json(metadata, payload)
    return payload


def profile_view(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in profile.items()
        if key
        in {
            "n_samples",
            "grams",
            "residual_second",
            "residual_mean",
            "position_second",
            "position_mean",
            "position_counts",
        }
    }


def load_reference() -> dict[str, Any]:
    payload = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    required = {"n_samples", "grams"}
    if not required.issubset(payload):
        raise ValueError(f"invalid C5 reference: {REFERENCE}")
    payload["grams"] = {int(key): value for key, value in payload["grams"].items()}
    return payload


def ensure_reference(
    samples: list[c4.PreparedSample], device: str
) -> dict[str, Any]:
    if REFERENCE.is_file():
        reference = load_reference()
        if int(reference["n_samples"]) == len(samples):
            return reference
        raise RuntimeError("C5 reference sample count mismatch")
    with lock(REFERENCE.with_suffix(".lock")):
        if REFERENCE.is_file():
            return load_reference()
        model_path = s3.require_model("opd", 0)
        model = campaign.load_model(model_path, device)
        try:
            print("[C5 reference] collecting base E_if profile", flush=True)
            profile = campaign.collect_profile(
                model,
                samples,
                list(c4.LAYERS),
                device,
                keep_factors=False,
                keep_residual_samples=False,
            )
            reference = profile_view(profile)
            atomic_torch(REFERENCE, reference)
            del profile
        finally:
            campaign.unload_model(model)
            gc.collect()
            torch.cuda.empty_cache()
    return load_reference()


def spectrum_metrics(values: np.ndarray) -> dict[str, float | int]:
    return {
        "effective_rank": c4.effective_rank(values),
        "r_eps_005": c4.functional_rank(values, 0.05),
        "r_eps_001": c4.functional_rank(values, 0.01),
        **{
            f"tail_energy_r{rank}": c4.tail_energy(values, rank)
            for rank in TAIL_RANKS
        },
    }


@torch.inference_mode()
def measure(
    model,
    current_profile: dict[str, Any],
    reference: dict[str, Any],
    device: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    current_scales = campaign.scaling_by_group(
        current_profile, list(c4.LAYERS), device
    )
    base_scales = campaign.scaling_by_group(
        reference, list(c4.LAYERS), device
    )
    spectra: dict[str, np.ndarray] = {}
    rows = []
    try:
        for layer in c4.LAYERS:
            for module in c4.MODULES:
                group = c4.MODULE_TO_GROUP[module]
                weight = campaign.module_at(
                    model, layer, module
                ).weight.detach().to(device=device, dtype=torch.float32)
                for track, scale in (
                    ("per_checkpoint", current_scales[layer][group]),
                    ("frozen_base", base_scales[layer][group]),
                ):
                    values = (
                        torch.linalg.svdvals(weight @ scale)
                        .float()
                        .cpu()
                        .numpy()
                    )
                    key = (
                        f"{track}__L{layer}__"
                        + module.replace(".", "__")
                    )
                    spectra[key] = values
                    metrics = spectrum_metrics(values)
                    for epsilon, rank_key in (
                        (0.05, "r_eps_005"),
                        (0.01, "r_eps_001"),
                    ):
                        rows.append(
                            {
                                "track": track,
                                "layer": layer,
                                "module": module,
                                "epsilon": epsilon,
                                "effective_rank": metrics["effective_rank"],
                                "r_epsilon": metrics[rank_key],
                                "tail_energy_r32": metrics["tail_energy_r32"],
                                "tail_energy_r64": metrics["tail_energy_r64"],
                                "tail_energy_r128": metrics["tail_energy_r128"],
                                "tail_energy_r256": metrics["tail_energy_r256"],
                                "spectrum_key": key,
                            }
                        )
                    del values
                del weight
                torch.cuda.empty_cache()
    finally:
        current_scales.clear()
        base_scales.clear()
        gc.collect()
        torch.cuda.empty_cache()
    return spectra, rows


def cell_paths(arm: str, step: int) -> tuple[Path, Path]:
    label = "base" if step == 0 else arm
    root = MEASUREMENTS / label / s3.step_label(step)
    return root / "summary.json", root / "spectra.npz"


def cell_complete(arm: str, step: int, corpus_hash: str) -> bool:
    summary, spectra = cell_paths(arm, step)
    if not summary.is_file() or not spectra.is_file():
        return False
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
        return (
            payload.get("schema_version") == 1
            and payload.get("corpus_sha256") == corpus_hash
            and len(payload.get("rows", []))
            == len(c4.LAYERS) * len(c4.MODULES) * 2 * len(EPSILONS)
        )
    except (OSError, json.JSONDecodeError):
        return False


def write_spectra(path: Path, spectra: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **spectra)
    os.replace(temporary, path)


def run_cells(args: argparse.Namespace) -> None:
    corpus_meta = prepare_corpus()
    tokenizer = AutoTokenizer.from_pretrained(
        str(s3.BASE_MODEL), trust_remote_code=True
    )
    samples = c4.prepare_samples(
        CORPUS,
        tokenizer,
        corpus_id=TASK,
        window_seed=c4.WINDOW_SEED,
        max_context_tokens=c4.MAX_CONTEXT_TOKENS,
    )
    reference = ensure_reference(samples, args.device)
    cells = [("opd", 0)] + [
        (arm, step)
        for arm in args.arms
        for step in args.steps
        if step != 0
    ]
    seen = set()
    for arm, step in cells:
        key = ("base", 0) if step == 0 else (arm, step)
        if key in seen:
            continue
        seen.add(key)
        if cell_complete(arm, step, corpus_meta["corpus_sha256"]):
            print(f"[C5 cached] {key}", flush=True)
            continue
        summary_path, spectra_path = cell_paths(arm, step)
        with lock(summary_path.with_suffix(".lock")):
            if cell_complete(arm, step, corpus_meta["corpus_sha256"]):
                continue
            path = s3.require_model(arm, step)
            print(f"[C5] {key} model={path}", flush=True)
            model = campaign.load_model(path, args.device)
            try:
                if step == 0:
                    profile = reference
                else:
                    profile = campaign.collect_profile(
                        model,
                        samples,
                        list(c4.LAYERS),
                        args.device,
                        keep_factors=False,
                        keep_residual_samples=False,
                    )
                spectra, rows = measure(
                    model, profile, reference, args.device
                )
                write_spectra(spectra_path, spectra)
                s3.atomic_json(
                    summary_path,
                    {
                        "schema_version": 1,
                        "task": "C5",
                        "arm": "base" if step == 0 else arm,
                        "step": step,
                        "probe": TASK,
                        "n_samples": len(samples),
                        "corpus_sha256": corpus_meta["corpus_sha256"],
                        "model_path": str(path),
                        "spectra_path": str(spectra_path),
                        "spectra_sha256": s3.sha256_file(spectra_path),
                        "rows": rows,
                    },
                )
                del spectra, rows
                if step != 0:
                    del profile
            finally:
                campaign.unload_model(model)
                gc.collect()
                torch.cuda.empty_cache()


def finalize() -> None:
    corpus_meta = json.loads(
        CORPUS.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    rows = []
    inventory = []
    for arm in s3.ARMS:
        for step in s3.STEPS:
            summary_path, spectra_path = cell_paths(arm, step)
            if not cell_complete(arm, step, corpus_meta["corpus_sha256"]):
                raise FileNotFoundError(f"incomplete C5 cell {arm}/{step}")
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            for row in payload["rows"]:
                rows.append(
                    {
                        "arm": arm,
                        "step": step,
                        "task_id": TASK,
                        "probe_type": "E",
                        "domain": "instruction_following",
                        "n_samples": payload["n_samples"],
                        **row,
                    }
                )
            inventory.append(
                {
                    "arm": arm,
                    "step": step,
                    "summary_path": str(summary_path),
                    "summary_sha256": s3.sha256_file(summary_path),
                    "spectra_path": str(spectra_path),
                    "spectra_sha256": s3.sha256_file(spectra_path),
                }
            )
    expected = (
        len(s3.ARMS)
        * len(s3.STEPS)
        * len(c4.LAYERS)
        * len(c4.MODULES)
        * 2
        * len(EPSILONS)
    )
    if len(rows) != expected:
        raise RuntimeError(f"C5 rows {len(rows)} != {expected}")

    base = {
        (row["track"], row["layer"], row["module"], row["epsilon"]): row
        for row in rows
        if row["arm"] == "opd" and row["step"] == 0
    }
    for row in rows:
        reference = base[
            (
                row["track"],
                row["layer"],
                row["module"],
                row["epsilon"],
            )
        ]
        row["effective_rank_base"] = reference["effective_rank"]
        row["effective_rank_delta"] = (
            row["effective_rank"] - reference["effective_rank"]
        )
        row["r_epsilon_base"] = reference["r_epsilon"]
        row["r_epsilon_delta"] = (
            row["r_epsilon"] - reference["r_epsilon"]
        )

    output = s3.MINI / "C5_eif_m1_geometry.csv"
    inventory_output = s3.MINI / "C5_eif_spectra_inventory.csv"
    s3.atomic_csv(output, rows)
    s3.atomic_csv(inventory_output, inventory)
    manifest = s3.MINI / "C5_eif_geometry_manifest.json"
    s3.atomic_json(
        manifest,
        {
            "schema_version": 1,
            "status": "complete",
            "contract": s3.artifact(s3.CONTRACT),
            "corpus": corpus_meta,
            "arms": list(s3.ARMS),
            "steps": list(s3.STEPS),
            "layers": list(c4.LAYERS),
            "tracks": ["per_checkpoint", "frozen_base"],
            "factor_retention": False,
            "outputs": [
                s3.artifact(output),
                s3.artifact(inventory_output),
            ],
        },
    )
    print(f"[C5 finalized] rows={len(rows)}", flush=True)


def synthetic_smoke() -> None:
    values = np.array([5.0, 2.0, 1.0, 0.1], dtype=np.float64)
    metrics = spectrum_metrics(values)
    if metrics["r_eps_005"] != 2 or metrics["r_eps_001"] != 3:
        raise RuntimeError(f"bad spectrum metrics: {metrics}")
    rows = s3.read_jsonl(s3.IFEVAL_INPUT)
    if len(rows) != 541 or "prompt" not in rows[0]:
        raise RuntimeError("IFEval source smoke failed")
    print(json.dumps({"status": "ok", "ifeval_rows": len(rows), **metrics}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("prepare", "cells", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--arms", default="all")
    parser.add_argument("--steps", default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    s3.assert_contract()
    if args.smoke:
        synthetic_smoke()
        return
    args.arms = s3.parse_names(args.arms, s3.ARMS)
    args.steps = s3.parse_ints(args.steps, s3.STEPS)
    if args.phase in ("prepare", "all"):
        prepare_corpus()
    if args.phase in ("cells", "all"):
        run_cells(args)
    if args.phase in ("finalize", "all"):
        finalize()


if __name__ == "__main__":
    main()
