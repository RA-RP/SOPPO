#!/usr/bin/env python3
"""Shared registry, provenance, and preflight helpers for Cycle 09 Stage 3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO = Path("/root/LLM-output-density")
AUTODL = Path("/root/autodl-tmp")
MINI = (
    REPO
    / "mypaper/local_experiment_results/"
    "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
RUN_ROOT = AUTODL / "cycle09_stage3"
CONTRACT = REPO / "mypaper/code/cycle09_stage3_execution_contract.json"

BASE_MODEL = AUTODL / "model/Qwen/Qwen3-4B-Base"
MODEL_ROOTS = {
    "opd": AUTODL / "cycle08_opd_trajectory/_merged_models",
    "sft": AUTODL / "cycle09_r3/sft_merged",
    "offkd": AUTODL / "cycle09_offkd/_merged_models",
    "seqkd": AUTODL / "cycle09_seqkd/_merged_models",
}
ARMS = ("opd", "sft", "offkd", "seqkd")
STEPS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
TRANSIENT_STEPS = (5, 10, 20, 40, 80)
LAYERS = (9, 18, 27)
MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

R4_ROOT = AUTODL / "cycle09_r4"
R4_FIXED = R4_ROOT / "corpora/fixed"
IFEVAL_INPUT = REPO / "Eval/tasks/data/ifeval/train.jsonl"
MMLUPRO_INPUT = AUTODL / "cycle09_s1/mmlupro_loglik/input/mmlupro_1400.jsonl"
OPD_RECONSTRUCTION = (
    AUTODL
    / "cycle09_s1/s1_5_opd_rollout/opd_step0_reconstructed_rollout.jsonl"
)
SFT_TRAIN = AUTODL / "cycle07_base_sft_trajectory/data_prep/train_5k.jsonl"
TEACHER_TRAIN = AUTODL / "cycle09_offkd/rollout/teacher_rollout.jsonl"
MATHCOT_PARQUET = AUTODL / "dataset/Math-CoT-20k/Math-CoT-20k.parquet"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def step_label(step: int) -> str:
    return f"step_{int(step):03d}"


def parse_names(value: str, allowed: Iterable[str]) -> list[str]:
    allowed_tuple = tuple(allowed)
    if value.strip().lower() == "all":
        return list(allowed_tuple)
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names).difference(allowed_tuple))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown names {unknown}; allowed={allowed_tuple}")
    return names


def parse_ints(value: str, allowed: Iterable[int]) -> list[int]:
    allowed_tuple = tuple(int(item) for item in allowed)
    if value.strip().lower() == "all":
        return list(allowed_tuple)
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    unknown = sorted(set(values).difference(allowed_tuple))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown steps {unknown}; allowed={allowed_tuple}")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def model_path(arm: str, step: int) -> Path:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if int(step) == 0:
        return BASE_MODEL
    return MODEL_ROOTS[arm] / step_label(step)


def _weight_files_from_index(index: Path) -> list[Path]:
    payload = json.loads(index.read_text(encoding="utf-8"))
    names = sorted(set(payload.get("weight_map", {}).values()))
    if not names:
        raise ValueError(f"empty weight_map: {index}")
    return [index.parent / name for name in names]


def model_integrity(path: Path) -> dict[str, Any]:
    config = path / "config.json"
    index_candidates = (
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin.index.json",
    )
    index = next((item for item in index_candidates if item.is_file()), None)
    error = None
    weights: list[Path] = []
    try:
        if not config.is_file() or config.stat().st_size == 0:
            raise FileNotFoundError(config)
        if index is not None:
            weights = _weight_files_from_index(index)
        else:
            weights = sorted(path.glob("*.safetensors")) or sorted(path.glob("pytorch_model*.bin"))
            if not weights:
                raise FileNotFoundError(f"no model weights under {path}")
        missing = [str(item) for item in weights if not item.is_file() or item.stat().st_size == 0]
        if missing:
            raise FileNotFoundError("missing/empty model shards: " + ", ".join(missing))
    except (OSError, ValueError, json.JSONDecodeError) as caught:
        error = str(caught)
    return {
        "path": str(path),
        "complete": error is None,
        "error": error,
        "config_bytes": config.stat().st_size if config.is_file() else 0,
        "index": str(index) if index else None,
        "weight_files": len(weights),
        "weight_bytes": sum(item.stat().st_size for item in weights if item.is_file()),
    }


def require_model(arm: str, step: int) -> Path:
    path = model_path(arm, step)
    check = model_integrity(path)
    if not check["complete"]:
        raise FileNotFoundError(f"incomplete model {arm}/{step}: {check['error']}")
    return path


def file_integrity(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    return {
        "path": str(path),
        "complete": bool(exists and size > 0),
        "bytes": size,
    }


def preflight_payload(tasks: list[str], arms: list[str], steps: list[int]) -> dict[str, Any]:
    task_files = {
        "c2": [R4_FIXED / "E_ood.jsonl"],
        "c3": [
            R4_FIXED / "legacy_S_math.jsonl",
            R4_FIXED / "E_ood.jsonl",
            R4_FIXED / "E_general.jsonl",
            R4_FIXED / "E_math_hard.jsonl",
            *(
                R4_ROOT / f"corpora/generated/S/bos/gen_seed_{seed}.jsonl"
                for seed in (3, 17, 31)
            ),
        ],
        "c5": [IFEVAL_INPUT],
        "c8": [OPD_RECONSTRUCTION, SFT_TRAIN, TEACHER_TRAIN],
        "c11": [MMLUPRO_INPUT],
        "c15": [
            AUTODL / "cycle07_base_sft_trajectory/cap_pilot/math500_samples.jsonl",
        ],
    }
    files = []
    for task in tasks:
        files.extend(file_integrity(path) | {"task": task} for path in task_files.get(task, []))

    required_cells = {(arm, step) for arm in arms for step in steps}
    models = [model_integrity(model_path(arm, step)) | {"arm": arm, "step": step}
              for arm, step in sorted(required_cells)]
    complete = all(item["complete"] for item in files + models)
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "tasks": tasks,
        "arms": arms,
        "steps": steps,
        "complete": complete,
        "files": files,
        "models": models,
    }


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def assert_contract() -> dict[str, Any]:
    if not CONTRACT.is_file():
        raise FileNotFoundError(f"missing frozen Stage-3 contract: {CONTRACT}")
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_before_stage3_results":
        raise ValueError(f"invalid Stage-3 contract status: {CONTRACT}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="c2,c3,c5,c8,c11,c15")
    parser.add_argument("--arms", default="all")
    parser.add_argument("--steps", default="all")
    parser.add_argument("--output", type=Path, default=RUN_ROOT / "preflight.json")
    args = parser.parse_args()
    tasks = [item.strip().lower() for item in args.tasks.split(",") if item.strip()]
    arms = parse_names(args.arms, ARMS)
    steps = parse_ints(args.steps, STEPS)
    assert_contract()
    payload = preflight_payload(tasks, arms, steps)
    atomic_json(args.output, payload)
    print(json.dumps({"complete": payload["complete"], "output": str(args.output)}))
    if not payload["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
