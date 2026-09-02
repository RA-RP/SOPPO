#!/usr/bin/env python3
"""Fail-closed validation and aggregate-only summary for Round4 smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METHODS = ("dpo", "sspo", "staticpe")
REQUIRED_METRICS = {
    "dpo": ("dpo/loss", "rewards/chosen", "rewards/rejected"),
    "sspo": ("sspo/loss_labeled", "sspo/loss_unlabeled", "sspo/loss_total", "sspo/gamma"),
    "staticpe": (
        "staticpe/loss_dpo",
        "staticpe/loss_pe",
        "staticpe/loss_total",
        "staticpe/p_mean",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--expected-alpaca-outputs", type=int, default=2)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric: {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite: {value!r}")
    return value


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty required file: {path}")


def latest_metric(log_history: list[dict[str, Any]], key: str) -> float:
    values = [entry[key] for entry in log_history if key in entry]
    if not values:
        raise KeyError(f"Metric never logged: {key}")
    return finite_number(values[-1], key)


def validate_training(method: str, adapter_dir: Path) -> dict[str, Any]:
    trainer_state_path = adapter_dir / "trainer_state.json"
    train_results_path = adapter_dir / "train_results.json"
    eval_results_path = adapter_dir / "eval_results.json"
    adapter_config_path = adapter_dir / "adapter_config.json"
    for path in (trainer_state_path, train_results_path, eval_results_path, adapter_config_path):
        require_file(path)
    adapter_weights = [
        path
        for path in (adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_model.bin")
        if path.is_file() and path.stat().st_size > 0
    ]
    if len(adapter_weights) != 1:
        raise RuntimeError(f"{method}: expected exactly one adapter weight file")

    state = read_json(trainer_state_path)
    if state.get("global_step") != 2:
        raise RuntimeError(f"{method}: expected global_step=2, got {state.get('global_step')!r}")
    history = state.get("log_history")
    if not isinstance(history, list) or not history:
        raise RuntimeError(f"{method}: trainer log_history is empty")
    eval_events = [entry for entry in history if "eval_loss" in entry]
    if len(eval_events) < 2:
        raise RuntimeError(f"{method}: expected eval at both optimizer steps, found {len(eval_events)}")
    for index, event in enumerate(eval_events):
        finite_number(event["eval_loss"], f"{method}.eval_loss[{index}]")

    metrics = {key: latest_metric(history, key) for key in REQUIRED_METRICS[method]}
    train_results = read_json(train_results_path)
    eval_results = read_json(eval_results_path)
    train_loss = finite_number(train_results.get("train_loss"), f"{method}.train_loss")
    eval_loss = finite_number(eval_results.get("eval_loss"), f"{method}.final_eval_loss")
    return {
        "global_step": 2,
        "eval_events": len(eval_events),
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "component_metrics_last": metrics,
        "adapter_weight_sha256": sha256_file(adapter_weights[0]),
    }


def validate_merged(method: str, merged_dir: Path) -> dict[str, Any]:
    for file_name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        require_file(merged_dir / file_name)
    weight_files = sorted(merged_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"{method}: merged model has no safetensors weights")
    return {
        "weight_file_count": len(weight_files),
        "weight_bytes": sum(path.stat().st_size for path in weight_files),
        "config_sha256": sha256_file(merged_dir / "config.json"),
    }


def validate_generation(method: str, method_export: Path, expected: int) -> dict[str, Any]:
    output_path = method_export / "alpacaeval_smoke_outputs.json"
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    require_file(output_path)
    require_file(manifest_path)
    outputs = read_json(output_path)
    if not isinstance(outputs, list) or len(outputs) != expected:
        raise RuntimeError(f"{method}: expected {expected} Alpaca outputs, got {len(outputs) if isinstance(outputs, list) else 'non-list'}")
    for index, row in enumerate(outputs):
        if not isinstance(row, dict):
            raise ValueError(f"{method}: output row {index} is not an object")
        for key in ("instruction", "output", "generator"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{method}: output row {index} has invalid {key}")
    manifest = read_json(manifest_path)
    if manifest.get("num_outputs") != expected:
        raise RuntimeError(f"{method}: generation manifest count mismatch")
    if manifest.get("output_sha256") != sha256_file(output_path):
        raise RuntimeError(f"{method}: generation output SHA mismatch")
    return {"outputs": expected, "output_sha256": sha256_file(output_path)}


def validate_judge(method: str, method_export: Path, run_id: str) -> dict[str, Any]:
    judge_dir = method_export / "alpacaeval_judge"
    leaderboard_candidates = list(judge_dir.rglob("leaderboard.csv"))
    if len(leaderboard_candidates) != 1:
        raise RuntimeError(f"{method}: expected one judge leaderboard, found {len(leaderboard_candidates)}")
    leaderboard = leaderboard_candidates[0]
    require_file(leaderboard)
    with leaderboard.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(f"{method}: AlpacaEval leaderboard has no rows")
    expected_generator = f"round4-{method}-{run_id}"
    matching_rows = [row for row in rows if expected_generator in row.values()]
    if len(matching_rows) != 1:
        raise RuntimeError(f"{method}: judge leaderboard does not contain exactly one smoke generator row")
    row = matching_rows[0]
    for required in ("win_rate", "length_controlled_winrate"):
        try:
            finite_number(float(row[required]), f"{method}.judge.{required}")
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{method}: judge leaderboard has no finite {required}") from error
    aggregate: dict[str, float | str] = {}
    for key, value in row.items():
        if key is None or value is None:
            continue
        try:
            numeric = float(value)
        except ValueError:
            if key.lower() in {"model", "generator"}:
                aggregate[key] = value
            continue
        if math.isfinite(numeric):
            aggregate[key] = numeric
    annotation_candidates = list(judge_dir.rglob("*annotations*.json"))
    if not annotation_candidates:
        raise FileNotFoundError(f"{method}: no judge annotation artifact found")
    return {"leaderboard": aggregate, "annotation_artifact_count": len(annotation_candidates)}


def main() -> None:
    args = parse_args()
    if args.expected_alpaca_outputs <= 0:
        raise ValueError("--expected-alpaca-outputs must be positive")
    run_root = Path(args.run_root).expanduser().resolve(strict=True)
    export_root = Path(args.export_root).expanduser().resolve(strict=True)
    plan_path = run_root / "SMOKE_PLAN.json"
    require_file(plan_path)
    plan = read_json(plan_path)
    if plan.get("run_id") != args.run_id or plan.get("code_commit") != args.code_commit:
        raise RuntimeError("Smoke plan is not bound to the requested run/commit")
    if plan.get("contract", {}).get("optimizer_steps") != 2:
        raise RuntimeError("Smoke plan does not require exactly two optimizer steps")

    methods: dict[str, Any] = {}
    for method in METHODS:
        adapter_dir = run_root / method / "adapter"
        method_export = export_root / method
        methods[method] = {
            "training": validate_training(method, adapter_dir),
            "merged": validate_merged(method, method_export / "merged"),
            "generation": validate_generation(method, method_export, args.expected_alpaca_outputs),
            "judge": validate_judge(method, method_export, args.run_id),
        }

    summary = {
        "schema": "round4-smoke-summary-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "run_id": args.run_id,
        "code_commit": args.code_commit,
        "paper_result": False,
        "methods": methods,
    }
    output = export_root / "SMOKE_SUMMARY.json"
    atomic_json(output, summary)
    print(json.dumps({"status": "PASS", "run_id": args.run_id, "summary": str(output)}, indent=2))


if __name__ == "__main__":
    main()
