"""Deterministic validation-only headroom and static-lambda selectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import yaml


def read_run(path: Path, require_checkpoint: bool = False) -> Dict:
    complete = path / "complete.json"
    best = path / "best.json"
    if not complete.is_file() or not best.is_file():
        raise FileNotFoundError(f"Incomplete training run: {path}")
    state = json.loads(complete.read_text(encoding="utf-8"))
    metric = json.loads(best.read_text(encoding="utf-8"))
    if state.get("status") != "succeeded":
        raise ValueError(f"Training run did not succeed: {path}")
    for key in ("val_accuracy", "val_brier"):
        value = float(metric[key])
        if not math_is_finite(value):
            raise ValueError(f"Non-finite {key}: {path}")
    step = int(metric["step"])
    checkpoint = path / "checkpoints" / f"step_{step:06d}"
    if require_checkpoint and not (checkpoint / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Selected LoRA adapter is missing: {checkpoint}")
    return {**metric, "run_dir": str(path), "checkpoint": str(checkpoint)}


def math_is_finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def atomic_json(path: Path, payload: Dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite selector output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def headroom(main_root: Path, output: Path) -> None:
    dpo10 = read_run(main_root / "dpo10", require_checkpoint=True)
    initial_path = main_root / "dpo10" / "initial_validation.json"
    if not initial_path.is_file():
        raise FileNotFoundError(f"DPO-10 initial base validation is missing: {initial_path}")
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    key = "raw_mean_logp_val_accuracy"
    brier_key = "raw_mean_logp_val_brier"
    expected_score = "simpo_mean_logp_delta_margin_free"
    for payload in (initial, dpo10):
        if any(
            metric not in payload or not math_is_finite(float(payload[metric]))
            for metric in (key, brier_key)
        ):
            raise ValueError("Headroom requires finite common-score raw mean-logp metrics")
        if payload.get("raw_mean_logp_score_type") != expected_score:
            raise ValueError("Headroom before/after score types do not match the frozen contract")
    if initial.get("checkpoint") != "frozen_qwen3_base_before_training":
        raise ValueError("Headroom baseline is not the explicitly adapter-disabled frozen base")
    if int(initial.get("val_samples", -1)) != int(dpo10.get("val_samples", -2)):
        raise ValueError("Headroom before/after validation sample counts differ")
    gap = float(dpo10[key]) - float(initial[key])
    payload = {
        "status": "succeeded" if gap >= 0.05 else "failed_headroom",
        "selection_split": "validation",
        "headroom": gap,
        "headroom_threshold": 0.05,
        "score_type": expected_score,
        "baseline": "frozen_qwen3_before_training",
        "trained": "dpo10",
        "base_validation": initial,
        "dpo10": dpo10,
    }
    atomic_json(output / "headroom_selection.json", payload)
    (output / "headroom_report.md").write_text(
        "# DPO headroom gate\n\n"
        f"- status: `{payload['status']}`\n"
        f"- DPO-10 minus frozen-base validation accuracy: `{gap:.4f}` (required `>=0.05`)\n"
        "- both sides use the same margin-free mean-response-logp A/B score.\n"
        "- DPO-100 is an oracle arm, not the headroom baseline; SFT is not part of v0.6.\n",
        encoding="utf-8",
    )
    if gap < 0.05:
        raise SystemExit("DPO-10 did not beat its frozen base by 0.05; downstream jobs remain blocked")


def static_lambda(main_root: Path, headroom_path: Path, output: Path) -> None:
    headroom_value = json.loads(headroom_path.read_text(encoding="utf-8"))
    if headroom_value.get("status") != "succeeded":
        raise ValueError("DPO headroom gate did not succeed")
    candidates = (0.1, 0.3, 0.5, 1.0)
    values = {
        value: read_run(main_root / f"soppo_pe_static_lambda_{value:.1f}", require_checkpoint=True)
        for value in candidates
    }
    selected = sorted(
        candidates,
        key=lambda value: (-values[value]["val_accuracy"], values[value]["val_brier"], value),
    )[0]
    payload = {
        "status": "succeeded",
        "selection_split": "validation",
        "selected_static_lambda": selected,
        "selected_method": f"soppo_pe_static_lambda_{selected:.1f}",
        "candidates": {str(key): value for key, value in values.items()},
        "headroom": headroom_value,
    }
    atomic_json(output / "lambda_selection.json", payload)
    with (output / "config_final.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "dpo_beta": 0.1,
                "simpo_beta": 10.0,
                "simpo_margin": 2.0,
                "gamma0": 1.0,
                "gamma_min": 2700 / (2700 + 24000),
                "gamma_decay": 0.01,
                "selected_static_lambda": selected,
                "dpo10_base_headroom": headroom_value["headroom"],
            },
            handle,
            sort_keys=True,
        )
    (output / "lambda_search_report.md").write_text(
        "# SOPPO-PE-static lambda selection\n\n"
        f"- selected lambda: `{selected}`\n"
        "- rule: higher validation accuracy, then lower Brier, then smaller lambda.\n"
        "- exponential PE is a separate preregistered arm and is not mixed into this selector.\n",
        encoding="utf-8",
    )


def checkpoint(run_dir: Path, output: Path) -> None:
    atomic_json(output, read_run(run_dir, require_checkpoint=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    head = sub.add_parser("headroom")
    head.add_argument("--main-root", required=True)
    head.add_argument("--output", required=True)
    lam = sub.add_parser("lambda")
    lam.add_argument("--main-root", required=True)
    lam.add_argument("--headroom", required=True)
    lam.add_argument("--output", required=True)
    ckpt = sub.add_parser("checkpoint")
    ckpt.add_argument("--run-dir", required=True)
    ckpt.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "headroom":
        headroom(Path(args.main_root), Path(args.output))
    elif args.command == "lambda":
        static_lambda(Path(args.main_root), Path(args.headroom), Path(args.output))
    else:
        checkpoint(Path(args.run_dir), Path(args.output))


if __name__ == "__main__":
    main()
