#!/usr/bin/env python3
"""Q1 alpha=.5 behavior keypoints using the current Qwen evaluation protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

import cycle09_block3_common as c
import cycle09_offkd_eval as evals
import cycle09_s1_12_mmlupro as mmlu
import cycle09_s1_9_ifeval as ifeval


ARM = "alpha05"
STEPS = (0, 5, 20, 40, 80, 160, 320)
ROOT = c.Q1_ROOT / "behavior"
MINI_SUMMARY = c.MINI / "qwen_alpha05_behavior_keypoints.csv"
MINI_MMLU_EXTRACT = c.MINI / "qwen_alpha05_mmlupro_extract_audit.csv"
MINI_MMLU_FLEX = c.MINI / "qwen_alpha05_mmlupro_flexible.csv"
MINI_IFEVAL = c.MINI / "qwen_alpha05_ifeval_breakdown.csv"
MANIFEST = c.MINI / "qwen_alpha05_behavior_manifest.json"


def model_path(step: int) -> Path:
    return c.QWEN_STUDENT if step == 0 else c.Q1_ROOT / "_merged_models" / f"step_{step:03d}"


def branch(smoke: bool) -> Path:
    return ROOT / ("smoke" if smoke else "formal")


def label(step: int) -> str:
    return f"step_{step:03d}"


def cell_root(step: int, smoke: bool) -> Path:
    return branch(smoke) / "cells" / label(step)


def result_json(path: Path) -> Path | None:
    results = sorted(path.rglob("results_*.json")) if path.is_dir() else []
    return results[-1] if results else None


def runner_args(smoke: bool, gpu_mem: float) -> SimpleNamespace:
    args = SimpleNamespace()
    args.run_root = c.Q1_ROOT
    args.eval_root = branch(smoke)
    args.gpu_mem = gpu_mem
    return args


def configure_runner() -> None:
    evals.model_path = lambda _run_root, step: model_path(int(step))


def run_math(args: SimpleNamespace, step: int, smoke: bool) -> Path:
    max_tokens, max_model_len = (256, 2048) if smoke else evals.math500_budget(step)
    return evals.run_think_eval(
        args,
        task="math500",
        step=step,
        n=2 if smoke else 500,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
    )


def run_mmlu(args: SimpleNamespace, step: int, smoke: bool) -> Path:
    output = args.eval_root / "lm_eval" / label(step) / "mmlu_pro"
    existing = result_json(output)
    if existing is not None:
        return existing
    output.mkdir(parents=True, exist_ok=True)
    command = evals.lm_eval_prefix() + [
        "--model", "vllm",
        "--model_args",
        f"pretrained={model_path(step)},dtype=bfloat16,gpu_memory_utilization={args.gpu_mem},max_model_len=4096",
        "--tasks", "mmlu_pro",
        "--num_fewshot", "0",
        "--batch_size", "auto",
        "--seed", str(evals.SEED),
        "--output_path", str(output),
        "--limit", "1" if smoke else "100",
        "--log_samples",
    ]
    evals.run_command(command, cwd=c.REPO)
    result = result_json(output)
    if result is None:
        raise RuntimeError(f"MMLU-Pro returned without results: {label(step)}")
    return result


def run_ifeval(args: SimpleNamespace, step: int, smoke: bool) -> Path:
    evals.run_ood_eval(args, step=step, tasks=("ifeval",), limit=1 if smoke else None)
    output = args.eval_root / "ood_expansion" / label(step)
    result = result_json(output)
    if result is None:
        raise RuntimeError(f"IFEval returned without results: {label(step)}")
    return result


def mmlu_rows(step: int, smoke: bool) -> pd.DataFrame:
    root = branch(smoke) / "lm_eval" / label(step) / "mmlu_pro"
    paths = sorted(root.rglob("samples_mmlu_pro_*.jsonl"))
    expected_files = 14
    if len(paths) != expected_files:
        raise RuntimeError(f"MMLU-Pro sample files={len(paths)} expected={expected_files}: {root}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        subject = path.name.split("samples_mmlu_pro_", 1)[1].rsplit("_", 1)[0]
        for record in c.read_jsonl(path):
            text = mmlu.response_text(record)
            prediction = mmlu.strict_prediction(record)
            flexible, tier = mmlu.flexible_extract(text)
            rows.append(
                {
                    "arm": ARM,
                    "step": step,
                    "subject": subject,
                    "target": str(record["target"]).upper(),
                    "strict_prediction": prediction,
                    "strict_extract_success": prediction is not None,
                    "strict_exact_match": float(record.get("exact_match", 0.0)),
                    "flexible_prediction": flexible,
                    "flexible_tier": tier,
                    "flexible_exact_match": float(flexible == str(record["target"]).upper()),
                    "response_text": text,
                    "response_chars": len(text),
                    "max_generation_tokens": mmlu.max_generation_tokens(record),
                }
            )
    expected_rows = 14 if smoke else 1400
    if len(rows) != expected_rows:
        raise RuntimeError(f"MMLU-Pro rows={len(rows)} expected={expected_rows}")
    tokenizer = AutoTokenizer.from_pretrained(c.QWEN_STUDENT, local_files_only=True, trust_remote_code=True)
    for start in range(0, len(rows), 64):
        group = rows[start : start + 64]
        encoded = tokenizer([row["response_text"] for row in group], add_special_tokens=False)
        for row, ids in zip(group, encoded["input_ids"], strict=True):
            row["response_tokens"] = len(ids)
            row["failure_shape"] = mmlu.failure_shape(
                row["response_text"], row["strict_extract_success"], row["response_tokens"], row["max_generation_tokens"]
            )
            del row["response_text"]
    return pd.DataFrame(rows)


def ifeval_rows(step: int, smoke: bool) -> pd.DataFrame:
    root = branch(smoke) / "ood_expansion" / label(step)
    paths = sorted(root.rglob("samples_ifeval_*.jsonl"))
    if len(paths) != 1:
        raise RuntimeError(f"IFEval sample files={len(paths)} expected=1: {root}")
    rows = ifeval.read_rows(paths[0])
    expected = 1 if smoke else 541
    if len(rows) != expected:
        raise RuntimeError(f"IFEval rows={len(rows)} expected={expected}")
    return pd.DataFrame(ifeval.build_breakdown([{"arm": ARM, "step": step, "rows": rows}]))


def metric(results: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in results:
            return results[key]
    return ""


def run_cell(step: int, smoke: bool, gpu_mem: float) -> dict[str, Any]:
    allowed_steps = (*STEPS, 5) if smoke else STEPS
    if step not in allowed_steps:
        raise ValueError(f"unsupported Q1 behavior step: {step}")
    target = cell_root(step, smoke) / "cell_manifest.json"
    cached = c.read_json(target, {})
    if cached.get("status") == "complete":
        return cached
    if not c.model_check(model_path(step))["complete"]:
        raise FileNotFoundError(f"Q1 merged model missing: {model_path(step)}")
    configure_runner()
    args = runner_args(smoke, gpu_mem)
    math = run_math(args, step, smoke)
    mmlu_result = run_mmlu(args, step, smoke)
    ifeval_result = run_ifeval(args, step, smoke)
    mmlu_frame = mmlu_rows(step, smoke)
    ifeval_frame = ifeval_rows(step, smoke)
    extract = mmlu.extraction_audit(mmlu_frame).iloc[0].to_dict()
    flexible = mmlu.flexible_table(mmlu_frame).iloc[0].to_dict()
    math_data = c.read_json(math)
    ifeval_data = c.read_json(ifeval_result).get("results", {}).get("ifeval", {})
    summary = {
        "arm": ARM,
        "step": step,
        "math500_n": math_data.get("n"),
        "math500_cap": math_data.get("max_tokens"),
        "math500_acc": math_data.get("acc"),
        "math500_trunc_rate": math_data.get("trunc_rate"),
        "math500_mean_response_len": math_data.get("mean_response_len"),
        "mmlu_pro_n": int(len(mmlu_frame)),
        "mmlu_pro_exact_match": flexible["exact_match"],
        "mmlu_pro_flexible": flexible["mmlu_pro_flexible"],
        "mmlu_pro_extract_fail_rate": extract["extract_fail_rate"],
        "ifeval_n": ifeval_data.get("sample_len", len(ifeval_frame)),
        "ifeval_prompt_strict": metric(ifeval_data, "prompt_level_strict_acc,none", "prompt_level_strict_acc"),
        "ifeval_instruction_strict": metric(ifeval_data, "inst_level_strict_acc,none", "inst_level_strict_acc"),
        "checkpoint_source_type": (
            "shared_base"
            if step == 0
            else "q1_stage_b_native"
            if step == 320
            else "q1_stage_a_native"
        ),
    }
    root = cell_root(step, smoke)
    c.atomic_csv(root / "mmlupro_samples.csv", mmlu_frame.to_dict("records"))
    c.atomic_csv(root / "mmlupro_extract.csv", [extract])
    c.atomic_csv(root / "mmlupro_flexible.csv", [flexible])
    c.atomic_csv(root / "ifeval_breakdown.csv", ifeval_frame.to_dict("records"))
    payload = {
        "schema_version": 1,
        "status": "complete",
        "arm": ARM,
        "step": step,
        "smoke": smoke,
        "model": c.model_check(model_path(step)),
        "summary": summary,
        "artifacts": {
            "math500": c.artifact(math),
            "mmlu_results": c.artifact(mmlu_result),
            "ifeval_results": c.artifact(ifeval_result),
            "mmlu_samples": c.artifact(root / "mmlupro_samples.csv"),
            "mmlu_extract": c.artifact(root / "mmlupro_extract.csv"),
            "mmlu_flexible": c.artifact(root / "mmlupro_flexible.csv"),
            "ifeval_breakdown": c.artifact(root / "ifeval_breakdown.csv"),
        },
        "created_utc": c.utc_now(),
    }
    c.atomic_json(target, payload)
    return payload


def finalize() -> dict[str, Any]:
    cells = [c.read_json(cell_root(step, False) / "cell_manifest.json", {}) for step in STEPS]
    if any(cell.get("status") != "complete" for cell in cells):
        raise RuntimeError("Q1 behavior cells are incomplete")
    summaries = [cell["summary"] for cell in cells]
    extracts = [c.read_json(cell_root(step, False) / "cell_manifest.json")["summary"] for step in STEPS]
    mmlu_extract = [pd.read_csv(cell_root(step, False) / "mmlupro_extract.csv").iloc[0].to_dict() for step in STEPS]
    mmlu_flexible = [pd.read_csv(cell_root(step, False) / "mmlupro_flexible.csv").iloc[0].to_dict() for step in STEPS]
    ifeval_frames = [pd.read_csv(cell_root(step, False) / "ifeval_breakdown.csv") for step in STEPS]
    c.atomic_csv(MINI_SUMMARY, summaries)
    c.atomic_csv(MINI_MMLU_EXTRACT, mmlu_extract)
    c.atomic_csv(MINI_MMLU_FLEX, mmlu_flexible)
    c.atomic_csv(MINI_IFEVAL, pd.concat(ifeval_frames, ignore_index=True).to_dict("records"))
    payload = {
        "schema_version": 1,
        "status": "complete",
        "task": "Q1 alpha=.5 behavior keypoints",
        "arm": ARM,
        "steps": list(STEPS),
        "protocol": {
            "math500": "Qwen as-run caps: steps 0/5/20 at 4096; steps 40+ at 16384",
            "mmlu_pro": "100/category, 1400 total, strict + frozen flexible extraction",
            "ifeval": "all 541 prompts, chat template, enable_thinking=false, native category audit",
        },
        "cells": [c.artifact(cell_root(step, False) / "cell_manifest.json") for step in STEPS],
        "outputs": [c.artifact(path) for path in (MINI_SUMMARY, MINI_MMLU_EXTRACT, MINI_MMLU_FLEX, MINI_IFEVAL)],
        "created_utc": c.utc_now(),
    }
    c.atomic_json(MANIFEST, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cell", "finalize"), required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    args = parser.parse_args()
    result = finalize() if args.phase == "finalize" else run_cell(args.step, args.smoke, args.gpu_mem)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
