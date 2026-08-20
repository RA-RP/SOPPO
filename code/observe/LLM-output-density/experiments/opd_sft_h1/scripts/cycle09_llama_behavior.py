#!/usr/bin/env python3
"""Run/finalize the shared-base eight-point Llama behavior replication."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import cycle09_block3_common as c
import cycle09_llama_model_export as export
import cycle09_offkd_eval as oe
import cycle09_s1_12_mmlupro as mmlu_audit
import cycle09_s1_9_ifeval as ifeval_audit


EVAL_ROOT = c.RUN_ROOT / "llama_behavior"
THINK_RUNNER = c.REPO / "Eval/component/think_math/runner_think.py"
TASKS = ("math500", "mmlu_pro", "ifeval")
SUMMARY_FIELDS = [
    "arm",
    "step",
    "task",
    "n",
    "accuracy",
    "standard_error",
    "cap_hit_rate",
    "eos_rate",
    "boxed_rate",
    "trunc_but_correct_rate",
    "strict_accuracy",
    "flexible_accuracy",
    "extract_failure_rate",
    "prompt_strict_accuracy",
    "instruction_strict_accuracy",
    "response_length_mean",
    "response_length_median",
    "response_length_p90",
    "max_new_tokens",
    "sample_source",
]


def math500_budget(step: int, smoke: bool) -> tuple[int, int]:
    if smoke:
        return 256, 2048
    # Match the established Qwen trajectory protocol: the early 0--20
    # checkpoints use the validated 4k budget, while 40+ uses 16k.
    return (4096, 6144) if step <= 20 else (16384, 18432)


def cell_root(arm: str, step: int, smoke: bool) -> Path:
    branch = "smoke" if smoke else "formal"
    label = "base" if step == 0 else arm
    return EVAL_ROOT / branch / label / f"step_{step:03d}"


def model_path(arm: str, step: int) -> Path:
    return export.merged_target(arm, step)


def result_json(root: Path) -> Path | None:
    paths = sorted(root.rglob("results_*.json")) if root.is_dir() else []
    return paths[-1] if paths else None


def run_command(command: list[str], cwd: Path) -> None:
    print("[CMD] " + " ".join(command), flush=True)
    environment = os.environ.copy()
    environment.update(
        {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    result = subprocess.run(command, cwd=cwd, env=environment)
    if result.returncode != 0:
        raise RuntimeError(f"command failed rc={result.returncode}: {' '.join(command)}")


def run_math(arm: str, step: int, root: Path, model: Path, smoke: bool, gpu_mem: float) -> Path:
    output = root / "math500"
    summary = output / f"step_{step:03d}.json"
    expected = 2 if smoke else 500
    if summary.is_file():
        try:
            generations = c.read_jsonl(output / f"step_{step:03d}_generations.jsonl")
            samples = c.read_jsonl(output / f"step_{step:03d}_samples.jsonl")
            if len(generations) == expected and len(samples) == expected:
                return summary
        except (OSError, ValueError):
            pass
        output.rename(output.with_name(f"math500_incomplete_{int(time.time())}"))
    max_tokens, max_model_len = math500_budget(step, smoke)
    command = [
        sys.executable,
        str(THINK_RUNNER),
        "--task",
        "math500",
        "--model",
        str(model),
        "--label",
        f"step_{step:03d}",
        "--n",
        "2" if smoke else "500",
        "--max-tokens",
        str(max_tokens),
        "--max-model-len",
        str(max_model_len),
        "--gpu-mem",
        str(gpu_mem),
        "--temperature",
        "0.6",
        "--top-p",
        "0.9",
        "--seed",
        "42",
        "--outdir",
        str(output),
    ]
    run_command(command, c.REPO)
    if not summary.is_file():
        raise RuntimeError(f"MATH500 summary absent: {summary}")
    return summary


def run_lm_eval(task: str, step: int, root: Path, model: Path, smoke: bool, gpu_mem: float) -> Path:
    output = root / task
    existing = result_json(output)
    if existing:
        if task == "mmlu_pro":
            expected_files = 14
            expected_rows = 14 if smoke else 1400
            paths = sample_files(output, "samples_mmlu_pro_*.jsonl")
            try:
                complete = len(paths) == expected_files and sum(len(c.read_jsonl(path)) for path in paths) == expected_rows
            except (OSError, ValueError):
                complete = False
        else:
            paths = sample_files(output, "samples_ifeval_*.jsonl")
            try:
                complete = len(paths) == 1 and len(ifeval_audit.read_rows(paths[0])) == (1 if smoke else 541)
            except (OSError, ValueError):
                complete = False
        if complete:
            return existing
        # Preserve incomplete/preempted sample logs, then rerun the formal cell.
        output.rename(output.with_name(f"{task}_incomplete_{int(time.time())}"))
    output.mkdir(parents=True, exist_ok=True)
    model_args = (
        f"pretrained={model},dtype=bfloat16,gpu_memory_utilization={gpu_mem},"
        "max_model_len=4096"
    )
    command = oe.lm_eval_prefix() + [
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        task,
        "--num_fewshot",
        "0",
        "--batch_size",
        "auto",
        "--seed",
        "42",
        "--output_path",
        str(output),
        "--log_samples",
    ]
    if task == "mmlu_pro":
        command.extend(["--limit", "1" if smoke else "100"])
        cwd = c.REPO
    elif task == "ifeval":
        command.extend(["--include_path", str(c.REPO / "Eval/tasks"), "--apply_chat_template"])
        if smoke:
            command.extend(["--limit", "1"])
        cwd = c.REPO / "Eval"
    else:
        raise ValueError(task)
    run_command(command, cwd)
    result = result_json(output)
    if result is None:
        raise RuntimeError(f"lm-eval result absent: {task}/step_{step:03d}")
    return result


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def math_summary(arm: str, step: int, path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    generations = c.read_jsonl(path.parent / f"step_{step:03d}_generations.jsonl")
    samples = c.read_jsonl(path.parent / f"step_{step:03d}_samples.jsonl")
    lengths = [int(row["resp_len"]) for row in generations]
    truncated_correct = sum(
        generation["finish"] == "length" and bool(sample["ok"])
        for generation, sample in zip(generations, samples, strict=True)
    )
    return {
        "arm": arm,
        "step": step,
        "task": "math500",
        "n": len(samples),
        "accuracy": summary["acc"],
        "standard_error": summary["stderr"],
        "cap_hit_rate": sum(row["finish"] == "length" for row in generations) / len(generations),
        "eos_rate": sum(row["finish"] == "stop" for row in generations) / len(generations),
        "boxed_rate": sum("\\boxed" in row["gen"] for row in generations) / len(generations),
        "trunc_but_correct_rate": truncated_correct / len(generations),
        "response_length_mean": statistics.fmean(lengths),
        "response_length_median": statistics.median(lengths),
        "response_length_p90": percentile(lengths, 0.90),
        "max_new_tokens": summary["max_tokens"],
        "sample_source": str(path.parent / f"step_{step:03d}_samples.jsonl"),
    }


def sample_files(root: Path, pattern: str) -> list[Path]:
    return sorted(root.rglob(pattern))


def response_lengths(tokenizer: Any, texts: list[str]) -> list[int]:
    lengths = []
    for start in range(0, len(texts), 128):
        encoded = tokenizer(
            texts[start : start + 128],
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def mmlu_summary(arm: str, step: int, root: Path, tokenizer: Any, smoke: bool) -> dict[str, Any]:
    paths = sample_files(root, "samples_mmlu_pro_*.jsonl")
    if len(paths) != 14:
        raise RuntimeError(f"MMLU-Pro sample files={len(paths)} expected=14: {root}")
    rows = []
    for path in paths:
        rows.extend(c.read_jsonl(path))
    expected = 14 if smoke else 1400
    if len(rows) != expected:
        raise RuntimeError(f"MMLU-Pro rows={len(rows)} expected={expected}")
    texts = [mmlu_audit.response_text(row) for row in rows]
    lengths = response_lengths(tokenizer, texts)
    strict = [mmlu_audit.strict_prediction(row) for row in rows]
    flexible = [mmlu_audit.flexible_extract(text)[0] for text in texts]
    targets = [str(row["target"]).upper() for row in rows]
    strict_ok = [prediction == target for prediction, target in zip(strict, targets, strict=True)]
    flexible_ok = [prediction == target for prediction, target in zip(flexible, targets, strict=True)]
    return {
        "arm": arm,
        "step": step,
        "task": "mmlu_pro",
        "n": len(rows),
        "strict_accuracy": statistics.fmean(strict_ok),
        "flexible_accuracy": statistics.fmean(flexible_ok),
        "extract_failure_rate": sum(prediction is None for prediction in strict) / len(strict),
        "response_length_mean": statistics.fmean(lengths),
        "response_length_median": statistics.median(lengths),
        "response_length_p90": percentile(lengths, 0.90),
        "sample_source": ";".join(map(str, paths)),
    }


def ifeval_summary(
    arm: str, step: int, root: Path, tokenizer: Any, smoke: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = sample_files(root, "samples_ifeval_*.jsonl")
    if len(paths) != 1:
        raise RuntimeError(f"IFEval sample files={len(paths)} expected=1: {root}")
    rows = ifeval_audit.read_rows(paths[0])
    expected = 1 if smoke else 541
    if len(rows) != expected:
        raise RuntimeError(f"IFEval rows={len(rows)} expected={expected}")
    texts = [ifeval_audit.response_text(row) for row in rows]
    lengths = response_lengths(tokenizer, texts)
    prompt = [bool(row["prompt_level_strict_acc"]) for row in rows]
    instruction = [bool(value) for row in rows for value in row["inst_level_strict_acc"]]
    categories = ifeval_audit.build_breakdown([{"arm": arm, "step": step, "rows": rows}])
    summary = {
        "arm": arm,
        "step": step,
        "task": "ifeval",
        "n": len(rows),
        "prompt_strict_accuracy": statistics.fmean(prompt),
        "instruction_strict_accuracy": statistics.fmean(instruction),
        "response_length_mean": statistics.fmean(lengths),
        "response_length_median": statistics.median(lengths),
        "response_length_p90": percentile(lengths, 0.90),
        "sample_source": str(paths[0]),
    }
    return summary, categories


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    if args.step == 0 and args.arm != "base":
        raise RuntimeError("step0 must be run once with --arm base")
    if args.step != 0 and args.arm not in c.ARMS:
        raise RuntimeError("nonzero behavior cells require a training arm")
    root = cell_root(args.arm, args.step, args.smoke)
    complete = root / "cell_manifest.json"
    cached = c.read_json(complete, {})
    if cached.get("status") == "complete":
        return cached
    model = model_path(args.arm, args.step)
    if not c.model_check(model)["complete"]:
        raise FileNotFoundError(f"model export incomplete: {model}")
    tokenizer = c.load_llama_tokenizer()
    math_path = run_math(args.arm, args.step, root, model, args.smoke, args.gpu_mem)
    run_lm_eval("mmlu_pro", args.step, root, model, args.smoke, args.gpu_mem)
    run_lm_eval("ifeval", args.step, root, model, args.smoke, args.gpu_mem)
    summaries = [math_summary(args.arm, args.step, math_path)]
    summaries.append(
        mmlu_summary(args.arm, args.step, root / "mmlu_pro", tokenizer, args.smoke)
    )
    ifeval, categories = ifeval_summary(
        args.arm, args.step, root / "ifeval", tokenizer, args.smoke
    )
    summaries.append(ifeval)
    c.atomic_csv(root / "summary.csv", [{field: row.get(field, "") for field in SUMMARY_FIELDS} for row in summaries], SUMMARY_FIELDS)
    c.atomic_csv(root / "ifeval_categories.csv", categories)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "arm": args.arm,
        "step": args.step,
        "model": c.model_check(model),
        "tasks": list(TASKS),
        "protocol": {
            "seed": 42,
            "math500": {
                "n": 2 if args.smoke else 500,
                "max_new_tokens": math500_budget(args.step, args.smoke)[0],
                "max_model_len": math500_budget(args.step, args.smoke)[1],
                "temperature": 0.6,
                "top_p": 0.9,
                "cap_protocol": "steps 0/5/20 at 4096; steps 40+ at 16384",
            },
            "mmlu_pro": {"n_per_category": 1 if args.smoke else 100, "chat_template": False, "num_fewshot": 0},
            "ifeval": {"n": 1 if args.smoke else 541, "apply_chat_template": True, "num_fewshot": 0},
        },
        "outputs": [c.artifact(root / "summary.csv"), c.artifact(root / "ifeval_categories.csv")],
        "created_utc": c.utc_now(),
    }
    c.atomic_json(complete, manifest)
    return manifest


def parse_names(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names or set(names).difference(allowed):
        raise ValueError(f"invalid names={value!r}; allowed={allowed}")
    return names


def parse_steps(value: str) -> tuple[int, ...]:
    steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not steps or set(steps).difference(c.MEASURED_CHECKPOINTS):
        raise ValueError(f"invalid measured steps={value!r}")
    return steps


def output_names(scope: str) -> tuple[Path, Path, Path]:
    if scope == "full":
        return (
            EVAL_ROOT / "llama_behavior_8ckpt.csv",
            EVAL_ROOT / "llama_ifeval_categories.csv",
            EVAL_ROOT / "llama_behavior_manifest.json",
        )
    stem = f"llama_{scope}"
    return (
        EVAL_ROOT / f"{stem}_behavior.csv",
        EVAL_ROOT / f"{stem}_ifeval_categories.csv",
        EVAL_ROOT / f"{stem}_behavior_manifest.json",
    )


def finalize(arms: tuple[str, ...], steps: tuple[int, ...], scope: str) -> dict[str, Any]:
    cells = [("base", 0)] + [
        (arm, step) for arm in arms for step in steps if step
    ]
    summaries = []
    categories = []
    manifests = []
    for arm, step in cells:
        root = cell_root(arm, step, False)
        manifest = c.read_json(root / "cell_manifest.json", {})
        if manifest.get("status") != "complete":
            raise RuntimeError(f"incomplete behavior cell: {arm}/{step}")
        manifests.append(c.artifact(root / "cell_manifest.json"))
        summaries.extend(pd.read_csv(root / "summary.csv").to_dict("records"))
        categories.extend(pd.read_csv(root / "ifeval_categories.csv").to_dict("records"))
    expected_task_rows = len(cells) * len(TASKS)
    if len(summaries) != expected_task_rows or len(categories) != len(cells) * 9:
        raise RuntimeError(
            f"behavior row-count drift: summary={len(summaries)}/{expected_task_rows} "
            f"categories={len(categories)}/{len(cells) * 9}"
        )
    local_summary, local_categories, manifest_path = output_names(scope)
    c.atomic_csv(local_summary, summaries, SUMMARY_FIELDS)
    c.atomic_csv(local_categories, categories)
    c.atomic_csv(c.MINI / local_summary.name, summaries, SUMMARY_FIELDS)
    c.atomic_csv(c.MINI / local_categories.name, categories)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "task": "Cycle09 block3 L2 Llama four-arm behavior",
        "scope": scope,
        "shared_base_cells": 1,
        "arms": list(arms),
        "nonzero_arm_cells": len(cells) - 1,
        "task_cells": expected_task_rows,
        "measured_checkpoints": list(steps),
        "cell_manifests": manifests,
        "outputs": [c.artifact(local_summary), c.artifact(local_categories)],
        "created_utc": c.utc_now(),
    }
    c.atomic_json(manifest_path, manifest)
    c.atomic_json(c.MINI / manifest_path.name, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cell", "finalize"), default="cell")
    parser.add_argument("--arm", choices=("base", *c.ARMS), default="base")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--arms", default=",".join(c.ARMS))
    parser.add_argument("--steps", default=",".join(map(str, c.MEASURED_CHECKPOINTS)))
    parser.add_argument("--scope", default="full")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = (
        finalize(parse_names(arguments.arms, c.ARMS), parse_steps(arguments.steps), arguments.scope)
        if arguments.phase == "finalize"
        else run_cell(arguments)
    )
    print(json.dumps(result, indent=2))
