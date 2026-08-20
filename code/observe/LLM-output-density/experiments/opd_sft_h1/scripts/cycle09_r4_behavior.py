#!/usr/bin/env python3
"""Rerun the selected MMLU-Pro cells with per-prompt logs for R4-3."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cycle09_r4_common as c

DEFAULT_STEPS = (0, 20, 40, 160)


def parse_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def result_file(output: Path) -> Path | None:
    candidates = sorted(output.rglob("results_*.json"))
    return candidates[-1] if candidates else None


def sample_files(output: Path) -> list[Path]:
    return sorted(output.rglob("samples_mmlu_pro_*.jsonl"))


def complete(output: Path) -> bool:
    return result_file(output) is not None and bool(sample_files(output))


def run_one(args: argparse.Namespace, arm: str, step: int) -> None:
    if step == 0 and arm == "sft":
        return
    output = args.run_root / "behavior/mmlu_pro" / arm / c.step_label(step)
    if complete(output):
        print(f"[Behavior skip] {arm}/{c.step_label(step)}", flush=True)
        return
    model = c.model_path(arm, step)
    output.mkdir(parents=True, exist_ok=True)
    model_args = (
        f"pretrained={model},dtype=bfloat16,"
        f"gpu_memory_utilization={args.gpu_mem},max_model_len={args.max_model_len}"
    )
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        "mmlu_pro",
        "--num_fewshot",
        "0",
        "--batch_size",
        "auto",
        "--seed",
        str(args.seed),
        "--output_path",
        str(output),
        "--limit",
        str(args.limit),
        "--log_samples",
    ]
    env = dict(os.environ)
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"[Behavior] {arm}/{c.step_label(step)} MMLU-Pro limit={args.limit}", flush=True)
    result = subprocess.run(cmd, cwd=str(c.REPO), env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"MMLU-Pro failed for {arm}/{c.step_label(step)} rc={result.returncode}"
        )
    if not complete(output):
        raise RuntimeError(f"MMLU-Pro output incomplete: {output}")


def aggregate_metric(path: Path) -> float | None:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    groups = data.get("groups", {})
    group = groups.get("mmlu_pro", {})
    for key, value in group.items():
        if key.startswith("exact_match") and not key.endswith("_stderr"):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    values = []
    for task in data.get("results", {}).values():
        for key, value in task.items():
            if key.startswith("exact_match") and "stderr" not in key:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    pass
                break
    return float(sum(values) / len(values)) if values else None


def summarize(args: argparse.Namespace) -> None:
    rows = []
    for arm in args.arms:
        for step in args.steps:
            source_arm = "opd" if step == 0 else arm
            output = args.run_root / "behavior/mmlu_pro" / source_arm / c.step_label(step)
            result = result_file(output)
            samples = sample_files(output)
            n_rows = sum(len(c.read_jsonl(path)) for path in samples)
            rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "source_arm": source_arm,
                    "status": "ok" if result is not None and samples else "missing",
                    "n_sample_files": len(samples),
                    "n_prompts": n_rows,
                    "aggregate_exact_match": aggregate_metric(result) if result else None,
                    "result_path": str(result) if result else "",
                    "sample_root": str(output),
                    "protocol": "lm_eval_vllm; 0-shot; no_chat_template; limit_per_subtask",
                }
            )
    c.write_csv_atomic(
        args.mini_root / "R4_behavior_prompt_manifest.csv",
        rows,
        list(rows[0]) if rows else [],
    )
    print(f"[Behavior summary] rows={len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-root", type=Path, default=c.RUN_ROOT)
    parser.add_argument("--mini-root", type=Path, default=c.MINI_ROOT)
    parser.add_argument("--arms", default=",".join(c.ARMS))
    parser.add_argument("--steps", default=",".join(map(str, DEFAULT_STEPS)))
    parser.add_argument("--limit", type=float, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()
    if args.all:
        args.run = args.summarize = True
    if args.smoke:
        args.run_root = args.run_root / "smoke_behavior"
        args.mini_root = args.mini_root / "smoke_behavior"
        args.arms = "opd"
        args.steps = "0"
        args.limit = 1
        args.run = args.summarize = True
    if not (args.run or args.summarize):
        parser.print_help()
        return
    args.arms = parse_names(args.arms)
    args.steps = parse_ints(args.steps)
    args.mini_root.mkdir(parents=True, exist_ok=True)
    if args.run:
        for arm in args.arms:
            for step in args.steps:
                run_one(args, arm, step)
    if args.summarize:
        summarize(args)


if __name__ == "__main__":
    main()
