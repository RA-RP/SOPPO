#!/usr/bin/env python3
"""Cycle 09 Round 3 OOD expansion runner (R3-6).

Runs IFEval and TruthfulQA-MC1 in one lm-eval/vLLM invocation per checkpoint
so the model is loaded once per arm/checkpoint. Results are descriptive
preservation readings only; this script does not create or modify claim gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path("/root/LLM-output-density")
SIDE = REPO / "experiments/opd_sft_h1"
if str(SIDE) not in sys.path:
    sys.path.insert(0, str(SIDE))

import cycle09_r2_unified_probe as r2  # noqa: E402

EVAL = REPO / "Eval"
DEFAULT_RUN = Path("/root/autodl-tmp/cycle09_r3")
DEFAULT_MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
ARMS = ("opd", "sft")
STEPS = (0, 5, 10, 20, 40, 160, 624)
TASKS = ("ifeval", "truthfulqa_mc1")


def step_label(step: int) -> str:
    return f"step_{int(step):03d}"


def parse_names(value: str, default: tuple[str, ...]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_ints(value: str, default: tuple[int, ...]) -> list[int]:
    if not value:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def lm_eval_binary() -> str:
    candidate = Path(sys.executable).resolve().with_name("lm_eval")
    return str(candidate) if candidate.exists() else (shutil.which("lm_eval") or "lm_eval")


def ensure_ifeval_dependencies(tasks: list[str]) -> None:
    if "ifeval" not in tasks:
        return
    try:
        import immutabledict  # noqa: F401
        import langdetect  # noqa: F401
        import nltk
        nltk.data.find("tokenizers/punkt_tab")
    except Exception as exc:
        raise RuntimeError(
            "IFEval requires langdetect, immutabledict, and NLTK punkt_tab before lm_eval starts"
        ) from exc


def result_json(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.rglob("results_*.json"))
    return candidates[-1] if candidates else None


def metrics_from_result(data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results", {})
    ifeval = results.get("ifeval", {})
    truthful = results.get("truthfulqa_mc1", {})
    return {
        "ifeval_prompt_strict": ifeval.get("prompt_level_strict_acc,none", ifeval.get("prompt_level_strict_acc", "")),
        "ifeval_instruction_strict": ifeval.get("inst_level_strict_acc,none", ifeval.get("inst_level_strict_acc", "")),
        "ifeval_prompt_loose": ifeval.get("prompt_level_loose_acc,none", ifeval.get("prompt_level_loose_acc", "")),
        "ifeval_instruction_loose": ifeval.get("inst_level_loose_acc,none", ifeval.get("inst_level_loose_acc", "")),
        "truthfulqa_mc1_acc": truthful.get("acc,none", truthful.get("acc", "")),
    }


def run_one(args: argparse.Namespace, arm: str, step: int) -> None:
    ensure_ifeval_dependencies(args.tasks)
    output = args.run_root / "ood_expansion" / arm / step_label(step)
    existing = result_json(output)
    if existing is not None:
        print(f"[Skip] {arm}/{step_label(step)} -> {existing}", flush=True)
        return

    model = r2.model_path_for(arm, step)
    if not (model / "config.json").exists():
        raise FileNotFoundError(f"Missing model: {model}")
    output.mkdir(parents=True, exist_ok=True)
    model_args = (
        f"pretrained={model},dtype=bfloat16,max_model_len={args.max_model_len},"
        f"gpu_memory_utilization={args.gpu_mem},enable_thinking=false"
    )
    cmd = [
        lm_eval_binary(),
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", ",".join(args.tasks),
        "--num_fewshot", "0",
        "--batch_size", "auto",
        "--output_path", str(output),
        "--include_path", str(EVAL / "tasks"),
        "--log_samples",
        "--apply_chat_template",
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    env = dict(os.environ)
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"[OOD] {arm}/{step_label(step)} tasks={args.tasks}", flush=True)
    result = subprocess.run(cmd, cwd=str(EVAL), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"lm_eval failed for {arm}/{step_label(step)} rc={result.returncode}")


def summarize(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for arm in args.arms:
        for step in args.steps:
            output = args.run_root / "ood_expansion" / arm / step_label(step)
            result = result_json(output)
            if result is None:
                rows.append(
                    {
                        "arm": arm,
                        "step": step,
                        "status": "missing",
                        "result_path": "",
                        "tasks": ",".join(args.tasks),
                        "protocol": "chat_template; enable_thinking=false; IFEval local YAML",
                    }
                )
                continue
            metrics = metrics_from_result(json.loads(result.read_text(encoding="utf-8")))
            rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "status": "ok",
                    "result_path": str(result),
                    "tasks": ",".join(args.tasks),
                    "protocol": "chat_template; enable_thinking=false; IFEval local YAML",
                    **metrics,
                }
            )
    fields = [
        "arm", "step", "status", "result_path", "tasks", "protocol",
        "ifeval_prompt_strict", "ifeval_instruction_strict", "ifeval_prompt_loose",
        "ifeval_instruction_loose", "truthfulqa_mc1_acc",
    ]
    path = args.mini_root / "R3_ood_expansion.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Summary] {path} rows={len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--mini-root", type=Path, default=DEFAULT_MINI)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--steps", default=",".join(map(str, STEPS)))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.run_root = args.run_root / "smoke_ood"
        args.mini_root = args.mini_root / "smoke_ood"
        args.run = True
        args.summarize = True
        args.arms = "opd"
        args.steps = "0"
        args.tasks = "ifeval"
        args.limit = 1
        args.max_model_len = 2048
    if args.all:
        args.run = True
        args.summarize = True
    if not (args.run or args.summarize):
        parser.print_help()
        return

    args.arms = parse_names(args.arms, ARMS)
    args.steps = parse_ints(args.steps, STEPS)
    args.tasks = parse_names(args.tasks, TASKS)
    unknown = set(args.tasks).difference(TASKS)
    if unknown:
        raise ValueError(f"Unsupported tasks: {sorted(unknown)}")
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.mini_root.mkdir(parents=True, exist_ok=True)
    r2.configure_roots(args.run_root, args.mini_root)
    print(f"[Plan] arms={args.arms} steps={args.steps} tasks={args.tasks}", flush=True)
    if args.dry_run:
        return
    if args.run:
        for arm in args.arms:
            for step in args.steps:
                run_one(args, arm, step)
    if args.summarize:
        summarize(args)


if __name__ == "__main__":
    main()

