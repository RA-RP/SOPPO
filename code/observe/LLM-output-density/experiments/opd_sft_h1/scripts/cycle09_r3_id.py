#!/usr/bin/env python3
"""Cycle 09 Round 3 ID completion runner (R3-7).

Implements the user-confirmed Numina cap pilot (N=64, final checkpoints,
12288/16384/24576), chooses one common cap by the 2pp next-cap rule, then
runs Numina N=200 at steps 40/160/624. AIME24 is secondary: N=30, cap 24576,
final plus each arm's MATH500 peak, averaged over ten sampling seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path("/root/LLM-output-density")
SIDE = REPO / "experiments/opd_sft_h1"
EVAL = REPO / "Eval"
COMPONENT = EVAL / "component"
for item in (SIDE, COMPONENT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import cycle09_r2_unified_probe as r2  # noqa: E402
from scorer import extract_pred  # noqa: E402
from scorer_v2 import score  # noqa: E402

DEFAULT_RUN = Path("/root/autodl-tmp/cycle09_r3")
DEFAULT_MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
NUMINA_PATH = Path("/root/autodl-tmp/prepared/NuminaMath-1___5/test.jsonl")
AIME_PATH = EVAL / "tasks/data/aime24/train.jsonl"
TRAJECTORY = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_08_h_opd_vs_sft_comparison/run_01/cap_unified_trajectory.csv"
)
ARMS = ("opd", "sft")
FINAL_STEP = 624
FORMAL_NUMINA_STEPS = (40, 160, 624)
PILOT_CAPS = (12288, 16384, 24576)
AIME_CAP = 24576
NUMINA_JUNK = {"", "not found", "notfound", "none", "nan", "proof"}
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."


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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_numina(n: int) -> list[dict[str, Any]]:
    rows = [
        row for row in read_jsonl(NUMINA_PATH)
        if str(row.get("answer", "")).strip().lower() not in NUMINA_JUNK
    ]
    records = [
        {
            "id": str(index),
            "problem": str(row["problem"]),
            "answer": str(row["answer"]),
        }
        for index, row in enumerate(rows)
    ]
    return records[:n]


def load_aime(n: int) -> list[dict[str, Any]]:
    rows = read_jsonl(AIME_PATH)
    records = [
        {
            "id": str(row.get("ID", index)),
            "problem": str(row["Problem"]),
            "answer": str(row["Answer"]),
        }
        for index, row in enumerate(rows)
    ]
    return records[:n]


def summary_path(root: Path, task: str, arm: str, step: int, cap: int, seed: int) -> Path:
    return root / "id_completion" / task / arm / step_label(step) / f"cap_{cap}" / f"seed_{seed}.json"


def samples_path(root: Path, task: str, arm: str, step: int, cap: int, seed: int) -> Path:
    return root / "id_completion" / task / arm / step_label(step) / f"cap_{cap}" / f"seed_{seed}_samples.jsonl"


def task_summary(
    task: str,
    arm: str,
    step: int,
    cap: int,
    seed: int,
    model: Path,
    rows: list[dict[str, Any]],
    outputs,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    correct = 0
    truncated = 0
    boxed = 0
    boxed_before_trunc = 0
    trunc_but_correct = 0
    response_lengths = []
    samples = []
    for row, output in zip(rows, outputs):
        completion = output.outputs[0]
        text = completion.text
        finish = completion.finish_reason
        ok = bool(score(text, row["answer"]))
        has_box = "\\boxed" in text
        is_truncated = finish == "length"
        correct += int(ok)
        truncated += int(is_truncated)
        boxed += int(has_box)
        boxed_before_trunc += int(is_truncated and has_box)
        trunc_but_correct += int(is_truncated and ok)
        response_lengths.append(len(completion.token_ids))
        samples.append(
            {
                "id": row["id"],
                "gold": row["answer"],
                "pred": extract_pred(text),
                "ok": ok,
                "finish": finish,
                "resp_len": len(completion.token_ids),
                "gen": text,
            }
        )
    total = len(rows)
    summary = {
        "task": task,
        "arm": arm,
        "step": int(step),
        "cap": int(cap),
        "seed": int(seed),
        "model": str(model),
        "n": total,
        "acc": correct / total if total else 0.0,
        "stderr": math.sqrt((correct / total) * (1 - correct / total) / total) if total else 0.0,
        "trunc_rate": truncated / total if total else 0.0,
        "boxed_rate": boxed / total if total else 0.0,
        "boxed_before_trunc_rate": boxed_before_trunc / truncated if truncated else 0.0,
        "trunc_but_correct": trunc_but_correct,
        "mean_response_len": sum(response_lengths) / total if total else 0.0,
        "temperature": 0.6,
        "top_p": 0.9,
    }
    return summary, samples


def run_model_cases(
    args: argparse.Namespace,
    task: str,
    arm: str,
    step: int,
    rows: list[dict[str, Any]],
    caps: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    missing = [
        (cap, seed)
        for cap in caps
        for seed in seeds
        if not summary_path(args.run_root, task, arm, step, cap, seed).exists()
    ]
    if not missing:
        return [
            read_json(summary_path(args.run_root, task, arm, step, cap, seed))
            for cap in caps for seed in seeds
        ]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = r2.model_path_for(arm, step)
    if not (model / "config.json").exists():
        raise FileNotFoundError(f"Missing model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["problem"] + INSTRUCTION}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    completed = []
    try:
        for cap, seed in missing:
            sampling = SamplingParams(
                temperature=0.6,
                top_p=0.9,
                max_tokens=cap,
                seed=seed,
            )
            outputs = llm.generate(prompts, sampling)
            summary, samples = task_summary(task, arm, step, cap, seed, model, rows, outputs)
            write_json(summary_path(args.run_root, task, arm, step, cap, seed), summary)
            target_samples = samples_path(args.run_root, task, arm, step, cap, seed)
            target_samples.parent.mkdir(parents=True, exist_ok=True)
            tmp = target_samples.with_suffix(target_samples.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            os.replace(tmp, target_samples)
            completed.append(summary)
            print(
                f"[{task}] {arm}/{step_label(step)} cap={cap} seed={seed} "
                f"acc={summary['acc']:.3f} trunc={summary['trunc_rate']:.3f}",
                flush=True,
            )
    finally:
        del llm, tokenizer
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return [
        read_json(summary_path(args.run_root, task, arm, step, cap, seed))
        for cap in caps for seed in seeds
    ]


def sample_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def paired_cap_rows(args: argparse.Namespace, arm: str, cap: int, next_cap: int) -> dict[str, Any]:
    left = sample_rows(samples_path(args.run_root, "numina_pilot", arm, FINAL_STEP, cap, args.seed))
    right = sample_rows(samples_path(args.run_root, "numina_pilot", arm, FINAL_STEP, next_cap, args.seed))
    if len(left) != len(right):
        raise ValueError("Pilot cap samples cannot be paired by row")
    if [row["id"] for row in left] != [row["id"] for row in right]:
        raise ValueError("Pilot cap sample IDs are not aligned")
    next_only = sum(int((not a["ok"]) and b["ok"]) for a, b in zip(left, right))
    current_only = sum(int(a["ok"] and (not b["ok"])) for a, b in zip(left, right))
    return {
        "arm": arm,
        "cap": cap,
        "next_cap": next_cap,
        "paired_n": len(left),
        "next_minus_current_acc": (sum(row["ok"] for row in right) - sum(row["ok"] for row in left)) / len(left),
        "next_only_correct": next_only,
        "current_only_correct": current_only,
    }


def select_common_cap(args: argparse.Namespace, pilot: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    candidates = list(args.pilot_caps)
    comparisons = []
    selected = candidates[-1]
    for index, cap in enumerate(candidates[:-1]):
        next_cap = candidates[index + 1]
        rows = [paired_cap_rows(args, arm, cap, next_cap) for arm in args.arms]
        comparisons.extend(rows)
        if all(float(row["next_minus_current_acc"]) <= 0.02 for row in rows):
            selected = cap
            break
    report = {
        "rule": "shortest cap whose next-cap accuracy gain is <= 0.02 for both final-checkpoint arms",
        "selected_cap": selected,
        "pilot_summaries": pilot,
        "paired_comparisons": comparisons,
    }
    write_json(args.mini_root / "R3_numina_cap_selection.json", report)
    return report


def run_pilot(args: argparse.Namespace) -> None:
    rows = load_numina(args.pilot_n)
    pilot: dict[str, dict[int, dict[str, Any]]] = {}
    csv_rows = []
    for arm in args.arms:
        summaries = run_model_cases(
            args, "numina_pilot", arm, FINAL_STEP, rows, args.pilot_caps, [args.seed]
        )
        pilot[arm] = {int(item["cap"]): item for item in summaries}
        for item in summaries:
            csv_rows.append({"row_type": "cap_summary", **item})
    report = select_common_cap(args, pilot)
    for item in report["paired_comparisons"]:
        csv_rows.append({"row_type": "paired_cap_comparison", **item})
    fields = sorted({key for row in csv_rows for key in row})
    path = args.mini_root / "R3_numina_cap_pilot.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[Pilot] selected common Numina cap={report['selected_cap']}", flush=True)


def selected_cap(mini_root: Path) -> int:
    path = mini_root / "R3_numina_cap_selection.json"
    if not path.exists():
        raise FileNotFoundError("Run --pilot before --numina-formal, or provide --numina-cap")
    return int(read_json(path)["selected_cap"])


def peak_steps() -> dict[str, int]:
    with open(TRAJECTORY, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm and row.get("math500_acc")]
        if not arm_rows:
            raise ValueError(f"No MATH500 peak source rows for {arm}")
        result[arm] = int(max(arm_rows, key=lambda row: float(row["math500_acc"]))["step"])
    return result


def run_numina_formal(args: argparse.Namespace) -> None:
    cap = args.numina_cap or selected_cap(args.mini_root)
    rows = load_numina(args.numina_n)
    for arm in args.arms:
        for step in args.numina_steps:
            run_model_cases(args, "numina", arm, step, rows, [cap], [args.seed])


def run_aime(args: argparse.Namespace) -> None:
    rows = load_aime(args.aime_n)
    peaks = peak_steps()
    for arm in args.arms:
        steps = sorted({FINAL_STEP, peaks[arm]})
        for step in steps:
            run_model_cases(
                args,
                "aime24",
                arm,
                step,
                rows,
                [args.aime_cap],
                [args.seed + offset for offset in range(args.aime_seeds)],
            )


def collect_id_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    root = args.run_root / "id_completion"
    for path in sorted(root.rglob("seed_*.json")) if root.exists() else []:
        data = read_json(path)
        rows.append({"row_type": "seed", "result_path": str(path), **data})
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["task"], row["arm"], int(row["step"]), int(row["cap"]))
        grouped.setdefault(key, []).append(row)
    for (task, arm, step, cap), values in grouped.items():
        if len(values) <= 1:
            continue
        rows.append(
            {
                "row_type": "seed_mean",
                "task": task,
                "arm": arm,
                "step": step,
                "cap": cap,
                "n": values[0]["n"],
                "seed_count": len(values),
                "acc": sum(float(value["acc"]) for value in values) / len(values),
                "trunc_rate": sum(float(value["trunc_rate"]) for value in values) / len(values),
                "boxed_before_trunc_rate": sum(
                    float(value["boxed_before_trunc_rate"]) for value in values
                ) / len(values),
                "secondary_caveat": "AIME24 N=30, ten sampling seeds; secondary only",
            }
        )
    return rows


def summarize(args: argparse.Namespace) -> None:
    rows = collect_id_rows(args)
    fields = sorted({key for row in rows for key in row}) if rows else ["row_type", "task", "arm", "step", "cap"]
    path = args.mini_root / "R3_id_completion.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Summary] {path} rows={len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--numina-formal", action="store_true")
    parser.add_argument("--aime", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--mini-root", type=Path, default=DEFAULT_MINI)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--pilot-caps", default=",".join(map(str, PILOT_CAPS)))
    parser.add_argument("--pilot-n", type=int, default=64)
    parser.add_argument("--numina-n", type=int, default=200)
    parser.add_argument("--numina-steps", default=",".join(map(str, FORMAL_NUMINA_STEPS)))
    parser.add_argument("--numina-cap", type=int, default=None)
    parser.add_argument("--aime-n", type=int, default=30)
    parser.add_argument("--aime-cap", type=int, default=AIME_CAP)
    parser.add_argument("--aime-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.run_root = args.run_root / "smoke_id"
        args.mini_root = args.mini_root / "smoke_id"
        args.pilot = True
        args.summarize = True
        args.arms = "opd"
        args.pilot_caps = "16"
        args.pilot_n = 1
        # A single Numina prompt can exceed 512 tokens before generation.
        # Keep the smoke tiny while leaving enough context for a valid request.
        args.max_model_len = 2048
    if args.all:
        args.pilot = True
        args.numina_formal = True
        args.aime = True
        args.summarize = True
    if not (args.pilot or args.numina_formal or args.aime or args.summarize):
        parser.print_help()
        return

    args.arms = parse_names(args.arms, ARMS)
    args.pilot_caps = parse_ints(args.pilot_caps, PILOT_CAPS)
    args.numina_steps = parse_ints(args.numina_steps, FORMAL_NUMINA_STEPS)
    if args.pilot_n <= 0 or args.numina_n <= 0 or args.aime_n <= 0:
        raise ValueError("sample counts must be positive")
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.mini_root.mkdir(parents=True, exist_ok=True)
    r2.configure_roots(args.run_root, args.mini_root)
    print(
        f"[Plan] arms={args.arms} pilot_caps={args.pilot_caps} "
        f"numina_n={args.numina_n} aime_n={args.aime_n}",
        flush=True,
    )
    if args.dry_run:
        return
    if args.pilot:
        run_pilot(args)
    if args.numina_formal:
        run_numina_formal(args)
    if args.aime:
        run_aime(args)
    if args.summarize:
        summarize(args)


if __name__ == "__main__":
    main()

