#!/usr/bin/env python3
"""Cycle 09 block 2 G2: seqKD behavior grid with immediate S1 audits."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import cycle09_offkd_eval as oe
import cycle09_s1_12_mmlupro as mmlu_audit
import cycle09_s1_9_ifeval as ifeval_audit


REPO = Path("/root/LLM-output-density")
RUN_ROOT = Path("/root/autodl-tmp/cycle09_seqkd")
EVAL_ROOT = RUN_ROOT / "eval"
MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
COPYBACK = MINI.parent / "seqkd"
PRIORITY_STEPS = (0, 5, 20, 40, 624, 10, 80, 160, 320, 480)
CRITICAL_STEPS = (0, 5, 20, 40, 624)
TASK_ORDER = ("math500", "mmlu_pro", "ifeval", "gpqa_diamond_zeroshot", "truthfulqa_mc1")
TRAJECTORY = MINI / "three_arm_full_trajectory.csv"
AUDIT_EXTRACT = MINI / "S1_mmlupro_extract_audit.csv"
AUDIT_FLEX = MINI / "S1_mmlupro_flexible.csv"
AUDIT_IFEVAL = MINI / "S1_ifeval_breakdown.csv"


def parse_steps(value: str) -> list[int]:
    steps = [int(item.strip()) for item in value.split(",") if item.strip()]
    unknown = sorted(set(steps) - set(PRIORITY_STEPS))
    if not steps or unknown or len(steps) != len(set(steps)):
        raise ValueError(f"invalid steps: {steps}; unknown={unknown}")
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default=",".join(map(str, PRIORITY_STEPS)))
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.steps = parse_steps(args.steps)
    args.run_root = RUN_ROOT
    args.eval_root = EVAL_ROOT / ("smoke" if args.smoke else "formal")
    args.copyback_root = COPYBACK
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def checkpoint_dir(step: int) -> Path:
    return RUN_ROOT / "checkpoints" / f"checkpoint-{step:06d}"


def direct_adapter_path(run_root: Path, step: int) -> Path:
    return run_root / "checkpoints" / f"checkpoint-{step:06d}"


def configure_shared_runner() -> None:
    oe.NATIVE_CHECKPOINT_STEPS = tuple(sorted(PRIORITY_STEPS))
    oe.BACKFILL_STEPS = ()
    oe.adapter_path = direct_adapter_path


def preflight(args: argparse.Namespace) -> None:
    manifest_path = RUN_ROOT / "checkpoints/training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    training = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not args.smoke
        and (training.get("status") != "complete" or int(training.get("completed_steps", -1)) != 624)
    ):
        raise RuntimeError(f"G1 is not complete: {training.get('status')} / {training.get('completed_steps')}")
    if list(training.get("checkpoint_grid", [])) != sorted(PRIORITY_STEPS):
        raise RuntimeError(f"G1 checkpoint grid drift: {training.get('checkpoint_grid')}")
    for step in args.steps:
        if step == 0:
            continue
        path = checkpoint_dir(step)
        for name in ("adapter_config.json", "adapter_model.safetensors", "complete.json"):
            if not (path / name).is_file():
                raise FileNotFoundError(path / name)
    free_gib = shutil.disk_usage(RUN_ROOT).free / 2**30
    if free_gib < 64:
        raise RuntimeError(f"only {free_gib:.1f} GiB free")
    print(f"[G2 preflight] steps={args.steps} free={free_gib:.1f}GiB", flush=True)


def result_path(root: Path) -> Path | None:
    return oe.result_json(root)


def run_lm_task(args: argparse.Namespace, step: int, task: str, *, smoke: bool) -> None:
    label = oe.step_label(step)
    if task in {"mmlu_pro", "gpqa_diamond_zeroshot"}:
        output = args.eval_root / "lm_eval" / label / task
        use_chat = False
    else:
        output = args.eval_root / "ood_expansion" / label / task
        use_chat = True
    if result_path(output) is not None:
        print(f"[G2 cached] {label}/{task}", flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"incomplete non-empty output requires audit: {output}")
    output.mkdir(parents=True, exist_ok=True)

    model_args = (
        f"pretrained={oe.model_path(args.run_root, step)},dtype=bfloat16,"
        f"gpu_memory_utilization={args.gpu_mem},max_model_len=4096"
    )
    if use_chat:
        model_args += ",enable_thinking=false"
    command = oe.lm_eval_prefix() + [
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", task,
        "--num_fewshot", "0",
        "--batch_size", "auto",
        "--seed", str(oe.SEED),
        "--output_path", str(output),
    ]
    if task == "mmlu_pro":
        command.extend(["--limit", "1" if smoke else "100", "--log_samples"])
    elif task == "gpqa_diamond_zeroshot" and smoke:
        command.extend(["--limit", "1"])
    elif use_chat:
        command.extend(
            [
                "--include_path", str(oe.EVAL_DIR / "tasks"),
                "--log_samples",
                "--apply_chat_template",
            ]
        )
        if smoke:
            command.extend(["--limit", "1"])
    oe.run_command(command, cwd=oe.EVAL_DIR if use_chat else REPO)
    if result_path(output) is None:
        raise RuntimeError(f"lm-eval returned without result: {label}/{task}")


def sample_files(root: Path, pattern: str) -> list[Path]:
    return sorted(root.rglob(pattern)) if root.is_dir() else []


def mmlu_frame(step: int, root: Path, tokenizer: Any, smoke: bool) -> pd.DataFrame:
    paths = sample_files(root, "samples_mmlu_pro_*.jsonl")
    expected_files = 14
    if len(paths) != expected_files:
        raise RuntimeError(f"MMLU sample files={len(paths)} expected={expected_files}: {root}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        subject = path.name.split("samples_mmlu_pro_", 1)[1].rsplit("_", 1)[0]
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                text = mmlu_audit.response_text(record)
                prediction = mmlu_audit.strict_prediction(record)
                flexible_prediction, tier = mmlu_audit.flexible_extract(text)
                rows.append(
                    {
                        "arm": "seqkd",
                        "step": step,
                        "subject": subject,
                        "target": str(record["target"]).upper(),
                        "strict_prediction": prediction,
                        "strict_extract_success": prediction is not None,
                        "strict_exact_match": float(record.get("exact_match", 0.0)),
                        "flexible_prediction": flexible_prediction,
                        "flexible_tier": tier,
                        "flexible_exact_match": float(flexible_prediction == str(record["target"]).upper()),
                        "response_text": text,
                        "response_chars": len(text),
                        "max_generation_tokens": mmlu_audit.max_generation_tokens(record),
                    }
                )
    expected_rows = 14 if smoke else 1400
    if len(rows) != expected_rows:
        raise RuntimeError(f"MMLU sample rows={len(rows)} expected={expected_rows}")
    for start in range(0, len(rows), 64):
        batch = rows[start : start + 64]
        encoded = tokenizer(
            [row["response_text"] for row in batch],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        for row, ids in zip(batch, encoded["input_ids"]):
            row["response_tokens"] = len(ids)
            row["failure_shape"] = mmlu_audit.failure_shape(
                row["response_text"],
                row["strict_extract_success"],
                row["response_tokens"],
                row["max_generation_tokens"],
            )
            del row["response_text"]
    return pd.DataFrame(rows)


def replace_rows(path: Path, new: pd.DataFrame, keys: list[str]) -> None:
    if path.is_file():
        old = pd.read_csv(path)
        selector = pd.Series(True, index=old.index)
        for key in keys:
            selector &= old[key].isin(new[key].unique())
        old = old[~selector]
        new = new.reindex(columns=old.columns)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new.copy()
    atomic_csv(path, combined.sort_values(keys, kind="stable").reset_index(drop=True))


def postprocess_mmlu(step: int, root: Path, tokenizer: Any, smoke: bool) -> None:
    frame = mmlu_frame(step, root, tokenizer, smoke)
    extract = mmlu_audit.extraction_audit(frame)
    flexible = mmlu_audit.flexible_table(frame)
    if smoke:
        print(extract.to_string(index=False), flush=True)
        print(flexible.to_string(index=False), flush=True)
        return
    replace_rows(AUDIT_EXTRACT, extract, ["arm", "step"])
    replace_rows(AUDIT_FLEX, flexible, ["arm", "step"])
    print(f"[G2 audit] MMLU seqkd/{step}: extract+flexible", flush=True)


def postprocess_ifeval(step: int, root: Path, smoke: bool) -> None:
    paths = sample_files(root, "samples_ifeval_*.jsonl")
    if len(paths) != 1:
        raise RuntimeError(f"IFEval sample files={len(paths)} expected=1: {root}")
    rows = ifeval_audit.read_rows(paths[0])
    expected = 1 if smoke else 541
    if len(rows) != expected:
        raise RuntimeError(f"IFEval rows={len(rows)} expected={expected}")
    breakdown = pd.DataFrame(
        ifeval_audit.build_breakdown([{"arm": "seqkd", "step": step, "rows": rows}])
    )
    if smoke:
        print(breakdown.to_string(index=False), flush=True)
        return
    replace_rows(AUDIT_IFEVAL, breakdown, ["arm", "step", "instruction_category"])
    print(f"[G2 audit] IFEval seqkd/{step}: {len(breakdown)} categories", flush=True)


def task_metrics(args: argparse.Namespace, step: int) -> dict[str, Any]:
    label = oe.step_label(step)
    output: dict[str, Any] = {}
    for task in ("mmlu_pro", "gpqa_diamond_zeroshot"):
        path = result_path(args.eval_root / "lm_eval" / label / task)
        if path:
            metrics = json.loads(path.read_text(encoding="utf-8")).get("results", {}).get(task, {})
            if task == "mmlu_pro":
                output["mmlu_pro_exact_match"] = oe.get_metric(metrics, "exact_match,custom-extract", "exact_match")
                output["mmlu_pro_n"] = metrics.get("sample_len", "")
            else:
                output["gpqa_diamond_acc"] = oe.get_metric(metrics, "acc,none", "acc")
                output["gpqa_diamond_n"] = metrics.get("sample_len", "")
    for task in ("ifeval", "truthfulqa_mc1"):
        path = result_path(args.eval_root / "ood_expansion" / label / task)
        if not path:
            continue
        metrics = json.loads(path.read_text(encoding="utf-8")).get("results", {}).get(task, {})
        if task == "ifeval":
            output.update(
                {
                    "ifeval_prompt_strict": oe.get_metric(metrics, "prompt_level_strict_acc,none", "prompt_level_strict_acc"),
                    "ifeval_instruction_strict": oe.get_metric(metrics, "inst_level_strict_acc,none", "inst_level_strict_acc"),
                    "ifeval_prompt_loose": oe.get_metric(metrics, "prompt_level_loose_acc,none", "prompt_level_loose_acc"),
                    "ifeval_instruction_loose": oe.get_metric(metrics, "inst_level_loose_acc,none", "inst_level_loose_acc"),
                    "ifeval_n": metrics.get("sample_len", ""),
                }
            )
        else:
            output["truthfulqa_mc1_acc"] = oe.get_metric(metrics, "acc,none", "acc")
            output["truthfulqa_mc1_n"] = metrics.get("sample_len", "")
    return output


def aggregate_step(args: argparse.Namespace, step: int) -> None:
    label = oe.step_label(step)
    math_path = args.eval_root / "generative" / label / "math500" / f"{label}.json"
    math = json.loads(math_path.read_text(encoding="utf-8"))
    row: dict[str, Any] = {
        "arm": "seqkd",
        "step": step,
        "checkpoint_source_type": "native_base" if step == 0 else "native_formal",
        "math500_n": math["n"],
        "math500_cap": math["max_tokens"],
        "math500_acc": math["acc"],
        "math500_trunc_rate": math["trunc_rate"],
        "math500_mean_response_len": math["mean_response_len"],
        "math500_source": str(math_path),
    }
    row.update(task_metrics(args, step))
    existing = pd.read_csv(TRAJECTORY)
    seq = existing[existing["arm"] == "seqkd"]
    old = existing[existing["arm"] != "seqkd"]
    seq = seq[seq["step"] != step]
    addition = pd.DataFrame([row]).reindex(columns=existing.columns)
    combined = pd.concat([old, seq, addition], ignore_index=True)
    arm_order = {"opd": 0, "sft": 1, "offkd": 2, "seqkd": 3}
    combined["_arm_order"] = combined["arm"].map(arm_order).fillna(99)
    combined = combined.sort_values(["_arm_order", "step"], kind="stable").drop(columns="_arm_order")
    atomic_csv(TRAJECTORY, combined)


def all_tasks_complete(args: argparse.Namespace, step: int) -> bool:
    label = oe.step_label(step)
    math = args.eval_root / "generative" / label / "math500" / f"{label}.json"
    roots = [
        args.eval_root / "lm_eval" / label / "mmlu_pro",
        args.eval_root / "ood_expansion" / label / "ifeval",
        args.eval_root / "lm_eval" / label / "gpqa_diamond_zeroshot",
        args.eval_root / "ood_expansion" / label / "truthfulqa_mc1",
    ]
    return math.is_file() and all(result_path(root) is not None for root in roots)


def update_manifest(args: argparse.Namespace, **updates: Any) -> dict[str, Any]:
    path = args.eval_root / "evaluation_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "schema_version": 1,
        "task": "Cycle 09 block 2 G2",
        "arm": "seqkd",
        "step_order": args.steps,
        "task_order": list(TASK_ORDER),
        "started_at": utc_now(),
        "protocol_source": "stage_plan_handoff.md second execution block",
    }
    data.update(updates)
    data["updated_at"] = utc_now()
    atomic_json(path, data)
    return data


def run(args: argparse.Namespace) -> None:
    configure_shared_runner()
    preflight(args)
    if args.dry_run:
        return
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(oe.BASE_MODEL, local_files_only=True)
    update_manifest(args, status="running", current_stage="merge_and_eval")
    for step in args.steps:
        label = oe.step_label(step)
        print(f"[G2] begin {label}", flush=True)
        oe.ensure_merged_model(args.run_root, step)
        max_tokens, max_model_len = oe.math500_budget(step)
        oe.run_think_eval(
            args,
            task="math500",
            step=step,
            n=2 if args.smoke else 500,
            max_tokens=256 if args.smoke else max_tokens,
            max_model_len=2048 if args.smoke else max_model_len,
        )
        run_lm_task(args, step, "mmlu_pro", smoke=args.smoke)
        postprocess_mmlu(
            step,
            args.eval_root / "lm_eval" / label / "mmlu_pro",
            tokenizer,
            args.smoke,
        )
        run_lm_task(args, step, "ifeval", smoke=args.smoke)
        postprocess_ifeval(
            step,
            args.eval_root / "ood_expansion" / label / "ifeval",
            args.smoke,
        )
        run_lm_task(args, step, "gpqa_diamond_zeroshot", smoke=args.smoke)
        run_lm_task(args, step, "truthfulqa_mc1", smoke=args.smoke)
        if not args.smoke:
            if not all_tasks_complete(args, step):
                raise RuntimeError(f"G2 incomplete cell: {step}")
            aggregate_step(args, step)
            completed = [value for value in args.steps if all_tasks_complete(args, value)]
            update_manifest(args, completed_steps=completed, current_step=step)
            if all(value in completed for value in CRITICAL_STEPS):
                atomic_json(
                    args.eval_root / "critical_steps_complete.json",
                    {"status": "complete", "steps": list(CRITICAL_STEPS), "created_at": utc_now()},
                )
        print(f"[G2] complete {label}", flush=True)
        if args.smoke:
            break
    if not args.smoke:
        update_manifest(args, status="complete", current_stage="complete", completed_at=utc_now())


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
