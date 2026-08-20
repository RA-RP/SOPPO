#!/usr/bin/env python3
"""Freeze the formal T-SUPPORT input inventory from training artifacts only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cycle09_stage3_followup_common as c


OUTPUT = c.scoped_run("contracts") / "support_inputs.json"
STEPS = (5, 20, 40, 80, 160, 320)
QWEN_TOKENIZER = "/root/autodl-tmp/model/Qwen/Qwen3-4B-Base"
LLAMA_TOKENIZER = "/root/autodl-tmp/model/Meta/modelscope/Llama-3.2-3B"


def fixed_cells(
    *,
    family: str,
    arm: str,
    path: Path,
    support_kind: str,
    shared_source_group: str,
    objective_kind: str,
    tokenizer_path: str,
    metrics_path: Path | None = None,
    **fields: Any,
) -> list[dict[str, Any]]:
    rows = []
    for step in STEPS:
        row = {
            "family": family,
            "arm": arm,
            "step": step,
            "path": str(path),
            "support_kind": support_kind,
            "shared_source_group": shared_source_group,
            "objective_kind": objective_kind,
            "tokenizer_path": tokenizer_path,
            "notes": "fixed support reused across checkpoints; repeated rows are descriptive, not independent",
            **fields,
        }
        if metrics_path is not None:
            row["metrics_path"] = str(metrics_path)
        rows.append(row)
    return rows


def dynamic_cells(
    *,
    family: str,
    arm: str,
    root: Path,
    tokenizer_path: str,
    support_kind: str,
    metrics_path: Path | None,
    missing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cells = []
    for step in STEPS:
        path = root / f"{step}.jsonl"
        if not path.is_file():
            reason = (
                "declared terminal rollout gap after checkpoint commit"
                if step in (160, 320)
                else "training rollout artifact was not retained"
            )
            missing.append(
                {
                    "family": family,
                    "arm": arm,
                    "step": step,
                    "expected_path": str(path),
                    "reason": reason,
                }
            )
            continue
        row: dict[str, Any] = {
            "family": family,
            "arm": arm,
            "step": step,
            "path": str(path),
            "support_kind": support_kind,
            "objective_kind": "forward_kl",
            "tokenizer_path": tokenizer_path,
            "response_field": "output",
            "max_response_tokens": 10240,
            "notes": "on-policy batch physically saved at this training update",
        }
        if metrics_path is not None:
            row["metrics_path"] = str(metrics_path)
        cells.append(row)
    return cells


def write() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    qwen_sft = Path("/root/autodl-tmp/cycle07_base_sft_trajectory/data_prep/train_5k.jsonl")
    qwen_teacher = Path("/root/autodl-tmp/cycle09_offkd/rollout/teacher_rollout.jsonl")
    qwen_offkd_metrics = Path("/root/autodl-tmp/cycle09_offkd/checkpoints/train_metrics.jsonl")
    qwen_seqkd_metrics = Path("/root/autodl-tmp/cycle09_seqkd/checkpoints/train_metrics.jsonl")
    cells += fixed_cells(
        family="qwen3_4b",
        arm="sft",
        path=qwen_sft,
        support_kind="dataset_cot",
        shared_source_group="qwen_sft_dataset_fixed",
        objective_kind="cross_entropy",
        tokenizer_path=QWEN_TOKENIZER,
        response_field="text",
        text_after="<|im_start|>assistant\n",
        text_before="<|im_end|>",
    )
    cells += fixed_cells(
        family="qwen3_4b",
        arm="offkd",
        path=qwen_teacher,
        support_kind="teacher_rollout_fixed",
        shared_source_group="qwen_teacher_rollout_fixed",
        objective_kind="forward_kl",
        tokenizer_path=QWEN_TOKENIZER,
        metrics_path=qwen_offkd_metrics,
        response_field="generation",
        token_ids_field="generation_token_ids",
        max_prompt_tokens=1024,
        max_response_tokens=10240,
    )
    cells += fixed_cells(
        family="qwen3_4b",
        arm="seqkd",
        path=qwen_teacher,
        support_kind="teacher_rollout_fixed",
        shared_source_group="qwen_teacher_rollout_fixed",
        objective_kind="cross_entropy",
        tokenizer_path=QWEN_TOKENIZER,
        metrics_path=qwen_seqkd_metrics,
        response_field="generation",
        token_ids_field="generation_token_ids",
        max_prompt_tokens=1024,
        max_response_tokens=10240,
    )
    cells += dynamic_cells(
        family="qwen3_4b",
        arm="alpha05",
        root=Path("/root/autodl-tmp/cycle09_block3/qwen_alpha05/rollouts"),
        tokenizer_path=QWEN_TOKENIZER,
        support_kind="half_self_half_external_mixture",
        metrics_path=None,
        missing=missing,
    )
    for step in STEPS:
        missing.append(
            {
                "family": "qwen3_4b",
                "arm": "opd",
                "step": step,
                "expected_path": f"/root/autodl-tmp/cycle08_opd_trajectory/rollouts/{step}.jsonl",
                "reason": "original Cycle08 training rollouts were pruned; evaluation generations are forbidden substitutes",
            }
        )

    llama_sft = Path("/root/autodl-tmp/cycle09_block2/model2_llama/sft_data/records.jsonl")
    llama_teacher = Path("/root/autodl-tmp/cycle09_block2/model2_llama/rollout/teacher_rollout.jsonl")
    llama_metric_root = Path("/root/autodl-tmp/cycle09_block2/model2_llama/g6")
    cells += fixed_cells(
        family="llama3_2_3b",
        arm="sft",
        path=llama_sft,
        support_kind="dataset_cot",
        shared_source_group="llama_sft_dataset_fixed",
        objective_kind="cross_entropy",
        tokenizer_path=LLAMA_TOKENIZER,
        metrics_path=llama_metric_root / "sft/checkpoints/train_metrics.jsonl",
        token_ids_field="generation_token_ids",
        max_prompt_tokens=1024,
        max_response_tokens=10240,
    )
    cells += fixed_cells(
        family="llama3_2_3b",
        arm="offkd",
        path=llama_teacher,
        support_kind="teacher_rollout_fixed",
        shared_source_group="llama_teacher_rollout_fixed",
        objective_kind="forward_kl",
        tokenizer_path=LLAMA_TOKENIZER,
        metrics_path=llama_metric_root / "offkd/checkpoints/train_metrics.jsonl",
        response_field="generation",
        token_ids_field="generation_token_ids",
        max_prompt_tokens=1024,
        max_response_tokens=10240,
    )
    cells += fixed_cells(
        family="llama3_2_3b",
        arm="seqkd",
        path=llama_teacher,
        support_kind="teacher_rollout_fixed",
        shared_source_group="llama_teacher_rollout_fixed",
        objective_kind="cross_entropy",
        tokenizer_path=LLAMA_TOKENIZER,
        metrics_path=llama_metric_root / "seqkd/checkpoints/train_metrics.jsonl",
        response_field="generation",
        token_ids_field="generation_token_ids",
        max_prompt_tokens=1024,
        max_response_tokens=10240,
    )
    cells += dynamic_cells(
        family="llama3_2_3b",
        arm="opd",
        root=Path("/root/autodl-tmp/cycle09_block3/llama_opd/rollouts/raw"),
        tokenizer_path=LLAMA_TOKENIZER,
        support_kind="current_student_on_policy",
        metrics_path=Path(
            "/root/autodl-tmp/cycle09_block3/llama_opd/rollouts/canonical/"
            "llama_opd_step_metrics.csv"
        ),
        missing=missing,
    )

    absent_sources = sorted({cell["path"] for cell in cells if not Path(cell["path"]).is_file()})
    if absent_sources:
        raise FileNotFoundError(f"declared present support sources are absent: {absent_sources}")
    absent_metrics = sorted(
        {
            cell["metrics_path"]
            for cell in cells
            if cell.get("metrics_path") and not Path(cell["metrics_path"]).is_file()
        }
    )
    if absent_metrics:
        raise FileNotFoundError(f"declared training metric sources are absent: {absent_metrics}")

    payload = {
        "schema_version": 2,
        "status": "frozen_with_declared_missing_cells" if missing else "frozen",
        "checkpoint_grid": list(STEPS),
        "input_policy": "training support only; behavior/evaluation generations are forbidden",
        "cells": cells,
        "missing_cells": missing,
        "shared_source_groups": {
            "qwen_sft_dataset_fixed": {
                "arms": ["sft"],
                "steps": list(STEPS),
                "physical_sources": 1,
            },
            "qwen_teacher_rollout_fixed": {
                "arms": ["offkd", "seqkd"],
                "steps": list(STEPS),
                "physical_sources": 1,
            },
            "llama_sft_dataset_fixed": {
                "arms": ["sft"],
                "steps": list(STEPS),
                "physical_sources": 1,
            },
            "llama_teacher_rollout_fixed": {
                "arms": ["offkd", "seqkd"],
                "steps": list(STEPS),
                "physical_sources": 1,
            },
        },
        "alpha05_source_separated_companion": str(
            c.MINI / "qwen_alpha05_stage_b_support_stats.csv"
        ),
        "created_utc": c.utc_now(),
    }
    existing = c.read_json(OUTPUT, {})
    if existing:
        comparable_existing = {key: value for key, value in existing.items() if key != "created_utc"}
        comparable_new = {key: value for key, value in payload.items() if key != "created_utc"}
        if comparable_existing != comparable_new:
            raise RuntimeError(
                "frozen support inventory drift; write a versioned contract instead of overwriting"
            )
    else:
        c.atomic_json(OUTPUT, payload)
    return {
        "status": "complete",
        "output": c.artifact(OUTPUT),
        "cells": len(cells),
        "missing_cells": len(missing),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("write",), required=True)
    parser.parse_args()
    print(json.dumps(write(), indent=2))
