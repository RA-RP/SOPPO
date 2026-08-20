#!/usr/bin/env python3
"""Assemble the OPD/SFT/offKD ten-checkpoint behavioral trajectory.

This is a provenance-preserving join. It does not recompute scores or interpret
the readings. Checkpoint-wide tasks must be complete at all ten points; Numina
and AIME retain their previously approved sparse protocols.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

REPO = Path("/root/LLM-output-density")
STEPS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
ARMS = ("opd", "sft", "offkd")
EVAL_ROOTS = {
    "opd": Path("/root/autodl-tmp/cycle08_opd_trajectory/eval"),
    "sft": Path("/root/autodl-tmp/cycle07_base_sft_trajectory/eval"),
}
CAP_RETEST = Path("/root/autodl-tmp/cap_unified_retest")
R3_ROOT = Path("/root/autodl-tmp/cycle09_r3")
OFFKD_ROOT = Path("/root/autodl-tmp/cycle09_offkd/eval")
COPYBACK = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/offkd"
)
MINI = COPYBACK.parent / "mini"
NUMINA_STEPS = (40, 160, 624)
NUMINA_CAP = 12288
AIME_CAP = 24576
AIME_SEEDS = tuple(range(42, 52))
MANDATORY_FIELDS = (
    "math500_acc",
    "gpqa_diamond_acc",
    "mmlu_pro_exact_match",
    "ifeval_prompt_strict",
    "truthfulqa_mc1_acc",
)


def step_label(step: int) -> str:
    return f"step_{step:03d}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def result_json(output: Path) -> Path | None:
    candidates = sorted(output.rglob("results_*.json")) if output.exists() else []
    return candidates[-1] if candidates else None


def metric(data: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return ""


def math_summary(arm: str, step: int) -> tuple[dict[str, Any], Path]:
    label = step_label(step)
    if step <= 20:
        path = EVAL_ROOTS[arm] / label / "math500" / f"{label}.json"
    else:
        path = CAP_RETEST / arm / label / "math500" / f"{label}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {arm} MATH500 result at step {step}: {path}")
    return read_json(path), path


def parse_lm_eval(arm: str, step: int) -> dict[str, Any]:
    label = step_label(step)
    root = EVAL_ROOTS[arm] / label
    gpqa_path = result_json(root / "gpqa")
    mmlu_path = result_json(root / "mmlu_pro")
    ood_path = result_json(R3_ROOT / "ood_expansion" / arm / label)
    if not gpqa_path or not mmlu_path or not ood_path:
        raise FileNotFoundError(
            f"Missing lm_eval result for {arm}/{label}: "
            f"gpqa={gpqa_path} mmlu={mmlu_path} ood={ood_path}"
        )
    gpqa = read_json(gpqa_path)["results"]["gpqa_diamond_zeroshot"]
    mmlu = read_json(mmlu_path)["results"]["mmlu_pro"]
    ood = read_json(ood_path)["results"]
    ifeval = ood["ifeval"]
    truthful = ood["truthfulqa_mc1"]
    return {
        "gpqa_diamond_n": gpqa.get("sample_len", ""),
        "gpqa_diamond_acc": metric(gpqa, "acc,none", "acc"),
        "mmlu_pro_n": mmlu.get("sample_len", ""),
        "mmlu_pro_exact_match": metric(
            mmlu, "exact_match,custom-extract", "exact_match"
        ),
        "ifeval_n": ifeval.get("sample_len", ""),
        "ifeval_prompt_strict": metric(
            ifeval, "prompt_level_strict_acc,none", "prompt_level_strict_acc"
        ),
        "ifeval_instruction_strict": metric(
            ifeval, "inst_level_strict_acc,none", "inst_level_strict_acc"
        ),
        "ifeval_prompt_loose": metric(
            ifeval, "prompt_level_loose_acc,none", "prompt_level_loose_acc"
        ),
        "ifeval_instruction_loose": metric(
            ifeval, "inst_level_loose_acc,none", "inst_level_loose_acc"
        ),
        "truthfulqa_mc1_n": truthful.get("sample_len", ""),
        "truthfulqa_mc1_acc": metric(truthful, "acc,none", "acc"),
    }


def id_summary_path(task: str, arm: str, step: int, cap: int, seed: int) -> Path:
    return (
        R3_ROOT
        / "id_completion"
        / task
        / arm
        / step_label(step)
        / f"cap_{cap}"
        / f"seed_{seed}.json"
    )


def parse_sparse_id(arm: str, step: int) -> dict[str, Any]:
    row: dict[str, Any] = {}
    numina_path = id_summary_path("numina", arm, step, NUMINA_CAP, 42)
    if numina_path.exists():
        data = read_json(numina_path)
        row.update(
            {
                "numina_n": data["n"],
                "numina_cap": data["cap"],
                "numina_acc": data["acc"],
                "numina_trunc_rate": data["trunc_rate"],
                "numina_mean_response_len": data["mean_response_len"],
                "numina_status": "formal_sparse",
            }
        )
    paths = [
        id_summary_path("aime24", arm, step, AIME_CAP, seed)
        for seed in AIME_SEEDS
    ]
    if all(path.exists() for path in paths):
        values = [read_json(path) for path in paths]
        row.update(
            {
                "aime24_n": values[0]["n"],
                "aime24_cap": values[0]["cap"],
                "aime24_seed_count": len(values),
                "aime24_acc_seed_mean": sum(float(value["acc"]) for value in values)
                / len(values),
                "aime24_trunc_rate_seed_mean": sum(
                    float(value["trunc_rate"]) for value in values
                )
                / len(values),
                "aime24_status": "secondary_peak_or_final",
            }
        )
    elif any(path.exists() for path in paths):
        raise RuntimeError(f"Partial AIME seed set for {arm}/{step_label(step)}")
    return row


def baseline_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ("opd", "sft"):
        for step in STEPS:
            math, source = math_summary(arm, step)
            row = {
                "arm": arm,
                "step": step,
                "checkpoint_source_type": "native_formal",
                "math500_n": math.get("n", ""),
                "math500_cap": math.get("max_tokens", ""),
                "math500_acc": math.get("acc", ""),
                "math500_trunc_rate": math.get("trunc_rate", ""),
                "math500_mean_response_len": math.get("mean_response_len", ""),
                "math500_source": str(source),
            }
            row.update(parse_sparse_id(arm, step))
            row.update(parse_lm_eval(arm, step))
            rows.append(row)
    return rows


def offkd_rows() -> list[dict[str, Any]]:
    path = OFFKD_ROOT / "offkd_eval_trajectory.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing offKD trajectory: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed = {int(row["step"]) for row in rows}
    if observed != set(STEPS):
        raise ValueError(f"offKD steps={sorted(observed)} expected={list(STEPS)}")
    for row in rows:
        row["step"] = int(row["step"])
        row["checkpoint_source_type"] = (
            "numerical_backfill_from_landmark"
            if row["step"] in (80, 320, 480)
            else "native_formal"
        )
        row["math500_source"] = str(
            OFFKD_ROOT
            / "generative"
            / step_label(int(row["step"]))
            / "math500"
            / f"{step_label(int(row['step']))}.json"
        )
        if row.get("numina_acc"):
            row["numina_status"] = "formal_sparse"
        if row.get("aime24_acc_seed_mean"):
            row["aime24_status"] = "secondary_peak_or_final"
    return rows


def main() -> None:
    rows = baseline_rows() + offkd_rows()
    errors = []
    for row in rows:
        missing = [field for field in MANDATORY_FIELDS if row.get(field, "") == ""]
        if missing:
            errors.append(
                {"arm": row["arm"], "step": row["step"], "missing": missing}
            )
    observed = {(str(row["arm"]), int(row["step"])) for row in rows}
    expected = {(arm, step) for arm in ARMS for step in STEPS}
    if observed != expected:
        errors.append(
            {
                "missing_cells": sorted(expected.difference(observed)),
                "extra_cells": sorted(observed.difference(expected)),
            }
        )
    if errors:
        raise RuntimeError(f"Three-arm trajectory is incomplete: {errors[:8]}")

    fields = [
        "arm",
        "step",
        "checkpoint_source_type",
        "math500_n",
        "math500_cap",
        "math500_acc",
        "math500_trunc_rate",
        "math500_mean_response_len",
        "math500_source",
        "numina_n",
        "numina_cap",
        "numina_acc",
        "numina_trunc_rate",
        "numina_mean_response_len",
        "numina_status",
        "aime24_n",
        "aime24_cap",
        "aime24_seed_count",
        "aime24_acc_seed_mean",
        "aime24_trunc_rate_seed_mean",
        "aime24_status",
        "gpqa_diamond_n",
        "gpqa_diamond_acc",
        "mmlu_pro_n",
        "mmlu_pro_exact_match",
        "ifeval_n",
        "ifeval_prompt_strict",
        "ifeval_instruction_strict",
        "ifeval_prompt_loose",
        "ifeval_instruction_loose",
        "truthfulqa_mc1_n",
        "truthfulqa_mc1_acc",
    ]
    COPYBACK.mkdir(parents=True, exist_ok=True)
    output = COPYBACK / "three_arm_full_trajectory.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at_unix": time.time(),
        "arms": list(ARMS),
        "checkpoint_grid": list(STEPS),
        "row_count": len(rows),
        "checkpoint_wide_complete_tasks": [
            "math500",
            "gpqa_diamond_zeroshot",
            "mmlu_pro",
            "ifeval",
            "truthfulqa_mc1",
        ],
        "sparse_protocols_preserved": {
            "numina": {"steps": list(NUMINA_STEPS), "n": 200, "cap": NUMINA_CAP},
            "aime24": {
                "steps": "per-arm MATH500 peak plus final",
                "n": 30,
                "cap": AIME_CAP,
                "seeds": list(AIME_SEEDS),
                "status": "secondary",
            },
        },
        "offkd_backfill": {
            "steps": [80, 320, 480],
            "validation": str(
                OFFKD_ROOT.parent
                / "checkpoint_backfill"
                / "backfill_validation.json"
            ),
            "caveat": (
                "numerically equivalent replay from nearest landmark; "
                "not bitwise identical to uninterrupted updates"
            ),
        },
        "output": str(output),
    }
    manifest_path = COPYBACK / "three_arm_full_trajectory_manifest.json"
    write_json_atomic(manifest_path, manifest)
    MINI.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, MINI / output.name)
    shutil.copy2(manifest_path, MINI / manifest_path.name)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
