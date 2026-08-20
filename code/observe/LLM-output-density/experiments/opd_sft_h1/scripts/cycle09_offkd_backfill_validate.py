#!/usr/bin/env python3
"""Validate deterministic off-KD checkpoint backfills against the formal run."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

REPO = Path("/root/LLM-output-density")
EXP_ROOT = Path("/root/autodl-tmp/cycle09_offkd")
FORMAL_ROOT = EXP_ROOT / "checkpoints"
BACKFILL_ROOT = EXP_ROOT / "checkpoint_backfill"
COPYBACK = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/offkd"
)
BRANCHES = {
    "from_040": {"resume_step": 40, "end_step": 80},
    "from_160": {"resume_step": 160, "end_step": 480},
}
CHECKPOINT_SOURCES = {
    80: BACKFILL_ROOT / "from_040/checkpoint-000080",
    320: BACKFILL_ROOT / "from_160/checkpoint-000320",
    480: BACKFILL_ROOT / "from_160/checkpoint-000480",
}
EXACT_FIELDS = (
    "epoch_zero_based",
    "batch_in_epoch_zero_based",
    "response_tokens",
)
FLOAT_FIELDS = (
    "loss",
    "student_top32_mass",
    "teacher_top32_mass",
    "grad_norm_before_clip",
)
NUMERICAL_TOLERANCES = {
    "loss": 1e-3,
    "student_top32_mass": 5e-5,
    "teacher_top32_mass": 0.0,
}
GRAD_CLIP = 1.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            step = int(row["step"])
            if step in rows:
                raise ValueError(f"Duplicate metric step {step} in {path}")
            rows[step] = row
    return rows


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def validate_checkpoint(step: int, path: Path, errors: list[str]) -> dict[str, Any]:
    required = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "trainer_state.pt",
        "complete.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        errors.append(f"step {step} missing checkpoint files: {missing}")
        return {"step": step, "path": str(path), "status": "missing", "missing": missing}
    complete = read_json(path / "complete.json")
    if int(complete.get("step", -1)) != step:
        errors.append(f"step {step} complete.json reports {complete.get('step')}")
    sizes = {name: (path / name).stat().st_size for name in required}
    if any(size <= 0 for size in sizes.values()):
        errors.append(f"step {step} has an empty checkpoint file")
    return {
        "step": step,
        "path": str(path),
        "status": "ok",
        "file_sizes": sizes,
    }


def validate_branch(
    name: str,
    resume_step: int,
    end_step: int,
    formal: dict[int, dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    root = BACKFILL_ROOT / name
    manifest_path = root / "training_manifest.json"
    metrics_path = root / "train_metrics.jsonl"
    if not manifest_path.exists() or not metrics_path.exists():
        errors.append(f"{name} is missing manifest or metrics")
        return {"branch": name, "status": "missing"}
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        errors.append(f"{name} manifest status={manifest.get('status')}")
    if int(manifest.get("completed_steps", -1)) != end_step:
        errors.append(
            f"{name} completed_steps={manifest.get('completed_steps')} expected={end_step}"
        )
    branch = read_metrics(metrics_path)
    expected_steps = set(range(resume_step + 1, end_step + 1))
    missing_steps = sorted(expected_steps.difference(branch))
    extra_steps = sorted(set(branch).difference(expected_steps))
    if missing_steps:
        errors.append(f"{name} missing metric steps: {missing_steps[:8]}")
    if extra_steps:
        errors.append(f"{name} unexpected metric steps: {extra_steps[:8]}")

    max_abs_delta = {field: 0.0 for field in FLOAT_FIELDS}
    mismatches: list[dict[str, Any]] = []
    for step in sorted(expected_steps.intersection(branch).intersection(formal)):
        expected = formal[step]
        observed = branch[step]
        for field in EXACT_FIELDS:
            if observed.get(field) != expected.get(field):
                mismatches.append(
                    {
                        "step": step,
                        "field": field,
                        "formal": expected.get(field),
                        "backfill": observed.get(field),
                    }
                )
        for field in FLOAT_FIELDS:
            delta = abs(float(observed[field]) - float(expected[field]))
            max_abs_delta[field] = max(max_abs_delta[field], delta)
            if field == "grad_norm_before_clip":
                violation = max(float(observed[field]), float(expected[field])) >= GRAD_CLIP
            else:
                violation = delta > NUMERICAL_TOLERANCES[field]
            if violation:
                mismatches.append(
                    {
                        "step": step,
                        "field": field,
                        "formal": expected[field],
                        "backfill": observed[field],
                        "abs_delta": delta,
                    }
                )
    first_step = resume_step + 1
    first_step_bitwise_exact = all(
        branch[first_step].get(field) == formal[first_step].get(field)
        for field in EXACT_FIELDS + FLOAT_FIELDS
    )
    if not first_step_bitwise_exact:
        errors.append(f"{name} first post-resume step is not bitwise exact")
    formal_missing = sorted(expected_steps.difference(formal))
    if formal_missing:
        errors.append(f"formal metrics missing comparison steps: {formal_missing[:8]}")
    if mismatches:
        errors.append(f"{name} has {len(mismatches)} numerical-equivalence gate failures")
    return {
        "branch": name,
        "status": "pass" if not (missing_steps or extra_steps or formal_missing or mismatches or not first_step_bitwise_exact) else "fail",
        "resume_step": resume_step,
        "end_step": end_step,
        "compared_steps": len(expected_steps.intersection(branch).intersection(formal)),
        "numerical_tolerances": NUMERICAL_TOLERANCES,
        "grad_norm_gate": f"formal and replay both below clipping threshold {GRAD_CLIP}",
        "first_post_resume_step_bitwise_exact": first_step_bitwise_exact,
        "parity": "bitwise_exact" if all(value == 0.0 for value in max_abs_delta.values()) else "numerically_equivalent_not_bitwise_exact",
        "max_abs_delta": max_abs_delta,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:20],
        "resume_from": manifest.get("resume_from"),
    }


def main() -> None:
    errors: list[str] = []
    formal_manifest = read_json(FORMAL_ROOT / "training_manifest.json")
    if formal_manifest.get("status") != "complete":
        errors.append(f"formal training status={formal_manifest.get('status')}")
    if int(formal_manifest.get("completed_steps", -1)) != 624:
        errors.append(
            f"formal completed_steps={formal_manifest.get('completed_steps')} expected=624"
        )
    formal_metrics = read_metrics(FORMAL_ROOT / "train_metrics.jsonl")
    if set(formal_metrics) != set(range(1, 625)):
        errors.append("formal metrics do not contain exactly steps 1..624")

    branch_reports = [
        validate_branch(name, spec["resume_step"], spec["end_step"], formal_metrics, errors)
        for name, spec in BRANCHES.items()
    ]
    checkpoint_reports = [
        validate_checkpoint(step, path, errors)
        for step, path in CHECKPOINT_SOURCES.items()
    ]
    report = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "validated_at_unix": time.time(),
        "method": "first-step bitwise integrity plus bounded numerical replay equivalence",
        "caveat": "resumed bf16/CUDA updates are numerically equivalent but not bitwise identical to uninterrupted updates",
        "formal_root": str(FORMAL_ROOT),
        "branches": branch_reports,
        "checkpoints": checkpoint_reports,
        "errors": errors,
    }
    output = BACKFILL_ROOT / "backfill_validation.json"
    write_json_atomic(output, report)
    COPYBACK.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, COPYBACK / output.name)
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise RuntimeError("Backfill validation failed: " + "; ".join(errors[:8]))


if __name__ == "__main__":
    main()
