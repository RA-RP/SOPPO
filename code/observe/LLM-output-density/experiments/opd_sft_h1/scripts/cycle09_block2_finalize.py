#!/usr/bin/env python3
"""Validate Cycle 09 block-2 artifacts and append an idempotent raw handin."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/root/LLM-output-density")
BLOCK = Path("/root/autodl-tmp/cycle09_block2")
SEQKD = Path("/root/autodl-tmp/cycle09_seqkd")
MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
HANDIN = REPO / "mypaper/code/code_evolution.md"
OUTPUT = MINI / "block2_completion_manifest.json"
START_MARKER = "<!-- cycle09-block2-start -->"
END_MARKER = "<!-- cycle09-block2-end -->"
GRID = [0, 5, 10, 20, 40, 80, 160, 320, 480, 624]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path, predicate: bool, detail: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_json(path)
    if not predicate:
        raise RuntimeError(f"completion validation failed for {path}: {detail}; {payload}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing/empty artifact: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": csv_rows(path) if path.suffix == ".csv" else None,
        "sha256": sha256_file(path),
    }


def validate_manifests() -> dict[str, str]:
    paths = {
        "qwen_chain": BLOCK / "qwen_chain_status.json",
        "llama_chain": BLOCK / "llama_chain_status.json",
        "g1_seqkd": SEQKD / "checkpoints/training_manifest.json",
        "g2_eval": SEQKD / "eval/formal/evaluation_manifest.json",
        "g3_geometry": MINI / "seqkd_geometry_manifest.json",
        "g4_preflight": BLOCK / "model2_llama/g4_preflight/formal/manifest.json",
        "g5_rollout": BLOCK / "model2_llama/rollout/rollout_manifest.json",
        "g6_sft": BLOCK / "model2_llama/g6/sft/checkpoints/training_manifest.json",
        "g6_offkd": BLOCK / "model2_llama/g6/offkd/checkpoints/training_manifest.json",
        "g6_seqkd": BLOCK / "model2_llama/g6/seqkd/checkpoints/training_manifest.json",
        "g7_offkd_h": MINI / "offkd_h_geometry_manifest.json",
        "g8_ablation": MINI / "G8_adapter_ablation_manifest.json",
        "c1_direction": MINI / "C1_direction_all_probes_manifest.json",
        "c2_dose": MINI / "C2_dose_response_manifest.json",
    }
    payloads = {name: read_json(path) for name, path in paths.items()}
    checks = {
        "qwen_chain": payloads["qwen_chain"].get("status") == "complete",
        "llama_chain": payloads["llama_chain"].get("status") == "complete",
        "g1_seqkd": payloads["g1_seqkd"].get("status") == "complete"
        and int(payloads["g1_seqkd"].get("completed_steps", -1)) == 624,
        "g2_eval": payloads["g2_eval"].get("status") == "complete"
        and set(payloads["g2_eval"].get("completed_steps", [])) == set(GRID),
        "g3_geometry": payloads["g3_geometry"].get("status") == "complete"
        and payloads["g3_geometry"].get("steps") == GRID,
        "g4_preflight": payloads["g4_preflight"].get("status") == "complete"
        and payloads["g4_preflight"].get("decision") == "GO",
        "g5_rollout": int(payloads["g5_rollout"].get("n_prompts", -1)) == 5000
        and int(payloads["g5_rollout"].get("logprob_pass2", {}).get("topk", -1)) == 32,
        "g6_sft": payloads["g6_sft"].get("status") == "complete"
        and int(payloads["g6_sft"].get("completed_steps", -1)) == 624,
        "g6_offkd": payloads["g6_offkd"].get("status") == "complete"
        and int(payloads["g6_offkd"].get("completed_steps", -1)) == 624,
        "g6_seqkd": payloads["g6_seqkd"].get("status") == "complete"
        and int(payloads["g6_seqkd"].get("completed_steps", -1)) == 624,
        "g7_offkd_h": payloads["g7_offkd_h"].get("status") == "complete",
        "g8_ablation": payloads["g8_ablation"].get("status") == "complete",
        "c1_direction": payloads["c1_direction"].get("status") == "complete",
        "c2_dose": payloads["c2_dose"].get("status") == "complete",
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"incomplete block-2 manifests: {failed}")
    return {name: sha256_file(path) for name, path in paths.items()}


def handin_section(artifacts: list[dict[str, Any]]) -> str:
    lines = [
        START_MARKER,
        "",
        "## Cycle 09 Stage 1 - Second execution block raw handin",
        "",
        "Validated completion: Qwen G1/G2/G3/G8; Llama G4/G5/G6; pulled-forward G7; CPU C1/C2. "
        "This section records raw artifacts and provenance only; no interpretation or adjudication.",
        "",
        "| artifact | rows | sha256 |",
        "|---|---:|---|",
    ]
    for item in artifacts:
        rows = "-" if item["rows"] is None else str(item["rows"])
        lines.append(f"| `{item['path']}` | {rows} | `{item['sha256']}` |")
    lines.extend(
        [
            "",
            "Dual-card provenance: GPU0 retained the detached canonical supervisor and remained the sole "
            "writer of shared CSV/manifest outputs. GPU1 staged G2 Math500 cells for steps 320/480 and "
            "published them atomically after provenance validation; G3 used per-checkpoint locks and a "
            "single finalizer; G8 used per-config locks and a single finalizer. Seeds, caps, sample counts, "
            "checkpoint grid, adapter B@A fp32 path, base references, and fp64 theta numerics were unchanged.",
            "",
            END_MARKER,
        ]
    )
    return "\n".join(lines)


def update_handin(section: str) -> None:
    text = HANDIN.read_text(encoding="utf-8")
    if START_MARKER in text and END_MARKER in text:
        before, remainder = text.split(START_MARKER, 1)
        _old, after = remainder.split(END_MARKER, 1)
        updated = before.rstrip() + "\n\n" + section + after
    else:
        updated = text.rstrip() + "\n\n---\n\n" + section + "\n"
    temporary = HANDIN.with_suffix(".md.tmp")
    temporary.write_text(updated, encoding="utf-8")
    os.replace(temporary, HANDIN)


def main() -> None:
    manifest_hashes = validate_manifests()
    artifact_paths = [
        MINI / "three_arm_full_trajectory.csv",
        MINI / "S1_mmlupro_extract_audit.csv",
        MINI / "S1_mmlupro_flexible.csv",
        MINI / "S1_ifeval_breakdown.csv",
        MINI / "R4_v2_spectra_seqkd.csv",
        MINI / "R4_m1_tail_ec.csv",
        MINI / "R4_m2_output_drift.csv",
        MINI / "R5_theta_reps.csv",
        MINI / "G8_adapter_ablation.csv",
        MINI / "S1_direction_analysis.csv",
        MINI / "C2_dose_response.csv",
    ]
    artifacts = [artifact(path) for path in artifact_paths]
    update_handin(handin_section(artifacts))
    payload = {
        "status": "complete",
        "task": "Cycle 09 Stage 1 second execution block",
        "completed_at": utc_now(),
        "validated_grid": GRID,
        "manifest_sha256": manifest_hashes,
        "artifacts": artifacts,
        "code_evolution": {
            "path": str(HANDIN),
            "sha256": sha256_file(HANDIN),
            "marker": START_MARKER,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(f"[block2 finalize] complete: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
