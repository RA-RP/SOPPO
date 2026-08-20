#!/usr/bin/env python3
"""CPU-only D8 verifier and final handoff writer for Cycle 09."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path("/root/LLM-output-density")
ROOT = Path("/root/autodl-tmp/cycle09_relative_functional_contraction")
FINAL = ROOT / "final"
AUDIT = ROOT / "audit"
MINI = REPO / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def json_data(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 << 20), b""):
            hasher.update(part)
    return hasher.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    os.replace(temporary, path)


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, records: list[dict[str, Any]]) -> None:
    require(bool(records), "branch-code records cannot be empty")
    fields = sorted({key for record in records for key in record})
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def meta(path: Path, stage: str) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "stage": stage,
        "name": path.name,
        "path": str(path),
        "rows": len(csv_rows(path)) if path.suffix == ".csv" else None,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def selector(rows: list[dict[str, str]], model: str, layer: str, track: str) -> dict[str, dict[str, str]]:
    selected = [
        row for row in rows
        if row.get("model") == model and row.get("layer") == layer
        and row.get("epsilon") == "0.05" and row.get("track") == track
    ]
    return {row["arm"]: row for row in selected if row.get("arm")}


def build_branch_code() -> dict[str, Any]:
    signed = csv_rows(FINAL / "relative_contraction_signed_nll_correlations.csv")
    detrended = csv_rows(FINAL / "relative_contraction_signed_nll_detrended.csv")
    interaction = csv_rows(FINAL / "relative_contraction_signed_nll_interaction.csv")
    llama_signed = selector(signed, "llama", "14", "legacy_llama")
    qwen_signed = selector(signed, "qwen", "18", "d4_merged_state")
    llama_detrended = selector(detrended, "llama", "14", "legacy_llama")
    qwen_detrended = selector(detrended, "qwen", "18", "d4_merged_state")
    require(set(("opd", "sft", "offkd", "seqkd")).issubset(llama_signed), "missing Llama headline signed-NLL arms")
    require(set(("opd", "sft", "offkd", "seqkd")).issubset(qwen_signed), "missing Qwen headline signed-NLL arms")
    lin = [
        row for row in interaction
        if row.get("model") == "llama" and row.get("layer") == "14"
        and row.get("epsilon") == "0.05" and row.get("track") == "legacy_llama"
    ]
    qin = [
        row for row in interaction
        if row.get("model") == "qwen" and row.get("layer") == "18"
        and row.get("epsilon") == "0.05" and row.get("track") == "d4_merged_state"
    ]
    require(len(lin) == 1 and len(qin) == 1, "missing headline interaction rows")
    qwen_offline_positive = all(float(qwen_signed[arm]["spearman"]) > 0.0 for arm in ("sft", "offkd", "seqkd"))
    opd_positive = float(llama_signed["opd"]["spearman"]) > 0.0 and float(qwen_signed["opd"]["spearman"]) > 0.0
    detrended_positive = float(llama_detrended["opd"]["spearman"]) > 0.0 and float(qwen_detrended["opd"]["spearman"]) > 0.0
    interaction_positive = float(lin[0]["opd_c_interaction"]) > 0.0 and float(qin[0]["opd_c_interaction"]) > 0.0
    if qwen_offline_positive:
        code, rule = "C", "Qwen D4 headline signed-NLL Spearman is positive for SFT, off-KD, and seqKD."
    elif opd_positive and detrended_positive and interaction_positive:
        code, rule = "A", "Both headline OPD, detrended OPD, and interaction signs are positive."
    elif opd_positive and not detrended_positive:
        code, rule = "B", "Headline OPD association is positive but the frozen detrended sign check fails."
    else:
        code, rule = "D", "No prior frozen branch condition is satisfied."
    return {
        "branch_code": code,
        "selection_rule": rule,
        "headline_epsilon": 0.05,
        "llama_opd_signed_spearman": float(llama_signed["opd"]["spearman"]),
        "qwen_opd_signed_spearman": float(qwen_signed["opd"]["spearman"]),
        "llama_opd_detrended_spearman": float(llama_detrended["opd"]["spearman"]),
        "qwen_opd_detrended_spearman": float(qwen_detrended["opd"]["spearman"]),
        "llama_opd_interaction": float(lin[0]["opd_c_interaction"]),
        "qwen_opd_interaction": float(qin[0]["opd_c_interaction"]),
        "qwen_sft_signed_spearman": float(qwen_signed["sft"]["spearman"]),
        "qwen_offkd_signed_spearman": float(qwen_signed["offkd"]["spearman"]),
        "qwen_seqkd_signed_spearman": float(qwen_signed["seqkd"]["spearman"]),
        "coder_scope": "mechanical_pre_registered_branch_code_no_theory_interpretation",
    }


def main() -> None:
    d2_manifest = AUDIT / "d2_completion_audit_manifest.json"
    d3_manifest = AUDIT / "d3_model_c_manifest.json"
    d4_manifest = ROOT / "d4_merged_state/formal/finalize_manifest.json"
    d5_manifest = ROOT / "d5_fairness/formal/finalize_manifest.json"
    d5d7_tables_manifest = MINI / "d5_d7_tables_manifest.json"
    parity_manifest = MINI / "qwen_merged_state_parity_manifest.json"
    d5d7_handoff_manifest = MINI / "d5_d7_raw_handoff_manifest.json"

    d2, d3, d4, d5, d5d7, parity = (
        json_data(path) for path in (d2_manifest, d3_manifest, d4_manifest, d5_manifest, d5d7_tables_manifest, parity_manifest)
    )
    require(d2.get("status") == "complete" and d2.get("grid_rows") == 1984, "D2 audit validation failed")
    require(d3.get("status") == "complete", "D3 is not complete")
    require(d3.get("schema_version") == "cycle09_d3_model_c_v2_qwen_d4_merged_state", "D3 was not rerun after D4")
    require(d3.get("by_model") == {"llama": 96, "qwen": 144}, "D3 model coverage mismatch")
    require(d3.get("grouped_rows") == 54 and d3.get("prediction_rows") == 2160, "D3 output count mismatch")
    require(d4.get("status") == "complete" and d5.get("status") == "complete" and d5d7.get("status") == "complete", "D4/D5/D5-D7 status mismatch")
    require(parity.get("status") == "complete_with_declared_opd_block", "D4 parity status mismatch")

    qmodule = csv_rows(FINAL / "qwen_d4_merged_state_module_audit.csv")
    qcells = csv_rows(FINAL / "qwen_d4_merged_state_all_cells.csv")
    qout = csv_rows(FINAL / "qwen_d4_merged_state_outputs.csv")
    require((len(qmodule), len(qcells), len(qout)) == (4144, 592, 148), "D4 row count mismatch")
    require(Counter(row["arm"] for row in qout) == Counter({"base": 4, "opd": 36, "sft": 36, "offkd": 36, "seqkd": 36}), "D4 arm coverage mismatch")

    d5module = csv_rows(FINAL / "d5_fairness_update_module.csv")
    d5equal = csv_rows(FINAL / "d5_fairness_update_equal7.csv")
    fairmodels = csv_rows(FINAL / "relative_contraction_fair_common_grid_models.csv")
    require(len(d5module) == 6720 and len(d5equal) == 960, "D5 row count mismatch")
    require(len(fairmodels) == 120 and {row["status"] for row in fairmodels} == {"complete"}, "D5 common-grid table incomplete")

    d6_paths = [
        FINAL / "relative_contraction_epsilon_layer_correlations.csv",
        FINAL / "relative_contraction_module_correlations.csv",
        FINAL / "relative_contraction_within_arm_checkpoint_domain.csv",
        FINAL / "relative_contraction_within_domain_checkpoint_arm.csv",
        FINAL / "relative_contraction_demeaned_correlations.csv",
    ]
    d7_paths = [
        FINAL / "relative_contraction_signed_nll_correlations.csv",
        FINAL / "relative_contraction_signed_nll_detrended.csv",
        FINAL / "relative_contraction_signed_nll_stepwise.csv",
        FINAL / "relative_contraction_signed_nll_grouped_models.csv",
        FINAL / "relative_contraction_signed_nll_interaction.csv",
        FINAL / "relative_contraction_signed_nll_predictions.csv",
    ]
    require(all(path.is_file() and csv_rows(path) for path in d6_paths + d7_paths), "D6/D7 file missing")
    for path in (FINAL / "relative_contraction_signed_nll_grouped_models.csv", FINAL / "relative_contraction_signed_nll_interaction.csv"):
        rows = csv_rows(path)
        headline_rows = [
            row for row in rows
            if (row.get("model"), row.get("layer"), row.get("epsilon"), row.get("track"))
            in {("llama", "14", "0.05", "legacy_llama"), ("qwen", "18", "0.05", "d4_merged_state")}
        ]
        require(len(headline_rows) == 2 and {row["status"] for row in headline_rows} == {"complete"}, f"headline D7 incomplete in {path.name}")

    branch = build_branch_code()
    branch_path = MINI / "relative_contraction_signed_nll_branch_codes.csv"
    atomic_csv(branch_path, [branch])

    artifact_groups = {
        "D2": [
            MINI / "relative_contraction_gap_path_audit.csv",
            MINI / "relative_contraction_recovered_artifacts.csv",
            MINI / "relative_contraction_unrecoverable_registry.csv",
            d2_manifest,
        ],
        "D3": [
            MINI / "relative_contraction_model_c_full_grouped.csv",
            MINI / "relative_contraction_model_c_full_predictions.csv",
            d3_manifest,
        ],
        "Llama_primary_sources": [
            FINAL / "relative_functional_contraction_all_cells.csv",
            FINAL / "relative_contraction_matched_cumulative_outputs.csv",
            FINAL / "relative_contraction_matched_stepwise_outputs.csv",
        ],
        "D4": [
            FINAL / "qwen_d4_merged_state_module_audit.csv",
            FINAL / "qwen_d4_merged_state_all_cells.csv",
            FINAL / "qwen_d4_merged_state_outputs.csv",
            d4_manifest,
            MINI / "qwen_merged_state_numeric_protocol.json",
            MINI / "qwen_merged_state_parity_audit.csv",
            parity_manifest,
        ],
        "D5": [
            FINAL / "d5_fairness_update_module.csv",
            FINAL / "d5_fairness_update_equal7.csv",
            d5_manifest,
            FINAL / "relative_contraction_fair_common_grid_models.csv",
            FINAL / "relative_contraction_fair_common_grid_predictions.csv",
        ],
        "D6": d6_paths,
        "D7": d7_paths + [branch_path],
        "D5_D7_aggregation": [d5d7_tables_manifest],
    }
    artifacts = [meta(path, stage) for stage, paths in artifact_groups.items() for path in paths]

    target_matrix = {
        "definition": "nonbase matched state/output cells at headline layer and epsilon=0.05",
        "llama": {
            "arms": ["opd", "sft", "offkd", "seqkd"],
            "checkpoints": [5, 20, 40, 80, 160, 320],
            "probes": ["E_general", "E_math", "E_ood", "E_if"],
            "expected_cells": 96,
            "completed_cells": 96,
            "completion_fraction": 1.0,
        },
        "qwen": {
            "arms": ["opd", "sft", "offkd", "seqkd"],
            "checkpoints": [5, 10, 20, 40, 80, 160, 320, 480, 624],
            "probes": ["E_general", "E_math", "E_ood", "E_if"],
            "expected_cells": 144,
            "completed_cells": 144,
            "completion_fraction": 1.0,
        },
        "combined": {"expected_cells": 240, "completed_cells": 240, "completion_fraction": 1.0},
        "common_grid": {
            "llama": [5, 20, 40, 80, 160, 320],
            "qwen": [5, 10, 20, 40, 80, 160, 320, 480, 624],
        },
    }
    completion = {
        "D2_inventory_path_protocol_audit": {"status": "complete", "grid_rows": 1984},
        "D3_model_c_full_availability": {"status": "complete_post_d4_rerun", "llama_cells": 96, "qwen_cells": 144, "grouped_cv_rows": 54, "prediction_rows": 2160},
        "D4_qwen_merged_state": {"status": "complete", "module_epsilon_rows": 4144, "equal7_rows": 592, "output_rows": 148, "parity_status": parity["status"]},
        "D5_common_grid_weight_fairness": {"status": "complete", "module_rows": 6720, "equal7_rows": 960, "fair_model_rows": 120},
        "D6_sensitivity": {"status": "complete", "files": len(d6_paths)},
        "D7_readout": {"status": "complete_headline_with_declared_legacy_outer_layer_deferrals", "files": len(d7_paths), "mechanical_branch_code": branch["branch_code"]},
    }

    handoff = MINI / "full_relative_functional_contraction_completion_handoff.md"
    manifest = MINI / "full_relative_functional_contraction_completion_manifest.json"
    atomic_text(handoff, f"""# Cycle 09 full relative functional contraction completion handoff

Status: COMPLETE_CORE_MATRIX  
Created: {now()}  
Scope: D2-D8 raw completion delivery. No new forward, training, behavior evaluation, or Theory interpretation.

## Completion Matrix

| model | arms | checkpoints | probes | matched state/output cells | completion |
| --- | --- | --- | --- | ---: | ---: |
| Llama | OPD, SFT, off-KD, seqKD | 5, 20, 40, 80, 160, 320 | E_general, E_math, E_ood, E_if | 96 / 96 | 100% |
| Qwen | OPD, SFT, off-KD, seqKD | 5, 10, 20, 40, 80, 160, 320, 480, 624 | E_general, E_math, E_ood, E_if | 144 / 144 | 100% |
| Combined | core arms, model-specific grids | as above | four core probes | 240 / 240 | 100% |

## D2-D7 Completion

| item | status | raw coverage |
| --- | --- | --- |
| D2 path/protocol audit | complete | 1,984 audit cells |
| D3 Model-C full availability | complete after D4 rerun | 54 grouped CV rows; 2,160 predictions |
| D4 Qwen merged state/output | complete | 4,144 module-epsilon; 592 equal-seven; 148 output rows |
| D4.1 parity | complete with declared OPD adapter block | 528 PASS_EXACT; 12 BLOCKED_MISSING_OPD_ADAPTER |
| D5 W/C/WS fairness | complete | 6,720 module rows; 960 equal-seven rows; 120 model rows |
| D6 sensitivity | complete | five formal tables |
| D7 readout | headline complete | six tables plus branch-code table |
| D8 final handoff | issued | this document and its JSON manifest |

## Mechanical D7.1 Branch Code

| code | frozen headline scope | selection rule |
| --- | --- | --- |
| {branch["branch_code"]} | epsilon=.05; Llama L14, Qwen L18 | {branch["selection_rule"]} |

This is a mechanical branch record from frozen raw D7 tables, not an additional theoretical or causal interpretation.

## Declared Boundaries

- Qwen OPD parity at steps 5/160/624 is blocked only for independent adapter B@A reconstruction. The primary full merged-state matrix and fixed-token outputs are complete; no merged-minus-base substitute was used.
- Four legacy Qwen outer-layer interaction rows are retained as DEFERRED_INSUFFICIENT_COMMON_GRID. They are not the D4 L18 primary track; the two primary headline interaction rows are complete.
- Per-file rows, SHA-256, paths, protocol references, full-availability/common-grid coverage, and predecessor handoffs are in:
{manifest}
""")

    payload = {
        "schema_version": "cycle09_full_relative_functional_contraction_completion_v1",
        "status": "COMPLETE_CORE_MATRIX",
        "created_utc": now(),
        "scope": "D2-D8 raw completion handoff",
        "completion": completion,
        "target_matrix": target_matrix,
        "d7_mechanical_branch_code": branch,
        "declared_boundaries": {
            "qwen_opd_parity": {"pass_exact_rows": 528, "blocked_rows": 12, "reason": "adapter B@A absent for independent reconstruction only"},
            "qwen_legacy_outer_layer_interaction": {"deferred_rows": 4, "status": "DEFERRED_INSUFFICIENT_COMMON_GRID", "primary_track_affected": False},
        },
        "protocol_references": {
            "qwen_numeric_protocol": str(MINI / "qwen_merged_state_numeric_protocol.json"),
            "d3_manifest": str(d3_manifest),
            "d4_manifest": str(d4_manifest),
            "d5_d7_tables_manifest": str(d5d7_tables_manifest),
        },
        "prior_handoffs": {
            "partial_c1_c5": str(MINI / "full_relative_functional_contraction_theory_handoff.md"),
            "d5_d7_raw": str(MINI / "d5_d7_raw_theory_handoff.md"),
            "d4_parity_addendum": str(MINI / "d4_parity_addendum_theory_handoff.md"),
        },
        "artifacts": artifacts + [meta(handoff, "D8_handoff_document")],
    }
    atomic_json(manifest, payload)

    prior = MINI / "d5_d7_raw_theory_handoff.md"
    prior_text = prior.read_text()
    old = "| D8 final completion handoff | not issued |"
    new = f"| D8 final completion handoff | issued: {handoff.name} |"
    if old in prior_text:
        atomic_text(prior, prior_text.replace(old, new, 1))
    elif new not in prior_text:
        fail("unexpected historical D8 status row")
    prior_manifest = json_data(d5d7_handoff_manifest)
    prior_manifest.setdefault("completion", {})["D8_final_completion_handoff"] = {
        "status": "issued",
        "completion_status": "COMPLETE_CORE_MATRIX",
        "path": str(handoff),
        "manifest": str(manifest),
    }
    prior_manifest["updated_utc"] = now()
    atomic_json(d5d7_handoff_manifest, prior_manifest)

    evolution = REPO / "mypaper/code/code_evolution.md"
    marker = "<!-- cycle09-d8-completion-handoff -->"
    existing = evolution.read_text()
    if marker not in existing:
        atomic_text(evolution, existing.rstrip() + f"""

---

{marker}

## Cycle 09 D3 v2 and D8 full relative-functional-contraction completion

Reran D3 after D4 using only the completed Qwen merged-state and matching fixed-token
outputs. The full-availability Model-C input is now 96 Llama plus 144 Qwen matched
state/output cells; all 54 grouped CV rows completed and 2,160 predictions were
written. The CPU-only D8 verifier checked D2-D7 coverage, rows, hashes, headline
table statuses, and the mechanical D7.1 branch code. It then issued the final
COMPLETE_CORE_MATRIX handoff, retaining the declared Qwen OPD parity boundary.

Raw Theory handoff:
mini/full_relative_functional_contraction_completion_handoff.md.
Machine-readable manifest:
mini/full_relative_functional_contraction_completion_manifest.json.

<!-- cycle09-d8-completion-handoff-end -->
""")

    print(json.dumps({
        "status": "COMPLETE_CORE_MATRIX",
        "handoff": str(handoff),
        "manifest": str(manifest),
        "branch_code": branch["branch_code"],
        "target_cells": target_matrix["combined"],
        "artifact_count": len(payload["artifacts"]),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
