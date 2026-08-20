#!/usr/bin/env python3
"""Cycle 09 D2--D8 completion helpers.

The first phases are deliberately CPU-only.  They inventory existing state and
output artifacts, distinguish a recoverable legacy result from a D4.1-valid
merged-state result, and rerun Model-C without restricting it to the W/WS
common grid.  They never overwrite the historical C0--C5 tables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO = Path("/root/LLM-output-density")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cycle09_block3_common as b3  # noqa: E402
import cycle09_offkd_eval as offkd_eval  # noqa: E402
import cycle09_relative_functional_contraction as legacy  # noqa: E402
import cycle09_stage3_common as qstage  # noqa: E402


ROOT = b3.AUTODL / "cycle09_relative_functional_contraction"
FINAL = ROOT / "final"
AUDIT = ROOT / "audit"
MINI = REPO / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"

STAGE4_STATE = b3.AUTODL / "cycle09_stage4_state_displacement/outputs/state_rank_full_cells.csv"
H2_CORE = (
    b3.AUTODL
    / "cycle09_stage3_followup/partitions/partition_probe_qwen_20260723/H2_probe_core"
    / "PROBE_CORE_r_epsilon.csv"
)
H2_MANIFEST = H2_CORE.with_name("PROBE_CORE_manifest.json")
H2_CONTRACT = H2_CORE.with_name("probe_core_contract.json")

CORE_PROBES = ("E_general", "E_math", "E_ood", "E_if")
EPSILONS = (0.01, 0.025, 0.05, 0.10)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 << 20), b""):
            value.update(part)
    return value.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def qwen_steps(arm: str) -> tuple[int, ...]:
    if arm == "base":
        return (0,)
    return tuple(step for step in qstage.STEPS if step != 0)


def llama_steps(arm: str) -> tuple[int, ...]:
    if arm == "base":
        return (0,)
    return tuple(step for step in b3.MEASURED_CHECKPOINTS if step != 0)


def grid() -> Iterable[dict[str, Any]]:
    for model, steps_for, layers in (
        ("llama", llama_steps, legacy.LAYER_SCOPE["llama"]),
        ("qwen", qwen_steps, (18,)),
    ):
        for arm in ("base", *legacy.ARMS):
            for checkpoint in steps_for(arm):
                for probe in CORE_PROBES:
                    for layer in layers:
                        for epsilon in EPSILONS:
                            yield {
                                "model": model,
                                "arm": arm,
                                "checkpoint": checkpoint,
                                "probe_name": probe,
                                "layer": layer,
                                "epsilon": epsilon,
                            }


def qwen_adapter_path(arm: str, checkpoint: int) -> Path | None:
    if checkpoint == 0 or arm in ("base", "opd"):
        return None
    if arm == "sft":
        return b3.AUTODL / "cycle07_base_sft_trajectory/checkpoints" / f"step_{checkpoint:03d}"
    if arm == "offkd":
        return offkd_eval.adapter_path(b3.AUTODL / "cycle09_offkd", checkpoint)
    if arm == "seqkd":
        return b3.AUTODL / "cycle09_seqkd/checkpoints" / f"checkpoint-{checkpoint:06d}"
    raise ValueError(f"unsupported Qwen arm: {arm}")


def adapter_record(arm: str, checkpoint: int) -> dict[str, Any]:
    adapter = qwen_adapter_path(arm, checkpoint)
    if adapter is None:
        return {
            "adapter_path": None,
            "adapter_available": False,
            "adapter_reason": "not_applicable",
        }
    required = [adapter / "adapter_config.json", adapter / "adapter_model.safetensors"]
    if arm in ("offkd", "seqkd"):
        required.append(adapter / "complete.json")
    missing = [str(path) for path in required if not path.is_file()]
    return {
        "adapter_path": str(adapter),
        "adapter_available": not missing,
        "adapter_reason": "complete_adapter" if not missing else f"missing_adapter_files:{','.join(missing)}",
        "adapter_config_sha256": digest(adapter / "adapter_config.json"),
        "adapter_weights_sha256": digest(adapter / "adapter_model.safetensors"),
        "adapter_completion_sha256": digest(adapter / "complete.json"),
        "temporary_merged_target": str(ROOT / "d4_materialized/qwen" / arm / f"step_{checkpoint:03d}"),
    }


def model_record(model: str, arm: str, checkpoint: int) -> dict[str, Any]:
    try:
        available, reason = legacy.model_is_available(model, arm, checkpoint)
        path = legacy.path_for(model, arm, checkpoint)
    except Exception as error:  # Preserve the failure rather than infer a path.
        available, reason, path = False, f"path_resolution_error:{type(error).__name__}:{error}", Path()
    config = path / "config.json"
    index = next(
        (candidate for candidate in (path / "model.safetensors.index.json", path / "pytorch_model.bin.index.json") if candidate.is_file()),
        None,
    )
    result = {
        "model_path": str(path),
        "model_available": bool(available),
        "model_reason": reason,
        "config_sha256": digest(config),
        "weight_index_sha256": digest(index) if index else None,
        "full_weight_hash_status": "DEFERRED_TO_D4_MATERIALIZATION",
    }
    if model == "qwen":
        adapter = adapter_record(arm, checkpoint)
        result |= adapter
        if not available and adapter["adapter_available"]:
            result["model_recovery_status"] = "MATERIALIZABLE_FROM_COMPLETE_ADAPTER"
        elif available:
            result["model_recovery_status"] = "EXISTING_MERGED_MODEL"
        else:
            result["model_recovery_status"] = "UNRECOVERABLE_MODEL_AND_ADAPTER"
    else:
        result["model_recovery_status"] = "EXISTING_MERGED_MODEL" if available else "UNRECOVERABLE_MODEL_AND_ADAPTER"
    return result


def _legacy_state_index() -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    source = legacy.load_state_sources()
    index: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in source.to_dict("records"):
        key = (
            row["model"], row["arm"], int(row["checkpoint"]), row["probe_name"],
            int(row["layer"]), round(float(row["epsilon"]), 6),
        )
        index[key].append(row)
    return index


def _stage4_index() -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    if not STAGE4_STATE.is_file():
        return {}
    frame = pd.read_csv(STAGE4_STATE)
    frame = frame[frame["model"].eq("qwen")].copy()
    index: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in frame.to_dict("records"):
        key = (
            "qwen", str(row["arm"]), int(row["checkpoint"]), str(row["probe_name"]),
            int(row["layer"]), round(float(row["epsilon"]), 6),
        )
        index[key].append(row)
    return index


def _h2_index() -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    if not H2_CORE.is_file():
        return {}
    frame = pd.read_csv(H2_CORE)
    index: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in frame.to_dict("records"):
        probe = str(row["probe"])
        if probe not in CORE_PROBES:
            continue
        key = (
            "qwen", str(row["arm"]), int(row["step"]), probe,
            int(row["layer"]), round(float(row["epsilon"]), 6),
        )
        index[key].append(row)
    return index


def output_record(model: str, arm: str, checkpoint: int, probe: str) -> dict[str, Any]:
    metadata = legacy.output_meta_path(model, arm, checkpoint, probe, 0, 0)
    payload = read_json(metadata, {})
    if payload.get("status") != "complete":
        return {"output_status": "MISSING", "output_metadata": str(metadata)}
    return {
        "output_status": "LEGACY_CANDIDATE" if model == "qwen" else "AVAILABLE_COMPLETE",
        "output_metadata": str(metadata),
        "output_sha256": payload.get("sha256"),
        "output_sample_ids_sha256": payload.get("sample_ids_sha256"),
        "output_source_sample_ids_sha256": payload.get("source_sample_ids_sha256"),
        "output_protocol_status": (
            "MATCHED_SAMPLE_IDS" if payload.get("sample_ids_sha256") == payload.get("source_sample_ids_sha256")
            else "SAMPLE_ID_MISMATCH"
        ),
    }


def audit(_: argparse.Namespace) -> dict[str, Any]:
    legacy_index = _legacy_state_index()
    stage4_index = _stage4_index()
    h2_index = _h2_index()
    rows: list[dict[str, Any]] = []
    recovered: list[dict[str, Any]] = []
    unrecoverable: list[dict[str, Any]] = []

    for cell in grid():
        key = (*[cell[name] for name in ("model", "arm", "checkpoint", "probe_name", "layer")], round(float(cell["epsilon"]), 6))
        model = model_record(cell["model"], cell["arm"], int(cell["checkpoint"]))
        old_rows = legacy_index.get(key, [])
        stage4_rows = stage4_index.get(key, [])
        h2_rows = h2_index.get(key, [])
        candidates: list[dict[str, Any]] = []
        if old_rows:
            candidates.append({
                "kind": "legacy_relative_contraction_state",
                "rows": len(old_rows),
                "source_names": sorted({str(item["source_name"]) for item in old_rows}),
                "source_paths": sorted({str(item["source_path"]) for item in old_rows}),
                "admission": "D4.1_VALIDATION_REQUIRED" if cell["model"] == "qwen" else "CANDIDATE_EXISTING_PROTOCOL",
            })
        if stage4_rows:
            candidates.append({
                "kind": "stage4_qwen_state_rank",
                "rows": len(stage4_rows),
                "sample_count": sorted({int(item["sample_count"]) for item in stage4_rows}),
                "support_ruler": sorted({str(item["support_ruler"]) for item in stage4_rows}),
                "weight_arithmetic": sorted({str(item["weight_arithmetic"]) for item in stage4_rows}),
                "delta_w_source": sorted({str(item["delta_w_source"]) for item in stage4_rows}),
                "admission": "D4.1_VALIDATION_REQUIRED",
            })
        if h2_rows:
            candidates.append({
                "kind": "h2_qwen_exact_math500",
                "rows": len(h2_rows),
                "normalization": sorted({str(item["normalization"]) for item in h2_rows}),
                "manifest": str(H2_MANIFEST),
                "contract": str(H2_CONTRACT),
                "admission": "D4.1_REFORWARD_REQUIRED_IN_MEMORY_PEFT_LEGACY",
            })
        out = output_record(cell["model"], cell["arm"], int(cell["checkpoint"]), cell["probe_name"])
        state_status = "MISSING_STATE_RANK"
        if cell["model"] == "qwen" and candidates:
            state_status = "LEGACY_STATE_CANDIDATE_REQUIRES_D4.1"
        elif cell["model"] == "llama" and old_rows:
            state_status = "CANDIDATE_EXISTING_PROTOCOL"
        if not model["model_available"] and model["model_recovery_status"] == "UNRECOVERABLE_MODEL_AND_ADAPTER":
            state_status = "BLOCKED_MISSING_UPSTREAM_MODEL"
            unrecoverable.append(cell | model | {"reason": "model_unavailable"})
        row = cell | model | out | {
            "state_status": state_status,
            "candidate_count": len(candidates),
            "candidates_json": json.dumps(candidates, ensure_ascii=True, sort_keys=True),
            "audit_utc": now(),
        }
        rows.append(row)
        for candidate in candidates:
            recovered.append(cell | model | {
                "artifact_kind": candidate["kind"],
                "artifact_rows": candidate["rows"],
                "admission": candidate["admission"],
                "detail_json": json.dumps(candidate, ensure_ascii=True, sort_keys=True),
            })

    common_fields = sorted({field for row in rows for field in row})
    recovered_fields = sorted({field for row in recovered for field in row}) or ["model", "arm", "checkpoint"]
    blocked_fields = sorted({field for row in unrecoverable for field in row}) or ["model", "arm", "checkpoint", "reason"]
    write_csv(MINI / "relative_contraction_gap_path_audit.csv", rows, common_fields)
    write_csv(MINI / "relative_contraction_recovered_artifacts.csv", recovered, recovered_fields)
    write_csv(MINI / "relative_contraction_unrecoverable_registry.csv", unrecoverable, blocked_fields)
    manifest = {
        "schema_version": "cycle09_d2_audit_v1",
        "status": "complete",
        "created_utc": now(),
        "grid_rows": len(rows),
        "state_status_counts": pd.Series([row["state_status"] for row in rows]).value_counts().to_dict(),
        "output_status_counts": pd.Series([row["output_status"] for row in rows]).value_counts().to_dict(),
        "stage4_state_sha256": digest(STAGE4_STATE),
        "h2_core_sha256": digest(H2_CORE),
        "h2_manifest_sha256": digest(H2_MANIFEST),
        "h2_contract_sha256": digest(H2_CONTRACT),
        "outputs": [
            str(MINI / "relative_contraction_gap_path_audit.csv"),
            str(MINI / "relative_contraction_recovered_artifacts.csv"),
            str(MINI / "relative_contraction_unrecoverable_registry.csv"),
        ],
    }
    write_json(AUDIT / "d2_completion_audit_manifest.json", manifest)
    return manifest


def _model_c_input() -> pd.DataFrame:
    legacy_state = pd.read_csv(FINAL / "relative_functional_contraction_all_cells.csv")
    legacy_outputs = pd.read_csv(FINAL / "relative_contraction_matched_cumulative_outputs.csv")
    llama_state = legacy_state[
        legacy_state["model"].eq("llama")
        & legacy_state["module_count"].eq(7)
        & np.isclose(legacy_state["epsilon"], 0.05)
        & legacy_state["layer"].eq(legacy.HEADLINE_LAYER["llama"])
    ].copy()
    llama_outputs = legacy_outputs[legacy_outputs["model"].eq("llama")].copy()
    llama = llama_state.merge(
        llama_outputs,
        on=["model", "arm", "checkpoint", "probe_name"],
        how="inner",
    )

    # D4 is the only admissible Qwen state/output source after its merged-state
    # completion; do not re-admit pre-D4 legacy Qwen rows into Model-C.
    qwen_state = pd.read_csv(FINAL / "qwen_d4_merged_state_all_cells.csv")
    qwen_outputs = pd.read_csv(FINAL / "qwen_d4_merged_state_outputs.csv")
    qwen_state = qwen_state[
        qwen_state["module_count"].eq(7)
        & np.isclose(qwen_state["epsilon"], 0.05)
        & qwen_state["layer"].eq(18)
    ].copy()
    qwen = qwen_state.merge(
        qwen_outputs,
        on=["model", "arm", "checkpoint", "probe_name"],
        how="inner",
    )

    joined = pd.concat([llama, qwen], ignore_index=True, sort=False)
    joined = joined[
        joined["arm"].isin(legacy.ARMS) & joined["probe_name"].isin(CORE_PROBES)
    ].copy()
    return joined.rename(columns={"relative_functional_contraction_equal7": "c_epsilon"})


def _append_leave_out(
    frame: pd.DataFrame, target: str, rows: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> None:
    result, predicted = legacy.grouped_oof(frame, ["c_epsilon"], target, "checkpoint", "leave_one_checkpoint_out")
    rows.append({"feature_set": "Model-C", "target": target, "evaluation_protocol": "leave_one_checkpoint_out"} | result)
    predictions.extend([{ "feature_set": "Model-C" } | item for item in predicted])
    for arm in sorted(frame["arm"].unique()):
        train, test = frame[frame["arm"] != arm], frame[frame["arm"] == arm]
        result, predicted = legacy.train_predict(train, test, ["c_epsilon"], target, f"leave_one_arm_out:{arm}")
        rows.append({"feature_set": "Model-C", "target": target, "evaluation_protocol": f"leave_one_arm_out:{arm}"} | result)
        predictions.extend([{ "feature_set": "Model-C", "held_out_arm": arm } | item for item in predicted])
    for probe in sorted(frame["probe_name"].unique()):
        train, test = frame[frame["probe_name"] != probe], frame[frame["probe_name"] == probe]
        result, predicted = legacy.train_predict(train, test, ["c_epsilon"], target, f"leave_one_domain_out:{probe}")
        rows.append({"feature_set": "Model-C", "target": target, "evaluation_protocol": f"leave_one_domain_out:{probe}"} | result)
        predictions.extend([{ "feature_set": "Model-C", "held_out_domain": probe } | item for item in predicted])


def model_c(_: argparse.Namespace) -> dict[str, Any]:
    frame = _model_c_input()
    result_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for model, local in frame.groupby("model", sort=True):
        for target in ("cumulative_kl_base_to_current", "delta_nll_cumulative", "absolute_delta_nll_cumulative"):
            current_rows: list[dict[str, Any]] = []
            current_predictions: list[dict[str, Any]] = []
            _append_leave_out(local, target, current_rows, current_predictions)
            coverage = {
                "model": model,
                "rows_available": int(len(local)),
                "arms": ",".join(sorted(local["arm"].unique())),
                "checkpoints": ",".join(str(int(value)) for value in sorted(local["checkpoint"].unique())),
                "probes": ",".join(sorted(local["probe_name"].unique())),
                "state_protocols": ",".join(sorted(local["source_protocol"].astype(str).unique())),
                "coverage_kind": "full_availability_per_feature_set_existing_cells",
            }
            result_rows.extend([coverage | item for item in current_rows])
            prediction_rows.extend([coverage | item for item in current_predictions])
    write_csv(
        MINI / "relative_contraction_model_c_full_grouped.csv",
        result_rows,
        sorted({field for row in result_rows for field in row}),
    )
    write_csv(
        MINI / "relative_contraction_model_c_full_predictions.csv",
        prediction_rows,
        sorted({field for row in prediction_rows for field in row}),
    )
    manifest = {
        "schema_version": "cycle09_d3_model_c_v2_qwen_d4_merged_state",
        "status": "complete",
        "created_utc": now(),
        "input_rows": int(len(frame)),
        "by_model": frame.groupby("model").size().to_dict(),
        "grouped_rows": len(result_rows),
        "prediction_rows": len(prediction_rows),
        "state_sources": {
            "llama": str(FINAL / "relative_functional_contraction_all_cells.csv"),
            "qwen": str(FINAL / "qwen_d4_merged_state_all_cells.csv"),
        },
        "output_sources": {
            "llama": str(FINAL / "relative_contraction_matched_cumulative_outputs.csv"),
            "qwen": str(FINAL / "qwen_d4_merged_state_outputs.csv"),
        },
        "note": "Qwen uses only the D4 merged-state/full-output track; legacy Qwen state rows are excluded.",
    }
    write_json(AUDIT / "d3_model_c_manifest.json", manifest)
    return manifest


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "model-c", "cpu-all"), required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result: dict[str, Any] = {}
    if args.phase in ("audit", "cpu-all"):
        result["audit"] = audit(args)
    if args.phase in ("model-c", "cpu-all"):
        result["model_c"] = model_c(args)
    print(json.dumps({"status": "complete", "phases": list(result), "created_utc": now()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
