#!/usr/bin/env python3
"""Build the complete Cycle 09 second-execution-block Theory handoff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

import cycle09_block2_intermediate_handoff as common


MINI = common.MINI
GRID = common.FULL_GRID
COMPLETION = MINI / "block2_completion_manifest.json"
HANDOFF = MINI / "mini_block2_theory_handoff.md"
HANDOFF_MANIFEST = MINI / "block2_theory_handoff_manifest.json"

G1_MANIFEST = (
    common.ROOT
    / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/seqkd/training_manifest.json"
)
G2_MANIFEST = Path("/root/autodl-tmp/cycle09_seqkd/eval/formal/evaluation_manifest.json")
G3_MANIFEST = MINI / "seqkd_geometry_manifest.json"
G8_MANIFEST = MINI / "G8_adapter_ablation_manifest.json"
G8_SOURCE = MINI / "G8_adapter_ablation.csv"
SPECTRA = MINI / "R4_v2_spectra_seqkd.csv"
M1 = MINI / "R4_m1_tail_ec.csv"
M2 = MINI / "R4_m2_output_drift.csv"
THETA = MINI / "R5_theta_reps.csv"

SNAPSHOTS = {
    "g2_behavior": MINI / "block2_final_g2_behavior.csv",
    "g2_extract": MINI / "block2_final_g2_mmlupro_extract.csv",
    "g2_flexible": MINI / "block2_final_g2_mmlupro_flexible.csv",
    "g2_ifeval": MINI / "block2_final_g2_ifeval_breakdown.csv",
    "g3_l18": MINI / "block2_final_g3_l18_summary.csv",
    "g4_examples": MINI / "block2_final_g4_examples5.csv",
    "g8": MINI / "block2_final_g8_adapter_ablation.csv",
}

G3_FIELDS = [
    "arm", "step", "task_id", "probe_type", "domain", "generation_seed",
    "layer", "module_count", "effective_rank_mean", "r_epsilon_005_mean",
    "r_epsilon_delta_mean", "drift_core_mean", "tail_energy_r32_mean",
    "m2_x0_mean", "theta_module_count", "theta_u_max_deg_mean",
    "theta_u_mean_deg_mean", "theta_v_max_deg_mean", "theta_v_mean_deg_mean",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=usecols)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    common.write_csv_atomic(path, list(frame.columns), frame.to_dict(orient="records"))


def md(frame: pd.DataFrame) -> str:
    return common.markdown_table(list(frame.columns), frame.to_dict(orient="records"))


def with_step(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["_step"] = frame["step"].astype(float).astype(int)
    return frame


def sort_grid(frame: pd.DataFrame, task_order: list[str] | None = None) -> pd.DataFrame:
    frame = frame.copy()
    frame["_grid_order"] = frame["_step"].map({step: i for i, step in enumerate(GRID)})
    columns = ["_grid_order"]
    if task_order is not None:
        frame["_task_order"] = frame["task_id"].map(
            {task: i for i, task in enumerate(task_order)}
        )
        columns.append("_task_order")
    return frame.sort_values(columns).drop(
        columns=["_grid_order", "_task_order"], errors="ignore"
    )


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def format_numbers(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = frame[column].map(
            lambda value: "" if pd.isna(value) else format(float(value), ".17g")
        )
    return frame


def build_g2() -> dict[str, pd.DataFrame]:
    trajectory = with_step(read_csv(common.TRAJECTORY))
    extract = with_step(read_csv(common.MMLU_EXTRACT))
    flexible = with_step(read_csv(common.MMLU_FLEXIBLE))
    ifeval = with_step(read_csv(common.IFEVAL_BREAKDOWN))

    trajectory = trajectory[
        (trajectory["arm"] == "seqkd") & trajectory["_step"].isin(GRID)
    ]
    extract = extract[(extract["arm"] == "seqkd") & extract["_step"].isin(GRID)]
    flexible = flexible[
        (flexible["arm"] == "seqkd") & flexible["_step"].isin(GRID)
    ]
    ifeval = ifeval[(ifeval["arm"] == "seqkd") & ifeval["_step"].isin(GRID)]

    require(len(trajectory) == 10, f"G2 trajectory rows={len(trajectory)}")
    require(len(extract) == 10, f"G2 extract rows={len(extract)}")
    require(len(flexible) == 10, f"G2 flexible rows={len(flexible)}")
    require(len(ifeval) == 90, f"G2 IFEval rows={len(ifeval)}")
    for step in GRID:
        require((ifeval["_step"] == step).sum() == 9, f"IFEval step {step} incomplete")

    behavior = trajectory.merge(
        flexible[["_step", "mmlu_pro_flexible", "strict_extract_fail_rate"]],
        on="_step",
        validate="one_to_one",
    ).merge(
        extract[["_step", "n_extract_fail"]],
        on="_step",
        validate="one_to_one",
    )
    behavior = sort_grid(behavior)[common.BEHAVIOR_COLUMNS]
    extract = sort_grid(extract)[
        [column for column in extract.columns if not column.startswith("_")]
    ]
    flexible = sort_grid(flexible)[
        [column for column in flexible.columns if not column.startswith("_")]
    ]
    ifeval = sort_grid(ifeval).sort_values(["_step", "instruction_category"])[
        [column for column in ifeval.columns if not column.startswith("_")]
    ]
    return {
        "behavior": behavior,
        "extract": extract,
        "flexible": flexible,
        "ifeval": ifeval,
    }


def build_g3(g3_manifest: Mapping[str, Any]) -> pd.DataFrame:
    spectra = read_csv(
        SPECTRA,
        [
            "arm", "step", "task_id", "probe_type", "domain", "generation_seed",
            "track", "layer", "module", "effective_rank", "r_eps_005",
            "tail_energy_r32",
        ],
    )
    spectra = numeric(
        spectra[
            (spectra["arm"] == "seqkd")
            & (spectra["track"] == "frozen_base")
            & (spectra["layer"].astype(int) == 18)
        ],
        ["step", "layer", "effective_rank", "r_eps_005", "tail_energy_r32"],
    )
    keys = [
        "arm", "step", "task_id", "probe_type", "domain", "generation_seed", "layer"
    ]
    summary = (
        spectra.groupby(keys, dropna=False, sort=False)
        .agg(
            module_count=("module", "nunique"),
            effective_rank_mean=("effective_rank", "mean"),
            r_epsilon_005_mean=("r_eps_005", "mean"),
            tail_energy_r32_mean=("tail_energy_r32", "mean"),
        )
        .reset_index()
    )
    require(len(spectra) == 490 and len(summary) == 70, "G3 spectra mismatch")
    require((summary["module_count"] == 7).all(), "G3 spectra module mismatch")

    m1 = read_csv(
        M1,
        [
            "arm", "step", "task_id", "track", "layer", "module", "epsilon",
            "r_epsilon_delta", "drift_core",
        ],
    )
    m1 = numeric(
        m1[
            (m1["arm"] == "seqkd")
            & (m1["track"] == "frozen_base")
            & (m1["layer"].astype(int) == 18)
            & (m1["epsilon"] == "0.05")
        ],
        ["step", "r_epsilon_delta", "drift_core"],
    )
    m1_summary = (
        m1.groupby(["step", "task_id"], sort=False)
        .agg(
            modules=("module", "nunique"),
            r_epsilon_delta_mean=("r_epsilon_delta", "mean"),
            drift_core_mean=("drift_core", "mean"),
        )
        .reset_index()
    )
    require(len(m1) == 490 and (m1_summary["modules"] == 7).all(), "G3 M1 mismatch")

    m2 = read_csv(
        M2,
        ["arm", "step", "task_id", "layer", "module", "reference", "m2_output_drift"],
    )
    m2 = numeric(
        m2[
            (m2["arm"] == "seqkd")
            & (m2["layer"].astype(int) == 18)
            & (m2["reference"] == "X0_primary")
        ],
        ["step", "m2_output_drift"],
    )
    m2_summary = (
        m2.groupby(["step", "task_id"], sort=False)
        .agg(modules=("module", "nunique"), m2_x0_mean=("m2_output_drift", "mean"))
        .reset_index()
    )
    require(len(m2) == 490 and (m2_summary["modules"] == 7).all(), "G3 M2 mismatch")

    theta = read_csv(
        THETA,
        [
            "arm", "step", "probe", "track", "layer", "module", "epsilon",
            "theta_u_max_deg", "theta_u_mean_deg", "theta_v_max_deg",
            "theta_v_mean_deg",
        ],
    )
    theta = numeric(
        theta[
            (theta["arm"] == "seqkd")
            & (theta["track"] == "frozen_base")
            & (theta["layer"].astype(int) == 18)
            & (theta["epsilon"] == "0.05")
        ],
        [
            "step", "theta_u_max_deg", "theta_u_mean_deg",
            "theta_v_max_deg", "theta_v_mean_deg",
        ],
    ).rename(columns={"probe": "task_id"})
    theta_summary = (
        theta.groupby(["step", "task_id"], sort=False)
        .agg(
            theta_module_count=("module", "nunique"),
            theta_u_max_deg_mean=("theta_u_max_deg", "mean"),
            theta_u_mean_deg_mean=("theta_u_mean_deg", "mean"),
            theta_v_max_deg_mean=("theta_v_max_deg", "mean"),
            theta_v_mean_deg_mean=("theta_v_mean_deg", "mean"),
        )
        .reset_index()
    )
    require(
        len(theta) == 441 and (theta_summary["theta_module_count"] == 7).all(),
        "G3 theta mismatch",
    )

    summary = (
        summary.merge(
            m1_summary.drop(columns=["modules"]),
            on=["step", "task_id"],
            validate="one_to_one",
        )
        .merge(
            m2_summary.drop(columns=["modules"]),
            on=["step", "task_id"],
            validate="one_to_one",
        )
        .merge(theta_summary, on=["step", "task_id"], how="left", validate="one_to_one")
    )
    summary["theta_module_count"] = summary["theta_module_count"].fillna(0).astype(int)
    summary["_step"] = summary["step"].astype(int)
    summary = sort_grid(summary, list(g3_manifest["probes"]))
    number_columns = [
        "effective_rank_mean", "r_epsilon_005_mean", "r_epsilon_delta_mean",
        "drift_core_mean", "tail_energy_r32_mean", "m2_x0_mean",
        "theta_u_max_deg_mean", "theta_u_mean_deg_mean",
        "theta_v_max_deg_mean", "theta_v_mean_deg_mean",
    ]
    summary = format_numbers(summary, number_columns)
    summary["step"] = summary["step"].astype(int).astype(str)
    summary["layer"] = summary["layer"].astype(int).astype(str)
    summary["module_count"] = summary["module_count"].astype(int).astype(str)
    summary["theta_module_count"] = summary["theta_module_count"].astype(str)
    return summary[G3_FIELDS]


def compact_g4_examples(path: Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(rows) == 5, f"G4 examples rows={len(rows)}")
    columns = [
        "protocol", "index", "gold", "prediction", "correct", "finish_reason", "n_tokens"
    ]
    return pd.DataFrame([{column: row.get(column, "") for column in columns} for row in rows])


def validate_completion(completion: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(completion.get("status") == "complete", "Completion manifest incomplete")
    require(completion.get("validated_grid") == GRID, "Completion grid mismatch")
    excluded = {str(HANDOFF), str(HANDOFF_MANIFEST)}
    artifacts = []
    for item in completion.get("artifacts", []):
        if item["path"] in excluded:
            continue
        path = Path(item["path"])
        require(path.is_file(), f"Missing artifact: {path}")
        require(path.stat().st_size == int(item["bytes"]), f"Size mismatch: {path}")
        require(common.sha256(path) == item["sha256"], f"Hash mismatch: {path}")
        artifacts.append(dict(item))
    require(len(artifacts) == 11, f"Source artifact count={len(artifacts)}")
    return artifacts


def main() -> None:
    generated_at = common.utc_now()
    completion = common.load_json(COMPLETION)
    completion_artifacts = validate_completion(completion)

    if G2_MANIFEST.is_file():
        g2_manifest = common.load_json(G2_MANIFEST)
        require(
            g2_manifest.get("status") == "complete"
            and set(g2_manifest.get("completed_steps", [])) == set(GRID),
            "G2 manifest incomplete",
        )
        g2_provenance = f"runtime manifest retained: {G2_MANIFEST}"
    else:
        original_hash = completion["manifest_sha256"]["g2_eval"]
        require(len(original_hash) == 64, "G2 preserved manifest hash is malformed")
        g2_provenance = (
            f"runtime manifest not migrated: {G2_MANIFEST}; "
            f"original SHA-256 preserved by completion manifest: {original_hash}"
        )
    g3_manifest = common.load_json(G3_MANIFEST)
    require(
        g3_manifest.get("status") == "complete"
        and g3_manifest.get("steps") == GRID
        and g3_manifest.get("rows_written")
        == {"spectra_seqkd": 2940, "m1": 5880, "m2": 3150, "theta": 3969},
        "G3 manifest incomplete",
    )
    g8_manifest = common.load_json(G8_MANIFEST)
    require(g8_manifest.get("status") == "complete", "G8 manifest incomplete")

    g2 = build_g2()
    g3 = build_g3(g3_manifest)
    g8 = read_csv(G8_SOURCE)
    require(len(g8) == 8, f"G8 rows={len(g8)}")
    require(common.sha256(G8_SOURCE) == g8_manifest["output_sha256"], "G8 hash mismatch")

    g4 = common.load_json(common.G4_MANIFEST)
    require(
        g4.get("status") == "complete" and g4.get("decision") == "GO",
        "G4 manifest incomplete",
    )
    g4_examples_path = Path(g4["artifacts"]["examples_5"])
    g4_examples = compact_g4_examples(g4_examples_path)

    write_frame(SNAPSHOTS["g2_behavior"], g2["behavior"])
    write_frame(SNAPSHOTS["g2_extract"], g2["extract"])
    write_frame(SNAPSHOTS["g2_flexible"], g2["flexible"])
    write_frame(SNAPSHOTS["g2_ifeval"], g2["ifeval"])
    write_frame(SNAPSHOTS["g3_l18"], g3)
    write_frame(SNAPSHOTS["g4_examples"], g4_examples)
    write_frame(SNAPSHOTS["g8"], g8)

    g1 = common.load_json(G1_MANIFEST)
    require(g1.get("status") == "complete" and g1.get("completed_steps") == 624, "G1 incomplete")
    g5 = common.load_json(common.G5_MANIFEST)
    require(g5.get("n_prompts") == 5000, "G5 incomplete")
    g6 = {}
    for arm in ("sft", "offkd", "seqkd"):
        source = common.load_json(
            common.G6_ROOT / arm / "checkpoints/training_manifest.json"
        )
        require(
            source.get("status") == "complete" and source.get("completed_steps") == 624,
            f"G6 {arm} incomplete",
        )
        g6[arm] = source
    g7 = common.load_json(common.G7_MANIFEST)
    c1 = common.load_json(common.C1_MANIFEST)
    c2 = common.load_json(common.C2_MANIFEST)
    require(g7.get("status") == "complete", "G7 incomplete")
    require(c1.get("status") == "complete", "C1 incomplete")
    require(c2.get("status") == "complete", "C2 incomplete")
    c2_frame = read_csv(common.C2_CSV)
    require(len(c2_frame) == 9, f"C2 rows={len(c2_frame)}")

    completion_frame = pd.DataFrame(
        [
            ["G1", "complete", "Qwen seqKD training; ten native checkpoints"],
            ["G2", "complete", "five behavior tasks and CPU audits; ten checkpoints"],
            ["G3", "complete", "seven static-probe task IDs; ten checkpoints; layers 9/18/27"],
            ["G4", "complete", "Llama zero-shot/four-shot preflight; recorded decision GO"],
            ["G5", "complete", "5000-prompt two-pass rollout; raw top-32"],
            ["G6", "complete", "Llama SFT/off-KD/seqKD training; ten checkpoints each"],
            ["G7", "complete", "off-KD H_bos/H_ood seven-step geometry"],
            ["G8", "complete", "eight adapter-layer configurations"],
            ["C1", "complete", "direction analysis extended to all static probes"],
            ["C2", "complete", "nine-row dose-response table"],
        ],
        columns=["item", "status", "raw_scope"],
    )
    g1_fields = [
        "status", "completed_steps", "eligible_prompts", "batch_size", "epochs",
        "steps_per_epoch", "training_seconds_this_invocation", "checkpoint_grid",
        "resume_from", "student_model", "rollout_dir",
    ]
    g1_frame = pd.DataFrame([{key: g1.get(key, "") for key in g1_fields}])
    g4_frame = pd.DataFrame(g4["summary"])
    g5_frame = pd.DataFrame(
        [{
            "n_prompts": g5["n_prompts"],
            "truncation_rate": g5["truncation_rate"],
            "n_truncated": g5["n_truncated"],
            "truncated_kept": g5["truncated_kept"],
            "has_boxed_rate": g5["has_boxed_rate"],
            "length_mean": g5["length_stats"]["mean"],
            "length_median": g5["length_stats"]["median"],
            "length_p90": g5["length_stats"]["p90"],
            "length_max": g5["length_stats"]["max"],
            "pass2_minutes": g5["timing_minutes"]["pass2_logprobs"],
            "topk": g5["logprob_pass2"]["topk"],
            "raw_convention": g5["logprob_pass2"]["convention"],
        }]
    )
    g6_frame = pd.DataFrame(
        [{
            "arm": arm,
            "status": source["status"],
            "completed_steps": source["completed_steps"],
            "eligible_prompts": source["eligible_prompts"],
            "batch_size": source["batch_size"],
            "epochs": source["epochs"],
            "steps_per_epoch": source["steps_per_epoch"],
            "training_seconds_this_invocation": source["training_seconds_this_invocation"],
            "loss_name": source["loss"]["name"],
            "checkpoint_grid": source["checkpoint_grid"],
            "student_model": source["student_model"],
            "data_source": source["data_source"],
        } for arm, source in g6.items()]
    )
    g7_frame = pd.DataFrame(
        [
            ["status", g7["status"]],
            ["steps", g7["steps"]],
            ["domains", g7["domains"]],
            ["generation_seeds", g7["generation_seeds"]],
            ["n_generated_per_cell", g7["n_generated_per_cell"]],
            ["dW_track", g7["dW_track"]],
            ["theta_numerics", g7["theta_numerics"]],
            ["spectra_rows", g7["rows"]["spectra"]],
            ["m1_rows", g7["rows"]["m1"]],
            ["m2_rows", g7["rows"]["m2"]],
            ["theta_rows", g7["rows"]["theta"]],
        ],
        columns=["field", "value"],
    )
    c1_frame = pd.DataFrame(
        [{
            "output": name,
            "path": info["path"],
            "rows": info["rows"],
            "sha256": info["sha256"],
        } for name, info in c1["outputs"].items()]
    )
    artifact_frame = pd.DataFrame(
        [{
            "path": item["path"],
            "rows": "" if item["rows"] is None else item["rows"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        } for item in completion_artifacts]
    )
    passport = pd.DataFrame(
        [
            ["status", "COMPLETE"],
            ["data_access_level", "raw"],
            ["generated_utc", generated_at],
            ["source_specification", str(common.SPEC)],
            ["validated_grid", GRID],
            ["reporting_guard", "raw readings and provenance only; no interpretation or adjudication"],
        ],
        columns=["field", "value"],
    )
    g3_protocol = pd.DataFrame(
        [
            ["status", g3_manifest["status"]],
            ["steps", g3_manifest["steps"]],
            ["probes", g3_manifest["probes"]],
            ["layers", g3_manifest["protocol"]["layers"]],
            ["spectra_rows", g3_manifest["rows_written"]["spectra_seqkd"]],
            ["m1_rows", g3_manifest["rows_written"]["m1"]],
            ["m2_rows", g3_manifest["rows_written"]["m2"]],
            ["theta_rows", g3_manifest["rows_written"]["theta"]],
            ["dW_track", g3_manifest["dW_track"]],
            ["base_whitening", g3_manifest["base_whitening"]],
            ["theta_numerics", g3_manifest["theta_numerics"]],
            ["windowing", g3_manifest["windowing"]],
        ],
        columns=["field", "value"],
    )
    audit_inventory = pd.DataFrame(
        [{
            "snapshot": name,
            "path": str(SNAPSHOTS[name]),
            "rows": len(g2[frame_name]),
            "sha256": common.sha256(SNAPSHOTS[name]),
        } for name, frame_name in [
            ("g2_extract", "extract"),
            ("g2_flexible", "flexible"),
            ("g2_ifeval", "ifeval"),
        ]]
    )

    sections = [
        "# Cycle 09 Stage 1 Second Execution Block - Final Theory Handoff",
        "",
        "## Material Passport", "", md(passport),
        "",
        "## Completion Register", "", md(completion_frame),
        "",
        "## G1 SeqKD Training", "", md(g1_frame),
        "", f"Durable manifest: {G1_MANIFEST}.",
        "",
        "## G2 Ten-Checkpoint Behavior", "", md(g2["behavior"]),
        "", f"Immutable snapshot: {SNAPSHOTS['g2_behavior']}.",
        "",
        "### G2 Audit Snapshots", "", md(audit_inventory),
        "", f"Manifest provenance: {g2_provenance}.",
        "",
        "## G3 SeqKD Geometry", "", md(g3_protocol),
        "",
        "L18, seven-module arithmetic means; theta cells are absent at base step 0 by protocol:",
        "", md(g3),
        "", f"Immutable summary: {SNAPSHOTS['g3_l18']}. Full spectra: {SPECTRA}.",
        "", f"Manifest: {G3_MANIFEST}.",
        "",
        "## G4 Llama Preflight", "", md(g4_frame),
        "", f"Frozen gate: {g4['frozen_gate']}. Recorded decision: {g4['decision']}.",
        "", md(g4_examples),
        "", f"Compact examples: {SNAPSHOTS['g4_examples']}. Full five generations: {g4_examples_path}.",
        "", f"Manifest: {common.G4_MANIFEST}.",
        "",
        "## G5 Llama Two-Pass Rollout", "", md(g5_frame),
        "", f"Manifest: {common.G5_MANIFEST}.",
        "",
        "## G6 Llama Offline Training", "", md(g6_frame),
        "",
        "Llama evaluation and geometry are outside this execution block per the confirmed specification.",
        "",
        "## G7 off-KD H Geometry", "", md(g7_frame),
        "", f"Manifest: {common.G7_MANIFEST}.",
        "",
        "## G8 Adapter Layer-Group Ablation", "", md(g8),
        "", f"Immutable snapshot: {SNAPSHOTS['g8']}. Manifest: {G8_MANIFEST}.",
        "",
        "## C1 Direction Analysis", "", md(c1_frame),
        "", f"Manifest: {common.C1_MANIFEST}.",
        "",
        "## C2 Dose-Response Raw Table", "", md(c2_frame),
        "", f"Manifest: {common.C2_MANIFEST}.",
        "",
        "## Validated Source Artifacts", "", md(artifact_frame),
        "", f"Completion manifest: {COMPLETION}.",
        "",
        "No task in the confirmed second execution block remains pending.",
        "",
    ]
    common.write_text_atomic(HANDOFF, "\n".join(sections))

    snapshot_records = {
        name: common.file_record(path) for name, path in SNAPSHOTS.items()
    }
    source_manifests = [
        G1_MANIFEST,
        G3_MANIFEST,
        common.G4_MANIFEST,
        common.G5_MANIFEST,
        common.G6_ROOT / "sft/checkpoints/training_manifest.json",
        common.G6_ROOT / "offkd/checkpoints/training_manifest.json",
        common.G6_ROOT / "seqkd/checkpoints/training_manifest.json",
        common.G7_MANIFEST,
        G8_MANIFEST,
        common.C1_MANIFEST,
        common.C2_MANIFEST,
    ]
    if G2_MANIFEST.is_file():
        source_manifests.append(G2_MANIFEST)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at": generated_at,
        "task": "Cycle 09 Stage 1 second execution block final Theory handoff",
        "source_specification": str(common.SPEC),
        "reporting_guard": "raw readings and provenance only; no interpretation or adjudication",
        "validated_grid": GRID,
        "completed_items": ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "C1", "C2"],
        "pending_items": [],
        "handoff": {"path": str(HANDOFF), "sha256": common.sha256(HANDOFF)},
        "snapshots": snapshot_records,
        "snapshot_rows": {
            "g2_behavior": len(g2["behavior"]),
            "g2_mmlupro_extract": len(g2["extract"]),
            "g2_mmlupro_flexible": len(g2["flexible"]),
            "g2_ifeval_breakdown": len(g2["ifeval"]),
            "g3_l18_summary": len(g3),
            "g4_examples": len(g4_examples),
            "g8_adapter_ablation": len(g8),
        },
        "source_manifest_sha256": {
            str(path): common.sha256(path) for path in source_manifests
        },
        "source_manifest_sha256_preserved_at_completion": completion["manifest_sha256"],
        "g2_manifest_provenance": g2_provenance,
        "validated_source_artifacts": completion_artifacts,
        "resource_use_for_handoff": "CPU only; no GPU inference or training",
    }
    common.write_json_atomic(HANDOFF_MANIFEST, manifest)
    print(json.dumps({
        "status": "complete",
        "handoff": str(HANDOFF),
        "handoff_sha256": manifest["handoff"]["sha256"],
        "manifest": str(HANDOFF_MANIFEST),
        "snapshot_rows": manifest["snapshot_rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
