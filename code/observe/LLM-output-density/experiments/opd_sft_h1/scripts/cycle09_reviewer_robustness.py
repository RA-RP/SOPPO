#!/usr/bin/env python3
"""Reviewer-robustness RR0 inventory for Cycle 09.

This script is intentionally read-only. It inventories artifacts for the
reviewer-robustness handoff and gates later tasks by explicit artifact status.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


REPO = Path("/root/LLM-output-density")
AUTODL = Path("/root/autodl-tmp")
MINI = (
    REPO
    / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"
)
OUT = MINI / "reviewer_robustness"
STAGE4 = AUTODL / "cycle09_stage4_state_displacement"
RFC = AUTODL / "cycle09_relative_functional_contraction"
D10 = RFC / "d10_llama_numeric_parity/formal/final"
D11 = RFC / "d11_pk_tpnt/formal/final"
QWEN_FINAL = RFC / "final"

MODELS = {"qwen": 18, "llama": 14}
ARMS = ["opd", "sft", "offkd", "seqkd"]
STEPS = [20, 40, 80]
PROBES = ["E_general", "E_math", "E_ood", "E_if"]
EPSILONS = [0.01, 0.025, 0.05, 0.10]


@dataclass(frozen=True)
class Cell:
    model: str
    arm: str
    checkpoint: int
    probe_name: str

    @property
    def layer(self) -> int:
        return MODELS[self.model]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return str(path)


def sha256_file(path: Path, max_bytes: int = 512 * 1024 * 1024) -> str:
    if not path.exists() or not path.is_file():
        return ""
    size = path.stat().st_size
    if size > max_bytes:
        return f"SKIPPED_SIZE_{size}"
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def state_tables() -> dict[str, pd.DataFrame]:
    qwen = load_csv(QWEN_FINAL / "qwen_d4_merged_state_all_cells.csv")
    llama = load_csv(D10 / "d10_5_integrated_state_equal7.csv")
    return {"qwen": qwen, "llama": llama}


def output_tables() -> dict[str, pd.DataFrame]:
    qwen = load_csv(QWEN_FINAL / "qwen_d4_merged_state_outputs.csv")
    llama = load_csv(D10 / "d10_5_integrated_outputs.csv")
    return {"qwen": qwen, "llama": llama}


def formal_cell_source(tables: dict[str, pd.DataFrame], cell: Cell) -> tuple[bool, str, str]:
    df = tables.get(cell.model, pd.DataFrame())
    if df.empty:
        return False, "", "missing formal state table"
    mask = (
        (df["model"].astype(str) == cell.model)
        & (df["arm"].astype(str) == cell.arm)
        & (df["checkpoint"].astype(int) == cell.checkpoint)
        & (df["probe_name"].astype(str) == cell.probe_name)
        & (df["layer"].astype(int) == cell.layer)
        & (df["epsilon"].astype(float).round(6) == 0.05)
    )
    base = (
        (df["model"].astype(str) == cell.model)
        & (df["arm"].astype(str) == "base")
        & (df["checkpoint"].astype(int) == 0)
        & (df["probe_name"].astype(str) == cell.probe_name)
        & (df["layer"].astype(int) == cell.layer)
        & (df["epsilon"].astype(float).round(6) == 0.05)
    )
    if not mask.any():
        return False, "", "missing current D10/D4 formal state cell"
    if not base.any():
        return False, "", "missing matching D10/D4 base state cell"
    row = df[mask].iloc[0]
    return True, str(row.get("source_protocol", "")), str(row.get("source_name", ""))


def sample_count(outputs: dict[str, pd.DataFrame], cell: Cell) -> int | None:
    df = outputs.get(cell.model, pd.DataFrame())
    if df.empty or "sample_count" not in df.columns:
        return None
    mask = (
        (df["model"].astype(str) == cell.model)
        & (df["arm"].astype(str) == cell.arm)
        & (df["checkpoint"].astype(int) == cell.checkpoint)
        & (df["probe_name"].astype(str) == cell.probe_name)
    )
    if not mask.any():
        base = (
            (df["model"].astype(str) == cell.model)
            & (df["arm"].astype(str) == "base")
            & (df["checkpoint"].astype(int) == 0)
            & (df["probe_name"].astype(str) == cell.probe_name)
        )
        if not base.any():
            return None
        return int(df[base].iloc[0]["sample_count"])
    return int(df[mask].iloc[0]["sample_count"])


def direction_path(cell: Cell, *, centered: bool = False, base: bool = False) -> Path:
    arm = "base" if base else cell.arm
    step = 0 if base else cell.checkpoint
    suffix = "main.centered.direction.pt" if centered else "main.direction.pt"
    return (
        STAGE4
        / "cells"
        / cell.model
        / arm
        / f"step_{step:03d}"
        / f"{cell.probe_name}.L{cell.layer}.{suffix}"
    )


def cell_json_path(cell: Cell, *, centered: bool = False, base: bool = False) -> Path:
    return Path(str(direction_path(cell, centered=centered, base=base)).replace(".direction.pt", ".json"))


def has_full_grid_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def find_sample_factor(cell: Cell) -> Path | None:
    # Deliberately narrow: only accept formal Stage-4 factor-like bundles. Older
    # R4/T1/qwen-alpha factors are not D10/D4 matched-grid factors.
    root = STAGE4 / "cells" / cell.model / cell.arm / f"step_{cell.checkpoint:03d}"
    if not root.exists():
        return None
    patterns = [
        f"{cell.probe_name}.L{cell.layer}.main.factor.pt",
        f"{cell.probe_name}.L{cell.layer}.main.factors.pt",
        f"{cell.probe_name}.L{cell.layer}.main.samples.pt",
        f"{cell.probe_name}.L{cell.layer}.main.sample_factors.pt",
    ]
    for pat in patterns:
        p = root / pat
        if p.exists():
            return p
    return None


def centered_available(cell: Cell) -> bool:
    cur = direction_path(cell, centered=True).exists()
    base = direction_path(cell, centered=True, base=True).exists()
    return cur and base


def top32_sources() -> list[dict[str, Any]]:
    candidates = [
        {
            "model": "llama",
            "pipeline": "model2_llama",
            "path": AUTODL
            / "cycle09_block2/model2_llama/rollout/teacher_top32_logprob.npz",
            "manifest": AUTODL / "cycle09_block2/model2_llama/rollout/rollout_manifest.json",
        },
        {
            "model": "qwen",
            "pipeline": "offkd",
            "path": AUTODL / "cycle09_offkd/rollout/teacher_top32_logprob.npz",
            "manifest": AUTODL / "cycle09_offkd/rollout/rollout_manifest.json",
        },
        {
            "model": "qwen",
            "pipeline": "alpha05_frozen_external",
            "path": AUTODL / "cycle09_block3/qwen_alpha05/frozen_external/top32_logprob.npy",
            "manifest": AUTODL / "cycle09_block3/qwen_alpha05/frozen_external",
        },
        {
            "model": "llama",
            "pipeline": "frozen_self",
            "path": AUTODL
            / "cycle09_stage3_followup/H5_frozen_self/frozen_store/top32_logprob.npy",
            "manifest": AUTODL / "cycle09_stage3_followup/H5_frozen_self/rollout/rollout_manifest.json",
        },
    ]
    rows: list[dict[str, Any]] = []
    for c in candidates:
        path = Path(c["path"])
        manifest = Path(c["manifest"])
        m = read_json(manifest) if manifest.is_file() else {}
        convention = json.dumps(m.get("logprob_pass2", {}), ensure_ascii=False)
        raw = "RAW" in convention and "temperature=1.0" in convention
        rows.append(
            {
                **c,
                "path": rel(path),
                "manifest": rel(manifest),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path, max_bytes=64 * 1024 * 1024),
                "raw_logprob_manifest": raw,
                "renormalized_blocker": (
                    "" if raw else "manifest does not prove raw full-vocab teacher log probabilities"
                ),
            }
        )
    return rows


def raw_activation_sources() -> list[dict[str, Any]]:
    paths = [
        AUTODL / "cycle09_block3/llama_geometry/llama_early_raw_representation_suite.csv",
        AUTODL / "cycle09_block3/llama_geometry/llama_early_320_raw_representation_suite.csv",
        AUTODL / "cycle09_block3/llama_geometry/llama_opd_early_raw_representation_suite.csv",
        AUTODL
        / "cycle09_stage3_followup/H5_frozen_self/geometry/llama_frozen_self_raw_representation_suite.csv",
        AUTODL / "cycle09_stage3_followup/H5_frozen_self/landmark_raw_representation.csv",
        MINI / "C2_raw_er_bootstrap.csv",
        MINI / "C2_raw_er_transient_structure.csv",
        MINI / "R5_raw_er_fixed.csv",
        MINI / "R5_raw_er_fixed_ckpt.csv",
    ]
    rows = []
    for p in paths:
        rows.append(
            {
                "path": rel(p),
                "exists": p.exists(),
                "bytes": p.stat().st_size if p.exists() else 0,
                "sha256": sha256_file(p),
            }
        )
    return rows


def rollout_sources() -> list[dict[str, Any]]:
    paths = [
        AUTODL / "cycle09_block2/model2_llama/rollout/teacher_rollout.jsonl",
        AUTODL / "cycle09_offkd/rollout/teacher_rollout.jsonl",
        AUTODL / "cycle09_stage3_followup/H5_frozen_self/rollout/teacher_rollout.jsonl",
        AUTODL / "cycle09_s1/s1_5_opd_rollout/opd_step0_reconstructed_rollout.jsonl",
    ]
    return [
        {
            "path": rel(p),
            "exists": p.exists(),
            "bytes": p.stat().st_size if p.exists() else 0,
            "sha256": sha256_file(p, max_bytes=64 * 1024 * 1024),
        }
        for p in paths
    ]


def artifact_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    named = [
        ("formal_state", "qwen_d4_state_equal7", QWEN_FINAL / "qwen_d4_merged_state_all_cells.csv"),
        ("formal_output", "qwen_d4_outputs", QWEN_FINAL / "qwen_d4_merged_state_outputs.csv"),
        ("formal_state", "llama_d10_5_state_equal7", D10 / "d10_5_integrated_state_equal7.csv"),
        ("formal_state", "llama_d10_5_state_module", D10 / "d10_5_integrated_state_module.csv"),
        ("formal_output", "llama_d10_5_outputs", D10 / "d10_5_integrated_outputs.csv"),
        ("d11_features", "d11_same_cell_feature_matrix", D11 / "d11_same_cell_feature_matrix.csv"),
        ("centered_partial", "d10_5_a5_centered", D10 / "d10_5_a5_centered_state_cells.csv"),
        ("sample_inventory", "state_displacement_sample_count_bootstrap", STAGE4 / "cpu/state_displacement_sample_count_bootstrap.csv"),
    ]
    for category, label, path in named:
        rows.append(
            {
                "category": category,
                "label": label,
                "path": rel(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path),
                "notes": "",
            }
        )
    for p in sorted((STAGE4 / "cells").glob("*/*/step_*/*.direction.pt")):
        if ".smoke." in p.name:
            continue
        rows.append(
            {
                "category": "direction_spectrum",
                "label": p.name,
                "path": rel(p),
                "exists": True,
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p, max_bytes=64 * 1024 * 1024),
                "notes": "Stage4 direction.pt",
            }
        )
    for item in top32_sources():
        rows.append(
            {
                "category": "top32",
                "label": item["pipeline"],
                "path": item["path"],
                "exists": item["exists"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
                "notes": "raw_logprob_manifest="
                + str(item["raw_logprob_manifest"])
                + "; "
                + item["renormalized_blocker"],
            }
        )
    for item in raw_activation_sources():
        rows.append(
            {
                "category": "raw_activation",
                "label": Path(item["path"]).name,
                "path": item["path"],
                "exists": item["exists"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
                "notes": "",
            }
        )
    for item in rollout_sources():
        rows.append(
            {
                "category": "rollout_text",
                "label": Path(item["path"]).name,
                "path": item["path"],
                "exists": item["exists"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
                "notes": "",
            }
        )
    return rows


def rr4_ready_for_model(model: str) -> tuple[bool, str, str]:
    sources = [s for s in top32_sources() if s["model"] == model and s["exists"]]
    raw = [s for s in sources if s["raw_logprob_manifest"]]
    if raw:
        return True, raw[0]["path"], "RAW teacher log probabilities according to rollout manifest"
    if sources:
        return False, sources[0]["path"], sources[0]["renormalized_blocker"]
    return False, "", "missing top32 teacher logprob arrays"


def rr5_ready_for_model(model: str) -> tuple[bool, str, str]:
    d11 = D11 / "d11_same_cell_feature_matrix.csv"
    raw = [Path(r["path"]) for r in raw_activation_sources() if r["exists"]]
    if not d11.exists():
        return False, "", "missing exact D11 same-cell feature matrix"
    if not raw:
        return False, "", "missing raw-activation feature tables"
    # Coverage is intentionally audited later by RR5; RR0 only gates presence.
    return True, ";".join(rel(p) for p in raw[:4]), "requires exact-key join audit before fitting"


def rr6_ready() -> tuple[bool, str, str]:
    current = AUTODL / "cycle09_block3/llama_behavior/formal/opd"
    frozen = AUTODL / "cycle09_stage3_followup/H5_frozen_self/rollout/teacher_rollout.jsonl"
    if current.exists() and frozen.exists():
        return True, rel(frozen), "matched prompt/checkpoint coverage still requires RR6 audit"
    if frozen.exists():
        return False, rel(frozen), "current OPD rollout text path for frozen-self checkpoints not identified"
    return False, "", "missing frozenSelf0-KD rollout text"


def build_coverage() -> list[dict[str, Any]]:
    tables = state_tables()
    outputs = output_tables()
    rows: list[dict[str, Any]] = []
    for task in ["RR1A", "RR1B", "RR2", "RR3", "RR4", "RR5", "RR6"]:
        for model in MODELS:
            for arm in ARMS:
                for ckpt in STEPS:
                    for probe in PROBES:
                        cell = Cell(model, arm, ckpt, probe)
                        formal_ok, protocol, source_name = formal_cell_source(tables, cell)
                        n = sample_count(outputs, cell)
                        cur_dir = direction_path(cell)
                        base_dir = direction_path(cell, base=True)
                        spectrum_ok = cur_dir.exists() and base_dir.exists()
                        factor = find_sample_factor(cell)
                        centered = centered_available(cell)
                        status = "BLOCKED_MISSING_ARTIFACT"
                        reason = ""
                        input_path = ""
                        input_sha = ""
                        formal_source = source_name
                        if not formal_ok:
                            reason = protocol or source_name
                        elif task in {"RR1A", "RR1B"}:
                            if factor:
                                status = "READY_REUSE"
                                input_path = rel(factor)
                                input_sha = sha256_file(factor)
                            else:
                                status = "READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT"
                                input_path = rel(cur_dir) if cur_dir.exists() else ""
                                input_sha = sha256_file(cur_dir, max_bytes=64 * 1024 * 1024)
                                reason = "missing formal per-sample factor bundle; new forward required for exact sample bootstrap"
                        elif task == "RR2":
                            if spectrum_ok:
                                status = "READY_REUSE"
                                input_path = rel(cur_dir)
                                input_sha = sha256_file(cur_dir, max_bytes=64 * 1024 * 1024)
                            else:
                                reason = "missing Stage4 current or base direction.pt singular spectrum"
                        elif task == "RR3":
                            status = "READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT"
                            input_path = rel(cur_dir) if cur_dir.exists() else ""
                            input_sha = sha256_file(cur_dir, max_bytes=64 * 1024 * 1024)
                            reason = "RR3 is new forward and requires explicit Theory GO"
                        elif task == "RR4":
                            ok, p, note = rr4_ready_for_model(model)
                            status = "READY_REUSE" if ok else "BLOCKED_MISSING_ARTIFACT"
                            input_path = p
                            input_sha = sha256_file(Path(p), max_bytes=64 * 1024 * 1024) if p else ""
                            reason = note
                        elif task == "RR5":
                            ok, p, note = rr5_ready_for_model(model)
                            status = "READY_REUSE" if ok else "BLOCKED_MISSING_ARTIFACT"
                            input_path = p
                            reason = note
                        elif task == "RR6":
                            ok, p, note = rr6_ready()
                            if model == "llama":
                                status = "READY_REUSE" if ok else "BLOCKED_MISSING_ARTIFACT"
                                input_path = p
                                input_sha = sha256_file(Path(p), max_bytes=64 * 1024 * 1024) if p else ""
                                reason = note
                            else:
                                status = "BLOCKED_PROTOCOL_MISMATCH"
                                reason = "RR6 is Llama OPD vs frozenSelf0-KD only"
                        rows.append(
                            {
                                "task": task,
                                "model": model,
                                "arm": arm,
                                "checkpoint": ckpt,
                                "probe_name": probe,
                                "layer": cell.layer,
                                "formal_source": formal_source,
                                "input_path": input_path,
                                "input_sha256": input_sha,
                                "protocol_id": protocol,
                                "has_spectrum": bool(spectrum_ok),
                                "has_sample_factors": bool(factor),
                                "has_centered": bool(centered),
                                "sample_count": n if n is not None else "",
                                "status": status,
                                "blocker_reason": reason,
                            }
                        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def protocol_map(coverage: list[dict[str, Any]], inventory: list[dict[str, Any]]) -> dict[str, Any]:
    cov = pd.DataFrame(coverage)
    inv = pd.DataFrame(inventory)
    task_counts = (
        cov.groupby(["task", "status"], dropna=False)
        .size()
        .reset_index(name="n_cells")
        .to_dict(orient="records")
    )
    return {
        "created_utc": now(),
        "git_commit": git_commit(),
        "handoff": rel(REPO / "mypaper/code/cycle09_reviewer_robustness_handoff.md"),
        "formal_grid": {
            "models": MODELS,
            "arms": ARMS,
            "checkpoints": STEPS,
            "probes": PROBES,
            "epsilons": EPSILONS,
        },
        "numeric_protocols": {
            "qwen": {
                "formal_source": rel(QWEN_FINAL / "qwen_d4_merged_state_all_cells.csv"),
                "protocol": "D4.1_current_merged_state; BF16 forward, FP64 Gram/eigh/SVD, FP32 W S matmul per handoff",
            },
            "llama": {
                "formal_source": rel(D10 / "d10_5_integrated_state_equal7.csv"),
                "protocol": "D10_bf16_forward_fp64_eigh_svd; BF16 forward, FP32 hidden/Gram, FP64 eigh/SVD per handoff",
            },
        },
        "task_status_counts": task_counts,
        "artifact_counts": (
            inv.groupby(["category", "exists"], dropna=False)
            .size()
            .reset_index(name="n")
            .to_dict(orient="records")
            if not inv.empty
            else []
        ),
    }


def blockers_markdown(coverage: list[dict[str, Any]]) -> str:
    df = pd.DataFrame(coverage)
    lines = [
        "# RR0 Blockers",
        "",
        f"Created UTC: {now()}",
        "",
        "## Status Counts",
        "",
        df.groupby(["task", "status"]).size().reset_index(name="n").to_markdown(index=False),
        "",
        "## Blocker Reasons",
        "",
    ]
    blocked = df[df["status"] != "READY_REUSE"].copy()
    if blocked.empty:
        lines.append("No blockers.")
    else:
        summary = (
            blocked.groupby(["task", "status", "blocker_reason"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["task", "status", "n"], ascending=[True, True, False])
        )
        lines.append(summary.to_markdown(index=False))
    lines += [
        "",
        "## Immediate Gate",
        "",
        "- RR1A/RR1B cells are not READY_REUSE unless formal per-sample factor bundles exist.",
        "- RR3 remains new-forward and requires explicit Theory GO.",
        "- READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT cells must not be launched without Theory GO.",
    ]
    return "\n".join(lines) + "\n"


def task_estimates(coverage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(coverage)
    counts = df.groupby(["task", "status"]).size().unstack(fill_value=0)
    rows = []
    estimates = {
        "RR1A": ("blocked/recompute", "0 now; 2-6h if new forward+1024 draws approved", "0 now; scratch depends on factors"),
        "RR1B": ("blocked/recompute", "0 now; 1-4h if new forward+200 subsets approved", "0 now; scratch depends on factors"),
        "RR2": ("reuse", "10-40 min CPU if all spectrum rows parse cleanly", "<2 GB compact outputs"),
        "RR3": ("requires GO", "2-6h GPU new forward after GO", "tens of GB transient factors/profiles"),
        "RR4": ("reuse/partial", "10-60 min CPU depending on top32 array size", "<5 GB compact/by-sequence outputs"),
        "RR5": ("reuse", "10-45 min CPU after exact join", "<1 GB"),
        "RR6": ("reuse/optional", "10-40 min CPU if matched text found", "<1 GB"),
    }
    for task, (tier, wall, scratch) in estimates.items():
        row = {
            "task": task,
            "tier": tier,
            "ready_reuse_cells": int(counts.get("READY_REUSE", pd.Series()).get(task, 0)),
            "ready_recompute_cells": int(
                counts.get("READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT", pd.Series()).get(task, 0)
            ),
            "blocked_cells": int(
                len(df[(df["task"] == task) & (~df["status"].isin(["READY_REUSE", "READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT"]))])
            ),
            "estimated_wall_time": wall,
            "estimated_scratch": scratch,
        }
        rows.append(row)
    return rows


def run_rr0() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = artifact_inventory()
    coverage = build_coverage()
    write_csv(OUT / "RR0_artifact_inventory.csv", inventory)
    write_csv(OUT / "RR0_grid_coverage.csv", coverage)
    write_csv(OUT / "RR0_task_estimates.csv", task_estimates(coverage))
    (OUT / "RR0_protocol_map.json").write_text(
        json.dumps(protocol_map(coverage, inventory), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "RR0_blockers.md").write_text(blockers_markdown(coverage), encoding="utf-8")
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": "COMPLETE_RR0_INVENTORY",
        "outputs": {
            "artifact_inventory": rel(OUT / "RR0_artifact_inventory.csv"),
            "grid_coverage": rel(OUT / "RR0_grid_coverage.csv"),
            "protocol_map": rel(OUT / "RR0_protocol_map.json"),
            "blockers": rel(OUT / "RR0_blockers.md"),
            "task_estimates": rel(OUT / "RR0_task_estimates.csv"),
        },
    }
    (OUT / "RR0_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def load_direction(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def energy_stats(singular: np.ndarray, rank: int, epsilon: float, tol: float = 1e-8) -> dict[str, float | str]:
    singular = np.asarray(singular, dtype=np.float64)
    energy = singular * singular
    total = float(energy.sum())
    if total <= 0:
        return {
            "tail_at_r": np.nan,
            "tail_at_r_minus_1": np.nan,
            "margin_below": np.nan,
            "margin_above": np.nan,
            "two_sided_tail_margin": np.nan,
            "stable_rank": np.nan,
            "entropy_effective_rank": np.nan,
            "top1_energy_share": np.nan,
            "top10_energy_share": np.nan,
            "zero_probability_count": int(len(energy)),
            "rank_spectrum_consistency": "INVALID_ZERO_TOTAL_ENERGY",
        }
    p = energy / total
    r = int(max(0, min(rank, len(p))))
    tail_at_r = float(p[r:].sum())
    tail_at_r_minus_1 = float(p[r - 1 :].sum()) if r > 0 else 1.0
    p_nonzero = p[p > 0]
    entropy = float(np.exp(-(p_nonzero * np.log(p_nonzero)).sum()))
    if tail_at_r <= epsilon + tol and tail_at_r_minus_1 > epsilon - tol:
        consistency = "PASS"
    else:
        consistency = "INVALID_RANK_SPECTRUM_MISMATCH"
    return {
        "tail_at_r": tail_at_r,
        "tail_at_r_minus_1": tail_at_r_minus_1,
        "margin_below": epsilon - tail_at_r,
        "margin_above": tail_at_r_minus_1 - epsilon,
        "two_sided_tail_margin": min(epsilon - tail_at_r, tail_at_r_minus_1 - epsilon),
        "stable_rank": float(total / energy.max()),
        "entropy_effective_rank": entropy,
        "top1_energy_share": float(p[:1].sum()),
        "top10_energy_share": float(p[:10].sum()),
        "zero_probability_count": int((p == 0).sum()),
        "rank_spectrum_consistency": consistency,
    }


def run_rr2() -> None:
    coverage = pd.read_csv(OUT / "RR0_grid_coverage.csv")
    ready = coverage[(coverage["task"] == "RR2") & (coverage["status"] == "READY_REUSE")]
    rows: list[dict[str, Any]] = []
    for _, cov in ready.iterrows():
        cell = Cell(
            str(cov["model"]),
            str(cov["arm"]),
            int(cov["checkpoint"]),
            str(cov["probe_name"]),
        )
        json_path = cell_json_path(cell)
        direction = load_direction(direction_path(cell))
        meta = read_json(json_path)
        meta_rows = meta.get("rows", [])
        by_mod_eps = {
            (r["module"], round(float(r["epsilon"]), 6)): r for r in meta_rows if r.get("centered") is False
        }
        for module, payload in direction.items():
            singular = payload.get("singular")
            if singular is None:
                continue
            singular_np = singular.detach().cpu().numpy()
            for eps in EPSILONS:
                rmeta = by_mod_eps.get((module, round(eps, 6)), {})
                rank = rmeta.get("displacement_rank")
                rank_source = "displacement_rank"
                if rank is None:
                    rank = rmeta.get("state_rank")
                    rank_source = "state_rank"
                if rank is None:
                    continue
                stats = energy_stats(singular_np, int(rank), eps)
                rows.append(
                    {
                        "model": cell.model,
                        "arm": cell.arm,
                        "checkpoint": cell.checkpoint,
                        "probe_name": cell.probe_name,
                        "layer": cell.layer,
                        "module": module,
                        "epsilon": eps,
                        "rank_at_epsilon": int(rank),
                        "rank_source": rank_source,
                        "spectrum_source": rel(direction_path(cell)),
                        "spectrum_quantity": "stage4_direction_singular_values",
                        "singular_values_stored": int(len(singular_np)),
                        **stats,
                    }
                )
    module_df = pd.DataFrame(rows)
    module_path = OUT / "RR2_spectrum_stability_module.csv"
    module_df.to_csv(module_path, index=False)
    num_cols = [
        "tail_at_r",
        "tail_at_r_minus_1",
        "margin_below",
        "margin_above",
        "two_sided_tail_margin",
        "stable_rank",
        "entropy_effective_rank",
        "top1_energy_share",
        "top10_energy_share",
        "zero_probability_count",
    ]
    equal7 = (
        module_df.groupby(["model", "arm", "checkpoint", "probe_name", "layer", "epsilon"], as_index=False)[num_cols]
        .mean()
    )
    equal7["module_count"] = (
        module_df.groupby(["model", "arm", "checkpoint", "probe_name", "layer", "epsilon"])["module"]
        .nunique()
        .to_numpy()
    )
    equal7_path = OUT / "RR2_spectrum_stability_equal7.csv"
    equal7.to_csv(equal7_path, index=False)

    ordering_rows = []
    for metric in ["stable_rank", "entropy_effective_rank"]:
        eps_df = equal7[equal7["epsilon"].round(6) == 0.05]
        for key, g in eps_df.groupby(["model", "checkpoint", "probe_name", "layer"]):
            arms = set(g["arm"])
            if set(ARMS).issubset(arms):
                gg = g.set_index("arm")
                vals = gg[metric]
                deepest = str(vals.idxmin())
                sorted_vals = vals.sort_values()
                margin = float(sorted_vals.iloc[1] - sorted_vals.iloc[0]) if len(sorted_vals) > 1 else np.nan
                ordering_rows.append(
                    {
                        "metric": metric,
                        "model": key[0],
                        "checkpoint": key[1],
                        "probe_name": key[2],
                        "layer": key[3],
                        "available_arms": ",".join(sorted(arms)),
                        "deepest_arm": deepest,
                        "opd_deepest": deepest == "opd",
                        "nearest_margin": margin,
                    }
                )
            else:
                ordering_rows.append(
                    {
                        "metric": metric,
                        "model": key[0],
                        "checkpoint": key[1],
                        "probe_name": key[2],
                        "layer": key[3],
                        "available_arms": ",".join(sorted(arms)),
                        "deepest_arm": "",
                        "opd_deepest": "",
                        "nearest_margin": np.nan,
                    }
                )
    ordering_path = OUT / "RR2_continuous_ordering.csv"
    pd.DataFrame(ordering_rows).to_csv(ordering_path, index=False)

    outputs = pd.concat(output_tables().values(), ignore_index=True)
    link = equal7[equal7["epsilon"].round(6) == 0.05].merge(
        outputs,
        on=["model", "arm", "checkpoint", "probe_name"],
        how="inner",
        suffixes=("", "_out"),
    )
    link_rows = []
    for metric in ["stable_rank", "entropy_effective_rank"]:
        for target in ["cumulative_kl_base_to_current", "absolute_delta_nll_cumulative"]:
            for (model, arm), g in link.groupby(["model", "arm"]):
                valid = g[[metric, target]].dropna()
                rho = valid[metric].rank().corr(valid[target].rank()) if len(valid) >= 3 else np.nan
                link_rows.append(
                    {
                        "model": model,
                        "arm": arm,
                        "metric": metric,
                        "target": target,
                        "n": int(len(valid)),
                        "spearman": rho,
                    }
                )
    link_path = OUT / "RR2_continuous_output_links.csv"
    pd.DataFrame(link_rows).to_csv(link_path, index=False)

    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": "COMPLETE_RR2_READY_REUSE_PARTIAL_GRID",
        "formal_protocol_id": "RR2_stage4_direction_reuse_after_RR0",
        "models": list(MODELS),
        "arms": ARMS,
        "checkpoints": STEPS,
        "probes": PROBES,
        "layers": MODELS,
        "epsilons": EPSILONS,
        "draws_and_seeds": {},
        "row_counts": {
            "module": int(len(module_df)),
            "equal7": int(len(equal7)),
            "ordering": int(len(ordering_rows)),
            "output_links": int(len(link_rows)),
        },
        "blocked_cells": int(len(coverage[(coverage["task"] == "RR2") & (coverage["status"] != "READY_REUSE")])),
        "input_paths_and_sha256": {
            "RR0_grid_coverage": sha256_file(OUT / "RR0_grid_coverage.csv"),
        },
        "output_sha256": {
            "RR2_spectrum_stability_module.csv": sha256_file(module_path),
            "RR2_spectrum_stability_equal7.csv": sha256_file(equal7_path),
            "RR2_continuous_ordering.csv": sha256_file(ordering_path),
            "RR2_continuous_output_links.csv": sha256_file(link_path),
        },
        "notes": "Uses Stage4 direction singular values exactly where RR0 marked READY_REUSE; Qwen non-OPD cells remain blocked.",
    }
    (OUT / "RR2_spectrum_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def load_top32_logprobs(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if path.suffix == ".npz":
        archive = np.load(path)
        return archive["top32_logprob"], archive["row_offsets"]
    return np.load(path, mmap_mode="r"), None


def summarize_array(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p01": float(np.quantile(x, 0.01)),
        "p05": float(np.quantile(x, 0.05)),
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "frac_below_0_90": float(np.mean(x < 0.90)),
        "frac_below_0_95": float(np.mean(x < 0.95)),
        "frac_below_0_99": float(np.mean(x < 0.99)),
    }


def top32_mass(logprobs: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    out = np.empty(logprobs.shape[0], dtype=np.float32)
    for start in range(0, logprobs.shape[0], chunk):
        end = min(start + chunk, logprobs.shape[0])
        out[start:end] = np.exp(logprobs[start:end]).sum(axis=1)
    return out


def run_rr4() -> None:
    summary_rows = []
    seq_rows = []
    coverage_rows = []
    for src in top32_sources():
        if not src["exists"]:
            coverage_rows.append({**src, "status": "BLOCKED_MISSING_ARTIFACT"})
            continue
        if not src["raw_logprob_manifest"]:
            coverage_rows.append({**src, "status": "BLOCKED_RENORMALIZED_TOPK"})
            continue
        path = Path(src["path"])
        logprobs, offsets = load_top32_logprobs(path)
        mass = top32_mass(logprobs)
        n_seq = int(offsets.shape[0]) if offsets is not None else ""
        summary_rows.append(
            {
                "model": src["model"],
                "arm_or_pipeline": src["pipeline"],
                "checkpoint_or_rollout_source": src["path"],
                "n_sequences": n_seq,
                "n_tokens": int(mass.shape[0]),
                **summarize_array(mass),
            }
        )
        if offsets is not None:
            prefix = np.concatenate([[0.0], np.cumsum(mass, dtype=np.float64)])
            for i, (start, end) in enumerate(offsets.astype(int)):
                vals = mass[start:end]
                seq_rows.append(
                    {
                        "model": src["model"],
                        "arm_or_pipeline": src["pipeline"],
                        "sequence_index": i,
                        "token_start": int(start),
                        "token_end": int(end),
                        "n_tokens": int(end - start),
                        "mean_retained_mass": float((prefix[end] - prefix[start]) / max(end - start, 1)),
                        "min_retained_mass": float(vals.min()) if len(vals) else np.nan,
                        "p05_retained_mass": float(np.quantile(vals, 0.05)) if len(vals) else np.nan,
                    }
                )
        coverage_rows.append({**src, "status": "READY_REUSE_RAW_LOGPROB"})
        del logprobs, mass
    summary_path = OUT / "RR4_top32_retained_mass_summary.csv"
    seq_path = OUT / "RR4_top32_retained_mass_by_sequence.csv"
    cov_path = OUT / "RR4_top32_coverage.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(seq_rows).to_csv(seq_path, index=False)
    pd.DataFrame(coverage_rows).to_csv(cov_path, index=False)
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": "COMPLETE_RR4_READY_REUSE",
        "formal_protocol_id": "RR4_raw_teacher_top32_logprob_reuse",
        "models": list(MODELS),
        "arms": ARMS,
        "checkpoints": [],
        "probes": [],
        "layers": {},
        "epsilons": [],
        "draws_and_seeds": {},
        "row_counts": {
            "summary": int(len(summary_rows)),
            "by_sequence": int(len(seq_rows)),
            "coverage": int(len(coverage_rows)),
        },
        "blocked_cells": int(sum(1 for r in coverage_rows if not str(r["status"]).startswith("READY"))),
        "input_paths_and_sha256": {r["path"]: r.get("sha256", "") for r in coverage_rows},
        "output_sha256": {
            "RR4_top32_retained_mass_summary.csv": sha256_file(summary_path),
            "RR4_top32_retained_mass_by_sequence.csv": sha256_file(seq_path),
            "RR4_top32_coverage.csv": sha256_file(cov_path),
        },
    }
    (OUT / "RR4_top32_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def raw_feature_table() -> pd.DataFrame:
    rows = []
    for p in [
        AUTODL / "cycle09_block3/llama_geometry/llama_early_raw_representation_suite.csv",
        AUTODL / "cycle09_block3/llama_geometry/llama_early_320_raw_representation_suite.csv",
        AUTODL / "cycle09_block3/llama_geometry/llama_opd_early_raw_representation_suite.csv",
    ]:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df.rename(columns={"step": "checkpoint", "probe": "probe_name"})
        df["model"] = "llama"
        rows.append(df)
    for p in [MINI / "R5_raw_er_fixed.csv", MINI / "R5_raw_er_fixed_ckpt.csv"]:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df.rename(columns={"step": "checkpoint", "task_id": "probe_name"})
        df["model"] = "qwen"
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    raw = pd.concat(rows, ignore_index=True)
    raw = raw[raw["probe_name"].isin(PROBES)]
    raw = raw[raw["layer"].isin(MODELS.values())]
    feature_cols = [
        c
        for c in [
            "normalized_entropy_effective_rank",
            "participation_ratio",
            "top1_explained_share",
            "top8_explained_share",
            "top32_explained_share",
            "raw_anisotropy",
            "centered_anisotropy",
            "linear_cka_vs_step0",
            "raw_er_unnormalized",
            "raw_er_normalized",
            "raw_top5_eigen_share",
            "raw_trace",
        ]
        if c in raw.columns
    ]
    keep = ["model", "arm", "checkpoint", "probe_name", "layer"] + feature_cols
    raw = raw[keep].drop_duplicates(["model", "arm", "checkpoint", "probe_name", "layer"])
    return raw


def run_rr5() -> None:
    d11 = pd.read_csv(D11 / "d11_same_cell_feature_matrix.csv")
    raw = raw_feature_table()
    keys = ["model", "arm", "checkpoint", "probe_name", "layer"]
    merged = d11.merge(raw, on=keys, how="left", indicator=True)
    feature_cols = [c for c in raw.columns if c not in keys]
    coverage = (
        merged.assign(has_raw_activation=merged["_merge"].eq("both"))
        .groupby(["model", "arm"], as_index=False)
        .agg(total_cells=("arm", "size"), matched_cells=("has_raw_activation", "sum"))
    )
    coverage["matched_fraction"] = coverage["matched_cells"] / coverage["total_cells"]
    coverage_path = OUT / "RR5_hybrid_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    full_model_arm = bool((coverage["matched_cells"] == coverage["total_cells"]).all()) if len(coverage) else False
    grouped_path = OUT / "RR5_hybrid_grouped_models.csv"
    pred_path = OUT / "RR5_hybrid_predictions.parquet"
    if not full_model_arm or not feature_cols:
        pd.DataFrame(
            [
                {
                    "status": "BLOCKED_INSUFFICIENT_COMMON_GRID",
                    "reason": "exact-key raw activation join does not cover every D11 model/arm cell",
                    "feature_columns_detected": ",".join(feature_cols),
                }
            ]
        ).to_csv(grouped_path, index=False)
        pd.DataFrame().to_parquet(pred_path, index=False)
        status = "BLOCKED_INSUFFICIENT_COMMON_GRID"
    else:
        # Reserved for a future complete grid; current handoff says to stop
        # rather than impute when coverage is incomplete.
        pd.DataFrame().to_csv(grouped_path, index=False)
        pd.DataFrame().to_parquet(pred_path, index=False)
        status = "COMPLETE_NO_MODELS_FIT"
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": status,
        "formal_protocol_id": "RR5_exact_D11_raw_activation_join",
        "models": list(MODELS),
        "arms": ARMS,
        "checkpoints": sorted(map(int, d11["checkpoint"].unique())),
        "probes": PROBES,
        "layers": MODELS,
        "epsilons": [0.05],
        "draws_and_seeds": {},
        "row_counts": {
            "coverage": int(len(coverage)),
            "d11_rows": int(len(d11)),
            "raw_rows": int(len(raw)),
            "exact_join_rows": int((merged["_merge"] == "both").sum()),
        },
        "blocked_cells": int((merged["_merge"] != "both").sum()),
        "input_paths_and_sha256": {
            rel(D11 / "d11_same_cell_feature_matrix.csv"): sha256_file(D11 / "d11_same_cell_feature_matrix.csv"),
        },
        "output_sha256": {
            "RR5_hybrid_coverage.csv": sha256_file(coverage_path),
            "RR5_hybrid_grouped_models.csv": sha256_file(grouped_path),
            "RR5_hybrid_predictions.parquet": sha256_file(pred_path),
        },
    }
    (OUT / "RR5_hybrid_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def response_stats(text: str, finish: str | None = None) -> dict[str, Any]:
    tokens = re.findall(r"\S+", text or "")
    bigrams = list(zip(tokens, tokens[1:]))
    fourgrams = list(zip(tokens, tokens[1:], tokens[2:], tokens[3:]))
    return {
        "response_tokens": len(tokens),
        "eos": finish == "stop",
        "truncated": finish == "length",
        "fourgram_repetition": 0.0 if not fourgrams else 1.0 - len(set(fourgrams)) / len(fourgrams),
        "distinct_2": 0.0 if not bigrams else len(set(bigrams)) / len(bigrams),
        "boxed": "\\boxed" in (text or ""),
        "think_tag": "<think>" in (text or "").lower() or "</think>" in (text or "").lower(),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_rr6() -> None:
    rows = []
    pair_rows = []
    coverage_rows = []
    for step in STEPS + [160]:
        opd_path = AUTODL / f"cycle09_block3/llama_behavior/formal/opd/step_{step:03d}/math500/step_{step:03d}_samples.jsonl"
        frozen_path = AUTODL / f"cycle09_stage3_followup/H5_frozen_self/behavior/formal/frozen_self/step_{step:03d}/math500/step_{step:03d}_samples.jsonl"
        opd = read_jsonl(opd_path)
        frozen = read_jsonl(frozen_path)
        matched = min(len(opd), len(frozen))
        same_gold = sum(1 for i in range(matched) if opd[i].get("gold") == frozen[i].get("gold"))
        coverage_rows.append(
            {
                "checkpoint": step,
                "dataset": "math500",
                "opd_path": rel(opd_path),
                "frozen_self_path": rel(frozen_path),
                "opd_n": len(opd),
                "frozen_self_n": len(frozen),
                "paired_n": matched,
                "same_gold_n": same_gold,
                "status": "READY_REUSE_MATCHED_BY_ROW" if matched and same_gold == matched else "BLOCKED_UNMATCHED_PROMPTS",
            }
        )
        if not matched or same_gold != matched:
            continue
        for arm_name, records in [("opd", opd), ("frozen_self", frozen)]:
            stat_rows = []
            for rec in records[:matched]:
                stat_rows.append(response_stats(rec.get("gen", ""), rec.get("finish")))
            sdf = pd.DataFrame(stat_rows)
            row = {"arm": arm_name, "checkpoint": step, "dataset": "math500", "n": matched}
            for col in sdf.columns:
                row[col + "_mean"] = float(sdf[col].mean())
            row["exact_duplicate_rate"] = 1.0 - len(set(rec.get("gen", "") for rec in records[:matched])) / matched
            rows.append(row)
        for i in range(matched):
            a = response_stats(opd[i].get("gen", ""), opd[i].get("finish"))
            b = response_stats(frozen[i].get("gen", ""), frozen[i].get("finish"))
            diff = {
                "checkpoint": step,
                "dataset": "math500",
                "pair_index": i,
                "gold": opd[i].get("gold"),
            }
            for key in a:
                diff[key + "_opd"] = a[key]
                diff[key + "_frozen_self"] = b[key]
                if isinstance(a[key], (int, float, bool)) and isinstance(b[key], (int, float, bool)):
                    diff[key + "_opd_minus_frozen_self"] = float(a[key]) - float(b[key])
            pair_rows.append(diff)
    text_path = OUT / "RR6_frozen_self_text_stats.csv"
    pair_path = OUT / "RR6_frozen_self_paired_differences.csv"
    cov_path = OUT / "RR6_frozen_self_coverage.csv"
    pd.DataFrame(rows).to_csv(text_path, index=False)
    pd.DataFrame(pair_rows).to_csv(pair_path, index=False)
    pd.DataFrame(coverage_rows).to_csv(cov_path, index=False)
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": "COMPLETE_RR6_MATH500_MATCHED_TEXT_STATS",
        "formal_protocol_id": "RR6_lamma_opd_vs_frozen_self_math500_row_matched",
        "models": ["llama"],
        "arms": ["opd", "frozen_self"],
        "checkpoints": STEPS + [160],
        "probes": [],
        "layers": {},
        "epsilons": [],
        "draws_and_seeds": {},
        "row_counts": {
            "text_stats": int(len(rows)),
            "paired_differences": int(len(pair_rows)),
            "coverage": int(len(coverage_rows)),
        },
        "blocked_cells": int(sum(1 for r in coverage_rows if not str(r["status"]).startswith("READY"))),
        "input_paths_and_sha256": {},
        "output_sha256": {
            "RR6_frozen_self_text_stats.csv": sha256_file(text_path),
            "RR6_frozen_self_paired_differences.csv": sha256_file(pair_path),
            "RR6_frozen_self_coverage.csv": sha256_file(cov_path),
        },
        "notes": "Uses matched math500 behavior samples only; broader rollout/text pools are not treated as paired.",
    }
    (OUT / "RR6_frozen_self_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def supersede_old_rr2() -> None:
    old_files = [
        "RR2_spectrum_stability_module.csv",
        "RR2_spectrum_stability_equal7.csv",
        "RR2_continuous_ordering.csv",
        "RR2_continuous_output_links.csv",
        "RR2_spectrum_manifest.json",
    ]
    manifest_path = OUT / "RR2_spectrum_manifest.json"
    manifest = read_json(manifest_path)
    manifest.update(
        {
            "status": "SUPERSEDED_WRONG_ESTIMAND_AND_EPSILON_IMPLEMENTATION",
            "superseded_utc": now(),
            "superseded_reason": (
                "Read main.direction.pt displacement spectrum, preferred displacement_rank, "
                "and earlier code hard-coded epsilon=0.05 for tail margins. It is not "
                "formal RR2 state-spectrum robustness for W_t S_{D,t}."
            ),
            "superseded_outputs": old_files,
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def state_spectrum_candidates(cell: Cell) -> list[Path]:
    root = STAGE4 / "cells" / cell.model / cell.arm / f"step_{cell.checkpoint:03d}"
    names = [
        f"{cell.probe_name}.L{cell.layer}.main.state_singular.pt",
        f"{cell.probe_name}.L{cell.layer}.main.state_spectrum.pt",
        f"{cell.probe_name}.L{cell.layer}.main.state_singular_values.pt",
        f"{cell.probe_name}.L{cell.layer}.main.state_singular.npy",
        f"{cell.probe_name}.L{cell.layer}.main.state_spectrum.npy",
        f"{cell.probe_name}.L{cell.layer}.main.state_singular_values.npy",
    ]
    return [root / name for name in names]


def run_rr2s_state_preflight() -> None:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for arm in ARMS:
            for checkpoint in STEPS:
                for probe in PROBES:
                    cell = Cell(model, arm, checkpoint, probe)
                    candidates = state_spectrum_candidates(cell)
                    existing = [p for p in candidates if p.exists()]
                    formal_ok, protocol, source_name = formal_cell_source(state_tables(), cell)
                    rows.append(
                        {
                            "model": model,
                            "arm": arm,
                            "checkpoint": checkpoint,
                            "probe_name": probe,
                            "layer": cell.layer,
                            "formal_state_cell_available": formal_ok,
                            "formal_source": source_name,
                            "formal_protocol": protocol,
                            "state_spectrum_path": rel(existing[0]) if existing else "",
                            "candidate_paths_checked": ";".join(rel(p) for p in candidates),
                            "status": (
                                "READY_REUSE_FULL_STATE_SPECTRUM"
                                if existing
                                else "READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT"
                            ),
                            "blocker_reason": ""
                            if existing
                            else "missing complete W_t S_{D,t} singular values or equivalent energy statistics",
                        }
                    )
    preflight = pd.DataFrame(rows)
    preflight_path = OUT / "RR2S_state_spectrum_preflight.csv"
    preflight.to_csv(preflight_path, index=False)

    for name in [
        "RR2S_state_spectrum_stability_module.csv",
        "RR2S_state_spectrum_stability_equal7.csv",
        "RR2S_continuous_ordering.csv",
        "RR2S_continuous_output_links.csv",
    ]:
        pd.DataFrame().to_csv(OUT / name, index=False)

    ready = int((preflight["status"] == "READY_REUSE_FULL_STATE_SPECTRUM").sum())
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": "READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT" if ready == 0 else "PARTIAL_READY_REUSE_REQUIRES_STATE_SPECTRUM_RUNNER",
        "formal_protocol_id": "RR2S_state_spectrum_WtSdt_complete_singular_values_required",
        "models": list(MODELS),
        "arms": ARMS,
        "checkpoints": STEPS,
        "probes": PROBES,
        "layers": MODELS,
        "epsilons": EPSILONS,
        "numeric_protocol": (
            "Formal RR2S requires complete singular values of A_state=W_t S_{D,t}; "
            "tail margin/stable rank/entropy effective rank must be computed from full state spectrum. "
            "Existing D10/D4 rank tables are not sufficient."
        ),
        "row_counts": {
            "preflight_cells": int(len(preflight)),
            "ready_full_state_spectrum_cells": ready,
            "ready_recompute_cells": int((preflight["status"] == "READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT").sum()),
        },
        "blocked_cells": int((preflight["status"] != "READY_REUSE_FULL_STATE_SPECTRUM").sum()),
        "estimated_recompute": {
            "gpu_wall_time": "2-6h on 1x96G for shared formal forward/state-spectrum cache; 1x32G likely slower and tighter",
            "vram": "32G minimum uncertain for both Qwen/Llama full state SVD cache; 96G preferred",
            "ram": "64-128GB CPU RAM preferred for full singular/stat aggregation",
            "scratch": "tens of GB transient; compact CSV/manifest outputs <2GB if singular values are streamed/reduced",
            "do_not_run_without": "explicit Theory GO",
        },
        "recoverable_command": (
            "python experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness.py "
            "--rr2s-forward-cache --formal-grid qwen,llama --arms opd,sft,offkd,seqkd "
            "--steps 20,40,80 --probes E_general,E_math,E_ood,E_if --save-state-spectrum"
        ),
        "atomic_completion_condition": (
            "For every model/arm/checkpoint/probe/layer/module, either complete state singular values "
            "or exact total/tail/stable-rank/entropy sufficient statistics exist with SHA256 provenance."
        ),
        "input_paths_and_sha256": {
            rel(QWEN_FINAL / "qwen_d4_merged_state_all_cells.csv"): sha256_file(QWEN_FINAL / "qwen_d4_merged_state_all_cells.csv"),
            rel(D10 / "d10_5_integrated_state_equal7.csv"): sha256_file(D10 / "d10_5_integrated_state_equal7.csv"),
        },
        "output_sha256": {
            "RR2S_state_spectrum_preflight.csv": sha256_file(preflight_path),
        },
    }
    (OUT / "RR2S_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_rr2d_displacement_auxiliary() -> None:
    rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    coverage = pd.read_csv(OUT / "RR0_grid_coverage.csv")
    ready = coverage[(coverage["task"] == "RR2") & (coverage["status"] == "READY_REUSE")]
    for _, cov in ready.iterrows():
        cell = Cell(str(cov["model"]), str(cov["arm"]), int(cov["checkpoint"]), str(cov["probe_name"]))
        if cell.model != "llama":
            blocked.append(
                {
                    "model": cell.model,
                    "arm": cell.arm,
                    "checkpoint": cell.checkpoint,
                    "probe_name": cell.probe_name,
                    "layer": cell.layer,
                    "status": "BLOCKED_TOP128_NOT_FULL_DIRECTION_SPECTRUM",
                    "reason": "Qwen direction spectrum is stored as top-128 only; continuous/tail metrics are not valid.",
                }
            )
            continue
        direction = load_direction(direction_path(cell))
        meta = read_json(cell_json_path(cell))
        by_mod_eps = {
            (r["module"], round(float(r["epsilon"]), 6)): r
            for r in meta.get("rows", [])
            if r.get("centered") is False
        }
        for module, payload in direction.items():
            singular = payload.get("singular")
            if singular is None:
                continue
            singular_np = singular.detach().cpu().numpy()
            for eps in EPSILONS:
                rank = by_mod_eps.get((module, round(eps, 6)), {}).get("displacement_rank")
                if rank is None:
                    continue
                rows.append(
                    {
                        "model": cell.model,
                        "arm": cell.arm,
                        "checkpoint": cell.checkpoint,
                        "probe_name": cell.probe_name,
                        "layer": cell.layer,
                        "module": module,
                        "epsilon": eps,
                        "rank_at_epsilon": int(rank),
                        "rank_source": "displacement_rank",
                        "spectrum_source": rel(direction_path(cell)),
                        "spectrum_quantity": "activation_conditioned_update_displacement_singular_values",
                        "singular_values_stored": int(len(singular_np)),
                        **energy_stats(singular_np, int(rank), eps),
                    }
                )
    aux = pd.DataFrame(rows)
    aux_path = OUT / "RR2D_displacement_spectrum_auxiliary.csv"
    blocked_path = OUT / "RR2D_blocked_cells.csv"
    aux.to_csv(aux_path, index=False)
    pd.DataFrame(blocked).to_csv(blocked_path, index=False)
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": "COMPLETE_RR2D_LLAMA_DISPLACEMENT_AUXILIARY_WITH_QWEN_BLOCKED",
        "formal_protocol_id": "RR2D_activation_conditioned_update_displacement_auxiliary",
        "not_formal_rr2s": True,
        "numeric_protocol": (
            "Uses complete Llama Stage4 main.direction.pt displacement singular values and displacement_rank. "
            "This is not r_epsilon(W_t S_t) state-spectrum robustness."
        ),
        "models": ["llama"],
        "arms": ARMS,
        "checkpoints": STEPS,
        "probes": PROBES,
        "epsilons": EPSILONS,
        "row_counts": {
            "auxiliary_rows": int(len(aux)),
            "blocked_rows": int(len(blocked)),
        },
        "blocked_cells": int(len(blocked)),
        "input_paths_and_sha256": {
            "RR0_grid_coverage.csv": sha256_file(OUT / "RR0_grid_coverage.csv"),
        },
        "output_sha256": {
            "RR2D_displacement_spectrum_auxiliary.csv": sha256_file(aux_path),
            "RR2D_blocked_cells.csv": sha256_file(blocked_path),
        },
    }
    (OUT / "RR2D_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ar = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    br = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    if np.std(ar) == 0 or np.std(br) == 0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def oof_ridge(X: np.ndarray, y: np.ndarray, groups: np.ndarray, row_ids: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    preds = np.full(len(y), np.nan, dtype=float)
    folds = []
    for test_group in sorted(set(groups)):
        train = groups != test_group
        test = groups == test_group
        if train.sum() <= X.shape[1] or test.sum() == 0:
            continue
        mean = X[train].mean(axis=0)
        std = X[train].std(axis=0)
        std[std == 0] = 1.0
        Xt = (X[train] - mean) / std
        Xv = (X[test] - mean) / std
        Xtd = np.column_stack([np.ones(Xt.shape[0]), Xt])
        Xvd = np.column_stack([np.ones(Xv.shape[0]), Xv])
        penalty = np.eye(Xtd.shape[1]) * 1e-6
        penalty[0, 0] = 0.0
        beta = np.linalg.pinv(Xtd.T @ Xtd + penalty) @ Xtd.T @ y[train]
        preds[test] = Xvd @ beta
        folds.append(
            {
                "fold": f"checkpoint_{int(test_group)}",
                "train_checkpoints": ",".join(map(str, sorted(set(groups[train])))),
                "test_checkpoints": str(int(test_group)),
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
                "test_row_ids": ",".join(map(str, row_ids[test])),
            }
        )
    return preds, folds


def oof_logistic(X: np.ndarray, y: np.ndarray, groups: np.ndarray, row_ids: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    probs = np.full(len(y), np.nan, dtype=float)
    folds = []
    for test_group in sorted(set(groups)):
        train = groups != test_group
        test = groups == test_group
        if train.sum() == 0 or test.sum() == 0 or len(set(y[train])) < 2:
            continue
        mean = X[train].mean(axis=0)
        std = X[train].std(axis=0)
        std[std == 0] = 1.0
        Xt = (X[train] - mean) / std
        Xv = (X[test] - mean) / std
        Xtd = np.column_stack([np.ones(Xt.shape[0]), Xt])
        Xvd = np.column_stack([np.ones(Xv.shape[0]), Xv])
        beta = np.zeros(Xtd.shape[1], dtype=float)
        lr = 0.1
        l2 = 1e-4
        for _ in range(2000):
            z = np.clip(Xtd @ beta, -40, 40)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = Xtd.T @ (p - y[train]) / train.sum()
            grad[1:] += l2 * beta[1:]
            beta -= lr * grad
        probs[test] = 1.0 / (1.0 + np.exp(-np.clip(Xvd @ beta, -40, 40)))
        folds.append(
            {
                "fold": f"checkpoint_{int(test_group)}",
                "train_checkpoints": ",".join(map(str, sorted(set(groups[train])))),
                "test_checkpoints": str(int(test_group)),
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
                "test_row_ids": ",".join(map(str, row_ids[test])),
            }
        )
    return probs, folds


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return wins / total


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(pred) & np.isfinite(y)
    yy = y[valid]
    pp = pred[valid]
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    ss_res = float(np.sum((yy - pp) ** 2))
    return {
        "n_oof": int(valid.sum()),
        "r2_oof": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "mae_oof": float(np.mean(np.abs(yy - pp))) if len(yy) else float("nan"),
        "spearman_oof": rank_corr(yy, pp),
    }


def classification_metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(prob)
    yy = y[valid].astype(int)
    pp = np.clip(prob[valid], 1e-6, 1 - 1e-6)
    pred = (pp >= 0.5).astype(int)
    pos = yy == 1
    neg = yy == 0
    tpr = float(np.mean(pred[pos] == 1)) if pos.any() else float("nan")
    tnr = float(np.mean(pred[neg] == 0)) if neg.any() else float("nan")
    return {
        "n_oof": int(valid.sum()),
        "auc_oof": auc_score(yy, pp),
        "log_loss_oof": float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))) if len(yy) else float("nan"),
        "balanced_accuracy_oof": float(np.nanmean([tpr, tnr])),
    }


def run_rr5_llama_models() -> None:
    d11 = pd.read_csv(D11 / "d11_same_cell_feature_matrix.csv")
    raw = raw_feature_table()
    keys = ["model", "arm", "checkpoint", "probe_name", "layer"]
    feature_cols_a = [
        c
        for c in [
            "normalized_entropy_effective_rank",
            "participation_ratio",
            "top1_explained_share",
            "top8_explained_share",
            "top32_explained_share",
            "raw_anisotropy",
            "centered_anisotropy",
            "linear_cka_vs_step0",
        ]
        if c in raw.columns
    ]
    feature_cols_c = ["c_epsilon"]
    feature_cols_pk = [c for c in ["p_k4", "p_k8", "p_k16", "p_k32"] if c in d11.columns]
    merged = d11.merge(raw[keys + feature_cols_a], on=keys, how="left", indicator=True)
    merged["has_raw_activation"] = merged["_merge"].eq("both")
    coverage = (
        merged.groupby(["model", "arm"], as_index=False)
        .agg(total_cells=("arm", "size"), matched_cells=("has_raw_activation", "sum"))
    )
    coverage["matched_fraction"] = coverage["matched_cells"] / coverage["total_cells"]
    coverage_path = OUT / "RR5_hybrid_coverage.csv"
    coverage.to_csv(coverage_path, index=False)

    llama = merged[(merged["model"] == "llama") & (merged["has_raw_activation"])].copy()
    needed = feature_cols_a + feature_cols_c + feature_cols_pk + [
        "cumulative_kl_base_to_current",
        "absolute_delta_nll_cumulative",
        "delta_nll_cumulative",
    ]
    common = llama.dropna(subset=needed).copy().reset_index(drop=True)
    common["row_id"] = np.arange(len(common))
    common["is_opd"] = (common["arm"] == "opd").astype(int)
    common_path = OUT / "RR5_llama_common_grid.csv"
    common.to_csv(common_path, index=False)

    blocks = {
        "A": feature_cols_a,
        "C": feature_cols_c,
        "Pk": feature_cols_pk,
        "A+C": feature_cols_a + feature_cols_c,
        "Pk+A": feature_cols_pk + feature_cols_a,
        "Pk+C": feature_cols_pk + feature_cols_c,
        "Pk+A+C": feature_cols_pk + feature_cols_a + feature_cols_c,
    }
    targets = {
        "cumulative_kl_base_to_current": "regression",
        "absolute_delta_nll_cumulative": "regression",
        "delta_nll_cumulative": "regression",
        "is_opd": "classification",
    }
    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    if common["checkpoint"].nunique() < 2:
        status = "BLOCKED_INSUFFICIENT_CHECKPOINT_GROUPS"
    else:
        status = "COMPLETE_RR5_LLAMA_ONLY_EXACT_COMMON_GRID"
        for target, task_type in targets.items():
            per_target_metrics: dict[str, dict[str, float]] = {}
            for block, cols in blocks.items():
                subset = common.dropna(subset=cols + [target]).copy()
                X = subset[cols].to_numpy(dtype=float)
                y = subset[target].to_numpy(dtype=float)
                groups = subset["checkpoint"].to_numpy(dtype=int)
                row_ids = subset["row_id"].to_numpy(dtype=int)
                if task_type == "regression":
                    pred, folds = oof_ridge(X, y, groups, row_ids)
                    m = regression_metrics(y, pred)
                    pred_col = "y_pred"
                else:
                    pred, folds = oof_logistic(X, y, groups, row_ids)
                    m = classification_metrics(y, pred)
                    pred_col = "y_prob"
                per_target_metrics[block] = m
                row = {
                    "model": "llama",
                    "target": target,
                    "task_type": task_type,
                    "feature_block": block,
                    "features": ",".join(cols),
                    "n_common": int(len(subset)),
                    "n_checkpoint_groups": int(subset["checkpoint"].nunique()),
                    "checkpoint_groups": ",".join(map(str, sorted(subset["checkpoint"].unique()))),
                    **m,
                }
                metric_rows.append(row)
                for fold in folds:
                    fold_rows.append(
                        {
                            "model": "llama",
                            "target": target,
                            "task_type": task_type,
                            "feature_block": block,
                            **fold,
                        }
                    )
                for i, (_, src) in enumerate(subset.iterrows()):
                    pred_rows.append(
                        {
                            "model": "llama",
                            "target": target,
                            "task_type": task_type,
                            "feature_block": block,
                            "row_id": int(src["row_id"]),
                            "arm": src["arm"],
                            "checkpoint": int(src["checkpoint"]),
                            "probe_name": src["probe_name"],
                            "layer": int(src["layer"]),
                            "y_true": float(y[i]),
                            pred_col: float(pred[i]) if np.isfinite(pred[i]) else np.nan,
                        }
                    )
            for row in metric_rows:
                if row["target"] != target:
                    continue
                for baseline in ["A", "C", "Pk"]:
                    base = per_target_metrics.get(baseline, {})
                    if task_type == "regression":
                        row[f"delta_r2_vs_{baseline}"] = row.get("r2_oof", np.nan) - base.get("r2_oof", np.nan)
                        row[f"mae_reduction_vs_{baseline}"] = base.get("mae_oof", np.nan) - row.get("mae_oof", np.nan)
                        row[f"delta_spearman_vs_{baseline}"] = row.get("spearman_oof", np.nan) - base.get("spearman_oof", np.nan)
                    else:
                        row[f"delta_auc_vs_{baseline}"] = row.get("auc_oof", np.nan) - base.get("auc_oof", np.nan)
                        row[f"log_loss_reduction_vs_{baseline}"] = base.get("log_loss_oof", np.nan) - row.get("log_loss_oof", np.nan)
                        row[f"delta_balanced_accuracy_vs_{baseline}"] = (
                            row.get("balanced_accuracy_oof", np.nan) - base.get("balanced_accuracy_oof", np.nan)
                        )

    metrics_path = OUT / "RR5_hybrid_grouped_models.csv"
    folds_path = OUT / "RR5_hybrid_fold_metrics.csv"
    pred_path = OUT / "RR5_hybrid_predictions.parquet"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(fold_rows).to_csv(folds_path, index=False)
    pd.DataFrame(pred_rows).to_parquet(pred_path, index=False)
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": status,
        "formal_protocol_id": "RR5_llama_only_exact_key_common_grid_checkpoint_grouped_oof",
        "join_policy": "strict exact-key join only; no imputation; no nearest checkpoint; no probe replacement",
        "standardization": "feature mean/std fit on train checkpoint folds only",
        "models": ["llama"],
        "qwen_status": "BLOCKED_RAW_ACTIVATION_EXACT_GRID_MISSING_FOR_OPD_SFT_SEQKD",
        "feature_blocks": blocks,
        "targets": targets,
        "row_counts": {
            "coverage": int(len(coverage)),
            "llama_common_grid": int(len(common)),
            "model_metrics": int(len(metric_rows)),
            "fold_rows": int(len(fold_rows)),
            "predictions": int(len(pred_rows)),
        },
        "blocked_cells": int((merged[(merged["model"] == "qwen")]["_merge"] != "both").sum()),
        "input_paths_and_sha256": {
            rel(D11 / "d11_same_cell_feature_matrix.csv"): sha256_file(D11 / "d11_same_cell_feature_matrix.csv"),
        },
        "output_sha256": {
            "RR5_llama_common_grid.csv": sha256_file(common_path),
            "RR5_hybrid_coverage.csv": sha256_file(coverage_path),
            "RR5_hybrid_grouped_models.csv": sha256_file(metrics_path),
            "RR5_hybrid_fold_metrics.csv": sha256_file(folds_path),
            "RR5_hybrid_predictions.parquet": sha256_file(pred_path),
        },
    }
    (OUT / "RR5_hybrid_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_top32_logprobs_with_offsets(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if path.suffix == ".npz":
        archive = np.load(path)
        offsets = archive["row_offsets"] if "row_offsets" in archive.files else None
        return archive["top32_logprob"], offsets
    arr = np.load(path, mmap_mode="r")
    offset_candidates = [
        path.with_name("row_offsets.npy"),
        path.parent / "row_offsets.npy",
        path.parent / "response_offsets.npy",
    ]
    offsets = None
    for candidate in offset_candidates:
        if candidate.exists():
            offsets = np.load(candidate, mmap_mode="r")
            break
    return arr, offsets


def summarize_mass(raw_mass: np.ndarray) -> dict[str, float]:
    clamped = np.clip(raw_mass, 0.0, 1.0)
    omitted = np.maximum(0.0, 1.0 - raw_mass)
    out = {}
    for prefix, vals in [
        ("raw", raw_mass),
        ("clamped", clamped),
        ("omitted", omitted),
    ]:
        s = summarize_array(vals)
        out.update({f"{prefix}_{k}": v for k, v in s.items()})
    return out


def run_rr4_provenance() -> None:
    summary_rows = []
    seq_rows = []
    coverage_rows = []
    for src in top32_sources():
        arm = src["pipeline"]
        arm_resolution = "pipeline_name_only"
        if src["model"] == "llama" and src["pipeline"] == "model2_llama":
            arm = "offkd"
            arm_resolution = "resolved_from_training_manifest_cycle09_block2_model2_llama_g6_offkd"
        if not src["exists"]:
            coverage_rows.append({**src, "resolved_arm": arm, "arm_resolution": arm_resolution, "status": "BLOCKED_MISSING_ARTIFACT"})
            continue
        if not src["raw_logprob_manifest"]:
            coverage_rows.append({**src, "resolved_arm": arm, "arm_resolution": arm_resolution, "status": "BLOCKED_RENORMALIZED_TOPK"})
            continue
        path = Path(src["path"])
        logprobs, offsets = load_top32_logprobs_with_offsets(path)
        mass = top32_mass(logprobs)
        n_seq = int(offsets.shape[0]) if offsets is not None else ""
        summary_rows.append(
            {
                "model": src["model"],
                "arm_or_pipeline": src["pipeline"],
                "resolved_arm": arm,
                "arm_resolution": arm_resolution,
                "checkpoint_or_rollout_source": src["path"],
                "n_sequences": n_seq,
                "n_tokens": int(mass.shape[0]),
                "weighting": "token_weighted",
                **summarize_mass(mass),
            }
        )
        if offsets is not None:
            for i, (start, end) in enumerate(offsets.astype(int)):
                vals = mass[start:end]
                if len(vals) == 0:
                    continue
                seq_rows.append(
                    {
                        "model": src["model"],
                        "arm_or_pipeline": src["pipeline"],
                        "resolved_arm": arm,
                        "sequence_index": i,
                        "token_start": int(start),
                        "token_end": int(end),
                        "n_tokens": int(end - start),
                        "mean_retained_mass_raw": float(vals.mean()),
                        "mean_retained_mass_clamped": float(np.clip(vals, 0, 1).mean()),
                        "mean_omitted_mass": float(np.maximum(0, 1 - vals).mean()),
                        "min_retained_mass_raw": float(vals.min()),
                        "p05_retained_mass_raw": float(np.quantile(vals, 0.05)),
                    }
                )
        coverage_rows.append({**src, "resolved_arm": arm, "arm_resolution": arm_resolution, "status": "READY_REUSE_RAW_LOGPROB"})
        del logprobs, mass
    summary_path = OUT / "RR4_top32_retained_mass_summary.csv"
    seq_path = OUT / "RR4_top32_retained_mass_by_sequence.csv"
    cov_path = OUT / "RR4_top32_coverage.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(seq_rows).to_csv(seq_path, index=False)
    pd.DataFrame(coverage_rows).to_csv(cov_path, index=False)
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": "COMPLETE_RR4_PROVENANCE_SUPPLEMENTED",
        "formal_protocol_id": "RR4_token_weighted_raw_top32_retained_mass_with_clamped_and_omitted_mass",
        "numeric_protocol": (
            "raw retained mass m32=sum(exp(raw teacher top32 logprobs)); raw values preserved; "
            "clamped_mass=clip(m32,0,1); omitted_mass=max(0,1-m32). Token-weighted summaries."
        ),
        "not_exact_full_vocab_kl": True,
        "models": list(MODELS),
        "row_counts": {
            "summary": int(len(summary_rows)),
            "by_sequence": int(len(seq_rows)),
            "coverage": int(len(coverage_rows)),
        },
        "blocked_cells": int(sum(1 for r in coverage_rows if not str(r["status"]).startswith("READY"))),
        "input_paths_and_sha256": {r["path"]: r.get("sha256", "") for r in coverage_rows},
        "output_sha256": {
            "RR4_top32_retained_mass_summary.csv": sha256_file(summary_path),
            "RR4_top32_retained_mass_by_sequence.csv": sha256_file(seq_path),
            "RR4_top32_coverage.csv": sha256_file(cov_path),
        },
    }
    (OUT / "RR4_top32_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_rr6_readout_bootstrap(draws: int = 2000, seed: int = 42) -> None:
    old_pair = OUT / "RR6_frozen_self_paired_differences.csv"
    old_stats = OUT / "RR6_frozen_self_text_stats.csv"
    old_cov = OUT / "RR6_frozen_self_coverage.csv"
    if not old_pair.exists() or not old_stats.exists():
        run_rr6()
    pair = pd.read_csv(old_pair)
    stats = pd.read_csv(old_stats).rename(columns=lambda c: c.replace("response_tokens", "response_whitespace_tokens"))
    pair = pair.rename(columns=lambda c: c.replace("response_tokens", "response_whitespace_tokens"))
    stats_path = OUT / "RR6_matched_math500_readout_stats.csv"
    effects_path = OUT / "RR6_matched_math500_paired_effects.csv"
    boot_path = OUT / "RR6_matched_math500_bootstrap_ci.csv"
    cov_path = OUT / "RR6_matched_math500_coverage.csv"
    stats.to_csv(stats_path, index=False)
    if old_cov.exists():
        pd.read_csv(old_cov).to_csv(cov_path, index=False)
    metrics = [
        "response_whitespace_tokens",
        "eos",
        "truncated",
        "fourgram_repetition",
        "distinct_2",
        "boxed",
        "think_tag",
    ]
    effect_rows = []
    boot_rows = []
    rng = np.random.default_rng(seed)
    for (checkpoint, dataset), g in pair.groupby(["checkpoint", "dataset"]):
        n = len(g)
        if n == 0:
            continue
        sample_idx = rng.integers(0, n, size=(draws, n))
        for metric in metrics:
            col = metric + "_opd_minus_frozen_self"
            if col not in g.columns:
                continue
            vals = g[col].to_numpy(dtype=float)
            boots = vals[sample_idx].mean(axis=1)
            effect_rows.append(
                {
                    "checkpoint": int(checkpoint),
                    "dataset": dataset,
                    "metric": metric,
                    "n_pairs": int(n),
                    "mean_opd_minus_frozen_self": float(np.mean(vals)),
                    "median_opd_minus_frozen_self": float(np.median(vals)),
                    "sign_probability_gt0": float(np.mean(vals > 0)),
                    "sign_probability_lt0": float(np.mean(vals < 0)),
                }
            )
            boot_rows.append(
                {
                    "checkpoint": int(checkpoint),
                    "dataset": dataset,
                    "metric": metric,
                    "draws": draws,
                    "seed": seed,
                    "mean": float(np.mean(vals)),
                    "ci95_low": float(np.quantile(boots, 0.025)),
                    "ci95_high": float(np.quantile(boots, 0.975)),
                    "bootstrap_mean": float(np.mean(boots)),
                    "bootstrap_std": float(np.std(boots)),
                    "sign_probability_bootstrap_gt0": float(np.mean(boots > 0)),
                    "sign_probability_bootstrap_lt0": float(np.mean(boots < 0)),
                }
            )
    pd.DataFrame(effect_rows).to_csv(effects_path, index=False)
    pd.DataFrame(boot_rows).to_csv(boot_path, index=False)
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": "COMPLETE_RR6_MATCHED_BEHAVIORAL_READOUT_DIAGNOSTICS",
        "formal_protocol_id": "RR6_matched_math500_behavioral_readout_not_training_mediator",
        "models": ["llama"],
        "arms": ["opd", "frozen_self"],
        "checkpoints": STEPS + [160],
        "dataset": "math500",
        "draws_and_seeds": {"paired_item_bootstrap_draws": draws, "seed": seed},
        "token_count_protocol": "response_whitespace_tokens from regex \\S+; no tokenizer token ids found or regenerated",
        "non_mediator_note": (
            "These are matched MATH500 eval behavior readouts, not training rollout support; "
            "length/EOS/truncation/repetition/boxed effects change direction across checkpoints and do not identify a single stable mediator."
        ),
        "row_counts": {
            "stats": int(len(stats)),
            "paired_effects": int(len(effect_rows)),
            "bootstrap_ci": int(len(boot_rows)),
            "paired_rows": int(len(pair)),
        },
        "input_paths_and_sha256": {
            "RR6_frozen_self_text_stats.csv": sha256_file(old_stats),
            "RR6_frozen_self_paired_differences.csv": sha256_file(old_pair),
        },
        "output_sha256": {
            "RR6_matched_math500_readout_stats.csv": sha256_file(stats_path),
            "RR6_matched_math500_paired_effects.csv": sha256_file(effects_path),
            "RR6_matched_math500_bootstrap_ci.csv": sha256_file(boot_path),
            "RR6_matched_math500_coverage.csv": sha256_file(cov_path),
        },
    }
    (OUT / "RR6_readout_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_rr1_rr3_joint_preflight() -> None:
    rows = []
    for model in MODELS:
        for arm in ARMS:
            for checkpoint in STEPS:
                for probe in PROBES:
                    cell = Cell(model, arm, checkpoint, probe)
                    factor = find_sample_factor(cell)
                    formal_ok, protocol, source_name = formal_cell_source(state_tables(), cell)
                    rows.append(
                        {
                            "model": model,
                            "arm": arm,
                            "checkpoint": checkpoint,
                            "probe_name": probe,
                            "layer": cell.layer,
                            "formal_cell_available": formal_ok,
                            "formal_source": source_name,
                            "formal_protocol": protocol,
                            "existing_sample_factor_path": rel(factor) if factor else "",
                            "can_reuse_existing_pt_profile": bool(factor),
                            "missing_per_sample_information": (
                                "" if factor else "per-sample module hidden/factor rows before Gram aggregation; centered and uncentered sample covariance inputs"
                            ),
                            "status": "READY_REUSE_SAMPLE_FACTOR" if factor else "READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT",
                        }
                    )
    df = pd.DataFrame(rows)
    path = OUT / "RR1_RR3_joint_forward_preflight.csv"
    df.to_csv(path, index=False)
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": "PREFLIGHT_ONLY_NO_FORWARD_STARTED",
        "formal_protocol_id": "RR1A_RR1B_RR3_shared_formal_forward_cache",
        "grid_counts": {
            "checkpoint_model_arm_probe_profiles": int(len(df)),
            "models": len(MODELS),
            "arms": len(ARMS),
            "checkpoints": len(STEPS),
            "probes": len(PROBES),
            "ready_reuse_existing_pt_profile": int(df["can_reuse_existing_pt_profile"].sum()),
            "ready_recompute": int((~df["can_reuse_existing_pt_profile"]).sum()),
        },
        "shared_cache_schema": {
            "identity": ["model", "arm", "checkpoint", "probe_name", "layer", "module", "sample_id"],
            "arrays": [
                "window_token_mean_hidden_or_factor_fp32",
                "uncentered_second_moment_contribution_fp32",
                "centered_covariance_inputs_fp32",
            ],
            "metadata": [
                "prompt_source",
                "sample_count",
                "window_v2_manifest",
                "generation_seed",
                "window_seed",
                "checkpoint_sha256",
                "profile_sha256",
            ],
        },
        "estimated_cost": {
            "gpu_wall_time": "2-6h shared extraction on 1x96G; avoid separate forwards for RR1A/RR1B/RR3",
            "bootstrap_cpu_time": "RR1A 1024 draws plus RR1B subsets and RR3 covariance postprocess: ~1-4h CPU",
            "vram": "96G preferred; 32G may require smaller batches and more wall time",
            "ram": "64-128GB CPU RAM recommended",
            "scratch": "tens of GB transient per-sample cache; compact final tables <2GB",
        },
        "recoverable_command": (
            "python experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness.py "
            "--rr1-rr3-forward-cache --formal-grid qwen,llama --arms opd,sft,offkd,seqkd "
            "--steps 20,40,80 --probes E_general,E_math,E_ood,E_if --atomic-cache"
        ),
        "atomic_completion_condition": (
            "Every profile has a complete manifest and temp files are atomically renamed only after all module/sample arrays and SHA256 hashes are written."
        ),
        "input_paths_and_sha256": {
            "RR0_grid_coverage.csv": sha256_file(OUT / "RR0_grid_coverage.csv"),
        },
        "output_sha256": {
            "RR1_RR3_joint_forward_preflight.csv": sha256_file(path),
        },
    }
    (OUT / "RR1_RR3_joint_forward_preflight_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_correction_handoff() -> None:
    def table(path: Path, n: int = 20) -> str:
        if not path.exists() or path.stat().st_size == 0:
            return "_missing_"
        try:
            df = pd.read_csv(path)
        except Exception:
            return "_unreadable_"
        if df.empty:
            return "_empty_"
        return df.head(n).to_markdown(index=False)

    rr2s = read_json(OUT / "RR2S_manifest.json")
    rr2d = read_json(OUT / "RR2D_manifest.json")
    rr4 = read_json(OUT / "RR4_top32_manifest.json")
    rr5 = read_json(OUT / "RR5_hybrid_manifest.json")
    rr6 = read_json(OUT / "RR6_readout_manifest.json")
    rr13 = read_json(OUT / "RR1_RR3_joint_forward_preflight_manifest.json")
    lines = [
        "# Reviewer Robustness Theory Handoff",
        "",
        "```yaml",
        "status: CORRECTION_PASS_COMPLETE_WITH_NEW_FORWARD_BLOCKERS",
        f"created_utc: {now()}",
        f"script: {rel(REPO / 'experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness.py')}",
        "command_correction: python experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness.py --correction-pass",
        f"output_root: {rel(OUT)}/",
        "guard: zero training; no new forward; raw readings only; no theory adjudication; no paper/human_read edits",
        f"git_commit: {git_commit()}",
        "```",
        "",
        "## Superseded Results",
        "",
        "| file group | status | reason |",
        "|---|---|---|",
        "| RR2_spectrum_stability_module/equal7/ordering/output_links + RR2_spectrum_manifest | SUPERSEDED_WRONG_ESTIMAND_AND_EPSILON_IMPLEMENTATION | used displacement spectrum, not W_t S_{D,t} state spectrum; previous epsilon margin implementation was invalid |",
        "",
        "## Formal Usable / Auxiliary / Blocked",
        "",
        "| item | status | use class | row counts / blocker |",
        "|---|---|---|---|",
        f"| RR2S state spectrum | {rr2s.get('status','')} | needs new forward before formal use | ready full-state spectra={rr2s.get('row_counts',{}).get('ready_full_state_spectrum_cells','')}; recompute={rr2s.get('row_counts',{}).get('ready_recompute_cells','')} |",
        f"| RR2D displacement spectrum | {rr2d.get('status','')} | auxiliary only | rows={rr2d.get('row_counts',{}).get('auxiliary_rows','')}; blocked={rr2d.get('blocked_cells','')} |",
        f"| RR5 Llama-only hybrid | {rr5.get('status','')} | formal Llama-only exact common-grid result | common rows={rr5.get('row_counts',{}).get('llama_common_grid','')}; metric rows={rr5.get('row_counts',{}).get('model_metrics','')} |",
        f"| RR6 matched Math500 readout | {rr6.get('status','')} | formal matched behavioral-readout diagnostics | bootstrap rows={rr6.get('row_counts',{}).get('bootstrap_ci','')} |",
        f"| RR4 top32 retained mass | {rr4.get('status','')} | provenance-supplemented retained-mass diagnostic | summary rows={rr4.get('row_counts',{}).get('summary','')}; blocked={rr4.get('blocked_cells','')} |",
        f"| RR1/RR3 joint forward preflight | {rr13.get('status','')} | plan only; no forward started | profiles={rr13.get('grid_counts',{}).get('checkpoint_model_arm_probe_profiles','')}; recompute={rr13.get('grid_counts',{}).get('ready_recompute','')} |",
        "",
        "## RR2S State-Spectrum Preflight",
        "",
        table(OUT / "RR2S_state_spectrum_preflight.csv", 12),
        "",
        "## RR2D Displacement Auxiliary",
        "",
        table(OUT / "RR2D_displacement_spectrum_auxiliary.csv", 12),
        "",
        "## RR5 Llama-Only Hybrid Models",
        "",
        "Common grid:",
        "",
        table(OUT / "RR5_llama_common_grid.csv", 8),
        "",
        "Model metrics:",
        "",
        table(OUT / "RR5_hybrid_grouped_models.csv", 28),
        "",
        "Fold rows:",
        "",
        table(OUT / "RR5_hybrid_fold_metrics.csv", 16),
        "",
        "## RR6 Matched Behavioral-Readout Diagnostics",
        "",
        "Stats:",
        "",
        table(OUT / "RR6_matched_math500_readout_stats.csv", 12),
        "",
        "Paired effects:",
        "",
        table(OUT / "RR6_matched_math500_paired_effects.csv", 28),
        "",
        "Bootstrap CI:",
        "",
        table(OUT / "RR6_matched_math500_bootstrap_ci.csv", 28),
        "",
        "## RR4 Top-32 Retained Mass Provenance",
        "",
        table(OUT / "RR4_top32_retained_mass_summary.csv", 8),
        "",
        "Coverage:",
        "",
        table(OUT / "RR4_top32_coverage.csv", 8),
        "",
        "## RR1/RR3 Joint Forward Preflight",
        "",
        table(OUT / "RR1_RR3_joint_forward_preflight.csv", 12),
        "",
        "## Output Files",
        "",
    ]
    for p in sorted(OUT.glob("RR*S*")) + sorted(OUT.glob("RR2D*")) + sorted(OUT.glob("RR5*")) + sorted(OUT.glob("RR6_matched*")) + [OUT / "RR6_readout_manifest.json", OUT / "RR4_top32_manifest.json"]:
        if p.exists():
            lines.append(f"- `{p.name}`")
    (OUT / "reviewer_robustness_theory_handoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_correction_return() -> None:
    path = REPO / "mypaper/code/cycle09_reviewer_robustness_handoff.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "# Cycle09 Reviewer Robustness Handoff\n"
    marker = "## 14. Correction Return: 2026-07-27"
    block = f"""

{marker}

Theory复核后的 CPU correction pass 已完成；未修改论文、`human_read-ch.md` 或理论结论。

输出目录：

```text
{rel(OUT)}/
```

执行命令：

```bash
python experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness.py --correction-pass
```

状态：

| 项 | 状态 |
|---|---|
| 旧 RR2 | `SUPERSEDED_WRONG_ESTIMAND_AND_EPSILON_IMPLEMENTATION` |
| RR2S state spectrum | `READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT`，未启动 new forward |
| RR2D displacement spectrum | auxiliary only，Llama only；Qwen top-128 blocked |
| RR5 | Llama-only exact common-grid grouped models 完成；Qwen raw activation exact-grid blocked |
| RR6 | 改名为 matched behavioral-readout diagnostics，并加入 paired item-bootstrap CI |
| RR4 | provenance/clamped/omitted mass 补充完成；Qwen alpha=.5 仍 blocked |
| RR1/RR3 | shared forward/cache preflight only，未启动 new forward |

正式 theory handoff：

```text
{rel(OUT / 'reviewer_robustness_theory_handoff.md')}
```
"""
    if marker in text:
        text = text[: text.index(marker)].rstrip() + block
    else:
        text = text.rstrip() + block
    path.write_text(text + "\n", encoding="utf-8")


def append_code_evolution_correction() -> None:
    path = REPO / "mypaper/code/code_evolution.md"
    entry = f"""

### 2026-07-27 Reviewer Robustness Correction Pass

- Script updated: `experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness.py`.
- Old RR2 manifest marked `SUPERSEDED_WRONG_ESTIMAND_AND_EPSILON_IMPLEMENTATION`.
- Added RR2S state-spectrum preflight, RR2D displacement-spectrum auxiliary, RR5 Llama-only exact common-grid fitting, RR6 matched Math500 readout bootstrap, RR4 retained-mass provenance supplement, and RR1/RR3 shared forward preflight.
- Output handoff: `mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/reviewer_robustness/reviewer_robustness_theory_handoff.md`.
"""
    text = path.read_text(encoding="utf-8") if path.exists() else "# Code Evolution\n"
    if "### 2026-07-27 Reviewer Robustness Correction Pass" not in text:
        path.write_text(text.rstrip() + entry + "\n", encoding="utf-8")


def run_correction_pass() -> None:
    if not (OUT / "RR0_grid_coverage.csv").exists():
        run_rr0()
    supersede_old_rr2()
    run_rr2s_state_preflight()
    run_rr2d_displacement_auxiliary()
    run_rr5_llama_models()
    run_rr6_readout_bootstrap()
    run_rr4_provenance()
    run_rr1_rr3_joint_preflight()
    write_correction_handoff()
    append_correction_return()
    append_code_evolution_correction()
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": "CORRECTION_PASS_COMPLETE_WITH_NEW_FORWARD_BLOCKERS",
        "handoff": rel(OUT / "reviewer_robustness_theory_handoff.md"),
    }
    (OUT / "reviewer_robustness_correction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def run_ready_reuse() -> None:
    if not (OUT / "RR0_grid_coverage.csv").exists():
        run_rr0()
    run_rr2()
    run_rr4()
    run_rr5()
    run_rr6()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rr0", action="store_true", help="Run RR0 inventory only.")
    parser.add_argument("--rr2", action="store_true", help="Run RR2 reuse analysis.")
    parser.add_argument("--rr4", action="store_true", help="Run RR4 reuse analysis.")
    parser.add_argument("--rr5", action="store_true", help="Run RR5 reuse coverage/model gate.")
    parser.add_argument("--rr6", action="store_true", help="Run RR6 reuse diagnostics.")
    parser.add_argument("--run-ready-reuse", action="store_true", help="Run all RR0-gated READY_REUSE tasks.")
    parser.add_argument("--correction-pass", action="store_true", help="Run Theory correction pass without new forward.")
    args = parser.parse_args()
    if args.rr0:
        run_rr0()
    elif args.rr2:
        run_rr2()
    elif args.rr4:
        run_rr4()
    elif args.rr5:
        run_rr5()
    elif args.rr6:
        run_rr6()
    elif args.run_ready_reuse:
        run_ready_reuse()
    elif args.correction_pass:
        run_correction_pass()
    else:
        parser.error("Choose --rr0, --rr2, --rr4, --rr5, --rr6, --run-ready-reuse, or --correction-pass.")


if __name__ == "__main__":
    main()
