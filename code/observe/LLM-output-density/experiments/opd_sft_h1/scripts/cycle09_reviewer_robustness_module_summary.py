#!/usr/bin/env python3
"""Formal tie-aware module-level robustness summary for reviewer-robustness.

This is a low-cost reuse-only postprocess.  It reads the formal RR2S and RR3
module-level tables and separates cells where OPD is the unique deepest arm,
tied for deepest, or strictly shallower than an offline arm.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path("/root/LLM-output-density")
OUT = (
    REPO
    / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/reviewer_robustness"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def sha256_file(path: Path, max_bytes: int = 512 * 1024 * 1024) -> str:
    if not path.is_file():
        return ""
    size = path.stat().st_size
    if size > max_bytes:
        return f"SKIPPED_SIZE_{size}"
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def module_level_cells(
    df: pd.DataFrame,
    *,
    analysis: str,
    source_table: str,
    metric: str,
    metric_direction: str,
    module_filter: str,
    formal_use_class: str,
) -> pd.DataFrame:
    if metric_direction != "max_is_deepest":
        raise ValueError(f"unsupported metric_direction: {metric_direction}")
    d = df[df["arm"].astype(str).ne("base")].copy()
    if module_filter == "non_qk":
        d = d[~d["module"].isin(["self_attn.q_proj", "self_attn.k_proj"])].copy()
    elif module_filter == "q_proj":
        d = d[d["module"].eq("self_attn.q_proj")].copy()
    elif module_filter == "k_proj":
        d = d[d["module"].eq("self_attn.k_proj")].copy()
    elif module_filter != "all":
        raise ValueError(f"unsupported module_filter: {module_filter}")

    keys = ["checkpoint", "probe_name", "module", "epsilon"]
    rows: list[dict[str, Any]] = []
    for key, g in d.groupby(keys, dropna=False):
        g = g.dropna(subset=[metric])
        if g.empty:
            continue
        best_value = float(g[metric].max())
        best_arms = sorted(
            g.loc[
                np.isclose(
                    g[metric].to_numpy(dtype=float),
                    best_value,
                    rtol=1e-9,
                    atol=1e-12,
                ),
                "arm",
            ]
            .astype(str)
            .unique()
        )
        opd = g[g["arm"].astype(str).eq("opd")]
        nearest_offline = float(g[~g["arm"].astype(str).eq("opd")][metric].max())
        opd_among_best = "opd" in best_arms
        opd_strict_deepest = opd_among_best and len(best_arms) == 1
        opd_tied_deepest = opd_among_best and len(best_arms) > 1
        offline_strictly_deeper = not opd_among_best
        checkpoint, probe_name, module, epsilon = key
        arm_values = {str(r.arm): float(getattr(r, metric)) for r in g.itertuples()}
        rows.append(
            {
                "analysis": analysis,
                "formal_use_class": formal_use_class,
                "source_table": source_table,
                "metric": metric,
                "metric_direction": metric_direction,
                "module_filter": module_filter,
                "model": "llama",
                "checkpoint": int(checkpoint),
                "probe_name": probe_name,
                "module": module,
                "epsilon": float(epsilon),
                "best_value": best_value,
                "best_arms": ",".join(best_arms),
                "opd_among_best": opd_among_best,
                "opd_strict_deepest": opd_strict_deepest,
                "opd_tied_deepest": opd_tied_deepest,
                "offline_strictly_deeper": offline_strictly_deeper,
                "opd_value": float(opd[metric].iloc[0]) if not opd.empty else np.nan,
                "nearest_offline_value": nearest_offline,
                "opd_minus_nearest_offline_margin": float(opd[metric].iloc[0] - nearest_offline) if not opd.empty else np.nan,
                "arm_values_json": json.dumps(arm_values, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    rr2s_path = OUT / "RR2S_llama_state_spectrum_module.csv"
    rr3_path = OUT / "RR3_llama_centered_module.csv"
    rr2s = pd.read_csv(rr2s_path)
    rr3 = pd.read_csv(rr3_path)
    specs = [
        {
            "analysis": "uncentered_r_epsilon",
            "source_table": rr2s_path.name,
            "df": rr2s,
            "metric": "absolute_contraction",
            "module_filter": "all",
            "formal_use_class": "formal_rr2s_llama_state_spectrum",
        },
        {
            "analysis": "uncentered_stable_rank",
            "source_table": rr2s_path.name,
            "df": rr2s,
            "metric": "stable_rank_contraction",
            "module_filter": "all",
            "formal_use_class": "formal_rr2s_llama_state_spectrum",
        },
        {
            "analysis": "uncentered_entropy_effective_rank",
            "source_table": rr2s_path.name,
            "df": rr2s,
            "metric": "entropy_effective_rank_contraction",
            "module_filter": "all",
            "formal_use_class": "formal_rr2s_llama_state_spectrum",
        },
        {
            "analysis": "centered_r_epsilon",
            "source_table": rr3_path.name,
            "df": rr3,
            "metric": "centered_absolute_contraction",
            "module_filter": "all",
            "formal_use_class": "formal_rr3_llama_centered_audit",
        },
        {
            "analysis": "centered_r_epsilon_non_qk_modules",
            "source_table": rr3_path.name,
            "df": rr3,
            "metric": "centered_absolute_contraction",
            "module_filter": "non_qk",
            "formal_use_class": "formal_rr3_llama_centered_audit",
        },
        {
            "analysis": "centered_r_epsilon_q_proj",
            "source_table": rr3_path.name,
            "df": rr3,
            "metric": "centered_absolute_contraction",
            "module_filter": "q_proj",
            "formal_use_class": "formal_rr3_llama_centered_audit",
        },
        {
            "analysis": "centered_r_epsilon_k_proj",
            "source_table": rr3_path.name,
            "df": rr3,
            "metric": "centered_absolute_contraction",
            "module_filter": "k_proj",
            "formal_use_class": "formal_rr3_llama_centered_audit",
        },
    ]
    cell_frames = [
        module_level_cells(
            spec["df"],
            analysis=spec["analysis"],
            source_table=spec["source_table"],
            metric=spec["metric"],
            metric_direction="max_is_deepest",
            module_filter=spec["module_filter"],
            formal_use_class=spec["formal_use_class"],
        )
        for spec in specs
    ]
    cells = pd.concat(cell_frames, ignore_index=True)
    summary = (
        cells.groupby(
            [
                "analysis",
                "formal_use_class",
                "source_table",
                "metric",
                "metric_direction",
                "module_filter",
            ],
            as_index=False,
        )
        .agg(
            opd_among_best_count=("opd_among_best", "sum"),
            opd_strict_deepest_count=("opd_strict_deepest", "sum"),
            opd_tied_deepest_count=("opd_tied_deepest", "sum"),
            offline_strictly_deeper_count=("offline_strictly_deeper", "sum"),
            total_cells=("opd_among_best", "size"),
            opd_among_best_fraction=("opd_among_best", "mean"),
            opd_strict_deepest_fraction=("opd_strict_deepest", "mean"),
            opd_tied_deepest_fraction=("opd_tied_deepest", "mean"),
            offline_strictly_deeper_fraction=("offline_strictly_deeper", "mean"),
            mean_opd_minus_nearest_offline_margin=("opd_minus_nearest_offline_margin", "mean"),
            min_opd_minus_nearest_offline_margin=("opd_minus_nearest_offline_margin", "min"),
            median_opd_minus_nearest_offline_margin=("opd_minus_nearest_offline_margin", "median"),
        )
    )
    partition_total = (
        summary["opd_strict_deepest_count"]
        + summary["opd_tied_deepest_count"]
        + summary["offline_strictly_deeper_count"]
    )
    if not partition_total.eq(summary["total_cells"]).all():
        raise AssertionError("strict/tied/offline cell counts do not partition total_cells")
    order = {spec["analysis"]: i for i, spec in enumerate(specs)}
    summary["_order"] = summary["analysis"].map(order)
    summary = summary.sort_values("_order").drop(columns=["_order"])

    cells_path = OUT / "RR_module_level_robustness_cells.csv"
    summary_path = OUT / "RR_module_level_robustness_summary.csv"
    manifest_path = OUT / "RR_module_level_robustness_manifest.json"
    cells.to_csv(cells_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": "COMPLETE_TIE_AWARE_MODULE_LEVEL_ROBUSTNESS_SUMMARY_REUSE_ONLY",
        "formal_protocol_id": "reviewer_robustness_module_level_opd_strict_tied_offline_summary",
        "numeric_protocol": "Reuse-only postprocess over formal RR2S Llama state-spectrum module table and formal RR3 Llama centered module table; no forward, no training.",
        "tie_policy": {
            "comparison": "np.isclose",
            "rtol": 1e-9,
            "atol": 1e-12,
            "opd_among_best": "OPD is one of the maximizing arms.",
            "opd_strict_deepest": "OPD is the unique maximizing arm.",
            "opd_tied_deepest": "OPD maximizes but at least one offline arm ties it.",
            "offline_strictly_deeper": "At least one offline arm exceeds OPD.",
        },
        "metric_direction": "max_is_deepest for contraction metrics; RR2D remains auxiliary and is not included in this formal module-level robustness count.",
        "status_notes": {
            "RR5_nested": "formal result",
            "RR5_fixed_regularization": "parity track only",
            "old_RR2": "superseded",
            "RR2D": "auxiliary only",
        },
        "row_counts": {
            "summary": int(len(summary)),
            "cells": int(len(cells)),
        },
        "input_paths_and_sha256": {
            rr2s_path.name: sha256_file(rr2s_path),
            rr3_path.name: sha256_file(rr3_path),
        },
        "output_sha256": {
            summary_path.name: sha256_file(summary_path),
            cells_path.name: sha256_file(cells_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    section = [
        "",
        "## Module-Level Robustness Summary",
        "",
        "Formal reuse-only, tie-aware module-level counts. These use the formal RR2S Llama state-spectrum module table and formal RR3 Llama centered module table; no forward/training was started. `opd_strict_deepest` means OPD is the unique maximizer; `opd_tied_deepest` means an offline arm attains the same maximum; `offline_strictly_deeper` means at least one offline arm exceeds OPD.",
        "",
        summary.to_markdown(index=False),
        "",
        "Status flags:",
        "",
        "- nested RR5 is the formal result;",
        "- old fixed-regularization RR5 is retained only as parity track;",
        "- old RR2 is superseded;",
        "- RR2D remains auxiliary only.",
        "",
        f"Manifest: `{manifest_path}`",
        "",
    ]
    handoff_path = OUT / "reviewer_robustness_theory_handoff.md"
    text = handoff_path.read_text(encoding="utf-8") if handoff_path.is_file() else "# Reviewer Robustness Theory Handoff\n"
    marker = "## Module-Level Robustness Summary"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n" + "\n".join(section)
    else:
        text = text.rstrip() + "\n" + "\n".join(section)
    handoff_path.write_text(text.rstrip() + "\n", encoding="utf-8")

    code_handoff = REPO / "mypaper/code/cycle09_reviewer_robustness_handoff.md"
    code_text = code_handoff.read_text(encoding="utf-8") if code_handoff.is_file() else "# Cycle09 Reviewer Robustness Handoff\n"
    code_marker = "## 16. Module-Level Robustness Return: 2026-07-27"
    code_block = f"""

{code_marker}

低成本收尾已完成并改为 tie-aware 口径：从正式 RR2S/RR3 module 表分别统计 OPD 严格最深、并列最深及 offline 严格更深；无 forward、无训练、未修改论文或 `human_read-ch.md`。

输出：

```text
{summary_path}
{cells_path}
{manifest_path}
```

正式读数：

{summary.to_markdown(index=False)}
"""
    if code_marker in code_text:
        code_text = code_text[: code_text.index(code_marker)].rstrip() + code_block
    else:
        code_text = code_text.rstrip() + code_block
    code_handoff.write_text(code_text.rstrip() + "\n", encoding="utf-8")

    evo = REPO / "mypaper/code/code_evolution.md"
    evo_text = evo.read_text(encoding="utf-8") if evo.is_file() else "# Code Evolution\n"
    evo_marker = "### 2026-07-28 Reviewer Robustness Module-Level Tie-Aware Correction"
    if evo_marker not in evo_text:
        evo_text = evo_text.rstrip() + f"""

{evo_marker}

- Updated `experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness_module_summary.py` to separate unique OPD wins, tied maxima, and offline wins.
- Regenerated `RR_module_level_robustness_summary.csv`, `RR_module_level_robustness_cells.csv`, and manifest from formal RR2S/RR3 module tables.
- Explicitly marked nested RR5 as formal, fixed-regularization RR5 as parity only, old RR2 as superseded, and RR2D as auxiliary.
"""
        evo.write_text(evo_text.rstrip() + "\n", encoding="utf-8")

    print(json.dumps({
        "status": manifest["status"],
        "summary": str(summary_path),
        "cells": str(cells_path),
        "manifest": str(manifest_path),
        "handoff": str(handoff_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
