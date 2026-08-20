#!/usr/bin/env python3
"""Cycle 08 CPU-only supplements: aggregate the already-computed OPD (cycle08) and
SFT (cycle07) geometry + principal-evidence CSVs into the missing summary files and
an OPD-vs-SFT comparison. No GPU, no model loading — purely reads on-disk CSVs.

Produces (under cycle08 geometry dir):
  geometry_summary.csv        arm,step,mean_effective_rank,mean_spectral_gap,mean_drift_from_base,mean_xs_log_spectrum_gap
  overlap_lift_summary.csv    arm,step,mean_overlap_lift,mean_jaccard,mean_uangle_mean_deg
  opd_vs_sft_geometry.md      side-by-side trajectory + D08 landmark table
"""
from __future__ import annotations
import csv, glob, statistics as st
from pathlib import Path

GRID = [0, 5, 10, 20, 40, 80, 160, 320, 480, 624]
ARMS = {
    "opd": Path("/root/autodl-tmp/cycle08_opd_trajectory"),
    "sft": Path("/root/autodl-tmp/cycle07_base_sft_trajectory"),
}
OUT = ARMS["opd"] / "geometry"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def geom_row(run_root: Path, step: int):
    """mean over (layer,module) of the 4 geometry metrics for one checkpoint."""
    p = run_root / "geometry" / f"geometry_metrics_step_{step:03d}.csv"
    if not p.exists():
        return None
    cols = {"effective_rank": [], "spectral_gap": [], "drift_from_base": [], "xs_log_spectrum_gap": []}
    for r in csv.DictReader(open(p)):
        for c in cols:
            v = _f(r.get(c))
            if v is not None:
                cols[c].append(v)
    if not cols["effective_rank"]:
        return None
    return {c: st.mean(v) for c, v in cols.items() if v}


def pe_row(run_root: Path, step: int):
    """mean OverlapLift/Jaccard/UAngle over layers*modules for one checkpoint (PE keyed by Source=step_NNN)."""
    label = f"step_{step:03d}"
    vals = {"OverlapLift": [], "Jaccard": [], "UAngleMeanDeg": []}
    for f in glob.glob(str(run_root / "principal_evidence" / "layer_*" / "principal_evidence.csv")):
        for r in csv.DictReader(open(f)):
            if r.get("Source") == label:
                for c in vals:
                    v = _f(r.get(c))
                    if v is not None:
                        vals[c].append(v)
    if not vals["OverlapLift"]:
        return None
    return {c: st.mean(v) for c, v in vals.items() if v}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- geometry_summary.csv ----
    geo = {}
    with open(OUT / "geometry_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "step", "mean_effective_rank", "mean_spectral_gap",
                    "mean_drift_from_base", "mean_xs_log_spectrum_gap"])
        for arm, root in ARMS.items():
            for s in GRID:
                r = geom_row(root, s)
                if r is None:
                    continue
                geo[(arm, s)] = r
                w.writerow([arm, s, r["effective_rank"], r["spectral_gap"],
                            r["drift_from_base"], r["xs_log_spectrum_gap"]])
    print(f"[supp] wrote {OUT/'geometry_summary.csv'} ({len(geo)} rows)", flush=True)

    # ---- overlap_lift_summary.csv ----
    ol = {}
    with open(OUT / "overlap_lift_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "step", "mean_overlap_lift", "mean_jaccard", "mean_uangle_mean_deg"])
        for arm, root in ARMS.items():
            for s in GRID:
                r = pe_row(root, s)
                if r is None:
                    continue
                ol[(arm, s)] = r
                w.writerow([arm, s, r["OverlapLift"], r["Jaccard"], r["UAngleMeanDeg"]])
    print(f"[supp] wrote {OUT/'overlap_lift_summary.csv'} ({len(ol)} rows)", flush=True)

    # ---- opd_vs_sft_geometry.md ----
    def landmarks(arm):
        rows = {s: geo[(arm, s)] for s in GRID if (arm, s) in geo}
        if not rows:
            return {}
        return {
            "argmax_eff_rank": max(rows, key=lambda s: rows[s]["effective_rank"]),
            "argmin_xs_log_gap": min(rows, key=lambda s: rows[s]["xs_log_spectrum_gap"]),
            "argmax_drift": max(rows, key=lambda s: rows[s]["drift_from_base"]),
        }
    with open(OUT / "opd_vs_sft_geometry.md", "w") as f:
        f.write("# OPD vs SFT — geometry & OverlapLift (from on-disk CSVs)\n\n")
        f.write("## effective_rank / xs_log_spectrum_gap / drift (mean over layers*modules)\n\n")
        f.write("| step | OPD eff_rank | SFT eff_rank | OPD xs_log_gap | SFT xs_log_gap | OPD drift | SFT drift |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for s in GRID:
            o, sf = geo.get(("opd", s)), geo.get(("sft", s))
            def g(d, k): return f"{d[k]:.4f}" if d else "-"
            f.write(f"| {s} | {g(o,'effective_rank')} | {g(sf,'effective_rank')} | "
                    f"{g(o,'xs_log_spectrum_gap')} | {g(sf,'xs_log_spectrum_gap')} | "
                    f"{g(o,'drift_from_base')} | {g(sf,'drift_from_base')} |\n")
        f.write("\n## OverlapLift (mean over layers*modules)\n\n")
        f.write("| step | OPD OverlapLift | SFT OverlapLift |\n|---|---|---|\n")
        for s in GRID:
            o, sf = ol.get(("opd", s)), ol.get(("sft", s))
            ov = f"{o['OverlapLift']:.4f}" if o else "-"
            sv = f"{sf['OverlapLift']:.4f}" if sf else "-"
            f.write(f"| {s} | {ov} | {sv} |\n")
        f.write("\n## D08 landmarks (SFT reference: triple co-location at step_20)\n\n")
        for arm in ("opd", "sft"):
            lm = landmarks(arm)
            f.write(f"- **{arm.upper()}**: argmax effective_rank = step_{lm.get('argmax_eff_rank','?')}, "
                    f"argmin xs_log_spectrum_gap = step_{lm.get('argmin_xs_log_gap','?')}, "
                    f"argmax drift_from_base = step_{lm.get('argmax_drift','?')}\n")
    print(f"[supp] wrote {OUT/'opd_vs_sft_geometry.md'}", flush=True)
    print("[supp] DONE", flush=True)


if __name__ == "__main__":
    main()
