#!/usr/bin/env python3
"""Cycle 09 mini T1: L18 layered geometry re-derivation + OPD-step_5 dip co-location adjudication.
Pure CSV analysis (zero GPU). Layers {9,18,27} only (that is all that was probed; full-36 = Tier B).
Headline layer = argmax over layers of max_step |ER_OPD - ER_SFT| (mean over modules) [QA2=a].
Outputs T1_layer_trajectories.csv + T1_landmarks_dip.md to mini/.
"""
import csv, statistics as st
from pathlib import Path

OPD = Path("/root/autodl-tmp/cycle08_opd_trajectory/geometry")
SFT = Path("/root/autodl-tmp/cycle07_base_sft_trajectory/geometry")
OUT = Path("/root/LLM-output-density/mypaper/local_experiment_results/"
           "cycle_09_aaai_competitiveness_completion/run_01/mini")
GRID = [0, 5, 10, 20, 40, 80, 160, 320, 480, 624]
LAYERS = [9, 18, 27]
MET = {"effective_rank": "ER", "xs_log_spectrum_gap": "xs_gap", "drift_from_base": "drift"}


def load(root):
    """(layer, step) -> {metric: mean over modules}"""
    out = {}
    for s in GRID:
        p = root / f"geometry_metrics_step_{s:03d}.csv"
        if not p.exists():
            continue
        acc = {L: {m: [] for m in MET} for L in LAYERS}
        for r in csv.DictReader(open(p)):
            L = int(r["layer"])
            if L in LAYERS:
                for m in MET:
                    acc[L][m].append(float(r[m]))
        for L in LAYERS:
            out[(L, s)] = {m: st.mean(v) for m, v in acc[L].items() if v}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    opd, sft = load(OPD), load(SFT)

    # trajectories CSV
    with open(OUT / "T1_layer_trajectories.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "layer", "step", "mean_ER", "mean_xs_gap", "mean_drift"])
        for arm, d in (("opd", opd), ("sft", sft)):
            for L in LAYERS:
                for s in GRID:
                    if (L, s) in d:
                        r = d[(L, s)]
                        w.writerow([arm, L, s, f"{r['effective_rank']:.4f}",
                                    f"{r['xs_log_spectrum_gap']:.6f}", f"{r['drift_from_base']:.6f}"])

    # headline layer = argmax_L max_step |ER_opd - ER_sft|
    sep = {}
    for L in LAYERS:
        diffs = [abs(opd[(L, s)]["effective_rank"] - sft[(L, s)]["effective_rank"])
                 for s in GRID if (L, s) in opd and (L, s) in sft]
        sep[L] = max(diffs) if diffs else 0.0
    headline = max(sep, key=sep.get)

    def argext(d, L, metric, fn):
        pts = [(s, d[(L, s)][metric]) for s in GRID if (L, s) in d]
        return fn(pts, key=lambda x: x[1])[0]

    def uptick(d, L, step):
        """local ER uptick at `step`: ER(step) > ER(prev) and > ER(next-with-tolerance)."""
        idx = GRID.index(step)
        prev, cur = d.get((L, GRID[idx - 1])), d.get((L, step))
        nxt = d.get((L, GRID[idx + 1])) if idx + 1 < len(GRID) else None
        if not cur or not prev:
            return None
        er = lambda x: x["effective_rank"]
        up_prev = er(cur) > er(prev)
        up_next = (nxt is None) or (er(cur) >= er(nxt))
        return bool(up_prev and up_next)

    lines = ["# Cycle 09 mini T1 — layered geometry + OPD dip adjudication (layers 9/18/27; full-36 = Tier B)\n",
             f"**Headline layer (max |ER_OPD−ER_SFT| over trajectory) = L{headline}** "
             f"(separation by layer: " + ", ".join(f"L{L}={sep[L]:.1f}" for L in LAYERS) + ").\n",
             "θ_r column deferred to round-2 (UV). Readings recorded, not interpreted (per guards).\n"]
    for L in LAYERS:
        star = "  ⟵ HEADLINE" if L == headline else ""
        lines.append(f"\n## Layer {L}{star}\n")
        lines.append("| arm | argmax ER | argmin ER | argmax xs_gap | argmin xs_gap | argmax drift | ER@0→@624 |")
        lines.append("|---|---|---|---|---|---|---|")
        for arm, d in (("OPD", opd), ("SFT", sft)):
            if (L, 0) not in d:
                continue
            e0 = d[(L, 0)]["effective_rank"]; e_end = d[(L, GRID[-1])]["effective_rank"]
            lines.append(f"| {arm} | {argext(d,L,'effective_rank',max)} | {argext(d,L,'effective_rank',min)} "
                         f"| {argext(d,L,'xs_log_spectrum_gap',max)} | {argext(d,L,'xs_log_spectrum_gap',min)} "
                         f"| {argext(d,L,'drift_from_base',max)} | {e0:.1f}→{e_end:.1f} |")
        # dip adjudication
        opd5 = uptick(opd, L, 5)
        sft20 = uptick(sft, L, 20)
        er = lambda d, s: d[(L, s)]["effective_rank"] if (L, s) in d else float("nan")
        lines.append(f"\n**Dip adjudication L{L}:** OPD step_5 local ER uptick? **{opd5}** "
                     f"(ER 0→5→10 = {er(opd,0):.1f}→{er(opd,5):.1f}→{er(opd,10):.1f}). "
                     f"SFT step_20 local ER uptick (bump)? **{sft20}** "
                     f"(ER 10→20→40 = {er(sft,10):.1f}→{er(sft,20):.1f}→{er(sft,40):.1f}).")
    (OUT / "T1_landmarks_dip.md").write_text("\n".join(lines) + "\n")
    print(f"[T1] headline layer = L{headline}; wrote T1_layer_trajectories.csv + T1_landmarks_dip.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
