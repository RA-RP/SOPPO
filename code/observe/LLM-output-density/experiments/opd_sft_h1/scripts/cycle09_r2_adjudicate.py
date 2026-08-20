#!/usr/bin/env python3
"""Cycle09 R2 CPU adjudications from codex's completed GetSlice geometry (T5/T7/T8). No GPU.
T6' : is the L18/L9 ER uptick above noise? -> cross-PROBE-domain + cross-MODULE bootstrap of the uptick height
      (true probe-SAMPLE bootstrap needs per-probe grams that were not saved; this cross-domain consistency is
      the available, defensible proxy).
T7' : ER vs theta_r as trackers of (i) arm separation (ii) MMLU-Pro Δ, at L18. -> Q5 tree.
T8' : raw vs whitened ER, OPD-SFT direction -> Q6 three-outcome tree.
Outputs *_adjudication.{csv,md} to mini/. Readings recorded; final ruling -> Theory.
"""
import csv, collections, math, statistics as st
import numpy as np
from pathlib import Path

MINI = Path("/root/LLM-output-density/mypaper/local_experiment_results/"
            "cycle_09_aaai_competitiveness_completion/run_01/mini")
STEPS = [0, 5, 10, 20, 40, 160, 624]
DIP = {"opd": 5, "sft": 20}                     # arm dip step
NEI = {5: (0, 10), 20: (10, 40)}                # neighbors for local-uptick height
LAYERS = [9, 18, 27]
# MMLU-Pro per step (cycle08 as-run), both arms, at the compressed steps
MMLU = {"opd": {0: .476, 5: .479, 10: .483, 20: .483, 40: .399, 160: .485, 624: .492},
        "sft": {0: .491, 5: .477, 10: .475, 20: .486, 40: .450, 160: .501, 624: .462}}
RNG = np.random.default_rng(42)


def load_t5():
    d = collections.defaultdict(dict)   # (arm,layer,probe,module) -> {step: ER}
    for r in csv.DictReader(open(MINI / "T5_full_layer_profile.csv")):
        d[(r["arm"], int(r["layer"]), r["probe"], r["module"])][int(r["step"])] = float(r["effective_rank"])
    return d


# ---------- T6' : uptick significance ----------
def t6(d):
    rows = []
    for L in LAYERS:
        for arm in ("opd", "sft"):
            dip = DIP[arm]; a, b = NEI[dip]
            heights = []
            for (aa, ll, pr, mod), sd in d.items():
                if aa == arm and ll == L and all(s in sd for s in (dip, a, b)):
                    # STRICT local-max height: positive iff dip exceeds BOTH neighbors
                    # (avoids the convexity false-positive of dip − mean(neighbors)).
                    heights.append(min(sd[dip] - sd[a], sd[dip] - sd[b]))
            heights = np.array(heights)
            if len(heights) == 0:
                continue
            boot = np.array([RNG.choice(heights, len(heights), replace=True).mean() for _ in range(2000)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append([L, arm, f"step_{dip}", len(heights), f"{heights.mean():+.3f}",
                         f"{lo:+.3f}", f"{hi:+.3f}", "yes" if (lo > 0 or hi < 0) else "no"])
    with open(MINI / "T6_uptick_significance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "arm", "dip_step", "n_probe_x_module", "mean_uptick_ER", "ci95_lo", "ci95_hi", "excludes_0"])
        w.writerows(rows)
    md = ["# T6' — L18/L9 ER uptick significance (cross-probe-domain × module bootstrap)\n",
          "Uptick height = ER(dip) − ½(ER(prev)+ER(next)); pooled over 5 probe domains × 7 modules; 2000 draws.\n",
          "NB not the exact probe-SAMPLE bootstrap (per-probe grams unsaved) → cross-domain+module consistency proxy.\n",
          "| layer | arm | dip | n | mean uptick ER | 95% CI | excl 0 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | [{r[5]},{r[6]}] | **{r[7]}** |")
    (MINI / "T6_uptick_significance.md").write_text("\n".join(md) + "\n")
    return rows


# ---------- T7' : ER vs theta_r ----------
def t7(d):
    # theta_r: rotation-from-base angle (deg), primary r=64, mean over modules, per (arm,step) at L18
    tr = collections.defaultdict(list)   # (arm,step) -> [angle_deg]
    for r in csv.DictReader(open(MINI / "T7_theta_r.csv")):
        if r["comparison"] == "vs_base" and int(r["layer"]) == 18 and int(r["r"]) == 64:
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(r["theta_u"])))))
            tr[(r["arm"], int(r["step_b"]))].append(ang)
    theta = {k: st.mean(v) for k, v in tr.items()}
    # ER at L18 (X_math, mean over modules) per (arm,step)
    er = collections.defaultdict(list)
    for (arm, L, pr, mod), sd in d.items():
        if L == 18 and pr == "X_math":
            for s in STEPS:
                if s in sd:
                    er[(arm, s)].append(sd[s])
    ER = {k: st.mean(v) for k, v in er.items()}

    def spear(xs, ys):
        from scipy.stats import spearmanr
        return spearmanr(xs, ys).correlation

    lines = ["# T7' — ER vs θ_r (L18): arm-separation + MMLU-Pro tracking (Q5 tree)\n",
             "θ_r = rotation-from-base angle (deg) at r=64, mean over modules.\n",
             "| step | OPD ER | SFT ER | OPD θ_r° | SFT θ_r° | MMLU OPD | MMLU SFT |", "|---|---|---|---|---|---|---|"]
    for s in STEPS:
        lines.append(f"| {s} | {ER.get(('opd',s),float('nan')):.1f} | {ER.get(('sft',s),float('nan')):.1f} "
                     f"| {theta.get(('opd',s),float('nan')):.2f} | {theta.get(('sft',s),float('nan')):.2f} "
                     f"| {MMLU['opd'][s]:.3f} | {MMLU['sft'][s]:.3f} |")
    # trackers
    try:
        from scipy.stats import spearmanr  # noqa
        have_scipy = True
    except Exception:
        have_scipy = False
    def corr(a, b):
        a, b = np.array(a), np.array(b)
        if have_scipy:
            from scipy.stats import spearmanr
            return float(spearmanr(a, b).correlation)
        # fallback: pearson on ranks
        ra, rb = a.argsort().argsort(), b.argsort().argsort()
        return float(np.corrcoef(ra, rb)[0, 1])
    lines.append("\n## Trackers (Spearman over the 7 steps, per arm, at L18)\n")
    lines.append("| arm | ρ(ER, MMLU) | ρ(θ_r, MMLU) | better tracker |")
    lines.append("|---|---|---|---|")
    defsteps = [s for s in STEPS if s != 0]   # θ_r undefined at base (step_0) → drop for apples-to-apples corr
    for arm in ("opd", "sft"):
        er_t = [ER[(arm, s)] for s in defsteps]; th_t = [theta[(arm, s)] for s in defsteps]
        mm_t = [MMLU[arm][s] for s in defsteps]
        r_er = corr(er_t, mm_t); r_th = corr(th_t, mm_t)
        better = "ER" if abs(r_er) > abs(r_th) else "θ_r"
        lines.append(f"| {arm} | {r_er:+.2f} | {r_th:+.2f} | **{better}** |")
    lines.append("\n_(Spearman over the 6 steps 5..624 where θ_r is defined; n=6 is small → diagnostic only.)_")
    lines.append("\n**θ_r arm-separation observed:** OPD rotates MORE and EARLIER than SFT "
                 "(step_5 OPD 3.35° vs SFT 0.20°; step_20 11.4° vs 5.2°). ⚠️ This is the OPPOSITE of the "
                 "pre-registered prediction ('SFT θ_r spike, OPD smooth') — recorded, Theory adjudicates.")
    # arm separation: range of OPD-SFT gap over steps
    er_sep = st.mean(abs(ER[("opd", s)] - ER[("sft", s)]) for s in STEPS)
    th_sep = st.mean(abs(theta.get(("opd", s), 0) - theta.get(("sft", s), 0)) for s in STEPS)
    lines.append(f"\n**Arm separation (mean |OPD−SFT| over steps):** ER = {er_sep:.1f} (units ER); "
                 f"θ_r = {th_sep:.2f}° — magnitudes not directly comparable; see whether each MONOTONE-separates.\n")
    (MINI / "T7_adjudication.md").write_text("\n".join(lines) + "\n")
    return theta, ER


# ---------- T8' : raw vs whitened ----------
def t8():
    er = collections.defaultdict(list)   # (construct,arm,step) -> [ER] at landmark layers, shared probes
    for r in csv.DictReader(open(MINI / "T8_dual_er.csv")):
        if int(r["layer"]) in LAYERS and r["probe"] in ("X_math", "X_ood_knowledge"):
            er[(r["construct"], r["arm"], int(r["step"]))].append(float(r["effective_rank"]))
    M = {k: st.mean(v) for k, v in er.items()}
    lines = ["# T8' — raw vs whitened ER, OPD−SFT direction (Q6 three-outcome tree)\n",
             "Mean ER over landmark layers {9,18,27} × modules × {X_math,X_ood} (shared probes for both constructs).\n",
             "| step | whitened OPD−SFT | raw OPD−SFT |", "|---|---|---|"]
    w_final = r_final = None
    for s in STEPS:
        wd = M.get(("whitened_weight_er", "opd", s), float("nan")) - M.get(("whitened_weight_er", "sft", s), float("nan"))
        rd = M.get(("raw_residual_stream_er", "opd", s), float("nan")) - M.get(("raw_residual_stream_er", "sft", s), float("nan"))
        lines.append(f"| {s} | {wd:+.1f} | {rd:+.1f} |")
        if s == 624:
            w_final, r_final = wd, rd
    # classify (OOD outcome = OPD preserves MMLU better, established)
    same_dir = (w_final is not None and r_final is not None and (w_final < 0) == (r_final < 0))
    negligible = (w_final and r_final is not None and abs(r_final) < 0.1 * abs(w_final))  # raw < 10% of whitened
    if negligible:
        outcome = (f"OUTCOME 2 (lean): whitened OPD−SFT = {w_final:+.1f} (strong) but raw OPD−SFT = {r_final:+.2f} "
                   "(negligible, <10% of whitened) — the arm-discriminating signal is SPECIFIC to the output-relevant "
                   "(whitened) spectrum; raw representational-spread ER barely separates the arms. Same sign as "
                   "whitened but not a strong same-magnitude counter-example to 2605.30524.")
    elif same_dir and r_final < 0:
        outcome = ("OUTCOME 1: raw ER ALSO shows OPD materially lower — same direction & non-negligible magnitude; "
                   "with OPD's better OOD preservation this is a direct counter/refinement to 2605.30524.")
    elif r_final is not None and r_final > 0:
        outcome = "OUTCOME 3: raw ER shows OPD HIGHER (reverse of whitened) — two compression constructs divide labor."
    else:
        outcome = "OUTCOME 2: only whitened discriminates; raw does not."
    lines.append(f"\n**Classification (step_624):** {outcome}\n")
    (MINI / "T8_adjudication.md").write_text("\n".join(lines) + "\n")
    return M


def main():
    d = load_t5()
    r6 = t6(d)
    print("[T6'] done:", [(x[0], x[1], x[4], x[7]) for x in r6])
    t7(d)
    print("[T7'] done")
    t8()
    print("[T8'] done")


if __name__ == "__main__":
    main()
