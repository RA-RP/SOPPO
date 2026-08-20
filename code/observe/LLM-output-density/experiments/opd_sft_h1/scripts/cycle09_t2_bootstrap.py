#!/usr/bin/env python3
"""Cycle 09 mini T2: B09 paired bootstrap 95% CI on OPD-vs-SFT MATH500 metrics [QA3: defs=a, draws=multi].
Paired over the shared 500 questions (verified gold-aligned). Mixed cap: early 0-20 @4096, late 40-624 @16384
(matches corrected trajectory). Metrics: final, peak, dip depth, AUC, non-term peak. Draws {256,1024,4096} to
show stability (per QA3=b). Zero GPU. Outputs T2_bootstrap_ci.csv to mini/.
"""
import json
import numpy as np
from pathlib import Path

C8 = Path("/root/autodl-tmp/cycle08_opd_trajectory/eval")
C7 = Path("/root/autodl-tmp/cycle07_base_sft_trajectory/eval")
RT = Path("/root/autodl-tmp/cap_unified_retest")
OUT = Path("/root/LLM-output-density/mypaper/local_experiment_results/"
           "cycle_09_aaai_competitiveness_completion/run_01/mini")
GRID = [0, 5, 10, 20, 40, 80, 160, 320, 480, 624]
EARLY = {0, 5, 10, 20}
DRAWS = [256, 1024, 4096]
SEED = 42


def is_ok(x): return str(x).lower() in ("true", "1")


def path_for(arm, step):
    lbl = f"step_{step:03d}"
    if step in EARLY:  # as-run 4096
        root = C8 if arm == "opd" else C7
        return root / lbl / "math500" / f"{lbl}_samples.jsonl"
    return RT / arm / lbl / "math500" / f"{lbl}_samples.jsonl"  # retest 16384


def load_arrays():
    ok, length, gold = {}, {}, {}
    for arm in ("opd", "sft"):
        for s in GRID:
            rows = [json.loads(l) for l in open(path_for(arm, s)) if l.strip()]
            ok[(arm, s)] = np.array([is_ok(r["ok"]) for r in rows], dtype=float)
            length[(arm, s)] = np.array([str(r.get("finish", "")) == "length" for r in rows], dtype=float)
            gold[(arm, s)] = [r["gold"] for r in rows]
    # verify pairing: same gold order across arms at each step
    for s in GRID:
        if gold[("opd", s)] != gold[("sft", s)]:
            raise SystemExit(f"[T2] ABORT: gold mismatch OPD vs SFT at step {s}")
    return ok, length


def metrics(ok, length, idx):
    """compute the 5 OPD-SFT diffs on resampled question indices idx."""
    def acc(arm, s): return ok[(arm, s)][idx].mean()
    def nt(arm, s): return length[(arm, s)][idx].mean()
    out = {}
    out["final"] = acc("opd", 624) - acc("sft", 624)
    # peak steps fixed from full data (computed once outside)
    out["peak"] = acc("opd", PEAK["opd"]) - acc("sft", PEAK["sft"])
    out["dip_depth"] = (acc("opd", 0) - acc("opd", 5)) - (acc("sft", 0) - acc("sft", 20))
    xs = np.array(GRID, dtype=float)
    auc_o = np.trapz([acc("opd", s) for s in GRID], xs)
    auc_s = np.trapz([acc("sft", s) for s in GRID], xs)
    out["auc"] = auc_o - auc_s
    out["nonterm_peak"] = max(nt("opd", s) for s in GRID) - max(nt("sft", s) for s in GRID)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ok, length = load_arrays()
    n = len(ok[("opd", 624)])
    global PEAK
    PEAK = {arm: max(GRID, key=lambda s: ok[(arm, s)].mean()) for arm in ("opd", "sft")}
    print(f"[T2] n={n}; peak steps OPD={PEAK['opd']} SFT={PEAK['sft']}")

    full = metrics(ok, length, np.arange(n))
    import csv
    rows = []
    for B in DRAWS:
        rng = np.random.default_rng(SEED)
        draws = {k: [] for k in full}
        for _ in range(B):
            idx = rng.integers(0, n, n)
            m = metrics(ok, length, idx)
            for k in m:
                draws[k].append(m[k])
        for k in full:
            arr = np.array(draws[k])
            lo, hi = np.percentile(arr, [2.5, 97.5])
            rows.append([k, B, f"{full[k]:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}",
                         "yes" if (lo > 0 or hi < 0) else "no"])
    with open(OUT / "T2_bootstrap_ci.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "n_draws", "opd_minus_sft", "ci95_lo", "ci95_hi", "excludes_0"])
        w.writerows(rows)
    print(f"[T2] wrote T2_bootstrap_ci.csv ({len(rows)} rows)")
    print(f"\n{'metric':<14}{'diff':>9}  95% CI (B=256 / 1024 / 4096)   excl0")
    for k in full:
        cis = {r[1]: (r[3], r[4], r[5]) for r in rows if r[0] == k}
        s = "  ".join(f"[{cis[B][0]},{cis[B][1]}]" for B in DRAWS)
        print(f"{k:<14}{full[k]:>+9.4f}  {s}   {cis[DRAWS[-1]][2]}")


if __name__ == "__main__":
    main()
