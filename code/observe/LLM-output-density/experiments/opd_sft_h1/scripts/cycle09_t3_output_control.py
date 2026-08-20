#!/usr/bin/env python3
"""Cycle 09 mini T3: C09 output-control decomposition (OPD side), parallel to cycle07 nontermination table.
Per (step, task): as_run_acc, finisher_acc, nonterm_pct, finish_n, trunc_but_correct, boxed_before_trunc_pct,
mean_resp_len. MATH500 at BOTH caps (early 0-20 @4096 as-run; 40-624 @16384 retest) [QA4=a]. numina/aime24 as-run.
Zero GPU. Outputs T3_c09_output_control.csv to mini/.
"""
import json
from pathlib import Path

C8 = Path("/root/autodl-tmp/cycle08_opd_trajectory/eval")
RT = Path("/root/autodl-tmp/cap_unified_retest/opd")
OUT = Path("/root/LLM-output-density/mypaper/local_experiment_results/"
           "cycle_09_aaai_competitiveness_completion/run_01/mini")
GRID = [0, 5, 10, 20, 40, 80, 160, 320, 480, 624]


def is_ok(x):
    return str(x).lower() in ("true", "1")


def load(p):
    if not p.exists():
        return None
    return [json.loads(l) for l in open(p) if l.strip()]


def stats(rows):
    n = len(rows)
    if n == 0:
        return None
    ok = [is_ok(r["ok"]) for r in rows]
    fin = [str(r.get("finish", "")) == "stop" for r in rows]  # stop=terminated
    length = [str(r.get("finish", "")) == "length" for r in rows]
    rl = [int(r.get("resp_len", 0)) for r in rows]
    acc = sum(ok) / n
    finish_n = sum(fin)
    finisher_acc = (sum(o for o, f in zip(ok, fin) if f) / finish_n) if finish_n else float("nan")
    nonterm = sum(length) / n
    trunc_correct = sum(1 for o, l in zip(ok, length) if o and l)
    boxed_before = sum(1 for r, l in zip(rows, length) if l and ("\\boxed" in str(r.get("gen", ""))))
    n_len = sum(length)
    boxed_pct = (boxed_before / n_len) if n_len else float("nan")
    return dict(n=n, acc=acc, finisher_acc=finisher_acc, nonterm_pct=nonterm, finish_n=finish_n,
                trunc_but_correct=trunc_correct, boxed_before_trunc_pct=boxed_pct,
                mean_resp_len=sum(rl) / n)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for s in GRID:
        lbl = f"step_{s:03d}"
        # math500 as-run 4096
        m = stats(load(C8 / lbl / "math500" / f"{lbl}_samples.jsonl") or [])
        if m:
            rows_out.append(("math500", s, 4096, m))
        # math500 retest 16384 (40-624)
        mr = load(RT / lbl / "math500" / f"{lbl}_samples.jsonl")
        if mr:
            rows_out.append(("math500", s, 16384, stats(mr)))
        # numina / aime24 as-run
        for task, cap in (("numina", 4096), ("aime24", 16384)):
            t = stats(load(C8 / lbl / task / f"{lbl}_samples.jsonl") or [])
            if t:
                rows_out.append((task, s, cap, t))

    import csv
    with open(OUT / "T3_c09_output_control.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "step", "cap", "n", "as_run_acc", "finisher_acc", "nonterm_pct",
                    "finish_n", "trunc_but_correct", "boxed_before_trunc_pct", "mean_resp_len"])
        for task, s, cap, d in rows_out:
            w.writerow([task, s, cap, d["n"], f"{d['acc']:.4f}", f"{d['finisher_acc']:.4f}",
                        f"{d['nonterm_pct']:.4f}", d["finish_n"], d["trunc_but_correct"],
                        f"{d['boxed_before_trunc_pct']:.4f}", f"{d['mean_resp_len']:.1f}"])
    print(f"[T3] wrote T3_c09_output_control.csv ({len(rows_out)} rows)")
    # console preview: math500 @16384
    print("\nMATH500 @16384 (OPD): step | acc | finisher_acc | nonterm% | trunc_ok | boxed_before% | meanlen")
    for task, s, cap, d in rows_out:
        if task == "math500" and cap == 16384:
            print(f"  {s:>3} | {d['acc']:.3f} | {d['finisher_acc']:.3f} | {d['nonterm_pct']:.2f} | "
                  f"{d['trunc_but_correct']:>3} | {d['boxed_before_trunc_pct']:.2f} | {d['mean_resp_len']:.0f}")


if __name__ == "__main__":
    main()
