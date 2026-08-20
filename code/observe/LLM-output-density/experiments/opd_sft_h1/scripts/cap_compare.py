#!/usr/bin/env python3
"""Compare as-run (cap 4096) vs recheck (cap 24576) on the SAME first-N problems.
math500 loads in fixed dataset order; --n takes the first N, so as-run[i] == recheck[i].
Match positionally, sanity-check with 'gold'. Usage: cap_compare.py <step>"""
import json, sys

RUN = "/root/autodl-tmp/cycle08_opd_trajectory"
step = sys.argv[1]

rc = [json.loads(l) for l in open(f"{RUN}/cap_recheck/step_{step}/step_{step}_samples.jsonl")]
ar = [json.loads(l) for l in open(f"{RUN}/eval/step_{step}/math500/step_{step}_samples.jsonl")]
rj = json.load(open(f"{RUN}/cap_recheck/step_{step}/step_{step}.json"))
aj = json.load(open(f"{RUN}/eval/step_{step}/math500/step_{step}.json"))
N = len(rc)
ar = ar[:N]                                    # same first-N problems

mism = sum(1 for a, r in zip(ar, rc) if str(a["gold"]) != str(r["gold"]))
def rate(rows, pred): return sum(1 for r in rows if pred(r)) / len(rows)
def boxed(r): return "\\boxed" in (r.get("gen") or "")

ar_acc, rc_acc = rate(ar, lambda r: r["ok"]), rate(rc, lambda r: r["ok"])
ar_tr,  rc_tr  = rate(ar, lambda r: r["finish"] == "length"), rate(rc, lambda r: r["finish"] == "length")
ar_bx,  rc_bx  = rate(ar, boxed), rate(rc, boxed)
ar_len = sum(r["resp_len"] for r in ar)/N; rc_len = sum(r["resp_len"] for r in rc)/N
rec  = sum(1 for a, r in zip(ar, rc) if not a["ok"] and r["ok"])   # as-run wrong -> recheck right
lost = sum(1 for a, r in zip(ar, rc) if a["ok"] and not r["ok"])   # as-run right -> recheck wrong

print(f"=== step_{step} MATH500  first-{N} (positional; gold-mismatch={mism})  as-run cap {aj['max_tokens']} vs recheck cap {rj['max_tokens']} ===")
print(f"{'metric':<15}{'as-run(4096)':>14}{'recheck(24576)':>16}{'Δ':>9}")
print(f"{'acc':<15}{ar_acc:>14.3f}{rc_acc:>16.3f}{rc_acc-ar_acc:>+9.3f}")
print(f"{'trunc_rate':<15}{ar_tr:>14.3f}{rc_tr:>16.3f}{rc_tr-ar_tr:>+9.3f}")
print(f"{'boxed_rate':<15}{ar_bx:>14.3f}{rc_bx:>16.3f}{rc_bx-ar_bx:>+9.3f}")
print(f"{'mean_resp_len':<15}{ar_len:>14.0f}{rc_len:>16.0f}{rc_len-ar_len:>+9.0f}")
print(f"flips: recovered(as-run X -> recheck OK)={rec}   lost(OK -> X)={lost}   net acc delta={ (rec-lost)/N:+.3f}")
print(f"(full-500 as-run acc was {aj['acc']:.3f}; recheck json acc {rj['acc']:.3f})")
