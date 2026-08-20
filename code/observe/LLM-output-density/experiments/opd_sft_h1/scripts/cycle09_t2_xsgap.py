#!/usr/bin/env python3
"""T2: xs_gap trajectory for the E_ood probe (L18), three arms.

Caliber note (reported to Theory, not decided here): the R5-A5 xs_gap pairs an X-type
probe against an S-type probe (X_opd_math<->S_math, legacy_S_math<->S_math). E_ood is an
external fixed probe and is neither, and off-KD never measured S_math, so the probe-axis
pairing is undefined/uncomputable for the third arm. The only three-arm-computable reading
is the time-axis gap against the arm's own step-0 spectrum, which is also the sibling of
the ER-vs-base offset that section 4 lists alongside it:

    xs_gap(t) = mean_i [ log sigma_i(t) - log sigma_i(0) ]        (signed, spec's literal formula)
    xs_gap_abs(t) = mean_i | log sigma_i(t) - log sigma_i(0) |    (R5-A5's abs implementation)

Both are reported. sigma clipped at 1e-12 exactly as R5-A5 does.
"""
import json
import numpy as np, pandas as pd
from pathlib import Path

M = Path('/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini')
MODULES = ("self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
           "mlp.gate_proj","mlp.up_proj","mlp.down_proj")

def gap(sig_t, sig_0, signed=True):
    x = np.clip(np.asarray(sig_t, dtype=np.float64), 1e-12, None)
    s = np.clip(np.asarray(sig_0, dtype=np.float64), 1e-12, None)
    w = min(x.size, s.size)
    d = np.log(x[:w]) - np.log(s[:w])
    return float(np.mean(d)) if signed else float(np.mean(np.abs(d)))

frames = []
for ch in pd.read_csv(M/'R4_v2_spectra_all.csv', chunksize=200000,
                      usecols=['arm','step','task_id','track','layer','module','sigma_json']):
    s = ch[(ch.layer==18)&(ch.track=='per_checkpoint')&(ch.task_id=='E_ood')]
    if len(s): frames.append(s)
off = pd.read_csv(M/'R4_v2_spectra_offkd.csv',
                  usecols=['arm','step','task_id','track','layer','module','sigma_json'])
frames.append(off[(off.layer==18)&(off.track=='per_checkpoint')&(off.task_id=='E_ood')])
d = pd.concat(frames, ignore_index=True)

rows = []
for arm in ('opd','sft','offkd'):
    a = d[d.arm==arm]
    base = {r.module: json.loads(r.sigma_json) for r in a[a.step==0].itertuples()}
    for step in sorted(a.step.unique()):
        cur = {r.module: json.loads(r.sigma_json) for r in a[a.step==step].itertuples()}
        signed = [gap(cur[m], base[m], True) for m in MODULES if m in cur and m in base]
        absv   = [gap(cur[m], base[m], False) for m in MODULES if m in cur and m in base]
        rows.append({"arm":arm, "step":int(step), "probe":"E_ood", "layer":18,
                     "track":"per_checkpoint", "n_modules":len(signed),
                     "xs_gap_signed_vs_base": float(np.mean(signed)),
                     "xs_gap_abs_vs_base": float(np.mean(absv)),
                     "caliber":"time-axis: mean_i(log sigma_i(t) - log sigma_i(0)), 7-module mean"})
out = pd.DataFrame(rows)
out.to_csv(M/'R5_xsgap_eood_three_arm.csv', index=False)
print(out.to_string(index=False))
