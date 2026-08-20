#!/usr/bin/env python3
"""S1-3: sample-level CI for the transient window (E_ood x {5,10,20,40,80} x three arms).

Completes T1's transient window. keep_factors=True collection where the R4 factor bundles
are missing (opd/sft@80, offkd@{5,10,20,40,80}), then a paired bootstrap (draws=256,
indices shared across arms/steps since E_ood is one fixed 128-sample corpus) over
    offset(arm, step, d) = ER(W_arm_step, gram_d) - ER(W_base, gram_d)
reporting the pairwise arm differences (offkd-opd / offkd-sft / opd-sft), L18, 7-module mean.
Readings only.
"""
import argparse, gc, math, json
from pathlib import Path
import numpy as np, torch

import cycle09_r4_common as c4
import cycle09_r4_campaign as camp
from utils.profiling_utils import _gram_to_svdllm_scaling_diag_matrix

LAYER, TASK = 18, "E_ood"
STEPS = (5, 10, 20, 40, 80)
ARMS = ("opd", "sft", "offkd")
FACTOR_ROOT = c4.RUN_ROOT / "scratch/bootstrap_factors"
OFFKD_MERGED = Path("/root/autodl-tmp/cycle09_offkd/_merged_models")
MINI = c4.MINI_ROOT
CACHE = Path("/root/autodl-tmp/cycle09_s1_3/cache")

def model_path(arm, step):
    if step == 0: return c4.BASE_MODEL
    if arm == "offkd":
        p = OFFKD_MERGED / c4.step_label(step)
        if not (p/"config.json").exists(): raise FileNotFoundError(p)
        return p
    return c4.model_path(arm, step)

def bundle_path(arm, step): return FACTOR_ROOT / arm / c4.step_label(step) / f"{TASK}.pt"

def collect(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(c4.BASE_MODEL))
    corpus = c4.RUN_ROOT / "corpora/fixed/E_ood.jsonl"
    samples = c4.prepare_samples(corpus, tok, corpus_id=TASK, window_seed=c4.WINDOW_SEED,
                                 max_context_tokens=c4.MAX_CONTEXT_TOKENS)
    pending = [(a, s) for a in ARMS for s in STEPS if not bundle_path(a, s).exists()]
    if not pending: print("[S1-3] all factor bundles present"); return
    print(f"[S1-3] collecting {len(pending)} bundles: {pending}", flush=True)
    for arm, step in pending:
        model = camp.load_model(model_path(arm, step), args.device)
        try:
            prof = camp.collect_profile(model, samples, [LAYER], args.device,
                                        keep_factors=True, keep_residual_samples=True)
            out = bundle_path(arm, step); out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"schema_version": 2, "arm": arm, "step": int(step), "task": TASK,
                        "layers": [LAYER], "sample_ids": [s.sample_id for s in samples],
                        "sample_factors": prof["sample_factors"],
                        "residual_samples": prof["residual_samples"],
                        "residual_sample_means": prof["residual_sample_means"]}, out)
            print(f"[S1-3] collected {arm}/{c4.step_label(step)} -> {out}", flush=True)
            del prof
        finally:
            camp.unload_model(model); gc.collect(); torch.cuda.empty_cache()

def draw_scalings(bundle, idx, device):
    out = {}
    sf = bundle["sample_factors"]
    for group in c4.GROUP_TO_MODULES:
        m = torch.cat([sf[int(i)][LAYER][group] for i in idx], 0).to(device, torch.float32)
        m.mul_(1.0/math.sqrt(len(idx)))
        out[group] = _gram_to_svdllm_scaling_diag_matrix(m.T @ m, cholesky_jitter=1e-5,
                                                         singular_floor=0.0).to(device, torch.float32)
        del m
    return out

@torch.no_grad()
def er_draws(arm, step, idx_all, device):
    """ER per draw per module for one (arm, step)."""
    b = torch.load(bundle_path(arm, step), map_location="cpu", weights_only=False)
    model = camp.load_model(model_path(arm, step), device)
    n, nm = len(idx_all), len(c4.MODULES)
    out = np.full((n, nm), np.nan)
    try:
        W = {m: camp.module_at(model, LAYER, m).weight.detach().float() for m in c4.MODULES}
        for d, idx in enumerate(idx_all):
            sc = draw_scalings(b, idx, device)
            for j, m in enumerate(c4.MODULES):
                sig = torch.linalg.svdvals(W[m] @ sc[c4.MODULE_TO_GROUP[m]])
                out[d, j] = c4.effective_rank(sig.cpu().numpy())
            sc.clear(); torch.cuda.empty_cache()
            if (d+1) % 64 == 0: print(f"[S1-3] {arm}/{c4.step_label(step)} {d+1}/{n}", flush=True)
    finally:
        camp.unload_model(model); del b; gc.collect(); torch.cuda.empty_cache()
    return out

def interval(v):
    f = v[np.isfinite(v)]
    if not f.size: return (np.nan,)*3
    return float(f.mean()), float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))

def bootstrap(args):
    CACHE.mkdir(parents=True, exist_ok=True)
    b0 = torch.load(bundle_path("opd", 0), map_location="cpu", weights_only=False)
    n = len(b0["sample_ids"]); del b0
    rng = np.random.default_rng(c4.stable_seed(args.seed, TASK))
    idx_all = rng.integers(0, n, size=(args.draws, n))
    print(f"[S1-3] bootstrap draws={args.draws} n_samples={n}", flush=True)

    def cached(arm, step):
        f = CACHE / f"{arm}__{c4.step_label(step)}.npy"
        if f.exists():
            a = np.load(f)
            if a.shape[0] == args.draws: print(f"[S1-3 cached] {arm}/{step}", flush=True); return a
        a = er_draws(arm, step, idx_all, args.device)
        np.save(f, a); return a

    base = cached("opd", 0)          # step 0 == base weights, shared by all arms
    off = {}
    for arm in ARMS:
        for step in STEPS:
            off[(arm, step)] = cached(arm, step) - base    # ER offset vs base, per draw/module

    rows = []
    mods = list(c4.MODULES) + ["mean_fixed_7_modules"]
    for step in STEPS:
        for j, mod in enumerate(mods):
            def series(arm):
                v = off[(arm, step)]
                return np.nanmean(v, 1) if mod == "mean_fixed_7_modules" else v[:, j]
            for arm in ARMS:
                m, lo, hi = interval(series(arm))
                rows.append({"task_id": TASK, "step": int(step), "layer": LAYER, "module": mod,
                             "metric": "er_offset_vs_base", "comparison": arm,
                             "bootstrap_unit": "sample; windows nested", "bootstrap_draws": args.draws,
                             "mean": m, "ci95_lo": lo, "ci95_hi": hi,
                             "ci_excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))})
            for a1, a2 in (("offkd","opd"), ("offkd","sft"), ("opd","sft")):
                m, lo, hi = interval(series(a1) - series(a2))
                rows.append({"task_id": TASK, "step": int(step), "layer": LAYER, "module": mod,
                             "metric": "er_offset_vs_base", "comparison": f"{a1}_minus_{a2}",
                             "bootstrap_unit": "sample; windows nested", "bootstrap_draws": args.draws,
                             "mean": m, "ci95_lo": lo, "ci95_hi": hi,
                             "ci_excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))})
    import pandas as pd
    d = pd.DataFrame(rows)
    d.to_csv(MINI/'S1_transient_ci.csv', index=False)
    c4.write_json_atomic(MINI/'S1_transient_ci_manifest.json', {
        "task": TASK, "layer": LAYER, "steps": list(STEPS), "arms": list(ARMS),
        "draws": args.draws, "seed": args.seed, "n_samples": int(n),
        "metric": "ER(W_arm_step, gram_draw) - ER(W_base, gram_draw)",
        "bootstrap_unit": "sample; windows nested",
        "indices_shared_across_arms_and_steps": True, "paired": True,
        "collected_bundles": "opd/sft@80 + offkd@{5,10,20,40,80}; others reused from R4"})
    print(f"[S1-3] rows={len(d)}", flush=True)
    g = d[(d.module=='mean_fixed_7_modules')]
    print(g[['step','comparison','mean','ci95_lo','ci95_hi','ci_excludes_zero']].round(3).to_string(index=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true"); ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--all", action="store_true"); ap.add_argument("--draws", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.all: a.collect = a.bootstrap = True
    if a.collect: collect(a)
    if a.bootstrap: bootstrap(a)

if __name__ == "__main__":
    main()
