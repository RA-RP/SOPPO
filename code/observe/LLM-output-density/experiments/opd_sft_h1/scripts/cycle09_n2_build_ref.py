#!/usr/bin/env python3
"""Build base whitening references for per-checkpoint probes (H / X_offkd).

The fixed probes already have cached references in R4's scratch/references; per-checkpoint
H corpora do not, and the geometry scripts need one per task_id. Idempotent: existing
references are skipped.
"""
import argparse, gc, sys
from pathlib import Path
import torch
import cycle09_r4_common as c4
import cycle09_r4_campaign as camp

REF_ROOT = c4.RUN_ROOT / "scratch/references"

def build(pairs, device="cuda"):
    todo = [(t, c) for t, c in pairs if not (REF_ROOT / f"{t}.pt").exists()]
    if not todo:
        print("[build-ref] all references present"); return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(c4.BASE_MODEL))
    model = camp.load_model(c4.BASE_MODEL, device)
    try:
        for task_id, corpus in todo:
            corpus = Path(corpus)
            if not corpus.exists():
                print(f"[build-ref] SKIP missing corpus {corpus}"); continue
            s = c4.prepare_samples(corpus, tok, corpus_id=task_id, window_seed=c4.WINDOW_SEED,
                                   max_context_tokens=c4.MAX_CONTEXT_TOKENS)
            # M1/M2/theta consume aggregate grams plus residual sample moments.
            # Per-sample SVD factors are bootstrap-only and would add ~2 GB per
            # H reference without being read by the geometry worker.
            prof = camp.collect_profile(model, s, list(c4.LAYERS), device,
                                        keep_factors=False, keep_residual_samples=True)
            out = REF_ROOT / f"{task_id}.pt"; out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"n_samples": prof["n_samples"], "grams": prof["grams"],
                        "residual_second": prof["residual_second"], "residual_mean": prof["residual_mean"],
                        "position_second": prof["position_second"], "position_mean": prof["position_mean"],
                        "position_counts": prof["position_counts"],
                        "residual_samples": prof["residual_samples"],
                        "residual_sample_means": prof["residual_sample_means"]}, out)
            print(f"[build-ref] {task_id} n={prof['n_samples']} -> {out}", flush=True)
            del prof, s; gc.collect(); torch.cuda.empty_cache()
    finally:
        camp.unload_model(model); gc.collect(); torch.cuda.empty_cache()

def h_pairs(arm, steps, domains, seeds):
    out = []
    for st in steps:
        for d in domains:
            for sd in seeds:
                out.append((f"H_{arm}_{d}__{c4.step_label(st)}__g{sd}",
                            c4.generated_corpus_path("H", d, sd, arm, st, c4.RUN_ROOT)))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--domains", default="bos,ood")
    ap.add_argument("--seeds", default="3,17,31")
    a = ap.parse_args()
    build(h_pairs(a.arm, [int(x) for x in a.steps.split(",")],
                  a.domains.split(","), [int(x) for x in a.seeds.split(",")]))
