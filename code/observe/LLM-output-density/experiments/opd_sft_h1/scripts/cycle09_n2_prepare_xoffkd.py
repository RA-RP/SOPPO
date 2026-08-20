#!/usr/bin/env python3
"""N-2 step 1a: register the teacher-rollout text as the fixed probe X_offkd_math,
then build its base whitening reference (the other probes' references already exist).

Sampling rule is copied from X_SFT (legacy_S_math): eligible region = response only,
question masked via eligible_start = len(prompt_token_ids). 32 samples, matching the
X/S probe scale. Provenance caveat: the teacher rollout was produced on opd_prompts_5k,
which overlaps X_SFT's gamma_s head by 26/32 — recorded, not corrected (X_offkd must
faithfully reflect off-KD's actual training input).
"""
import json, gc
from pathlib import Path
import torch
import cycle09_r4_common as c4
import cycle09_r4_campaign as camp

N = c4.N_GENERATED                    # 32, same scale as X_SFT / X_OPD
SRC = Path('/root/autodl-tmp/cycle09_offkd/rollout/teacher_rollout.jsonl')
TARGET = c4.RUN_ROOT / 'corpora/fixed/X_offkd_math.jsonl'
REF = c4.RUN_ROOT / 'scratch/references/X_offkd_math.pt'

def build_corpus():
    if TARGET.exists() and len(c4.read_jsonl(TARGET)) == N:
        print(f"[prep] corpus exists: {TARGET}"); return
    rows = []
    with open(SRC, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= N: break
            r = json.loads(line)
            pid, gid = list(r['prompt_token_ids']), list(r['generation_token_ids'])
            rows.append({
                "sample_id": f"x_offkd_{i:03d}", "probe_type": "X", "domain": "math",
                "source_kind": "teacher_rollout_question_masked",
                "prompt_text": r['prompt'], "generation_text": r['generation'],
                "prompt_token_ids": pid, "generation_token_ids": gid,
                "full_token_ids": pid + gid,
                "eligible_start": len(pid), "eligible_end": len(pid) + len(gid),
                "finish_reason": r.get('finish_reason'),
            })
    c4.write_jsonl_atomic(TARGET, rows)
    print(f"[prep] wrote {TARGET} n={len(rows)}")

def build_reference():
    if REF.exists():
        print(f"[prep] reference exists: {REF}"); return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(c4.BASE_MODEL))
    samples = c4.prepare_samples(TARGET, tok, corpus_id='X_offkd_math',
                                 window_seed=c4.WINDOW_SEED,
                                 max_context_tokens=c4.MAX_CONTEXT_TOKENS)
    model = camp.load_model(c4.BASE_MODEL, 'cuda')
    try:
        prof = camp.collect_profile(model, samples, list(c4.LAYERS), 'cuda',
                                    keep_factors=True, keep_residual_samples=True)
        REF.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"n_samples": prof["n_samples"], "grams": prof["grams"],
                    "residual_second": prof["residual_second"], "residual_mean": prof["residual_mean"],
                    "position_second": prof["position_second"], "position_mean": prof["position_mean"],
                    "position_counts": prof["position_counts"],
                    "sample_factors": prof["sample_factors"],
                    "residual_samples": prof["residual_samples"],
                    "residual_sample_means": prof["residual_sample_means"]}, REF)
        print(f"[prep] wrote reference {REF} n_samples={prof['n_samples']}")
    finally:
        camp.unload_model(model); gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    build_corpus(); build_reference()
