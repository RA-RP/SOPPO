#!/usr/bin/env python3
"""N-2 item 3 (T5): off-KD raw ER, ten points — R5-A6 / 2605.30524 replica.

Formula verbatim from the paper's section 3.1 (and R5-A6's implementation):
  H_bar = H - 1 mu^T ; Sigma = (n-1)^-1 H_bar^T H_bar ; p_i = lam_i / sum lam
  erank(Sigma) = d^-1 exp(-sum p_i log(p_i + eps))
Same probe corpora and window-v2 hierarchical weighting as the existing rows, so the
appended offkd rows match R5_raw_er_fixed.csv cell for cell.
"""
import gc
from pathlib import Path
import numpy as np, pandas as pd, torch
import cycle09_r4_common as c4
import cycle09_r4_campaign as camp
import cycle09_r5_common as c5

ARM = "offkd"
STEPS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
MINI = c4.MINI_ROOT
OFFKD_MERGED = Path("/root/autodl-tmp/cycle09_offkd/_merged_models")
PROBES = {  # same five static probes as the existing raw-ER rows
    "legacy_S_math": "corpora/fixed/legacy_S_math.jsonl",
    "E_ood": "corpora/fixed/E_ood.jsonl",
    "E_general": "corpora/fixed/E_general.jsonl",
    "E_math_hard": "corpora/fixed/E_math_hard.jsonl",
    "S_bos__g3": None,
}
CACHE = Path("/root/autodl-tmp/cycle09_n2/t5_cache")

def model_path(step):
    if step == 0: return c4.BASE_MODEL
    p = OFFKD_MERGED / c4.step_label(step)
    if not (p/"config.json").exists(): raise FileNotFoundError(p)
    return p

@torch.no_grad()
def main():
    from transformers import AutoTokenizer
    CACHE.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(c4.BASE_MODEL))
    samples = {}
    for name, rel in PROBES.items():
        corpus = (c4.RUN_ROOT/rel) if rel else c4.generated_corpus_path("S","bos",3,run_root=c4.RUN_ROOT)
        samples[name] = c4.prepare_samples(corpus, tok, corpus_id=name,
                                           window_seed=c4.WINDOW_SEED,
                                           max_context_tokens=c4.MAX_CONTEXT_TOKENS)
    rows = []
    for step in STEPS:
        f = CACHE/f"step_{step:03d}.csv"
        if f.exists():
            rows.extend(pd.read_csv(f).to_dict("records")); print(f"[T5 cached] step {step}", flush=True); continue
        model = camp.load_model(model_path(step), "cuda")
        srows = []
        try:
            for name, s in samples.items():
                prof = camp.collect_profile(model, s, list(c4.LAYERS), "cuda",
                                            keep_factors=False, keep_residual_samples=False)
                for layer in c4.LAYERS:
                    second = prof["residual_second"][layer].double()
                    mean = prof["residual_mean"][layer].double()
                    cov = second - torch.outer(mean, mean)
                    ev = torch.linalg.eigvalsh(cov).clamp_min(0).cpu().numpy()[::-1]
                    srows.append({"arm": ARM, "step": int(step), "task_id": name, "layer": int(layer),
                                  "raw_er_unnormalized": c4.effective_rank(ev),
                                  "raw_er_normalized": c5.normalized_effective_rank(ev),
                                  "raw_top5_eigen_share": c5.top_eigen_share(ev, 5),
                                  "raw_dim": int(ev.size), "raw_trace": float(ev.sum()),
                                  "protocol": "centered covariance; eps=1e-12; normalized erank (2605.30524 §3.1)",
                                  "protocol_note": "口径自定，未核原文"})
                del prof; gc.collect(); torch.cuda.empty_cache()
            pd.DataFrame(srows).to_csv(f, index=False)
            rows.extend(srows); print(f"[T5] step {step} done ({len(srows)} rows)", flush=True)
        finally:
            camp.unload_model(model); gc.collect(); torch.cuda.empty_cache()
    new = pd.DataFrame(rows)
    p = MINI/"R5_raw_er_fixed.csv"
    if p.exists():
        old = pd.read_csv(p); old = old[old["arm"] != ARM]
        out = pd.concat([old, new], ignore_index=True)
    else: out = new
    c4.write_csv_atomic(p, out.to_dict("records"), list(out.columns))
    print(f"[T5] appended offkd rows={len(new)} (total {len(out)})", flush=True)

if __name__ == "__main__":
    main()
