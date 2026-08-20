#!/usr/bin/env python3
"""Backfill H (and B1) generation at steps {80,320,480} so S1-7 covers the full ten-point grid.

Protocol is replicated verbatim from the R4 campaign (H) and R5 B-line (B1):
same prompt banks, same per-request seed rule stable_seed(batch_seed, probe_type, domain,
sample_id), same temperature/top_p/max_new_tokens. Only the missing steps are generated;
complete corpora are skipped, so this is idempotent.
"""
import gc, json
from pathlib import Path
import torch

import cycle09_r4_common as c4
import cycle09_r4_campaign as camp

STEPS = (80, 320, 480)
H_DOMAINS = ("ood", "general", "bos")
R4_ROOT = c4.RUN_ROOT
R5_ROOT = Path('/root/autodl-tmp/cycle09_r5')

def b1_path(step, seed):
    return R5_ROOT / "corpora/generated/X/sft" / c4.step_label(step) / "math" / f"gen_seed_{seed}.jsonl"

def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(c4.BASE_MODEL))
    banks = c4.prompt_banks(c4.N_GENERATED)

    jobs = []   # (arm, step) -> list of (target, probe_type, domain, seed)
    for arm in c4.ARMS:
        for step in STEPS:
            items = []
            for seed in c4.GENERATION_SEEDS:
                for dom in H_DOMAINS:
                    t = c4.generated_corpus_path("H", dom, seed, arm, step, R4_ROOT)
                    if not camp.complete_corpus(t, c4.N_GENERATED):
                        items.append((t, "H", dom, seed))
                if arm == "sft":                      # B1 = SFT self-gen on the training domain
                    t = b1_path(step, seed)
                    if not camp.complete_corpus(t, c4.N_GENERATED):
                        items.append((t, "X", "math", seed))
            if items:
                jobs.append((arm, step, items))
    if not jobs:
        print("[s1-hgen] nothing pending"); return

    for arm, step, items in jobs:
        print(f"[s1-hgen] {arm}/{c4.step_label(step)}: {len(items)} batches", flush=True)
        llm = LLM(model=str(c4.model_path(arm, step)), dtype="bfloat16",
                  gpu_memory_utilization=0.82, max_model_len=4096, seed=c4.WINDOW_SEED)
        try:
            for target, ptype, dom, seed in items:
                bank = banks[dom]
                prompts, params = [], []
                for it in bank:
                    prompts.append(camp.formatted_prompt(tok, it["prompt"], it["instruction"], dom))
                    params.append(SamplingParams(
                        temperature=c4.TEMPERATURE, top_p=c4.TOP_P, max_tokens=c4.MAX_NEW_TOKENS,
                        seed=c4.stable_seed(seed, ptype, dom, it["sample_id"])))
                outs = llm.generate(prompts, params)
                rows = []
                for it, formatted, o in zip(bank, prompts, outs):
                    comp = o.outputs[0]
                    pid, gid = list(o.prompt_token_ids), list(comp.token_ids)
                    rows.append({
                        "sample_id": it["sample_id"], "probe_type": ptype, "domain": dom,
                        "source_kind": ("checkpoint_nontraining_generation" if ptype == "H"
                                        else "sft_selfgen_training_domain"),
                        "arm": arm, "step": int(step), "generation_seed": int(seed),
                        "per_request_seed": c4.stable_seed(seed, ptype, dom, it["sample_id"]),
                        "prompt_text": it["prompt"], "formatted_prompt": formatted,
                        "generation_text": comp.text,
                        "prompt_token_ids": pid, "generation_token_ids": gid,
                        "full_token_ids": pid + gid,
                        "eligible_start": len(pid), "eligible_end": len(pid) + len(gid),
                        "finish_reason": comp.finish_reason,
                        "generation_config": {"temperature": c4.TEMPERATURE, "top_p": c4.TOP_P,
                                              "max_new_tokens": c4.MAX_NEW_TOKENS,
                                              "backfilled_by": "cycle09_s1_backfill_hgen.py"},
                    })
                c4.write_jsonl_atomic(target, rows)
                print(f"[s1-hgen] {target} n={len(rows)}", flush=True)
        finally:
            del llm; gc.collect(); torch.cuda.empty_cache()
    print("[s1-hgen] DONE", flush=True)

if __name__ == "__main__":
    main()
