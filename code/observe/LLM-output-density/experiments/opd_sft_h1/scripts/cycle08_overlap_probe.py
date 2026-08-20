#!/usr/bin/env python3
"""Cycle 08 Phase 0 — teacher/student top-k overlap probe.

OPD supervises the student on states the STUDENT visits, with the teacher's
next-token distribution. This probe measures the student/teacher top-k overlap
and the teacher's probability mass on shared tokens as a DIAGNOSTIC baseline.

NOTE (revised 2026-06-30): this is NOT a STOP-gate. Per Zhang et al., Rethinking
On-Policy Distillation (arXiv 2604.13016), there is no absolute overlap threshold
separating success from failure — the healthy signature is overlap RISING during
training (their successful runs go ~72%->91%); failure shows stagnant overlap.
A high teacher-mass-on-shared (~0.97 in our smoke) already indicates a healthy,
concentrated teacher signal. Cycle 08 therefore starts OPD from the base model
(step_000) unconditionally and judges viability by the overlap SLOPE over the
first ~20 training steps, not by this static base value. The static numbers below
are reported for the record only.

Method (on-policy aligned): each candidate student generates a short rollout on
the fixed probe prompts; on those rollout positions we compute student top-k and
teacher top-k (k=32) and report set-overlap, teacher mass on shared tokens, and
entropies. Models are loaded one at a time to fit a single 48G card.

Run with the density env python (transformers + the local models). Output:
  {EXP_ROOT}/phase0/overlap_probe.csv
  {EXP_ROOT}/phase0/selected_start.json
"""
import argparse
import json
from pathlib import Path

import torch

TEACHER = "/root/autodl-tmp/model/Qwen/Qwen3-8B"
BASE = "/root/autodl-tmp/model/Qwen/Qwen3-4B-Base"
C07_CKPT = Path("/root/autodl-tmp/cycle07_base_sft_trajectory/checkpoints")
PROBE_ROWS = Path("/root/autodl-tmp/cycle07_base_sft_trajectory/data_prep/probe_rows.jsonl")
EXP_ROOT = Path("/root/LLM-output-density/mypaper/local_experiment_results/"
                "cycle_08_h_opd_vs_sft_comparison/run_01")
INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
OVERLAP_THRESHOLD = 0.70
# candidate -> adapter step (None = base only)
CANDIDATES = [("step_000", None), ("step_005", 5), ("step_010", 10), ("step_020", 20)]


def _load_model(adapter_step):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    if adapter_step is not None:
        from peft import PeftModel
        adapter = C07_CKPT / f"step_{adapter_step:03d}"
        model = PeftModel.from_pretrained(model, str(adapter))
        model = model.merge_and_unload()
    model.eval()
    return model


@torch.no_grad()
def _topk_logprobs(model, input_ids, attn, positions, k):
    """Return (topk_ids, topk_logprobs, full_logprobs_at_positions) for given positions."""
    out = model(input_ids=input_ids, attention_mask=attn)
    logp = torch.log_softmax(out.logits.float(), dim=-1)  # (1, L, V)
    sel = logp[0, positions, :]                            # (P, V)
    tk = torch.topk(sel, k, dim=-1)                        # ids, logprobs
    return tk.indices, tk.values, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--n-rollout", type=int, default=256)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=str(EXP_ROOT / "phase0"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in PROBE_ROWS.open()][: args.n_prompts]
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["question"] + INSTR}],
                                       tokenize=False, add_generation_prompt=True) for r in rows]
    k = args.topk

    # ---- Phase A: each student candidate generates rollouts + its own top-k ----
    # store per candidate: list of (full_ids, rollout_positions, student_topk_ids, student_topk_logp)
    cand_data = {}
    for label, astep in CANDIDATES:
        print(f"[probe] student {label}: generating rollouts ...", flush=True)
        model = _load_model(astep)
        recs = []
        for p in prompts:
            enc = tok(p, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            plen = enc["input_ids"].shape[1]
            gen = model.generate(**enc, max_new_tokens=args.n_rollout, do_sample=args.temperature > 0,
                                 temperature=args.temperature, top_p=0.9,
                                 pad_token_id=tok.pad_token_id)
            full = gen[0]                                   # (L,)
            roll_pos = list(range(plen - 1, full.shape[0] - 1))  # positions predicting rollout tokens
            if not roll_pos:
                continue
            ids = full.unsqueeze(0)
            attn = torch.ones_like(ids)
            s_ids, s_logp, _ = _topk_logprobs(model, ids, attn, roll_pos, k)
            recs.append({"full": full.cpu(), "pos": roll_pos,
                         "s_ids": s_ids.cpu(), "s_logp": s_logp.cpu()})
        cand_data[label] = recs
        del model; torch.cuda.empty_cache()

    # ---- Phase B: load teacher once, score all candidates' rollout states ----
    print("[probe] loading teacher, scoring all rollout states ...", flush=True)
    teacher = _load_model_teacher()
    results = []
    for label, _ in CANDIDATES:
        recs = cand_data[label]
        overlaps, shared_mass, s_ent, t_ent = [], [], [], []
        for rec in recs:
            ids = rec["full"].unsqueeze(0).to("cuda")
            attn = torch.ones_like(ids)
            t_ids, t_logp, t_sel = _topk_logprobs(teacher, ids, attn, rec["pos"], k)
            t_ids, t_logp = t_ids.cpu(), t_logp.cpu()
            s_ids, s_logp = rec["s_ids"], rec["s_logp"]
            for j in range(len(rec["pos"])):
                sset, tset = set(s_ids[j].tolist()), set(t_ids[j].tolist())
                inter = sset & tset
                overlaps.append(len(inter) / k)
                # teacher prob mass on tokens shared with student top-k
                if inter:
                    tmap = {int(i): float(lp) for i, lp in zip(t_ids[j].tolist(), t_logp[j].tolist())}
                    shared_mass.append(float(sum(torch.exp(torch.tensor(tmap[i])) for i in inter)))
                else:
                    shared_mass.append(0.0)
                s_ent.append(float(-(s_logp[j].exp() * s_logp[j]).sum()))
                t_ent.append(float(-(t_logp[j].exp() * t_logp[j]).sum()))
        import statistics as st
        mean_ov = st.mean(overlaps) if overlaps else 0.0
        med_ov = st.median(overlaps) if overlaps else 0.0
        viable = mean_ov >= OVERLAP_THRESHOLD
        results.append({
            "checkpoint": label,
            "candidate_role": "cold_start_base" if label == "step_000" else "cold_start_sft_adapter",
            "n_prompts": len(recs), "top_k": k,
            "mean_student_teacher_topk_overlap": round(mean_ov, 4),
            "median_student_teacher_topk_overlap": round(med_ov, 4),
            "teacher_topk_mass_on_shared_tokens": round(st.mean(shared_mass) if shared_mass else 0.0, 4),
            "student_entropy_mean": round(st.mean(s_ent) if s_ent else 0.0, 4),
            "teacher_entropy_mean": round(st.mean(t_ent) if t_ent else 0.0, 4),
            "sample_generation_notes": f"rollout={args.n_rollout}tok temp={args.temperature}",
            "viability_verdict": "viable" if viable else "below_threshold",
        })
        print(f"[probe] {label}: mean_overlap={mean_ov:.3f} viable={viable}", flush=True)
    del teacher; torch.cuda.empty_cache()

    # ---- write CSV + selection ----
    import csv
    cols = list(results[0].keys())
    with open(outdir / "overlap_probe.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(results)

    # selection: step_000 if viable, else earliest viable in {005,010}; never 020+
    sel = None
    by_label = {r["checkpoint"]: r for r in results}
    for label in ["step_000", "step_005", "step_010"]:
        if by_label.get(label, {}).get("viability_verdict") == "viable":
            sel = label; break
    sel_step = {"step_000": 0, "step_005": 5, "step_010": 10}.get(sel)
    selection = {
        "selected_start_step": sel_step,
        "selected_start_label": sel,
        "selection_reason": (f"earliest checkpoint with mean top-k overlap >= {OVERLAP_THRESHOLD}"
                             if sel else f"NO candidate reached overlap {OVERLAP_THRESHOLD}; STOP, do not auto-start"),
        "overlap_threshold": OVERLAP_THRESHOLD,
        "teacher_path": TEACHER,
        "student_start_path": (BASE if sel == "step_000"
                               else f"{C07_CKPT}/{sel} (base+LoRA, to be merged)" if sel else None),
        "smoke_status": "pending",
        "all_candidates": {r["checkpoint"]: r["mean_student_teacher_topk_overlap"] for r in results},
    }
    (outdir / "selected_start.json").write_text(json.dumps(selection, indent=2))
    print(f"\n[probe] -> {outdir}/overlap_probe.csv")
    print(f"[probe] selected_start: {sel} (step {sel_step})")
    if sel is None:
        print("[probe] WARNING: no viable pre-dip start; per brief, STOP and ask the user.")


def _load_model_teacher():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        TEACHER, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    m.eval()
    return m


if __name__ == "__main__":
    main()
