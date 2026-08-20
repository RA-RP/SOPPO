#!/usr/bin/env python3
"""Cycle 07 cap-pilot (efficient v2) — decensor the math generation-length distribution.

Single probe checkpoint (step_080, the most-truncated point) at a large cap so the
true response-length distribution becomes visible, to choose the minimal max_token
for the re-test (smallest cap with truncation back to the clean-region ~6-8%).

v2 fixes vs v1: smaller N, per-task caps, and CHUNKED STREAMING writes so partial
data lands on disk continuously (monitorable + crash-safe), instead of one giant
non-terminating generate() call.
"""
import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path("/root/LLM-output-density")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "opd_sft_h1"))  # makes `scripts.*` importable
sys.path.insert(0, str(REPO / "Eval" / "component"))

from scorer_v2 import score          # noqa: E402
from scorer import extract_pred      # noqa: E402

BASE = Path("/root/autodl-tmp/model/Qwen/Qwen3-4B-Base")
RUN_ROOT = Path("/root/autodl-tmp/cycle07_base_sft_trajectory")
MATH500 = REPO / "Eval/tasks/data/hendrycks_math500/test.jsonl"
INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SEED = 42
OUTDIR = RUN_ROOT / "cap_pilot"
CANDIDATE_CAPS = [4096, 8192, 12288, 16384, 24576, 32768]


def merged_step080() -> str:
    from scripts.run_opd_minimal_closure import merge_lora_adapter
    adapter = RUN_ROOT / "checkpoints/step_080"
    merged = RUN_ROOT / "_merged_tmp/step_080_pilot"
    if not (merged / "config.json").exists():
        merge_lora_adapter(BASE, adapter, merged)
    return str(merged)


def load_math500(n: int) -> list:
    rows = [json.loads(l) for l in open(MATH500)]
    rows = [{"problem": r["problem"], "answer": r["answer"],
             "level": r.get("level"), "subject": r.get("subject")} for r in rows]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows[:n]


def load_aime24() -> list:
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/aime_2024", split="train")
    return [{"problem": r["Problem"], "answer": str(r["Answer"])} for r in ds]


def _summarize(samples: list, name: str, cap: int) -> dict:
    L = sorted(s["resp_len"] for s in samples)
    n = len(L)
    def pct(q): return L[min(int(q * n), n - 1)]
    hit = sum(s["finish"] == "length" for s in samples)
    sweep = {C: round(100 * sum(s["resp_len"] > C for s in samples) / n, 1) for C in CANDIDATE_CAPS}
    return {
        "task": name, "n": n, "probe_cap": cap,
        "p50": pct(.50), "p90": pct(.90), "p95": pct(.95), "p99": pct(.99),
        "max": L[-1], "mean": round(sum(L) / n, 1),
        "hit_probe_cap": f"{hit}/{n}",
        "acc_all": round(sum(s["ok"] for s in samples) / n, 4),
        "trunc_pct_at_cap": sweep,
    }


def run_task(llm, sp, tok, rows: list, name: str, cap: int, chunk: int) -> dict:
    from vllm import SamplingParams  # noqa: F401  (sp already built)
    samp_path = OUTDIR / f"{name}_samples.jsonl"
    samp_path.write_text("")  # fresh
    samples = []
    for i in range(0, len(rows), chunk):
        batch = rows[i:i + chunk]
        prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + INSTR}],
                                           tokenize=False, add_generation_prompt=True) for r in batch]
        outs = llm.generate(prompts, sp)
        with open(samp_path, "a") as f:
            for r, o in zip(batch, outs):
                s = {
                    "resp_len": len(o.outputs[0].token_ids),
                    "finish": o.outputs[0].finish_reason,
                    "ok": bool(score(o.outputs[0].text, r["answer"])),
                    "has_boxed": "\\boxed" in o.outputs[0].text,
                    "answer": r["answer"], "pred": extract_pred(o.outputs[0].text),
                }
                samples.append(s)
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        done = len(samples)
        partial = _summarize(samples, name, cap)
        print(f"[{name}] {done}/{len(rows)} | running p50={partial['p50']} p90={partial['p90']} "
              f"p95={partial['p95']} max={partial['max']} hit_cap={partial['hit_probe_cap']}", flush=True)
    return _summarize(samples, name, cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-math", type=int, default=60)
    ap.add_argument("--math-cap", type=int, default=24576)
    ap.add_argument("--aime-cap", type=int, default=31744)  # 32768 - prompt headroom
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    OUTDIR.mkdir(parents=True, exist_ok=True)
    model = merged_step080()
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=32768, trust_remote_code=True)

    def mk_sp(cap):
        return SamplingParams(temperature=0.6, top_p=0.9, max_tokens=cap, seed=SEED)

    results = []
    results.append(run_task(llm, mk_sp(args.math_cap), tok,
                            load_math500(args.n_math), "math500", args.math_cap, chunk=20))
    _flush_summary(results, args)
    results.append(run_task(llm, mk_sp(args.aime_cap), tok,
                            load_aime24(), "aime24", args.aime_cap, chunk=10))
    _flush_summary(results, args)

    print("\n" + "=" * 70)
    print(f"CAP PILOT v2 (step_080, seed={SEED})")
    for r in results:
        print(f"\n--- {r['task']} (N={r['n']}, probe_cap={r['probe_cap']}) ---")
        print(f"  resp_len: p50={r['p50']} p90={r['p90']} p95={r['p95']} p99={r['p99']} "
              f"max={r['max']} mean={r['mean']}  hit_probe_cap={r['hit_probe_cap']}")
        print(f"  acc_all(@probe_cap)={r['acc_all']}")
        print("  truncation% if cap = :")
        for C, t in r["trunc_pct_at_cap"].items():
            print(f"      {C:>6}: {t:>5}%")
    print("=" * 70)


def _flush_summary(results, args):
    (OUTDIR / "pilot_summary.json").write_text(json.dumps(
        {"checkpoint": "step_080", "seed": SEED, "version": "v2",
         "math_cap": args.math_cap, "aime_cap": args.aime_cap,
         "candidate_caps": CANDIDATE_CAPS, "results": results}, indent=2))


if __name__ == "__main__":
    main()
