"""MATH500 evaluation runner.

Protocol: chat template + enable_thinking=False. Scoring via math_verify on
\\boxed{} extraction, fixing the $-span extraction bug from lm_eval's default
hendrycks_math500 scorer (FINDING_05_math500_dollar_span_extraction_bug.md).

Importable API:
    from component.math500.runner import run
    summary = run(model="/path", label="mymodel", outdir="/out")

CLI (via run_eval.py):
    python run_eval.py --model PATH --label NAME --task math500 --outdir DIR
"""
import argparse
import json
import sys
from pathlib import Path

_COMPONENT = Path(__file__).resolve().parent.parent  # Eval/component/
if str(_COMPONENT) not in sys.path:
    sys.path.insert(0, str(_COMPONENT))

from scorer_v2 import score
from scorer import extract_pred

_EVAL_DIR = Path(__file__).resolve().parents[2]  # Eval/
TEST = _EVAL_DIR / "tasks/data/hendrycks_math500/test.jsonl"
INSTR = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def load_subset(n: int):
    rows = [json.loads(l) for l in open(TEST)]
    if n and n < len(rows):
        rows = rows[:n]
    return rows


def run(
    model: str,
    label: str,
    *,
    n: int = 0,
    outdir: str,
    max_tokens: int = 3072,
    gpu_mem: float = 0.80,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    seed: int = 42,
) -> dict:
    """Run MATH500 eval for one model. Returns summary dict."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_subset(n)
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompts = []
    for r in rows:
        msgs = [{"role": "user", "content": r["problem"] + INSTR}]
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p)

    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=gpu_mem,
              max_model_len=4096, trust_remote_code=True)
    sp = SamplingParams(temperature=temperature, top_p=top_p,
                        top_k=top_k, max_tokens=max_tokens, seed=seed)
    outs = llm.generate(prompts, sp)

    correct = boxed = trunc = 0
    with open(out / f"{label}_samples.jsonl", "w") as fsamp:
        for r, o in zip(rows, outs):
            text = o.outputs[0].text
            ok = bool(score(text, r["answer"]))
            if "\\boxed" in text:
                boxed += 1
            if o.outputs[0].finish_reason == "length":
                trunc += 1
            correct += int(ok)
            fsamp.write(json.dumps({"gold": r["answer"], "pred": extract_pred(text),
                                    "ok": ok, "level": r.get("level"), "subject": r.get("subject"),
                                    "finish": o.outputs[0].finish_reason,
                                    "gen": text}, ensure_ascii=False) + "\n")

    total = len(rows)
    summary = {"label": label, "model": model, "n": total,
               "acc": correct / total,
               "boxed_rate": boxed / total, "trunc_rate": trunc / total}
    (out / f"{label}.json").write_text(json.dumps(summary, indent=2))
    print(f"\n===== {label} (N={total}) =====")
    print(f"acc {summary['acc']:.3f} | boxed {summary['boxed_rate']:.3f} "
          f"| trunc {summary['trunc_rate']:.3f}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="MATH500 eval (called via run_eval.py)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="/root/autodl-tmp/eval_results/math500")
    args = ap.parse_args()
    run(
        model=args.model,
        label=args.label,
        n=args.n,
        outdir=args.outdir,
        max_tokens=args.max_tokens,
        gpu_mem=args.gpu_mem,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
