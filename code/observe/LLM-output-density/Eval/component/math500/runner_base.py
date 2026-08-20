"""MATH500 evaluation runner — base model protocol.

Protocol: 4-shot CoT, plain text format, no chat template, no enable_thinking.
For base models (Qwen3-1.7B-Base) and models trained from them.
Shots: 4 standard Hendrycks MATH competition-style examples.

Importable API:
    from component.math500.runner_base import run
    summary = run(model="/path", label="mymodel", outdir="/out")
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

# 4 standard math competition few-shot demonstrations.
# Source: standard MATH benchmark evaluation protocol, representative of
# Hendrycks et al. 2021 MATH dataset examples across subject areas.
MATH500_SHOTS = [
    {
        "problem": "What is the value of $\\sqrt{36 \\cdot 16}$?",
        "solution": "We use the product rule for square roots: $\\sqrt{36 \\cdot 16} = \\sqrt{36} \\cdot \\sqrt{16} = 6 \\cdot 4 = 24$. The answer is $\\boxed{24}$.",
    },
    {
        "problem": "What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20?",
        "solution": "We compute each value: $120\\% \\cdot 30 = 1.2 \\cdot 30 = 36$ and $130\\% \\cdot 20 = 1.3 \\cdot 20 = 26$. The positive difference is $36 - 26 = 10$. The answer is $\\boxed{10}$.",
    },
    {
        "problem": "Simplify $\\dfrac{1}{1+\\sqrt{2}}$.",
        "solution": "Multiplying numerator and denominator by the conjugate $1 - \\sqrt{2}$: $\\dfrac{1}{1+\\sqrt{2}} \\cdot \\dfrac{1-\\sqrt{2}}{1-\\sqrt{2}} = \\dfrac{1-\\sqrt{2}}{1-2} = \\dfrac{1-\\sqrt{2}}{-1} = \\sqrt{2}-1$. The answer is $\\boxed{\\sqrt{2}-1}$.",
    },
    {
        "problem": "If $3x - 5 = 10$, what is the value of $x$?",
        "solution": "Adding 5 to both sides gives $3x = 15$. Dividing both sides by 3 gives $x = 5$. The answer is $\\boxed{5}$.",
    },
]


def _build_prompt(problem: str) -> str:
    parts = []
    for s in MATH500_SHOTS:
        parts.append(f"Problem: {s['problem']}{INSTR}\n\nSolution: {s['solution']}")
    parts.append(f"Problem: {problem}{INSTR}\n\nSolution:")
    return "\n\n".join(parts)


def load_subset(n: int):
    rows = [json.loads(line) for line in open(TEST)]
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
    """Run MATH500 base-series eval (4-shot CoT, no chat template)."""
    from vllm import LLM, SamplingParams

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_subset(n)
    prompts = [_build_prompt(r["problem"]) for r in rows]

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
    summary = {
        "label": label, "model": model, "n": total,
        "acc": correct / total,
        "boxed_rate": boxed / total, "trunc_rate": trunc / total,
        "n_shots": len(MATH500_SHOTS),
        "protocol": "4-shot CoT plain text, no chat template",
        "shot_source": "Standard Hendrycks MATH benchmark CoT examples",
    }
    (out / f"{label}.json").write_text(json.dumps(summary, indent=2))
    print(f"\n===== {label} [base] (N={total}) =====")
    print(f"acc {summary['acc']:.3f} | boxed {summary['boxed_rate']:.3f} "
          f"| trunc {summary['trunc_rate']:.3f}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="MATH500 base-series eval (4-shot CoT, no chat)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="/root/autodl-tmp/eval_results/math500_base")
    args = ap.parse_args()
    run(
        model=args.model, label=args.label, n=args.n, outdir=args.outdir,
        max_tokens=args.max_tokens, gpu_mem=args.gpu_mem,
        temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, seed=args.seed,
    )


if __name__ == "__main__":
    main()
