"""GSM8K evaluation runner — base model protocol.

Protocol: 4-shot CoT, plain text format, no chat template, no enable_thinking.
For base models (Qwen3-1.7B-Base) and models trained from them.
Shots: 4 standard chain-of-thought examples (Wei et al. 2022, CoT paper).

Importable API:
    from component.gsm8k.runner_base import run
    summary = run(model="/path", label="mymodel", outdir="/out")
"""
import argparse
import json
import re
import sys
from pathlib import Path

_COMPONENT = Path(__file__).resolve().parent.parent  # Eval/component/
if str(_COMPONENT) not in sys.path:
    sys.path.insert(0, str(_COMPONENT))

from scorer import extract_pred  # last-\boxed{} aware extraction

GOLD_RE = re.compile(r"#### (-?[0-9.,]+)")
NUM_RE = re.compile(r"-?[0-9][0-9.,]*")

# 4 standard GSM8K chain-of-thought demonstrations (Wei et al. 2022, "Chain-of-Thought
# Prompting Elicits Reasoning in Large Language Models", NeurIPS 2022, Table 1).
GSM8K_SHOTS = [
    {
        "problem": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "solution": "We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.",
    },
    {
        "problem": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "solution": "Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.",
    },
    {
        "problem": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "solution": "Jason started with 20 lollipops. Then he gave some to Denny. Then he had 12 lollipops. So he gave Denny 20 - 12 = 8 lollipops. The answer is 8.",
    },
    {
        "problem": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "solution": "He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.",
    },
]


def _build_prompt(problem: str) -> str:
    parts = []
    for s in GSM8K_SHOTS:
        parts.append(f"Problem: {s['problem']}\n\nSolution: {s['solution']}")
    parts.append(f"Problem: {problem}\n\nSolution:")
    return "\n\n".join(parts)


def clean_num(s: str) -> str:
    s = s.strip().rstrip(".").replace("$", "").replace(",", "")
    if s.endswith("."):
        s = s[:-1]
    return s


def extract_pred_number(text: str):
    span = extract_pred(text)
    if span is None:
        return None
    matches = NUM_RE.findall(span)
    if not matches:
        return None
    return clean_num(matches[-1])


def load_subset(n: int):
    import datasets
    ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
    rows = [{"question": r["question"], "answer": r["answer"]} for r in ds]
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
    """Run GSM8K base-series eval (4-shot CoT, no chat template)."""
    from vllm import LLM, SamplingParams

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_subset(n)
    prompts = [_build_prompt(r["question"]) for r in rows]

    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=gpu_mem,
              max_model_len=4096, trust_remote_code=True)
    sp = SamplingParams(temperature=temperature, top_p=top_p,
                        top_k=top_k, max_tokens=max_tokens, seed=seed)
    outs = llm.generate(prompts, sp)

    correct = trunc = 0
    with open(out / f"{label}_samples.jsonl", "w") as fsamp:
        for r, o in zip(rows, outs):
            text = o.outputs[0].text
            gold_m = GOLD_RE.search(r["answer"])
            gold = clean_num(gold_m.group(1)) if gold_m else None
            pred = extract_pred_number(text)
            ok = pred is not None and gold is not None and pred == gold
            if o.outputs[0].finish_reason == "length":
                trunc += 1
            correct += int(ok)
            fsamp.write(json.dumps({"gold": gold, "pred": pred, "ok": ok,
                                    "finish": o.outputs[0].finish_reason,
                                    "gen": text}, ensure_ascii=False) + "\n")

    total = len(rows)
    summary = {
        "label": label, "model": model, "n": total,
        "acc": correct / total, "trunc_rate": trunc / total,
        "n_shots": len(GSM8K_SHOTS),
        "protocol": "4-shot CoT plain text, no chat template",
        "shot_source": "Wei et al. 2022 CoT paper (4 of 8 standard examples)",
    }
    (out / f"{label}.json").write_text(json.dumps(summary, indent=2))
    print(f"\n===== {label} [base] (N={total}) =====")
    print(f"acc {summary['acc']:.3f} | trunc {summary['trunc_rate']:.3f}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="GSM8K base-series eval (4-shot CoT, no chat)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="/root/autodl-tmp/eval_results/gsm8k_base")
    args = ap.parse_args()
    run(
        model=args.model, label=args.label, n=args.n, outdir=args.outdir,
        max_tokens=args.max_tokens, gpu_mem=args.gpu_mem,
        temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, seed=args.seed,
    )


if __name__ == "__main__":
    main()
