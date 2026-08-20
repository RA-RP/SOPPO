"""NuminaMath-1.5 test set evaluation runner — base model protocol.

Protocol: 4-shot CoT, plain text format, no chat template, no enable_thinking.
For base models (Qwen3-1.7B-Base) and models trained from them.
Shots: first 4 rows of test.jsonl (fixed; remaining rows are scored).

Importable API:
    from component.numina.runner_base import run
    summary = run(model="/path", label="mymodel", outdir="/out")
"""
import argparse
import json
import sys
from pathlib import Path

_COMPONENT = Path(__file__).resolve().parent.parent  # Eval/component/
if str(_COMPONENT) not in sys.path:
    sys.path.insert(0, str(_COMPONENT))

from scorer_v2 import score, is_mcq
from scorer import extract_pred

TEST = "/root/autodl-tmp/prepared/NuminaMath-1___5/test.jsonl"
INSTR = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

# Number of rows from the beginning of test.jsonl used as few-shot demonstrations.
# These rows are excluded from scoring.
N_SHOTS = 4

JUNK = {"", "not found", "notfound", "none", "nan", "proof"}


def _build_prompt(shots: list[dict], problem: str) -> str:
    """Build a 4-shot CoT plain-text prompt (no chat template).

    test.jsonl has 'answer' (final answer string), not a full solution.
    We use it as the Solution text; the boxed format keeps the scorer happy.
    """
    parts = []
    for s in shots:
        sol = s.get("solution") or s.get("answer", "")
        parts.append(f"Problem: {s['problem']}{INSTR}\n\nSolution: {sol}")
    parts.append(f"Problem: {problem}{INSTR}\n\nSolution:")
    return "\n\n".join(parts)


def load_data():
    """Return (shots, eval_rows). shots = first N_SHOTS rows; eval = the rest."""
    rows = [json.loads(line) for line in open(TEST)]
    rows = [r for r in rows if str(r.get("answer", "")).strip().lower() not in JUNK]
    shots = rows[:N_SHOTS]
    eval_rows = rows[N_SHOTS:]
    return shots, eval_rows


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
    """Run NuminaMath base-series eval (4-shot CoT, no chat template)."""
    from vllm import LLM, SamplingParams

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    shots, eval_rows = load_data()
    if n and n < len(eval_rows):
        eval_rows = eval_rows[:n]
    prompts = [_build_prompt(shots, r["problem"]) for r in eval_rows]

    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=gpu_mem,
              max_model_len=4096, trust_remote_code=True)
    sp = SamplingParams(temperature=temperature, top_p=top_p,
                        top_k=top_k, max_tokens=max_tokens, seed=seed)
    outs = llm.generate(prompts, sp)

    n_mcq = n_open = c_mcq = c_open = boxed_count = trunc = 0
    with open(out / f"{label}_samples.jsonl", "w") as fsamp:
        for r, o in zip(eval_rows, outs):
            text = o.outputs[0].text
            ok = bool(score(text, r["answer"]))
            mcq = is_mcq(str(r["answer"]).strip())
            if "\\boxed" in text:
                boxed_count += 1
            if o.outputs[0].finish_reason == "length":
                trunc += 1
            if mcq:
                n_mcq += 1; c_mcq += int(ok)
            else:
                n_open += 1; c_open += int(ok)
            fsamp.write(json.dumps({"gold": r["answer"], "pred": extract_pred(text),
                                    "ok": ok, "is_mcq": mcq,
                                    "finish": o.outputs[0].finish_reason,
                                    "gen": text}, ensure_ascii=False) + "\n")

    total = len(eval_rows)
    correct = c_mcq + c_open
    summary = {
        "label": label, "model": model,
        "n": total, "n_shots": N_SHOTS,
        "overall": correct / total,
        "open_acc": c_open / max(n_open, 1), "n_open": n_open,
        "mcq_acc": c_mcq / max(n_mcq, 1), "n_mcq": n_mcq,
        "boxed_rate": boxed_count / total, "trunc_rate": trunc / total,
        "shot_row_indices": list(range(N_SHOTS)),
        "protocol": "4-shot CoT plain text, no chat template",
    }
    (out / f"{label}.json").write_text(json.dumps(summary, indent=2))
    print(f"\n===== {label} [base] (N={total}, {N_SHOTS}-shot) =====")
    print(f"overall {summary['overall']:.3f} | open {summary['open_acc']:.3f} (N={n_open}) "
          f"| mcq {summary['mcq_acc']:.3f} (N={n_mcq}) | boxed {summary['boxed_rate']:.3f} "
          f"| trunc {summary['trunc_rate']:.3f}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="NuminaMath base-series eval (4-shot CoT, no chat)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="/root/autodl-tmp/eval_results/numina_base")
    args = ap.parse_args()
    run(
        model=args.model, label=args.label, n=args.n, outdir=args.outdir,
        max_tokens=args.max_tokens, gpu_mem=args.gpu_mem,
        temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, seed=args.seed,
    )


if __name__ == "__main__":
    main()
