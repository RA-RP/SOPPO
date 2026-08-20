#!/usr/bin/env python3
"""Unified evaluation CLI — single entry point for all eval tasks.

Dispatches to either a custom vLLM script (gsm8k / math500 / numina) or
lm-evaluation-harness (mmlu / truthfulqa / winogrande / arc_challenge).
Protocol defaults per task match the cycle05 field standards (see handin_05.md).

Usage
-----
Custom backend (default for gsm8k / math500 / numina):
    python run_eval.py --model PATH --label NAME --task gsm8k --outdir DIR

lm_eval backend (all OOD tasks; also available for gsm8k/math500/numina):
    python run_eval.py --model PATH --label NAME --task mmlu --outdir DIR
    python run_eval.py --model PATH --label NAME --task gsm8k --backend lm_eval --outdir DIR

Protocol defaults per task
--------------------------
  gsm8k        custom  chat+no-think  0-shot  (N=1319)
  math500      custom  chat+no-think  0-shot  (N=500)
  numina       custom  chat+no-think  0-shot  (N=1024)
  mmlu         lm_eval no-chat  no-think  5-shot  8-subject subset  acc
  truthfulqa   lm_eval chat+no-think  0-shot  acc
  winogrande   lm_eval no-chat  no-think  0-shot  acc
  arc_challenge lm_eval chat+no-think  25-shot  acc_norm
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

LMEVAL = str(Path(sys.executable).with_name("lm_eval"))
if not Path(LMEVAL).exists():
    LMEVAL = shutil.which("lm_eval") or "lm_eval"

# 8-subject MMLU subset — spans STEM, life science, humanities, reasoning.
# Avoids 10+ hour full-57-subject run. Score is macro-average. ~3 min/model.
MMLU_SUBSET = (
    "mmlu_abstract_algebra,mmlu_anatomy,mmlu_philosophy,mmlu_logical_fallacies,"
    "mmlu_high_school_mathematics,mmlu_world_religions,mmlu_clinical_knowledge,"
    "mmlu_computer_security"
)

# Per-task lm_eval defaults
_LM_DEFAULTS: dict[str, dict] = {
    "mmlu": {
        "lm_tasks": MMLU_SUBSET,
        "num_fewshot": 5,
        "apply_chat_template": False,
        "enable_thinking": None,   # base model mode — omit from model_args entirely
        "metric_path": ("mmlu", ("acc,none", "acc")),
    },
    "truthfulqa": {
        "lm_tasks": "truthfulqa_mc1",
        "num_fewshot": 0,
        "apply_chat_template": True,
        "enable_thinking": False,
        "metric_path": ("truthfulqa_mc1", ("acc,none", "acc")),
    },
    "winogrande": {
        "lm_tasks": "winogrande",
        "num_fewshot": 0,
        "apply_chat_template": False,
        "enable_thinking": False,
        "metric_path": ("winogrande", ("acc,none", "acc")),
    },
    "arc_challenge": {
        "lm_tasks": "arc_challenge",
        "num_fewshot": 25,
        "apply_chat_template": True,
        "enable_thinking": False,
        "metric_path": ("arc_challenge", ("acc_norm,none", "acc_norm", "acc,none", "acc")),
    },
}

CUSTOM_TASKS = {"gsm8k", "math500", "numina"}
LMEVAL_TASKS = set(_LM_DEFAULTS.keys())
ALL_TASKS = CUSTOM_TASKS | LMEVAL_TASKS


def _run_custom(args: argparse.Namespace) -> int:
    """Dispatch to component runner for gsm8k / math500 / numina."""
    task = args.task
    if task == "gsm8k":
        from component.gsm8k.runner import run
    elif task == "math500":
        from component.math500.runner import run
    elif task == "numina":
        from component.numina.runner import run
    else:
        print(f"[ERROR] Unknown custom task: {task}", file=sys.stderr)
        return 1

    kwargs: dict = dict(
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
    # numina uses different defaults; run() accepts them all, excess are ignored
    run(**kwargs)
    return 0


def _run_lmeval(args: argparse.Namespace) -> int:
    """Dispatch to lm-evaluation-harness."""
    task = args.task
    defaults = _LM_DEFAULTS.get(task, {})

    lm_tasks = args.lm_tasks or defaults.get("lm_tasks", task)
    num_fewshot = args.num_fewshot if args.num_fewshot is not None else defaults.get("num_fewshot", 0)
    apply_chat = args.apply_chat_template if args.apply_chat_template is not None else defaults.get("apply_chat_template", False)
    enable_think = defaults.get("enable_thinking")  # always from defaults; CLI override not exposed

    vllm_args = (
        f"pretrained={args.model},"
        f"dtype=bfloat16,"
        f"max_model_len={args.max_model_len},"
        f"gpu_memory_utilization={args.gpu_mem}"
    )
    if enable_think is not None:
        vllm_args += f",enable_thinking={str(enable_think).lower()}"

    out = Path(args.outdir) / args.label
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        LMEVAL,
        "--model", "vllm",
        "--model_args", vllm_args,
        "--tasks", lm_tasks,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", "auto",
        "--output_path", str(out),
        "--log_samples",
    ]
    if apply_chat:
        cmd.append("--apply_chat_template")

    print(f"[run_eval] lm_eval: task={task} label={args.label} fewshot={num_fewshot} "
          f"chat={apply_chat} think={enable_think}")
    print(f"[run_eval] cmd: {' '.join(cmd)}")

    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"[ERROR] lm_eval exited with code {rc}", file=sys.stderr)
    return rc


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, help="Path to model checkpoint")
    ap.add_argument("--label", required=True, help="Short identifier (used for output file names)")
    ap.add_argument("--task", required=True, choices=sorted(ALL_TASKS),
                    help="Evaluation task")
    ap.add_argument("--outdir", required=True,
                    help="Output directory (custom: writes <label>.json; lm_eval: writes <label>/)")
    ap.add_argument("--backend", choices=["custom", "lm_eval"], default=None,
                    help="Backend (default: custom for gsm8k/math500/numina, lm_eval for others)")

    # Custom backend options
    cg = ap.add_argument_group("custom backend options (gsm8k / math500 / numina)")
    cg.add_argument("--n", type=int, default=0, help="Sample limit (0=all)")
    cg.add_argument("--max-tokens", type=int, default=3072)
    cg.add_argument("--temperature", type=float, default=0.7)
    cg.add_argument("--top-p", type=float, default=0.8)
    cg.add_argument("--top-k", type=int, default=20)
    cg.add_argument("--seed", type=int, default=42)

    # Shared options
    ap.add_argument("--gpu-mem", type=float, default=0.80,
                    help="GPU memory utilization (vLLM)")
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="vLLM max_model_len (lm_eval backend only)")

    # lm_eval backend overrides
    lg = ap.add_argument_group("lm_eval backend overrides")
    lg.add_argument("--lm-tasks", default=None,
                    help="Override lm_eval --tasks string (e.g. custom MMLU subject list)")
    lg.add_argument("--num-fewshot", type=int, default=None,
                    help="Override number of few-shot examples")
    lg.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction, default=None,
                    help="Override chat template flag")

    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    # Resolve backend
    backend = args.backend
    if backend is None:
        backend = "custom" if args.task in CUSTOM_TASKS else "lm_eval"

    if backend == "custom":
        if args.task not in CUSTOM_TASKS:
            print(f"[ERROR] Task '{args.task}' has no custom backend. "
                  f"Use --backend lm_eval or choose from: {sorted(CUSTOM_TASKS)}", file=sys.stderr)
            return 1
        return _run_custom(args)
    else:
        return _run_lmeval(args)


if __name__ == "__main__":
    raise SystemExit(main())
