#!/usr/bin/env python3
"""Cycle 09 block 2 G4: Llama-3.2-3B-Base MATH500 launch gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch


REPO = Path("/root/LLM-output-density")
MODEL = Path("/root/autodl-tmp/model/Meta/modelscope/Llama-3.2-3B")
ROOT = Path("/root/autodl-tmp/cycle09_block2/model2_llama/g4_preflight")
MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
SEED = 42
FORMAL_N = 100
FEWSHOT_N = 4
MATH500 = REPO / "Eval/tasks/data/hendrycks_math500/test.jsonl"
ZERO_TEMPLATE = (
    "Solve the following mathematics problem. Show your reasoning and put the "
    "final answer within \\boxed{}.\n\nProblem:\n{problem}\n\nSolution:\n"
)
FOURSHOT_HEADER = (
    "Solve each mathematics problem. Show the reasoning and put every final "
    "answer within \\boxed{}. The first four solved examples demonstrate the "
    "required format.\n\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=FORMAL_N)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.n < 1 or args.n > FORMAL_N:
        parser.error(f"--n must be within 1..{FORMAL_N}")
    if args.smoke:
        args.n = 4
        args.max_tokens = 256
        args.output_root = ROOT / "smoke"
    else:
        if args.n != FORMAL_N:
            parser.error("formal G4 is frozen at N=100; use --smoke for engineering tests")
        args.output_root = ROOT / "formal"
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def fourshot_template(exemplars: list[dict[str, Any]]) -> str:
    blocks = [FOURSHOT_HEADER]
    for number, row in enumerate(exemplars, start=1):
        blocks.append(
            f"Example {number}\nProblem:\n{row['problem']}\n\n"
            f"Solution:\n{row['solution']}\n\n"
        )
    blocks.append("Problem:\n{problem}\n\nSolution:\n")
    return "".join(blocks)


def main() -> None:
    args = parse_args()
    required = (MODEL / "config.json", MODEL / "tokenizer.json")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPO / "Eval/component/think_math"))
    import runner_think
    from vllm import LLM, SamplingParams

    with MATH500.open(encoding="utf-8") as handle:
        all_rows = [json.loads(line) for line in handle if line.strip()]
    if len(all_rows) != 500:
        raise RuntimeError(f"MATH500 rows={len(all_rows)}, expected 500")
    eval_rows = all_rows[: args.n]
    exemplars = all_rows[-FEWSHOT_N:]
    four_template = fourshot_template(exemplars)
    templates = {"zero_shot": ZERO_TEMPLATE, "four_shot": four_template}
    template_path = args.output_root / "prompt_templates.txt"
    template_path.write_text(
        "===== ZERO_SHOT =====\n"
        + ZERO_TEMPLATE
        + "\n\n===== FOUR_SHOT =====\n"
        + four_template,
        encoding="utf-8",
    )

    llm = LLM(
        model=str(MODEL),
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=16384,
        seed=SEED,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=SEED,
    )
    summaries: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    for protocol, template in templates.items():
        prompts = [template.replace("{problem}", row["problem"]) for row in eval_rows]
        outputs = llm.generate(prompts, sampling)
        correct = extractable = boxed = truncated = 0
        token_lengths: list[int] = []
        for index, (row, prompt, output) in enumerate(zip(eval_rows, prompts, outputs)):
            candidate = output.outputs[0]
            prediction = runner_think.extract_pred(candidate.text)
            ok = bool(runner_think.score(candidate.text, row["answer"]))
            correct += int(ok)
            extractable += int(prediction is not None and str(prediction).strip() != "")
            boxed += int("\\boxed" in candidate.text)
            truncated += int(candidate.finish_reason == "length")
            token_lengths.append(len(candidate.token_ids))
            all_samples.append(
                {
                    "protocol": protocol,
                    "index": index,
                    "problem": row["problem"],
                    "gold": row["answer"],
                    "prediction": prediction,
                    "correct": ok,
                    "finish_reason": candidate.finish_reason,
                    "n_tokens": len(candidate.token_ids),
                    "prompt": prompt,
                    "generation": candidate.text,
                }
            )
        n = len(eval_rows)
        acc = correct / n
        summaries.append(
            {
                "protocol": protocol,
                "n": n,
                "acc": acc,
                "stderr": math.sqrt(max(acc * (1.0 - acc), 0.0) / n),
                "extractable_rate": extractable / n,
                "boxed_rate": boxed / n,
                "truncation_rate": truncated / n,
                "mean_generation_tokens": sum(token_lengths) / n,
                "max_generation_tokens": max(token_lengths),
            }
        )

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    samples_path = args.output_root / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for row in all_samples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    example_path = args.output_root / "generation_examples_5.jsonl"
    with example_path.open("w", encoding="utf-8") as handle:
        for row in all_samples[:5]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    frame = pd.DataFrame(summaries)
    summary_path = args.output_root / "summary.csv"
    atomic_csv(summary_path, frame)
    go = any(
        row["acc"] >= 0.10 and row["extractable_rate"] > 0.0 for row in summaries
    )
    decision = "SMOKE_ONLY" if args.smoke else ("GO" if go else "STOP")
    manifest = {
        "status": "smoke_complete" if args.smoke else "complete",
        "task": "Cycle 09 block 2 G4",
        "created_at": utc_now(),
        "model": str(MODEL),
        "model_config_sha256": sha256_file(MODEL / "config.json"),
        "dataset": str(MATH500),
        "n": len(eval_rows),
        "seed": SEED,
        "decoding": {"temperature": 0.0, "max_tokens": args.max_tokens},
        "fewshot_examples": [
            {"problem": row["problem"], "answer": row["answer"]}
            for row in exemplars
        ],
        "prompt_templates": str(template_path),
        "prompt_templates_sha256": sha256_file(template_path),
        "summary": summaries,
        "frozen_gate": "GO iff either protocol acc>=0.10 and an answer is extractable",
        "decision": decision,
        "artifacts": {
            "summary": str(summary_path),
            "samples": str(samples_path),
            "examples_5": str(example_path),
        },
    }
    manifest_path = args.output_root / "manifest.json"
    atomic_json(manifest_path, manifest)
    if not args.smoke:
        atomic_csv(MINI / "G4_llama_preflight.csv", frame)
        atomic_json(MINI / "G4_llama_preflight_manifest.json", manifest)
    print(frame.to_string(index=False), flush=True)
    print(f"[G4] decision={decision} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
