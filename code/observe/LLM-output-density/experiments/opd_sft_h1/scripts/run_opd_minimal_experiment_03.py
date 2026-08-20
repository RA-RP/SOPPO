#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_opd_minimal_experiment_03.py

TRL-first OPD-like 最小工程闭环（实验 03）

流程：
  Step 1: cold-start theta_0 — LoRA SFT(Teacher-Rollout) from Qwen3-1.7B，512 prompts
  Step 2: OPD-like distillation — TRL-OPD，1024 prompts，teacher=Qwen3-4B
  Step 3: SFT control — continued SFT from theta_0
  Step 4: Summarize

用法：
  python experiments/opd_sft_h1/scripts/run_opd_minimal_experiment_03.py
    --base-model /root/autodl-tmp/model/Qwen/Qwen3-1.7B
    --teacher-model /root/autodl-tmp/model/Qwen/Qwen3-4B
    --output-root /root/autodl-tmp/exp0609/opd_minimal_03
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("/root/LLM-output-density")
DEFAULT_EXP_ROOT = Path("/root/autodl-tmp/exp0609/opd_minimal_03")

NUMINA_PARQUET = "/root/autodl-tmp/prepared/NuminaMath-1___5/train.parquet"


def convert_parquet_to_jsonl(parquet_path: str, output_path: str, num_samples: int) -> None:
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42)
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({"problem": str(row["problem"])}, ensure_ascii=False) + "\n")
    print(f"[DATA] {len(df)} prompts -> {output_path}")


def run_trl_smoke(config_path: str) -> None:
    """Run run_trl_distill_smoke.py with given config."""
    env = {
        **subprocess.os.environ,
        "TRL_EXPERIMENTAL_SILENCE": "1",
        "TMPDIR": "/root/autodl-tmp/pip-tmp",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/opd_sft_h1/scripts/run_trl_distill_smoke.py"),
            "--config", config_path,
        ],
        env=env,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"TRL smoke failed (rc={result.returncode})")


def write_config(
    output_path: Path,
    *,
    name: str,
    output_root: str,
    base_model: str,
    teacher_model: str,
    student_checkpoint: str | None,
    prompt_jsonl: str,
    max_samples: int,
    max_steps: int,
    role_label: str,
    registry_method: str,
    pi_mix_lambda: float | None,
    use_teacher_server: bool = False,
    teacher_model_server_url: str | None = None,
) -> None:
    import yaml

    cfg = {
        "experiment": {
            "name": name,
            "output_root": output_root,
            "seed": 42,
        },
        "model": {
            "base_model": base_model,
            "cold_start_checkpoint": student_checkpoint,
            "student_start_checkpoint": student_checkpoint,
            "teacher_model": teacher_model,
        },
        "data": {
            "prompt_jsonl": prompt_jsonl,
            "prompt_text_field": "problem",
            "max_samples": max_samples,
        },
        "selection": {
            "selector_metric": "GSM8K",
            "valid_gain_threshold_points": 2.0,
            "checkpoint_policy": "latest_and_best_or_closest",
        },
        "eval": {
            "selector_tasks": ["gsm8k"],
            "full_eval_tasks": ["gsm8k", "hendrycks_math500", "mmlu", "winogrande", "truthfulqa_mc1", "truthfulqa_mc2"],
        },
        "trl": {
            "lmbda": 1.0,
            "beta": 0.5,
            "loss_top_k": 1,
            "max_length": 4096,
            "max_completion_length": 512,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 3.0e-5,
            "max_steps": max_steps,
            "save_steps": max(max_steps // 2, 10),
            "save_total_limit": 2,
            "logging_steps": 5,
            "gradient_checkpointing": False,
            "optim": "adamw_torch",
            "bf16": True,
            "model_init_kwargs": {"torch_dtype": "bfloat16"},
            "teacher_model_init_kwargs": {"torch_dtype": "bfloat16"},
            "peft": {
                "enabled": True,
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            },
            "use_vllm": False,
            "use_teacher_server": use_teacher_server,
            "teacher_model_server_url": teacher_model_server_url if use_teacher_server else None,
            "report_to": [],
        },
        "registry": {
            "method": registry_method,
            "role_label": role_label,
            "pi_mix_lambda": pi_mix_lambda,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"[CONFIG] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OPD Minimal Experiment 03")
    parser.add_argument("--base-model", default="/root/autodl-tmp/model/Qwen/Qwen3-1.7B")
    parser.add_argument("--teacher-model", default="/root/autodl-tmp/model/Qwen/Qwen3-4B")
    parser.add_argument("--num-samples-cold", type=int, default=512)
    parser.add_argument("--num-samples-distill", type=int, default=1024)
    parser.add_argument("--cold-steps", type=int, default=20)
    parser.add_argument("--distill-steps", type=int, default=50)
    parser.add_argument("--output-root", default=str(DEFAULT_EXP_ROOT))
    parser.add_argument("--use-teacher-server", action="store_true")
    parser.add_argument("--teacher-model-server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    exp_root = Path(args.output_root).expanduser().resolve()
    prompt_jsonl = "/tmp/opd_minimal_prompts.jsonl"
    max_samples = max(args.num_samples_cold, args.num_samples_distill)

    print("=" * 60)
    print("OPD Minimal Experiment 03")
    print("=" * 60)
    print(f"Base model:    {args.base_model}")
    print(f"Teacher model: {args.teacher_model}")
    print(f"Exp root:      {exp_root}")
    print(
        "Teacher mode:  "
        + (f"server ({args.teacher_model_server_url})" if args.use_teacher_server else "local")
    )
    print(f"Cold-start SFT(Teacher-Rollout): {args.num_samples_cold} samples, {args.cold_steps} steps")
    print(f"Distill:      {args.num_samples_distill} samples, {args.distill_steps} steps")
    print()

    if args.dry_run:
        print("[DRY RUN] would run experiment 03")
        return

    # Step 1: prepare data
    print("\n[Step 1/4] Preparing data...")
    convert_parquet_to_jsonl(NUMINA_PARQUET, prompt_jsonl, max_samples)

    # Step 2: cold-start theta_0 via teacher-guided prompt rollout/distillation.
    print("\n[Step 2/4] Cold-start theta_0: SFT(Teacher-Rollout)...")
    cold_root = exp_root / "step2_cold_start"
    cold_config = cold_root / "config_cold.yaml"
    write_config(
        cold_config,
        name="cold_start_theta0",
        output_root=str(cold_root),
        base_model=args.base_model,
        teacher_model=args.teacher_model,
        student_checkpoint=None,
        prompt_jsonl=prompt_jsonl,
        max_samples=args.num_samples_cold,
        max_steps=args.cold_steps,
        role_label="theta_0_SFT(Teacher-Rollout)",
        registry_method="cold_start",
        pi_mix_lambda=None,
        use_teacher_server=args.use_teacher_server,
        teacher_model_server_url=args.teacher_model_server_url,
    )
    # Check if already done
    if (cold_root / "checkpoint_output" / "adapter_config.json").exists():
        print("  [SKIP] cold-start already exists")
        theta_0_checkpoint = cold_root / "checkpoint_output"
    else:
        print("  Running cold-start SFT(Teacher-Rollout)...")
        run_trl_smoke(str(cold_config))
        theta_0_checkpoint = cold_root / "checkpoint_output"
        if not theta_0_checkpoint.exists():
            raise RuntimeError("Cold-start did not produce checkpoint_output")
    print(f"  [OK] theta_0 checkpoint: {theta_0_checkpoint}")

    # Step 3: OPD-like distillation
    print("\n[Step 3/4] OPD-like distillation...")
    distill_root = exp_root / "step3_opd_distill"
    distill_config = distill_root / "config_distill.yaml"
    write_config(
        distill_config,
        name="opd_distill",
        output_root=str(distill_root),
        base_model=args.base_model,
        teacher_model=args.teacher_model,
        student_checkpoint=str(theta_0_checkpoint),
        prompt_jsonl=prompt_jsonl,
        max_samples=args.num_samples_distill,
        max_steps=args.distill_steps,
        role_label="TRL-OPD-lmbda-1.0",
        registry_method="trl_opd_like",
        pi_mix_lambda=1.0,
        use_teacher_server=args.use_teacher_server,
        teacher_model_server_url=args.teacher_model_server_url,
    )
    print("  Running OPD distillation...")
    run_trl_smoke(str(distill_config))
    distill_checkpoint = distill_root / "checkpoint_output"
    print(f"  [OK] OPD distill checkpoint: {distill_checkpoint}")

    # Step 4: summary
    print("\n[Step 4/4] Summary...")
    summary = {
        "experiment": "opd_minimal_03",
        "timestamp": datetime.now().isoformat(),
        "base_model": args.base_model,
        "teacher_model": args.teacher_model,
        "cold_start": {
            "label": "SFT(Teacher-Rollout)",
            "checkpoint": str(theta_0_checkpoint),
            "samples": args.num_samples_cold,
            "steps": args.cold_steps,
        },
        "opd_distill": {"checkpoint": str(distill_checkpoint), "samples": args.num_samples_distill, "steps": args.distill_steps},
    }
    (exp_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  Summary: {exp_root / 'summary.json'}")
    print()
    print("=" * 60)
    print("Experiment 03 pipeline completed!")
    print("Next: run full eval + GetSlice + match (see README.md)")


if __name__ == "__main__":
    main()
