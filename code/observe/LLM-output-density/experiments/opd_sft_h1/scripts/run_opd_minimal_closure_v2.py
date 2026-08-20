#!/usr/bin/env python3
"""
run_opd_minimal_closure_v2.py — OPD 最小实验 03 的严谨版 closure

相对 v1 修复 5 个方法论缺陷：
  1. 统一 prompt 母池（一次采样切分，消除 bias）
  2. held-out eval loss + load_best_model_at_end（防过拟合）
  3. DataSize 语义统一为"实际训练消耗 prompt/样本数"
  4. 正确的 S/X 探针：S 按模型区分(teacher/student rollout/训练数据)，X 冻结共用(theta0 rollout)
  5. S×model 完整交叉矩阵

复用 v1 (run_opd_minimal_closure.py) 已验证的工具函数。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path("/root/LLM-output-density")
SIDECAR_ROOT = REPO_ROOT / "experiments/opd_sft_h1"
EXP_ROOT_DEFAULT = Path("/root/autodl-tmp/exp0609/opd_minimal_03_v2")
BASE_MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen3-1.7B")
TEACHER_MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen3-4B")
NUMINA_PARQUET = Path("/root/autodl-tmp/prepared/NuminaMath-1___5/train.parquet")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

# 复用 v1 工具函数
from scripts.run_opd_minimal_closure import (  # noqa: E402
    ModelSpec,
    TARGET_METRICS,
    OOD_BENCHMARKS,
    build_figures,
    cleanup_cuda,
    ensure_dir,
    link_eval_models,
    merge_lora_adapter,
    write_csv,
    write_json,
    write_jsonl,
    _as_float,
)
from Eval.allRunEval_vLLM import run_eval_vllm  # noqa: E402
from Eval.component.Eval2Res import eval2res  # noqa: E402


def run_full_eval_v2(exp_root: Path, specs: list[ModelSpec], *, eval_limit: int | None = None) -> Path:
    """v2 全量评估：与 v1 相同，但支持 eval_limit（每个 benchmark 限样本数，便于 smoke/中等规模提速）。"""
    model_root = link_eval_models(exp_root, specs)
    eval_root = ensure_dir(exp_root / "eval")
    config = {
        "eval": {
            "enable_eval": True,
            "enable_result": True,
            "model_output_root": str(model_root),
            "merge_root": str(model_root),
            "output_origin_root": str(eval_root / "origin"),
            "output_root": str(eval_root / "fix"),
            "csv_path": str(eval_root / "csv_results"),
            "batch_size": 6,
            "gpu_memory_utilization": 0.65,
            "vllm_model_len": 4096,
            "eval_limit": eval_limit,
            "fail_on_eval_error": True,
            "include_path": "Eval/tasks",
            "hf_home": "/root/.cache/huggingface",
            "hf_datasets_cache": "/root/.cache/huggingface/datasets",
            "hf_hub_offline": True,
            "hf_datasets_offline": True,
            "lm_tasks": ["gsm8k", "hendrycks_math500", "mmlu", "truthfulqa_mc1", "truthfulqa_mc2", "winogrande"],
            "target_metrics_csv": "target_metrics_results.csv",
            "target_metrics": TARGET_METRICS,
            "tasks": [spec.eval_task for spec in specs],
        }
    }
    config_path = eval_root / "eval_config.yaml"
    ensure_dir(config_path.parent)
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    run_eval_vllm(config)
    csv_path = eval_root / "csv_results" / "target_metrics_results.csv"
    if not csv_path.exists():
        eval2res(config)
    if not csv_path.exists():
        raise FileNotFoundError(f"target metrics CSV missing: {csv_path}")
    return csv_path
from opd_sft_h1.geometry_metrics import (  # noqa: E402
    effective_rank,
    log_spectrum_drift,
    spectral_gap,
    xs_log_spectrum_gap,
)
from opd_sft_h1.geometry_reader import read_geometry_rows  # noqa: E402


# =============================================================================
# 1. 统一 prompt 母池
# =============================================================================
def build_unified_pool(
    exp_root: Path,
    *,
    n_cold: int,
    n_opd: int,
    n_sft_max: int,
    n_heldout: int,
    n_probe: int,
    seed: int = 42,
) -> dict[str, Any]:
    """从 NuminaMath 一次性采样母池，切分为互不重叠的 slice 落盘。

    返回各 slice 的路径与切分元信息。
    """
    pool_dir = ensure_dir(exp_root / "pool")
    meta_path = pool_dir / "pool_meta.json"
    if meta_path.exists():
        print(f"[POOL] reuse existing pool: {meta_path}", flush=True)
        return json.loads(meta_path.read_text())

    df = pd.read_parquet(NUMINA_PARQUET)
    df = df[df["problem"].notna() & df["solution"].notna() & df["answer"].notna()].reset_index(drop=True)

    total_needed = n_sft_max + n_heldout + n_probe
    if len(df) < total_needed:
        raise ValueError(f"Pool too small: have {len(df)}, need {total_needed}")

    # 一次性打乱采样母集
    sample = df.sample(n=total_needed, random_state=seed).reset_index(drop=True)

    # 互不重叠切分：[train_sft_max] [heldout] [probe]
    sft_part = sample.iloc[:n_sft_max].reset_index(drop=True)
    heldout_part = sample.iloc[n_sft_max : n_sft_max + n_heldout].reset_index(drop=True)
    probe_part = sample.iloc[n_sft_max + n_heldout : n_sft_max + n_heldout + n_probe].reset_index(drop=True)

    # train_prompts.jsonl: prompt-only（cold/OPD/SFT 都从这里取前 N），取自 sft_part 母集
    train_prompts = pool_dir / "train_prompts.jsonl"
    write_jsonl(train_prompts, [{"problem": str(r["problem"])} for _, r in sft_part.iterrows()])

    # train_sft.jsonl: problem+solution(+answer)，SFT 监督标签，与 train_prompts 行对齐
    def _sft_row(r: pd.Series) -> dict[str, Any]:
        problem = str(r["problem"])
        solution = str(r["solution"])
        answer = str(r.get("answer") or "").strip()
        if answer and answer not in solution:
            solution = f"{solution}\n\nFinal answer: {answer}"
        return {"messages": [{"role": "user", "content": problem}, {"role": "assistant", "content": solution}]}

    train_sft = pool_dir / "train_sft.jsonl"
    write_jsonl(train_sft, [_sft_row(r) for _, r in sft_part.iterrows()])

    # heldout_eval.jsonl: held-out 验证集（prompt-only，OPD 算 JSD / SFT 算 CE 时都用这批 prompt 的标签）
    heldout = pool_dir / "heldout_eval.jsonl"
    write_jsonl(heldout, [_sft_row(r) for _, r in heldout_part.iterrows()])

    # probe_prompts.jsonl: 给 X/S rollout 用的 prompt（与训练/eval 不重叠）
    probe_prompts = pool_dir / "probe_prompts.jsonl"
    write_jsonl(
        probe_prompts,
        [{"problem": str(r["problem"]), "solution": str(r["solution"])} for _, r in probe_part.iterrows()],
    )

    meta = {
        "seed": seed,
        "total_sampled": int(total_needed),
        "n_cold": n_cold,
        "n_opd": n_opd,
        "n_sft_max": n_sft_max,
        "n_heldout": n_heldout,
        "n_probe": n_probe,
        "train_prompts": str(train_prompts),
        "train_sft": str(train_sft),
        "heldout_eval": str(heldout),
        "probe_prompts": str(probe_prompts),
    }
    write_json(meta_path, meta)
    print(f"[POOL] built unified pool -> {pool_dir}", flush=True)
    return meta


# =============================================================================
# 2. rollout 工具（teacher / student 离线生成）
# =============================================================================
def rollout_completions(
    model_dir: Path,
    prompts: list[str],
    *,
    max_new_tokens: int = 2048,
    seqlen: int = 1024,
    device: str = "cuda",
    temperature: float = 0.0,
) -> list[str]:
    """用 HF 模型对 prompts 逐条 greedy 生成 completion，生成到自然 EOS（不人为截断）。

    max_new_tokens 仅作安全上限（默认 2048），模型遇 EOS 自动停止以保留真实长度信息。
    返回纯生成文本（不含 prompt）。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()

    completions: list[str] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seqlen).to(device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        gen_ids = out[0, enc["input_ids"].shape[1] :]
        completions.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

    del model
    cleanup_cuda()
    return completions


def rollout_freeform_bos(
    model_dir: Path,
    n_samples: int,
    *,
    max_new_tokens: int = 2048,
    device: str = "cuda",
    temperature: float = 0.8,
    seed: int = 42,
) -> list[str]:
    """X-BOS 探针：纯 BOS token 起步、无任何 prompt 条件的自由生成（生成到自然 EOS，不截断）。

    用于"完全无条件输出分布"探针（论文依据）。因为无条件，必须采样(temperature>0)以避免
    每条都生成相同文本；固定 seed 保证可复现。返回 n_samples 条生成文本。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        # Qwen 系列无显式 bos，用 eos 作为序列起始 token（与无条件生成等价的最小起步）
        bos_id = tokenizer.eos_token_id

    torch.manual_seed(seed)
    completions: list[str] = []
    for _ in range(n_samples):
        input_ids = torch.tensor([[bos_id]], device=device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, **gen_kwargs)
        gen_ids = out[0, input_ids.shape[1] :]
        completions.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

    del model
    cleanup_cuda()
    return completions


# =============================================================================
# 3. 训练阶段
# =============================================================================
def _write_trl_config(
    cfg_path: Path,
    *,
    name: str,
    output_root: Path,
    student_checkpoint: str | None,
    prompt_jsonl: str,
    eval_jsonl: str,
    max_samples: int,
    max_steps: int,
    eval_steps: int,
    grad_accum: int,
    learning_rate: float,
    role_label: str,
    registry_method: str,
    pi_mix_lambda: float | None,
) -> None:
    cfg = {
        "experiment": {"name": name, "output_root": str(output_root), "seed": 42},
        "model": {
            "base_model": str(BASE_MODEL),
            "cold_start_checkpoint": student_checkpoint,
            "student_start_checkpoint": student_checkpoint,
            "teacher_model": str(TEACHER_MODEL),
        },
        "data": {
            "prompt_jsonl": prompt_jsonl,
            "eval_jsonl": eval_jsonl,
            "prompt_text_field": "problem",
            "max_samples": max_samples,
            "eval_max_samples": 64,
        },
        "trl": {
            "lmbda": 1.0,
            "beta": 0.5,
            "loss_top_k": 1,
            "max_length": 4096,
            "max_completion_length": 512,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": grad_accum,
            "per_device_eval_batch_size": 1,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "save_steps": eval_steps,
            "eval_steps": eval_steps,
            "save_total_limit": 2,
            "logging_steps": max(eval_steps // 2, 1),
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
            "use_teacher_server": False,
            "teacher_model_server_url": None,
            "report_to": [],
        },
        "registry": {"method": registry_method, "role_label": role_label, "pi_mix_lambda": pi_mix_lambda},
    }
    ensure_dir(cfg_path.parent)
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def train_opd_like(
    *,
    exp_root: Path,
    label: str,
    student_checkpoint: str | None,
    prompt_jsonl: str,
    eval_jsonl: str,
    max_samples: int,
    max_steps: int,
    grad_accum: int,
    eval_steps: int,
    role_label: str,
    registry_method: str,
    model_role: str,
) -> ModelSpec:
    """运行一次 OPD-like 蒸馏（cold-start 或 OPD），返回 merged ModelSpec。

    DataSize 语义：实际消耗 prompt 数 = max_steps * batch(1) * grad_accum。
    registry_method: 写入 trl 配置的 registry.method（必须是 ALLOWED_METHODS，如 cold_start/trl_opd_like）
    model_role: ModelSpec.role，用于后续 selection（theta0/opd/sft）
    """
    from opd_sft_h1.trl_runner import run_from_config

    out_root = ensure_dir(exp_root / label)
    adapter_dir = out_root / "checkpoint_output"
    merged_dir = adapter_dir / "merged_model"
    consumed = max_steps * 1 * grad_accum

    if (merged_dir / "config.json").exists():
        print(f"[SKIP] {label} merged exists: {merged_dir}", flush=True)
        return ModelSpec(label, str(consumed), model_role, merged_dir, adapter_dir)

    cfg_path = out_root / f"config_{label}.yaml"
    _write_trl_config(
        cfg_path,
        name=label,
        output_root=out_root,
        student_checkpoint=student_checkpoint,
        prompt_jsonl=prompt_jsonl,
        eval_jsonl=eval_jsonl,
        max_samples=max_samples,
        max_steps=max_steps,
        eval_steps=eval_steps,
        grad_accum=grad_accum,
        learning_rate=3.0e-5,
        role_label=role_label,
        registry_method=registry_method,
        pi_mix_lambda=1.0,
    )
    run_from_config(str(cfg_path))

    # run_from_config 内部已对 LoRA adapter 合并；确认 merged 存在
    if not (merged_dir / "config.json").exists():
        # student_checkpoint=None 时基模是 BASE_MODEL，否则是上一级 merged
        base_for_merge = Path(student_checkpoint) if student_checkpoint else BASE_MODEL
        merge_lora_adapter(base_for_merge, adapter_dir, merged_dir)
    return ModelSpec(label, str(consumed), model_role, merged_dir, adapter_dir)


def _sft_config_kwargs(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    import inspect

    sig = inspect.signature(cls.__init__)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    accepted = set(sig.parameters)
    return {k: v for k, v in kwargs.items() if k in accepted}


def train_sft_control(
    *,
    exp_root: Path,
    label: str,
    theta0_merged: Path,
    train_sft_jsonl: Path,
    heldout_jsonl: Path,
    num_samples: int,
    learning_rate: float = 1.0e-5,
    num_train_epochs: float = 3.0,
) -> ModelSpec:
    """continued SFT 对照：用 num_samples 条监督数据训练（扫 gain 点）。带 held-out CE eval + best ckpt。

    DataSize 语义：实际监督样本数 = num_samples。
    """
    out_root = ensure_dir(exp_root / "step4_sft_controls" / label)
    adapter_dir = out_root / "checkpoint_output"
    merged_dir = out_root / "merged_model"
    if (merged_dir / "config.json").exists():
        print(f"[SKIP] SFT {label} merged exists: {merged_dir}", flush=True)
        return ModelSpec(label, str(num_samples), "sft", merged_dir, adapter_dir)

    import torch  # noqa: F401
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(str(theta0_merged), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full = load_dataset("json", data_files=str(train_sft_jsonl), split="train")
    train_ds = full.select(range(min(num_samples, len(full))))
    eval_ds = load_dataset("json", data_files=str(heldout_jsonl), split="train")
    eval_ds = eval_ds.select(range(min(64, len(eval_ds))))

    def to_text(record: dict[str, Any]) -> dict[str, str]:
        return {"text": tokenizer.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=False)}

    train_ds = train_ds.map(to_text, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(to_text, remove_columns=eval_ds.column_names)

    eval_steps = max(num_samples // (16 * 4), 4)  # 大致每 ~4 个优化步评估一次
    sft_kwargs = {
        "output_dir": str(adapter_dir),
        "do_train": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "max_steps": -1,
        "max_grad_norm": 0.5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "weight_decay": 0.0,
        "eval_strategy": "steps",
        "eval_steps": eval_steps,
        "per_device_eval_batch_size": 1,
        "save_strategy": "steps",
        "save_steps": eval_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": max(eval_steps // 2, 1),
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "adamw_torch",
        "bf16": True,
        "report_to": "none",
        "max_length": 4096,
        "dataset_text_field": "text",
        "packing": False,
        "seed": 42,
        "remove_unused_columns": True,
    }
    args = SFTConfig(**_sft_config_kwargs(SFTConfig, sft_kwargs))
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    trainer = SFTTrainer(
        model=str(theta0_merged), args=args, train_dataset=train_ds,
        eval_dataset=eval_ds, processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    del trainer
    cleanup_cuda()
    merge_lora_adapter(theta0_merged, adapter_dir, merged_dir)
    return ModelSpec(label, str(num_samples), "sft", merged_dir, adapter_dir)


# =============================================================================
# 4. S/X 探针生成
# =============================================================================
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_probes(
    exp_root: Path,
    pool: dict[str, Any],
    theta0: ModelSpec,
    opd: ModelSpec,
    sft_specs: list[ModelSpec],
    *,
    n_probe: int = 32,
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    """生成 X(双探针,冻结共用) 与 S(按模型区分) 探针。所有 rollout 生成到自然 EOS,不截断。

    X(两个,都由 theta0 生成,冻结共用):
      X_prompt: theta0 对 probe_prompts rollout(有 prompt 条件) -> {"output": {"text": prompt+rollout}}
      X_bos:    theta0 纯 BOS 起步无条件自由生成 -> {"output": {"text": freeform}}
    S(按模型区分,代表各模型训练时对齐的目标分布):
      theta0: teacher(Qwen3-4B) 对 probe_prompts rollout -> {"question": prompt, "answer": teacher_rollout}
      opd:    opd 模型自己 rollout -> 同格式
      sft_*:  直接取该 SFT 的训练数据切片(problem->question, solution->answer)
    返回 {"x_probes": {variant: jsonl}, "s_roots": {source: Path}}
    """
    gs_root = ensure_dir(exp_root / "getslice")
    inputs_root = ensure_dir(gs_root / "inputs")

    probe_rows = _read_jsonl(Path(pool["probe_prompts"]))[:n_probe]
    probe_prompts = [str(r["problem"]) for r in probe_rows]

    x_probes: dict[str, Path] = {}

    # ---- X_prompt: theta0 有 prompt 条件 rollout(不截断) ----
    x_prompt_dir = ensure_dir(inputs_root / "X_prompt")
    x_prompt_jsonl = x_prompt_dir / "x_probe.jsonl"
    if not x_prompt_jsonl.exists():
        print(f"[PROBE-X_prompt] theta0 rollout {len(probe_prompts)} prompts (to EOS) ...", flush=True)
        x_comps = rollout_completions(theta0.model_dir, probe_prompts, max_new_tokens=max_new_tokens)
        write_jsonl(x_prompt_jsonl, [{"output": {"text": f"{p}\n{c}"}} for p, c in zip(probe_prompts, x_comps)])
    else:
        print(f"[PROBE-X_prompt] reuse {x_prompt_jsonl}", flush=True)
    x_probes["prompt"] = x_prompt_jsonl

    # ---- X_bos: theta0 纯 BOS 无条件自由生成(不截断) ----
    x_bos_dir = ensure_dir(inputs_root / "X_bos")
    x_bos_jsonl = x_bos_dir / "x_probe.jsonl"
    if not x_bos_jsonl.exists():
        print(f"[PROBE-X_bos] theta0 freeform-from-BOS x{len(probe_prompts)} (to EOS) ...", flush=True)
        b_comps = rollout_freeform_bos(theta0.model_dir, len(probe_prompts), max_new_tokens=max_new_tokens)
        write_jsonl(x_bos_jsonl, [{"output": {"text": c}} for c in b_comps])
    else:
        print(f"[PROBE-X_bos] reuse {x_bos_jsonl}", flush=True)
    x_probes["bos"] = x_bos_jsonl

    s_roots: dict[str, Path] = {}

    # ---- S: theta0 用 teacher rollout(不截断) ----
    s_theta0_dir = ensure_dir(inputs_root / "S" / "theta0" / "numina_math_probe")
    s_theta0 = s_theta0_dir / "gamma_s.jsonl"
    if not s_theta0.exists():
        print(f"[PROBE-S theta0] teacher rollout {len(probe_prompts)} prompts (to EOS) ...", flush=True)
        t_comps = rollout_completions(TEACHER_MODEL, probe_prompts, max_new_tokens=max_new_tokens)
        write_jsonl(s_theta0, [{"question": p, "answer": c} for p, c in zip(probe_prompts, t_comps)])
    s_roots["theta0"] = (inputs_root / "S" / "theta0")

    # ---- S: opd 用自己 rollout(不截断) ----
    s_opd_dir = ensure_dir(inputs_root / "S" / opd.source / "numina_math_probe")
    s_opd = s_opd_dir / "gamma_s.jsonl"
    if not s_opd.exists():
        print(f"[PROBE-S {opd.source}] student rollout {len(probe_prompts)} prompts (to EOS) ...", flush=True)
        o_comps = rollout_completions(opd.model_dir, probe_prompts, max_new_tokens=max_new_tokens)
        write_jsonl(s_opd, [{"question": p, "answer": c} for p, c in zip(probe_prompts, o_comps)])
    s_roots[opd.source] = (inputs_root / "S" / opd.source)

    # ---- S: 每个 SFT 用其训练数据切片 ----
    sft_rows = _read_jsonl(Path(pool["train_sft"]))
    for spec in sft_specs:
        s_sft_dir = ensure_dir(inputs_root / "S" / spec.source / "numina_math_probe")
        s_sft = s_sft_dir / "gamma_s.jsonl"
        if not s_sft.exists():
            n = int(spec.size)
            sel = sft_rows[:n][:n_probe]
            out_rows = []
            for r in sel:
                msgs = r["messages"]
                q = next((m["content"] for m in msgs if m["role"] == "user"), "")
                a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                out_rows.append({"question": q, "answer": a})
            write_jsonl(s_sft, out_rows)
        s_roots[spec.source] = (inputs_root / "S" / spec.source)

    return {"x_probes": x_probes, "s_roots": s_roots}


# =============================================================================
# 5. S×model 交叉矩阵 GetSlice
# =============================================================================
def run_getslice_cross(
    exp_root: Path,
    specs: list[ModelSpec],
    probes: dict[str, Any],
    *,
    target_layer: int = 14,
    seqlen: int = 1024,
    s_nsamples: int = 32,
    x_nsamples: int = 32,
) -> Path:
    """对每个 (model_i, S_probe_j) 组合跑一次 GetSlice S；对每个 (model_i, X_variant) 跑 X。
    X 探针(X_prompt / X_bos)由 theta0 生成,冻结共用。

    输出布局:
      getslice/outputs/{model_i}/step_{size}/S__{probe_j}/numina_math_probe/layer_N/sMat_*.json
      getslice/outputs/{model_i}/step_{size}/X__{variant}/layer_N/xMat_X.json   (variant: prompt/bos)
    """
    gs_root = exp_root / "getslice"
    output_root = ensure_dir(gs_root / "outputs")
    config_root = ensure_dir(gs_root / "configs")
    s_roots: dict[str, Path] = probes["s_roots"]
    x_probes: dict[str, Path] = probes["x_probes"]

    _base_cfg = {
        "tasks": ["numina_math_probe"],
        "DEV": "cuda",
        "model_seq_len": seqlen,
        "seed": 3,
        "target_layer": target_layer,
        "layer_gpu_chunk_size": 7,
        "single_layer_task_group_size": 1,
        "epsilon": 0.001,
        "svd_singular_floor": 0.0,
        "cholesky_jitter": 0.00001,
        "activation_cache_device": "cuda",
        "uv_dtype": "float32",
        "cleanup_intermediate": True,
        "skip_existing_outputs": True,
        "model_dtype": "float16",
        "trust_remote_code": True,
        "s_batch_size": 1,
        "x_batch_size": 1,
        "save_s_json_path": "sMat_{task}.json",
        "save_x_json_path": "xMat_X.json",
        "save_s_pt_path": None,
        "save_x_pt_path": None,
        "save_s_uv_path": None,
        "save_x_uv_path": None,
        "save_metrics_pt_path": None,
        "save_metrics_json_path": None,
    }

    for model_spec in specs:
        step_root = output_root / model_spec.source / f"step_{model_spec.size}"

        # ---- S: 每个 S 探针一次 ----
        for probe_source, s_root in s_roots.items():
            tag = f"S__{probe_source}"
            save_path = step_root / tag
            if (save_path / "numina_math_probe" / "sMat_numina_math_probe.json").exists():
                print(f"[SKIP] GetSlice {model_spec.source} x {tag}", flush=True)
                continue
            cfg_s = dict(_base_cfg)
            cfg_s.update({
                "model": str(model_spec.model_dir),
                "save_path": str(save_path),
                "mode": "s_only_svd",
                "s_nsamples": s_nsamples,
                "s_jsonl_path": str(s_root),
                "s_jsonl_file": "gamma_s.jsonl",
            })
            cfg_path = config_root / f"{model_spec.source}__{tag}__S.json"
            write_json(cfg_path, cfg_s)
            _run_slice(cfg_path)

        # ---- X: 两个 X 变体各一次 (X__prompt / X__bos) ----
        for variant, x_jsonl in x_probes.items():
            x_tag = f"X__{variant}"
            x_save_path = step_root / x_tag
            if (x_save_path / "X" / "xMat_X.json").exists():
                print(f"[SKIP] GetSlice {model_spec.source} x {x_tag}", flush=True)
                continue
            cfg_x = dict(_base_cfg)
            cfg_x.update({
                "model": str(model_spec.model_dir),
                "save_path": str(x_save_path),
                "mode": "x_only_svd",
                "x_nsamples": x_nsamples,
                "x_jsonl_path": str(x_jsonl),
            })
            cfg_path = config_root / f"{model_spec.source}__{x_tag}.json"
            write_json(cfg_path, cfg_x)
            _run_slice(cfg_path)

    return output_root


def _run_slice(cfg_path: Path) -> None:
    import subprocess

    env = os.environ.copy()
    env.update({"TMPDIR": "/root/autodl-tmp/pip-tmp", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})
    cmd = [sys.executable, str(REPO_ROOT / "GetSlice/slice.py"), "--config", str(cfg_path)]
    print(f"[SLICE] {cfg_path.name}", flush=True)
    result = subprocess.run(cmd, cwd=str(REPO_ROOT / "GetSlice"), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"GetSlice failed rc={result.returncode}: {cfg_path}")


# =============================================================================
# 6. 几何表（含 s_probe_source 交叉维度）
# =============================================================================
def build_geometry_tables_cross(exp_root: Path, getslice_root: Path) -> Path:
    """解析交叉矩阵布局，输出含 model/s_probe_source 维度的 geometry_metrics。

    source 形如 '{model}/step_{size}/S__{probe}/numina_math_probe/layer_N'
              或 '{model}/step_{size}/X/layer_N'
    """
    rows = read_geometry_rows(getslice_root)
    tables_dir = ensure_dir(exp_root / "tables")
    long_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    def parse_source(source: str, probe_dist: str) -> tuple[str, str | None, str | None]:
        """返回 (model, s_probe_source, x_variant)。
        S 路径含 'S__{probe}'，X 路径含 'X__{variant}'(prompt/bos)。
        """
        parts = source.split("/")
        model = parts[0] if parts else source
        s_probe = None
        x_variant = None
        if probe_dist == "S":
            for p in parts:
                if p.startswith("S__"):
                    s_probe = p[len("S__"):]
                    break
        elif probe_dist == "X":
            for p in parts:
                if p.startswith("X__"):
                    x_variant = p[len("X__"):]
                    break
        return model, s_probe, x_variant

    # 缓存：S 谱、X 谱(按 variant)、theta0 对角 S 谱
    s_sigma_by_key: dict[tuple, list[float]] = {}
    x_sigma_by_key: dict[tuple, list[float]] = {}  # (model, x_variant, layer, module) -> sigma
    theta0_diag_s: dict[tuple, list[float]] = {}  # (layer, module) -> theta0 在 theta0-S 探针下的谱

    # X-S gap 默认参考 X_prompt(与旧版口径一致；X_bos 也单独存)
    XS_REF_VARIANT = "prompt"

    parsed_rows = []
    for row in rows:
        model, s_probe, x_variant = parse_source(str(row["source"]), row["probe_distribution"])
        parsed_rows.append((row, model, s_probe, x_variant))
        if row["probe_distribution"] == "S" and s_probe is not None:
            s_sigma_by_key[(model, s_probe, row["layer"], row["module"])] = row["singular_values"]
            if model == "theta0" and s_probe == "theta0":
                theta0_diag_s[(row["layer"], row["module"])] = row["singular_values"]
        elif row["probe_distribution"] == "X" and x_variant is not None:
            x_sigma_by_key[(model, x_variant, row["layer"], row["module"])] = row["singular_values"]

    for row, model, s_probe, x_variant in parsed_rows:
        sigma = row["singular_values"]
        long_rows.append({
            **row, "model": model, "s_probe_source": s_probe, "x_variant": x_variant,
            "singular_values": json.dumps(sigma),
        })

        base_sigma = theta0_diag_s.get((row["layer"], row["module"])) if row["probe_distribution"] == "S" else None
        # X-S gap：对 S 行用参考 X 变体(X_prompt)的同模型谱
        x_ref = x_sigma_by_key.get((model, XS_REF_VARIANT, row["layer"], row["module"]))
        is_diagonal = (row["probe_distribution"] == "S" and s_probe == model)

        metric_rows.append(
            {
                "model": model,
                "s_probe_source": s_probe,
                "x_variant": x_variant,
                "is_diagonal": is_diagonal,
                "step": row["step"],
                "probe_distribution": row["probe_distribution"],
                "layer": row["layer"],
                "module": row["module"],
                "effective_rank": effective_rank(sigma),
                "spectral_gap_k": 1,
                "spectral_gap": spectral_gap(sigma, 1),
                "spectral_drift_from_theta0_diag": (
                    None if base_sigma is None else log_spectrum_drift(sigma, base_sigma)
                ),
                "X_S_spectrum_level_gap": (
                    xs_log_spectrum_gap(x_ref, sigma)
                    if row["probe_distribution"] == "S" and x_ref is not None
                    else None
                ),
                "principal_angle_status": "unavailable_no_uv",
                "singular_json_path": row["singular_json_path"],
            }
        )

    write_csv(tables_dir / "geometry_long.csv", long_rows)
    geometry_metrics = tables_dir / "geometry_metrics.csv"
    write_csv(geometry_metrics, metric_rows)
    return geometry_metrics


# =============================================================================
# 7. 评估表 + 选择 + 匹配
# =============================================================================
def build_eval_and_selection(exp_root: Path, specs: list[ModelSpec], metrics_csv: Path) -> dict[str, Any]:
    """构建 eval_trajectory / ood_penalty / matched pairs / selection.json。

    DataSize 列直接来自 ModelSpec.size（已统一为实际训练消耗量）。
    """
    spec_by_source = {s.source: s for s in specs}
    df = pd.read_csv(metrics_csv, encoding="utf-8-sig")
    base_rows = df[df["Source"].astype(str) == "theta0"]
    if base_rows.empty:
        raise ValueError("No theta0 baseline in metrics CSV")
    base = base_rows.iloc[0].to_dict()

    eval_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []
    for _, series in df.iterrows():
        row = series.to_dict()
        source = str(row["Source"])
        spec = spec_by_source.get(source)
        row["checkpoint_id"] = f"{source}_{row.get('DataSize')}"
        row["role"] = spec.role if spec else "unknown"
        row["checkpoint_path"] = str(spec.checkpoint_path) if spec else ""
        row["model_dir"] = str(spec.model_dir) if spec else ""
        gsm = _as_float(row.get("GSM8K"))
        row["GSM8K_gain"] = None if gsm is None else gsm - _as_float(base.get("GSM8K"))
        m500 = _as_float(row.get("MATH500"))
        row["MATH500_gain"] = None if m500 is None else m500 - _as_float(base.get("MATH500"))
        drops = []
        for bench in OOD_BENCHMARKS:
            score = _as_float(row.get(bench))
            base_score = _as_float(base.get(bench))
            drop = None if score is None or base_score is None else max(0.0, base_score - score)
            ood_rows.append({"checkpoint_id": row["checkpoint_id"], "Source": source, "benchmark": bench,
                             "score": score, "baseline_score": base_score, "drop": drop})
            if drop is not None:
                drops.append(drop)
        if drops:
            valid_scores = [_as_float(row.get(b)) for b in OOD_BENCHMARKS if _as_float(row.get(b)) is not None]
            row["OOD_lite_avg"] = sum(valid_scores) / len(valid_scores)
            row["OOD_lite_penalty_p2"] = sum(d**2 for d in drops) ** 0.5
            row["OOD_lite_penalty_p3"] = sum(d**3 for d in drops) ** (1.0 / 3.0)
            row["Worst_OOD_lite_drop"] = max(drops)
        else:
            row["OOD_lite_avg"] = row["OOD_lite_penalty_p2"] = row["OOD_lite_penalty_p3"] = row["Worst_OOD_lite_drop"] = None
        eval_rows.append(row)

    tables_dir = ensure_dir(exp_root / "tables")
    write_csv(tables_dir / "eval_trajectory.csv", eval_rows)
    write_csv(tables_dir / "ood_penalty.csv", ood_rows)

    opd_rows = [r for r in eval_rows if r["role"] == "opd"]
    sft_rows = [r for r in eval_rows if r["role"] == "sft"]
    if not opd_rows:
        raise ValueError("No OPD rows for selection")
    selected_opd = max(opd_rows, key=lambda r: _as_float(r.get("GSM8K")) if _as_float(r.get("GSM8K")) is not None else -1)
    opd_gain = _as_float(selected_opd.get("GSM8K_gain"))

    selected_sft = None
    best_gap = None
    for r in sft_rows:
        gain = _as_float(r.get("GSM8K_gain"))
        if gain is None or opd_gain is None:
            continue
        gap = abs(gain - opd_gain)
        if best_gap is None or gap < best_gap:
            best_gap, selected_sft = gap, r

    threshold = 0.02
    match_row = {
        "opd_run_id": selected_opd["Source"],
        "opd_checkpoint_id": selected_opd["checkpoint_id"],
        "opd_GSM8K_gain": selected_opd.get("GSM8K_gain"),
        "sft_run_id": selected_sft["Source"] if selected_sft else None,
        "sft_checkpoint_id": selected_sft["checkpoint_id"] if selected_sft else None,
        "sft_GSM8K_gain": selected_sft.get("GSM8K_gain") if selected_sft else None,
        "GSM8K_gain_gap": best_gap,
        "match_status": ("valid_match" if best_gap is not None and best_gap <= threshold
                         else ("nearest_match" if selected_sft else "no_sft_candidate")),
        "OOD_lite_penalty_p2_delta": (
            _as_float(selected_opd.get("OOD_lite_penalty_p2")) - _as_float(selected_sft.get("OOD_lite_penalty_p2"))
            if selected_sft and _as_float(selected_opd.get("OOD_lite_penalty_p2")) is not None
            and _as_float(selected_sft.get("OOD_lite_penalty_p2")) is not None else None
        ),
        "Worst_OOD_lite_drop_delta": (
            _as_float(selected_opd.get("Worst_OOD_lite_drop")) - _as_float(selected_sft.get("Worst_OOD_lite_drop"))
            if selected_sft and _as_float(selected_opd.get("Worst_OOD_lite_drop")) is not None
            and _as_float(selected_sft.get("Worst_OOD_lite_drop")) is not None else None
        ),
    }
    write_csv(tables_dir / "matched_gsm8k_pairs.csv", [match_row])
    write_csv(tables_dir / "main_matched_result.csv",
              [{**match_row, "selection_note": "best OPD by GSM8K; nearest SFT by GSM8K_gain (SFT swept by data size)"}])

    selection = {"opd": selected_opd, "sft": selected_sft, "match": match_row, "baseline": base}
    write_json(exp_root / "selection.json", selection)
    return {
        "eval_trajectory": tables_dir / "eval_trajectory.csv",
        "ood_penalty": tables_dir / "ood_penalty.csv",
        "matched_gsm8k_pairs": tables_dir / "matched_gsm8k_pairs.csv",
        "selection": selection,
    }


# =============================================================================
# 8. 图表 + registry + summary
# =============================================================================
def build_figures_cross(exp_root: Path) -> None:
    """交叉矩阵可视化：对角谱轨迹 + OOD vs drift 散点 + model×s_probe effective_rank heatmap。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir = ensure_dir(exp_root / "figures")
    eval_df = pd.read_csv(exp_root / "tables/eval_trajectory.csv", encoding="utf-8-sig")
    geom_df = pd.read_csv(exp_root / "tables/geometry_metrics.csv", encoding="utf-8-sig")

    s_geom = geom_df[geom_df["probe_distribution"] == "S"].copy()
    diag = s_geom[s_geom["is_diagonal"] == True].copy()  # noqa: E712

    # 1) 对角 spectral_gap 轨迹（每模型在自己探针下）
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, grp in diag.groupby("model"):
        g = grp.sort_values("module")
        ax.plot(g["module"].astype(str), g["spectral_gap"], marker="o", label=str(model))
    ax.set_ylabel("spectral_gap (diagonal S)")
    ax.set_xlabel("module")
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures_dir / "main_geometry_trajectory.png", dpi=160)
    plt.close(fig)

    # 2) OOD vs drift 散点（对角 drift 均值）
    drift = diag.groupby("model", as_index=False)["spectral_drift_from_theta0_diag"].mean()
    merged = eval_df[["Source", "OOD_lite_penalty_p2"]].merge(drift, left_on="Source", right_on="model", how="left")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(merged["spectral_drift_from_theta0_diag"], merged["OOD_lite_penalty_p2"])
    for _, r in merged.iterrows():
        ax.annotate(str(r["Source"]), (r["spectral_drift_from_theta0_diag"], r["OOD_lite_penalty_p2"]), fontsize=8)
    ax.set_xlabel("mean spectral drift (diag)")
    ax.set_ylabel("OOD_lite_penalty_p2")
    fig.tight_layout()
    fig.savefig(figures_dir / "main_ood_vs_geometry.png", dpi=160)
    plt.close(fig)

    # 3) model × s_probe effective_rank 交叉热力图（对所有 module 取均值）
    cross = s_geom.pivot_table(index="model", columns="s_probe_source", values="effective_rank", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cross.fillna(0.0).to_numpy(), aspect="auto")
    ax.set_yticks(range(len(cross.index)))
    ax.set_yticklabels(cross.index)
    ax.set_xticks(range(len(cross.columns)))
    ax.set_xticklabels(cross.columns, rotation=30, ha="right")
    ax.set_title("effective_rank: model (row) x S-probe (col)")
    fig.colorbar(im, ax=ax, label="effective_rank")
    fig.tight_layout()
    fig.savefig(figures_dir / "appendix_cross_effrank_heatmap.png", dpi=160)
    plt.close(fig)


def write_registry_and_summary(exp_root: Path, specs: list[ModelSpec], artifacts: dict[str, Any]) -> None:
    registry_dir = ensure_dir(exp_root / "registry")
    records = []
    for spec in specs:
        records.append({
            "run_id": spec.checkpoint_id,
            "checkpoint_id": spec.checkpoint_id,
            "trajectory_group_id": "opd_minimal_03_v2",
            "method": spec.role,
            "role_label": spec.source,
            "data_size": spec.size,
            "checkpoint_path": str(spec.checkpoint_path),
            "model_dir": str(spec.model_dir),
            "status": "completed",
        })
    write_jsonl(registry_dir / "run_registry.jsonl", records)
    write_jsonl(registry_dir / "checkpoints.jsonl", records)
    summary = {
        "experiment": "opd_minimal_03_v2",
        "status": "completed",
        "models": [{"source": s.source, "size": s.size, "role": s.role,
                    "model_dir": str(s.model_dir), "checkpoint_path": str(s.checkpoint_path)} for s in specs],
        "artifacts": {k: (str(v) if isinstance(v, Path) else v) for k, v in artifacts.items()},
    }
    write_json(exp_root / "summary.json", summary)


def shutdown_now() -> None:
    import subprocess
    subprocess.run(["/usr/bin/shutdown", "-h", "now"], check=False)


# =============================================================================
# main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="OPD minimal experiment 03 closure v2")
    parser.add_argument("--exp-root", default=str(EXP_ROOT_DEFAULT))
    parser.add_argument("--smoke", action="store_true", help="tiny smoke run")
    parser.add_argument("--shutdown-on-exit", action="store_true")
    parser.add_argument("--skip-getslice", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TMPDIR", "/root/autodl-tmp/pip-tmp")
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    ensure_dir(Path(os.environ["TMPDIR"]))

    exp_root = Path(args.exp_root).expanduser().resolve()
    ensure_dir(exp_root)

    if args.smoke:
        # max_new_tokens 是探针 rollout 的安全上限（遇 EOS 自动停，保留真实长度）
        cfg = dict(n_cold=8, n_opd_steps=2, opd_grad_accum=2, cold_grad_accum=2,
                   sft_sizes=[8, 16], n_heldout=8, n_probe=4, max_new_tokens=256, target_layer=14,
                   eval_limit=8, gs_seqlen=64, gs_nsamples=4)
    else:
        # 中等规模：OPD ~200 步(grad_accum=4 -> 800 prompt), SFT 扫数据量
        # max_new_tokens=2048：探针 rollout 生成到自然 EOS 不截断，保留长度信息
        cfg = dict(n_cold=512, n_opd_steps=200, opd_grad_accum=4, cold_grad_accum=4,
                   sft_sizes=[256, 512, 1024, 2048], n_heldout=64, n_probe=32, max_new_tokens=2048, target_layer=14,
                   eval_limit=200, gs_seqlen=512, gs_nsamples=16)

    artifacts: dict[str, Any] = {}
    try:
        n_sft_max = max(cfg["sft_sizes"])
        pool = build_unified_pool(
            exp_root, n_cold=cfg["n_cold"], n_opd=cfg["n_opd_steps"],
            n_sft_max=n_sft_max, n_heldout=cfg["n_heldout"], n_probe=cfg["n_probe"], seed=42,
        )

        # ---- theta0 cold-start ----
        cold_steps = max(cfg["n_cold"] // cfg["cold_grad_accum"], 1)
        theta0 = train_opd_like(
            exp_root=exp_root, label="step2_cold_start", student_checkpoint=None,
            prompt_jsonl=pool["train_prompts"], eval_jsonl=pool["heldout_eval"],
            max_samples=cfg["n_cold"], max_steps=cold_steps, grad_accum=cfg["cold_grad_accum"],
            eval_steps=max(cold_steps // 2, 1), role_label="theta0",
            registry_method="cold_start", model_role="theta0",
        )
        theta0 = ModelSpec("theta0", str(cfg["n_cold"]), "theta0", theta0.model_dir, theta0.checkpoint_path)

        # ---- OPD distill from theta0 ----
        opd_consumed = cfg["n_opd_steps"] * cfg["opd_grad_accum"]
        opd = train_opd_like(
            exp_root=exp_root, label="step3_opd_distill", student_checkpoint=str(theta0.model_dir),
            prompt_jsonl=pool["train_prompts"], eval_jsonl=pool["heldout_eval"],
            max_samples=n_sft_max, max_steps=cfg["n_opd_steps"], grad_accum=cfg["opd_grad_accum"],
            eval_steps=max(cfg["n_opd_steps"] // 4, 1), role_label="opd_lmbda1",
            registry_method="trl_opd_like", model_role="opd",
        )
        opd = ModelSpec("opd_lmbda1", str(opd_consumed), "opd", opd.model_dir, opd.checkpoint_path)

        # ---- SFT controls: 扫数据量 ----
        sft_specs: list[ModelSpec] = []
        for size in cfg["sft_sizes"]:
            spec = train_sft_control(
                exp_root=exp_root, label=f"sft_n{size}", theta0_merged=theta0.model_dir,
                train_sft_jsonl=Path(pool["train_sft"]), heldout_jsonl=Path(pool["heldout_eval"]),
                num_samples=size, learning_rate=1.0e-5,
            )
            sft_specs.append(spec)

        specs = [theta0, opd, *sft_specs]

        # ---- 全量评估 ----
        metrics_csv = run_full_eval_v2(exp_root, specs, eval_limit=cfg.get("eval_limit"))
        artifacts["target_metrics_csv"] = metrics_csv
        artifacts.update(build_eval_and_selection(exp_root, specs, metrics_csv))

        # ---- 探针 + 交叉 GetSlice + 几何 ----
        if not args.skip_getslice:
            probes = build_probes(exp_root, pool, theta0, opd, sft_specs,
                                  n_probe=cfg["n_probe"], max_new_tokens=cfg["max_new_tokens"])
            getslice_root = run_getslice_cross(
                exp_root, specs, probes, target_layer=cfg["target_layer"],
                seqlen=cfg["gs_seqlen"], s_nsamples=cfg["gs_nsamples"], x_nsamples=cfg["gs_nsamples"],
            )
            artifacts["geometry_metrics"] = build_geometry_tables_cross(exp_root, getslice_root)
            build_figures_cross(exp_root)

        write_registry_and_summary(exp_root, specs, artifacts)
        print(f"[DONE] closure v2 completed: {exp_root}", flush=True)
    except Exception as exc:
        write_json(exp_root / "closure_failed.json", {"error": repr(exc)})
        print(f"[FAILED] {exc!r}", flush=True)
        import traceback
        traceback.print_exc()
        if args.shutdown_on_exit:
            shutdown_now()
        raise
    if args.shutdown_on_exit:
        shutdown_now()


if __name__ == "__main__":
    main()
