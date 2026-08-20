#!/usr/bin/env python3
"""
run_cycle06.py — Cycle 06: SFT Feasibility and Degradation

Four phases run sequentially with skip-if-exists logic:
  Phase 1: build_unified_pool (n_sft_max=4096)
  Phase 2: train 8 SFT arms (instruct × 4 + base × 4)
  Phase 3: eval 10 models (instruct series + base series)
  Phase 4: geometry (GetSlice + weight export + principalEvidence)

Usage:
  python run_cycle06.py [--exp-root PATH] [--smoke] [--start-from-phase N]

  --exp-root: defaults to /root/autodl-tmp/cycle06_sft_feasibility_and_degradation/
  --smoke:    small-scale run for pipeline verification
  --start-from-phase: skip phases before N (1-4), useful after a crash
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/root/LLM-output-density")
SIDECAR_ROOT = REPO_ROOT / "experiments/opd_sft_h1"
EXP_ROOT_DEFAULT = Path("/root/autodl-tmp/cycle06_sft_feasibility_and_degradation")

INSTRUCT_BASE_MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen3-1.7B")
BASE_BASE_MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen3-1___7B-Base")
NUMINA_PARQUET = Path("/root/autodl-tmp/prepared/NuminaMath-1___5/train.parquet")
NUMINA_TEST = Path("/root/autodl-tmp/prepared/NuminaMath-1___5/test.jsonl")

GEOMETRY_LAYERS = [6, 14, 22]
SFT_SIZES = [512, 1024, 2048, 4096]

# 6 representative MMLU subtasks (~1090 items total) covering STEM / humanities /
# social-sciences / professional domains — sufficient for OOD degradation detection.
# lm_eval --limit does not propagate to group tasks, so we enumerate subtasks explicitly.
MMLU_SUBTASKS = ",".join([
    "mmlu_abstract_algebra",       # STEM / math  (~100)
    "mmlu_anatomy",                # STEM / bio    (~135)
    "mmlu_machine_learning",       # STEM / CS     (~112)
    "mmlu_high_school_world_history",  # humanities   (~237)
    "mmlu_marketing",              # social sci    (~234)
    "mmlu_professional_medicine",  # professional  (~272)
])

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))


# =============================================================================
# Utilities
# =============================================================================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_lm_eval(
    model_path: str,
    tasks: str,
    output_path: str,
    *,
    model_args_extra: str = "",
    num_fewshot: int | None = None,
    apply_chat_template: bool = False,
    limit: int | None = None,
) -> None:
    """Run lm_eval via subprocess."""
    # Tuned for 48 GB GPU: higher utilization leaves ample headroom for log_softmax;
    # longer max_model_len improves batching; CUDA graphs enabled for Ada arch.
    model_args = (
        f"pretrained={model_path},dtype=bfloat16,"
        f"gpu_memory_utilization=0.85,max_model_len=4096"
    )
    if model_args_extra:
        model_args += f",{model_args_extra}"

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", tasks,
        "--batch_size", "auto",
        "--output_path", output_path,
    ]
    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]
    if apply_chat_template:
        cmd += ["--apply_chat_template"]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    env = dict(os.environ,
               HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1",
               TMPDIR="/root/autodl-tmp/pip-tmp")
    print(f"[LM_EVAL] {tasks} -> {output_path}", flush=True)
    r = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"lm_eval failed rc={r.returncode}: tasks={tasks} model={model_path}")


# =============================================================================
# Phase 1: Data Pool
# =============================================================================
def phase1_build_pool(exp_root: Path, *, smoke: bool) -> dict:
    """Build unified pool and render instruct + base versions."""
    import pandas as pd

    pool_dir = ensure_dir(exp_root / "pool")
    meta_path = pool_dir / "pool_meta.json"
    if meta_path.exists():
        print(f"[POOL] reuse existing pool: {meta_path}", flush=True)
        return json.loads(meta_path.read_text())

    n_sft_max = 16 if smoke else 4096
    n_heldout = 8 if smoke else 64
    n_probe = 4 if smoke else 32
    seed = 42

    df = pd.read_parquet(NUMINA_PARQUET)
    df = df[df["problem"].notna() & df["solution"].notna()].reset_index(drop=True)

    total_needed = n_sft_max + n_heldout + n_probe
    if len(df) < total_needed:
        raise ValueError(f"Pool too small: have {len(df)}, need {total_needed}")

    sample = df.sample(n=total_needed, random_state=seed).reset_index(drop=True)
    sft_part = sample.iloc[:n_sft_max].reset_index(drop=True)
    heldout_part = sample.iloc[n_sft_max: n_sft_max + n_heldout].reset_index(drop=True)
    probe_part = sample.iloc[n_sft_max + n_heldout:].reset_index(drop=True)

    def _sft_row(r):
        problem = str(r["problem"])
        solution = str(r["solution"])
        answer = str(r.get("answer") or "").strip()
        if answer and answer not in solution:
            solution = f"{solution}\n\nFinal answer: {answer}"
        return {
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ]
        }

    # pool_instruct.jsonl — messages format for instruct series SFT
    pool_instruct = pool_dir / "pool_instruct.jsonl"
    write_jsonl(pool_instruct, [_sft_row(r) for _, r in sft_part.iterrows()])

    # pool_base.jsonl — plain text format for base series SFT
    pool_base = pool_dir / "pool_base.jsonl"
    write_jsonl(pool_base, [
        {"text": f"Problem: {str(r['problem'])}\n\nSolution: {str(r['solution'])}"}
        for _, r in sft_part.iterrows()
    ])

    # heldout_eval.jsonl — for eval_loss during training (messages format for both series)
    heldout_eval = pool_dir / "heldout_eval.jsonl"
    write_jsonl(heldout_eval, [_sft_row(r) for _, r in heldout_part.iterrows()])

    # heldout_eval_base.jsonl — plain text version for base series held-out eval_loss
    heldout_eval_base = pool_dir / "heldout_eval_base.jsonl"
    write_jsonl(heldout_eval_base, [
        {"text": f"Problem: {str(r['problem'])}\n\nSolution: {str(r['solution'])}"}
        for _, r in heldout_part.iterrows()
    ])

    # probe_prompts.jsonl — for GetSlice S probe
    probe_prompts = pool_dir / "probe_prompts.jsonl"
    write_jsonl(probe_prompts, [
        {"problem": str(r["problem"]), "solution": str(r["solution"])}
        for _, r in probe_part.iterrows()
    ])

    meta = {
        "seed": seed,
        "n_sft_max": n_sft_max,
        "n_heldout": n_heldout,
        "n_probe": n_probe,
        "pool_instruct": str(pool_instruct),
        "pool_base": str(pool_base),
        "heldout_eval": str(heldout_eval),
        "heldout_eval_base": str(heldout_eval_base),
        "probe_prompts": str(probe_prompts),
    }
    write_json(meta_path, meta)
    print(f"[POOL] built -> {pool_dir}  "
          f"(n_sft_max={n_sft_max}, n_heldout={n_heldout}, n_probe={n_probe})", flush=True)
    return meta


# =============================================================================
# Phase 2: SFT Training
# =============================================================================
def _sft_config_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    accepted = set(sig.parameters)
    return {k: v for k, v in kwargs.items() if k in accepted}


def train_sft_arm(
    *,
    exp_root: Path,
    label: str,
    start_model: Path,
    train_jsonl: Path,
    heldout_jsonl: Path,
    num_samples: int,
    is_base_series: bool,
) -> Path:
    """Train one SFT arm. Returns path to merged_model/.

    is_base_series=True: plain text format, dataset_text_field="text".
    is_base_series=False: messages format, SFTTrainer renders via chat template.
    """
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    out_root = ensure_dir(exp_root / label)
    adapter_dir = out_root / "checkpoint_output"
    merged_dir = out_root / "merged_model"

    if (merged_dir / "config.json").exists():
        print(f"[SKIP] {label} merged exists", flush=True)
        return merged_dir

    tokenizer = AutoTokenizer.from_pretrained(str(start_model), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_ds = load_dataset("json", data_files=str(train_jsonl), split="train")
    train_ds = full_ds.select(range(min(num_samples, len(full_ds))))
    eval_ds = load_dataset("json", data_files=str(heldout_jsonl), split="train")
    eval_ds = eval_ds.select(range(min(64, len(eval_ds))))

    if is_base_series:
        # plain text: field "text" is already rendered
        pass
    else:
        # messages format: render via chat template
        def to_text(record):
            return {
                "text": tokenizer.apply_chat_template(
                    record["messages"], tokenize=False, add_generation_prompt=False
                )
            }
        train_ds = train_ds.map(to_text, remove_columns=train_ds.column_names)
        eval_ds = eval_ds.map(to_text, remove_columns=eval_ds.column_names)

    eval_steps = max(num_samples // 8, 4)  # ~8 effective batch size (2 * 4 accum)
    sft_kwargs = {
        "output_dir": str(adapter_dir),
        "do_train": True,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1.0e-5,
        "num_train_epochs": 3,
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
        # SFTConfig-specific
        "max_seq_length": 4096,
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
        model=str(start_model), args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer, peft_config=peft_config,
    )
    print(f"[TRAIN] {label}  n={num_samples}  start={start_model}", flush=True)
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    del trainer
    cleanup_cuda()

    # Merge LoRA adapter into base model
    from scripts.run_opd_minimal_closure import merge_lora_adapter
    merge_lora_adapter(start_model, adapter_dir, merged_dir)

    print(f"[TRAIN] {label} done -> {merged_dir}", flush=True)
    return merged_dir


def phase2_train(exp_root: Path, pool: dict, *, smoke: bool) -> dict[str, Path]:
    """Train all 8 SFT arms. Returns {label: merged_model_path}."""
    sizes = [4] if smoke else SFT_SIZES
    models: dict[str, Path] = {
        "instruct_base": INSTRUCT_BASE_MODEL,
        "base_base": BASE_BASE_MODEL,
    }

    for n in sizes:
        label = f"instruct_sft_n{n}"
        merged = train_sft_arm(
            exp_root=exp_root,
            label=label,
            start_model=INSTRUCT_BASE_MODEL,
            train_jsonl=Path(pool["pool_instruct"]),
            heldout_jsonl=Path(pool["heldout_eval"]),
            num_samples=n,
            is_base_series=False,
        )
        models[label] = merged

    for n in sizes:
        label = f"base_sft_n{n}"
        merged = train_sft_arm(
            exp_root=exp_root,
            label=label,
            start_model=BASE_BASE_MODEL,
            train_jsonl=Path(pool["pool_base"]),
            heldout_jsonl=Path(pool["heldout_eval_base"]),
            num_samples=n,
            is_base_series=True,
        )
        models[label] = merged

    return models


# =============================================================================
# Phase 3: Evaluation
# =============================================================================
def _run_math_eval(
    runner_path: Path,
    model_path: str,
    label: str,
    outdir: Path,
    *,
    smoke: bool,
    gpu_mem: float = 0.85,
) -> dict:
    """Run a math eval runner as a subprocess to isolate GPU memory between calls.

    vLLM 0.18.0 has no LLM.__del__; running in-process leaks GPU memory across
    consecutive evals. Subprocess isolation is the only reliable fix.
    """
    result_json = outdir / f"{label}.json"
    if result_json.exists():
        print(f"[SKIP] {outdir.name}/{label}.json", flush=True)
        return json.loads(result_json.read_text())
    n = 8 if smoke else 0  # 0 = full dataset
    cmd = [
        sys.executable, str(runner_path),
        "--model", model_path,
        "--label", label,
        "--outdir", str(outdir),
        "--gpu-mem", str(gpu_mem),
    ]
    if n > 0:
        cmd += ["--n", str(n)]
    env = dict(os.environ, TMPDIR="/root/autodl-tmp/pip-tmp",
               HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1")
    print(f"[MATH_EVAL] {runner_path.name}  {label}", flush=True)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Math eval failed rc={r.returncode}: {runner_path.name} label={label}")
    return json.loads(result_json.read_text())


def _lm_eval_skip_check(output_path: Path, tasks: str) -> bool:
    """Return True if lm_eval output already exists.

    lm_eval writes results into <output_path>/<model_name>/results_*.json,
    not as task-named files, so we look for any results_*.json recursively.
    """
    if not output_path.exists():
        return False
    return bool(list(output_path.glob("**/results_*.json")))


def eval_instruct_series(exp_root: Path, models: dict[str, Path], *, smoke: bool) -> None:
    """Eval all 5 instruct-series models (instruct_base + 4 SFT arms)."""
    eval_root = ensure_dir(exp_root / "eval" / "instruct")
    instruct_labels = ["instruct_base"] + [f"instruct_sft_n{n}" for n in (SFT_SIZES if not smoke else [4])]

    runner_paths = {
        "numina": REPO_ROOT / "Eval/component/numina/runner.py",
        "gsm8k":  REPO_ROOT / "Eval/component/gsm8k/runner.py",
        "math500": REPO_ROOT / "Eval/component/math500/runner.py",
    }

    for label in instruct_labels:
        model_path = str(models[label])

        # Math generation tasks (each in its own subprocess to isolate GPU memory)
        for task, rpath in runner_paths.items():
            outdir = ensure_dir(eval_root / label / task)
            _run_math_eval(rpath, model_path, label, outdir, smoke=smoke)

        if smoke:
            continue  # skip lm_eval in smoke mode

        # OOD-lite: MMLU subset (6 subtasks, no chat, 5-shot)
        mmlu_out = eval_root / label / "mmlu"
        if not _lm_eval_skip_check(mmlu_out, MMLU_SUBTASKS):
            _run_lm_eval(model_path, MMLU_SUBTASKS, str(mmlu_out),
                         model_args_extra="enable_thinking=False",
                         num_fewshot=5, apply_chat_template=False)

        # OOD-lite: TruthfulQA (WITH chat template, 0-shot)
        tqa_out = eval_root / label / "truthfulqa"
        if not _lm_eval_skip_check(tqa_out, "truthfulqa_mc1"):
            _run_lm_eval(model_path, "truthfulqa_mc1", str(tqa_out),
                         model_args_extra="enable_thinking=False",
                         apply_chat_template=True)

        # OOD-lite: WinoGrande (no chat, 0-shot)
        wino_out = eval_root / label / "winogrande"
        if not _lm_eval_skip_check(wino_out, "winogrande"):
            _run_lm_eval(model_path, "winogrande", str(wino_out),
                         model_args_extra="enable_thinking=False",
                         apply_chat_template=False)

        # OOD-lite: ARC-challenge (no chat, 25-shot task default)
        arc_out = eval_root / label / "arc"
        if not _lm_eval_skip_check(arc_out, "arc_challenge"):
            _run_lm_eval(model_path, "arc_challenge", str(arc_out),
                         model_args_extra="enable_thinking=False",
                         apply_chat_template=False)


def eval_base_series(exp_root: Path, models: dict[str, Path], *, smoke: bool) -> None:
    """Eval all 5 base-series models (base_base + 4 SFT arms)."""
    eval_root = ensure_dir(exp_root / "eval" / "base")
    base_labels = ["base_base"] + [f"base_sft_n{n}" for n in (SFT_SIZES if not smoke else [4])]

    runner_paths = {
        "numina": REPO_ROOT / "Eval/component/numina/runner_base.py",
        "gsm8k":  REPO_ROOT / "Eval/component/gsm8k/runner_base.py",
        "math500": REPO_ROOT / "Eval/component/math500/runner_base.py",
    }

    for label in base_labels:
        model_path = str(models[label])

        # Math generation tasks (4-shot CoT, no chat template; each in own subprocess)
        for task, rpath in runner_paths.items():
            outdir = ensure_dir(eval_root / label / task)
            _run_math_eval(rpath, model_path, label, outdir, smoke=smoke)

        if smoke:
            continue

        # OOD-lite: all no chat, no enable_thinking arg for base models
        for task_name, lm_task, num_fs in [
            ("mmlu", MMLU_SUBTASKS, 5),
            ("truthfulqa", "truthfulqa_mc1", None),
            ("winogrande", "winogrande", None),
            ("arc", "arc_challenge", None),
        ]:
            task_out = eval_root / label / task_name
            if not _lm_eval_skip_check(task_out, lm_task):
                kwargs: dict = dict(apply_chat_template=False)
                if num_fs is not None:
                    kwargs["num_fewshot"] = num_fs
                _run_lm_eval(model_path, lm_task, str(task_out), **kwargs)


def phase3_eval(exp_root: Path, models: dict[str, Path], *, smoke: bool) -> None:
    eval_instruct_series(exp_root, models, smoke=smoke)
    eval_base_series(exp_root, models, smoke=smoke)


# =============================================================================
# Phase 4: Geometry
# =============================================================================
def _rollout_base(model_path: Path, prompts: list[str], *, max_new_tokens: int = 512) -> list[str]:
    """Generate completions from a base model using plain text (no chat template)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    completions = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0, enc["input_ids"].shape[1]:]
        completions.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    del model
    cleanup_cuda()
    return completions


def _rollout_instruct(model_path: Path, prompts: list[str], *, max_new_tokens: int = 512) -> list[str]:
    """Generate completions from instruct model using chat template."""
    from scripts.run_opd_minimal_closure_v2 import rollout_completions
    return rollout_completions(model_path, prompts, max_new_tokens=max_new_tokens)


def _build_s_probe(
    label: str,
    model_path: Path,
    pool: dict,
    inputs_root: Path,
    *,
    n_probe: int,
    is_base_series: bool,
) -> Path:
    """Build S probe for one model.

    Returns s_jsonl_path (= inputs_root/S/{label}), which is the parent
    of the task directory. GetSlice reads {s_jsonl_path}/{task}/gamma_s.jsonl.
    """
    s_dir = ensure_dir(inputs_root / "S" / label / "numina_math_probe")
    s_file = s_dir / "gamma_s.jsonl"
    s_jsonl_path = inputs_root / "S" / label  # returned value

    if s_file.exists():
        print(f"[SKIP] S probe {label}", flush=True)
        return s_jsonl_path

    probe_rows = read_jsonl(Path(pool["probe_prompts"]))[:n_probe]
    out_rows: list[dict] = []

    has_n = "_n" in label
    n_train = 0
    if has_n:
        try:
            n_train = int(label.split("_n")[-1])
        except ValueError:
            pass

    if has_n and n_train > 0:
        # SFT model: use training data slice as S probe
        if is_base_series:
            pool_rows = read_jsonl(Path(pool["pool_base"]))[:n_train][:n_probe]
            for r in pool_rows:
                text = r.get("text", "")
                if "\n\nSolution: " in text:
                    parts = text.split("\n\nSolution: ", 1)
                    q = parts[0].removeprefix("Problem: ")
                    a = parts[1]
                else:
                    q, a = text, ""
                out_rows.append({"question": q, "answer": a})
        else:
            pool_rows = read_jsonl(Path(pool["pool_instruct"]))[:n_train][:n_probe]
            for r in pool_rows:
                msgs = r.get("messages", [])
                q = next((m["content"] for m in msgs if m["role"] == "user"), "")
                a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                out_rows.append({"question": q, "answer": a})
    else:
        # Untrained base model: rollout from model as S probe
        prompts_txt = [str(r["problem"]) for r in probe_rows]
        if is_base_series:
            plain_prompts = [f"Problem: {p}\n\nSolution:" for p in prompts_txt]
            comps = _rollout_base(model_path, plain_prompts)
        else:
            comps = _rollout_instruct(model_path, prompts_txt)
        out_rows = [{"question": p, "answer": c} for p, c in zip(prompts_txt, comps)]

    write_jsonl(s_file, out_rows)
    print(f"[S PROBE] {label} -> {s_file}  ({len(out_rows)} rows)", flush=True)
    return s_jsonl_path


_GETSLICE_BASE_CFG = {
    "tasks": ["numina_math_probe"],
    "DEV": "cuda",
    "layer_gpu_chunk_size": 14,       # 48 GB GPU: double chunk size vs 32 GB
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
    # GetSlice's activation-capture (profiling_utils.Catcher) caches one sample per
    # [seqlen, hidden] slot and only squeezes the batch dim when it is 1, so the
    # profiling path requires batch_size=1. A larger value yields [B, seqlen, hidden]
    # tensors that cannot be assigned into the single-sample slot (RuntimeError).
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
    "seed": 3,
}


def _run_getslice_layer(
    model_path: Path,
    label: str,
    s_jsonl_path: Path,
    x_jsonl: Path,
    x_variant: str,
    exp_root: Path,
    layer: int,
    *,
    seqlen: int,
    n_probe: int,
) -> None:
    """Run GetSlice S and X for one (model, layer) pair.

    s_jsonl_path: directory containing {task}/gamma_s.jsonl
                  (i.e. GetSlice reads {s_jsonl_path}/numina_math_probe/gamma_s.jsonl)
    x_jsonl:      path to x_probe.jsonl (plain JSONL file)
    """
    gs_root = exp_root / "getslice"
    output_root = ensure_dir(gs_root / "outputs")
    config_root = ensure_dir(gs_root / "configs")
    step_root = output_root / label

    base_cfg = dict(_GETSLICE_BASE_CFG)
    base_cfg["model_seq_len"] = seqlen
    base_cfg["target_layer"] = layer

    # S probe
    s_save_path = step_root / f"layer_{layer}" / "S"
    s_mat_path = s_save_path / "numina_math_probe" / "sMat_numina_math_probe.json"
    if not s_mat_path.exists():
        cfg_s = dict(base_cfg)
        cfg_s.update({
            "model": str(model_path),
            "save_path": str(s_save_path),
            "mode": "s_only_svd",
            "s_nsamples": n_probe,
            "s_jsonl_path": str(s_jsonl_path),
            "s_jsonl_file": "gamma_s.jsonl",
        })
        cfg_path = config_root / f"{label}__layer{layer}__S.json"
        write_json(cfg_path, cfg_s)
        _run_slice(cfg_path)

    # X probe
    x_save_path = step_root / f"layer_{layer}" / f"X__{x_variant}"
    x_mat_path = x_save_path / "X" / "xMat_X.json"
    if not x_mat_path.exists():
        cfg_x = dict(base_cfg)
        cfg_x.update({
            "model": str(model_path),
            "save_path": str(x_save_path),
            "mode": "x_only_svd",
            "x_nsamples": n_probe,
            "x_jsonl_path": str(x_jsonl),
        })
        cfg_path = config_root / f"{label}__layer{layer}__X_{x_variant}.json"
        write_json(cfg_path, cfg_x)
        _run_slice(cfg_path)


def _run_slice(cfg_path: Path) -> None:
    env = dict(os.environ,
               TMPDIR="/root/autodl-tmp/pip-tmp",
               NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    cmd = [sys.executable, str(REPO_ROOT / "GetSlice/slice.py"), "--config", str(cfg_path)]
    print(f"[SLICE] {cfg_path.name}", flush=True)
    r = subprocess.run(cmd, cwd=str(REPO_ROOT / "GetSlice"), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"GetSlice failed rc={r.returncode}: {cfg_path}")


def _build_x_probe(
    ref_model: Path,
    series_tag: str,
    pool: dict,
    inputs_root: Path,
    *,
    n_probe: int,
    is_base_series: bool,
) -> Path:
    """Generate X_prompt probe from ref_model. Returns path to x_probe.jsonl."""
    x_dir = ensure_dir(inputs_root / f"X_{series_tag}")
    x_file = x_dir / "x_probe.jsonl"
    if x_file.exists():
        print(f"[SKIP] X probe {series_tag}", flush=True)
        return x_file

    probe_rows = read_jsonl(Path(pool["probe_prompts"]))[:n_probe]
    prompts = [str(r["problem"]) for r in probe_rows]
    print(f"[PROBE-X_{series_tag}] rollout {len(prompts)} from {ref_model}", flush=True)
    if is_base_series:
        plain_prompts = [f"Problem: {p}\n\nSolution:" for p in prompts]
        comps = _rollout_base(ref_model, plain_prompts)
        write_jsonl(x_file, [{"output": {"text": f"{p}\n{c}"}} for p, c in zip(prompts, comps)])
    else:
        comps = _rollout_instruct(ref_model, prompts)
        write_jsonl(x_file, [{"output": {"text": f"{p}\n{c}"}} for p, c in zip(prompts, comps)])
    return x_file


def phase4_geometry(exp_root: Path, pool: dict, models: dict[str, Path], *, smoke: bool) -> None:
    """Run GetSlice minimal geometry + weight export + principalEvidence."""
    n_probe = 4 if smoke else pool["n_probe"]
    seqlen = 64 if smoke else 512
    layers = GEOMETRY_LAYERS
    sizes = [4] if smoke else SFT_SIZES

    gs_root = ensure_dir(exp_root / "getslice")
    inputs_root = ensure_dir(gs_root / "inputs")

    instruct_labels = ["instruct_base"] + [f"instruct_sft_n{n}" for n in sizes]
    base_labels = ["base_base"] + [f"base_sft_n{n}" for n in sizes]

    # X probes (one per series, generated from untrained base)
    x_instruct = _build_x_probe(
        INSTRUCT_BASE_MODEL, "instruct", pool, inputs_root,
        n_probe=n_probe, is_base_series=False,
    )
    x_base = _build_x_probe(
        BASE_BASE_MODEL, "base", pool, inputs_root,
        n_probe=n_probe, is_base_series=True,
    )

    # S probes + GetSlice per model × layer
    for label in instruct_labels:
        s_path = _build_s_probe(label, models[label], pool, inputs_root,
                                n_probe=n_probe, is_base_series=False)
        for layer in layers:
            _run_getslice_layer(
                models[label], label, s_path, x_instruct, "instruct",
                exp_root, layer, seqlen=seqlen, n_probe=n_probe,
            )

    for label in base_labels:
        s_path = _build_s_probe(label, models[label], pool, inputs_root,
                                n_probe=n_probe, is_base_series=True)
        for layer in layers:
            _run_getslice_layer(
                models[label], label, s_path, x_base, "base",
                exp_root, layer, seqlen=seqlen, n_probe=n_probe,
            )

    # Weight export + principalEvidence (skip in smoke mode)
    if not smoke:
        _run_principal_evidence(exp_root, models, instruct_labels, base_labels, layers)


def _run_principal_evidence(
    exp_root: Path,
    models: dict[str, Path],
    instruct_labels: list[str],
    base_labels: list[str],
    layers: list[int],
) -> None:
    """Export weights and run principalEvidence for all layers and both series."""
    from scripts.export_weights import export_model_weights, MODULES

    weights_root = ensure_dir(exp_root / "weights")
    pe_root = ensure_dir(exp_root / "principal_evidence")

    sys.path.insert(0, str(REPO_ROOT))
    from AnalyseMat.principalEvidence import run_principal_evidence

    # Export base model weights (flat layout, used as base_model_npy_dir)
    instruct_base_weights = weights_root / "instruct_base"
    export_model_weights(str(INSTRUCT_BASE_MODEL), instruct_base_weights)

    base_base_weights = weights_root / "base_base"
    export_model_weights(str(BASE_BASE_MODEL), base_base_weights)

    # Export finetuned model weights (nested layout: {label}/{data_size}/)
    for label in instruct_labels:
        if label == "instruct_base":
            continue
        n = int(label.split("_n")[-1])
        out_dir = weights_root / label / str(n)
        export_model_weights(str(models[label]), out_dir)

    for label in base_labels:
        if label == "base_base":
            continue
        n = int(label.split("_n")[-1])
        out_dir = weights_root / label / str(n)
        export_model_weights(str(models[label]), out_dir)

    # Run principalEvidence for each layer and both series
    instruct_tasks = [[f"instruct_sft_n{n}", str(n)] for n in SFT_SIZES]
    base_tasks = [[f"base_sft_n{n}", str(n)] for n in SFT_SIZES]

    for layer in layers:
        # Instruct series
        cfg = {
            "analyse": {
                "base_model_npy_dir": str(instruct_base_weights),
                "npy_output_root": str(weights_root),
                "related_work": {
                    "enable": True,
                    "output_root": str(pe_root / "instruct"),
                    "target_layer": layer,
                    "target_modules": None,
                    "principal_rank_k": 50,
                    "principal_top_ratio": 0.01,
                    "save_png": True,
                },
                "tasks": instruct_tasks,
            }
        }
        print(f"[PE] instruct series layer={layer}", flush=True)
        run_principal_evidence(cfg)

        # Base series
        cfg_base = {
            "analyse": {
                "base_model_npy_dir": str(base_base_weights),
                "npy_output_root": str(weights_root),
                "related_work": {
                    "enable": True,
                    "output_root": str(pe_root / "base"),
                    "target_layer": layer,
                    "target_modules": None,
                    "principal_rank_k": 50,
                    "principal_top_ratio": 0.01,
                    "save_png": True,
                },
                "tasks": base_tasks,
            }
        }
        print(f"[PE] base series layer={layer}", flush=True)
        run_principal_evidence(cfg_base)


# =============================================================================
# Provenance
# =============================================================================
def write_provenance(exp_root: Path, models: dict[str, Path], pool: dict) -> None:
    """Write run_provenance.json with all paths, model versions, and shot specs."""
    import importlib.util

    def _load(runner_path):
        spec = importlib.util.spec_from_file_location("_runner", runner_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    gsm8k_base = _load(REPO_ROOT / "Eval/component/gsm8k/runner_base.py")
    math500_base = _load(REPO_ROOT / "Eval/component/math500/runner_base.py")
    numina_base = _load(REPO_ROOT / "Eval/component/numina/runner_base.py")

    provenance = {
        "cycle": "cycle_06_sft_feasibility_and_degradation",
        "exp_root": str(exp_root),
        "model_paths": {label: str(path) for label, path in models.items()},
        "pool": pool,
        "base_model_paths": {
            "instruct_base": str(INSTRUCT_BASE_MODEL),
            "base_base": str(BASE_BASE_MODEL),
        },
        "eval_protocol": {
            "instruct_series": "chat template + enable_thinking=False + 3072 tokens (0-shot for math)",
            "base_series": "plain text 4-shot CoT, no chat template, 3072 tokens",
            "mmlu_subtasks": MMLU_SUBTASKS,
            "mmlu_note": "6 representative subtasks (STEM/humanities/social-sci/professional); full MMLU excluded due to GPU memory constraint on 152K-vocab loglikelihood",
        },
        "base_series_shots": {
            "gsm8k": [s["problem"] for s in gsm8k_base.GSM8K_SHOTS],
            "gsm8k_source": "Wei et al. 2022 CoT paper (4 of 8 standard examples)",
            "math500": [s["problem"] for s in math500_base.MATH500_SHOTS],
            "math500_source": "Standard Hendrycks MATH benchmark CoT examples",
            "numina_shot_row_indices": list(range(numina_base.N_SHOTS)),
            "numina_source": f"First {numina_base.N_SHOTS} rows of {NUMINA_TEST} (excluded from scoring)",
        },
        "geometry_layers": GEOMETRY_LAYERS,
        "sft_sizes": SFT_SIZES,
    }

    logs_dir = ensure_dir(exp_root / "logs")
    write_json(logs_dir / "run_provenance.json", provenance)
    print(f"[PROV] written -> {logs_dir}/run_provenance.json", flush=True)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Cycle 06: SFT Feasibility and Degradation")
    ap.add_argument("--exp-root", type=Path, default=EXP_ROOT_DEFAULT,
                    help=f"Experiment root directory (default: {EXP_ROOT_DEFAULT})")
    ap.add_argument("--smoke", action="store_true",
                    help="Small-scale run for pipeline verification")
    ap.add_argument("--start-from-phase", type=int, default=1, choices=[1, 2, 3, 4],
                    help="Skip phases before this number (useful after a crash)")
    args = ap.parse_args()

    exp_root: Path = args.exp_root
    ensure_dir(exp_root)
    smoke: bool = args.smoke
    start: int = args.start_from_phase

    print(f"=== Cycle 06 === exp_root={exp_root}  smoke={smoke}  start_phase={start}", flush=True)

    # Verify model paths
    if not INSTRUCT_BASE_MODEL.exists():
        raise FileNotFoundError(f"Instruct base model not found: {INSTRUCT_BASE_MODEL}")
    if not BASE_BASE_MODEL.exists():
        raise FileNotFoundError(f"Base model not found: {BASE_BASE_MODEL}")
    print(f"[OK] instruct_base: {INSTRUCT_BASE_MODEL}", flush=True)
    print(f"[OK] base_base:     {BASE_BASE_MODEL}", flush=True)

    # Phase 1
    pool: dict = {}
    if start <= 1:
        print("\n=== Phase 1: Data Pool ===", flush=True)
        pool = phase1_build_pool(exp_root, smoke=smoke)
    else:
        meta_path = exp_root / "pool" / "pool_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"pool_meta.json not found (required when --start-from-phase > 1): {meta_path}")
        pool = json.loads(meta_path.read_text())
        print(f"[POOL] loaded from {meta_path}", flush=True)

    # Phase 2
    models: dict[str, Path] = {}
    if start <= 2:
        print("\n=== Phase 2: Training ===", flush=True)
        models = phase2_train(exp_root, pool, smoke=smoke)
    else:
        # Reconstruct model paths from expected locations
        models = {"instruct_base": INSTRUCT_BASE_MODEL, "base_base": BASE_BASE_MODEL}
        sizes = [4] if smoke else SFT_SIZES
        for n in sizes:
            for series in ["instruct", "base"]:
                label = f"{series}_sft_n{n}"
                merged = exp_root / label / "merged_model"
                if not (merged / "config.json").exists():
                    raise FileNotFoundError(f"Merged model not found: {merged}")
                models[label] = merged
        print(f"[MODELS] located {len(models)} model paths", flush=True)

    # Phase 3
    if start <= 3:
        print("\n=== Phase 3: Evaluation ===", flush=True)
        phase3_eval(exp_root, models, smoke=smoke)

    # Phase 4
    if start <= 4:
        print("\n=== Phase 4: Geometry ===", flush=True)
        phase4_geometry(exp_root, pool, models, smoke=smoke)

    # Provenance
    write_provenance(exp_root, models, pool)
    print("\n=== Cycle 06 complete ===", flush=True)


if __name__ == "__main__":
    main()
