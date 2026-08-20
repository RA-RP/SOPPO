#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path("/root/LLM-output-density")
SIDECAR_ROOT = REPO_ROOT / "experiments/opd_sft_h1"
EXP_ROOT_DEFAULT = Path("/root/autodl-tmp/exp0609/opd_minimal_03")
BASE_MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen3-1.7B")
NUMINA_PARQUET = Path("/root/autodl-tmp/prepared/NuminaMath-1___5/train.parquet")
REFERENCE_X_JSONL = REPO_ROOT / "MyFunc/dataset/X/Qwen.jsonl"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from Eval.allRunEval_vLLM import run_eval_vllm  # noqa: E402
from Eval.component.Eval2Res import eval2res  # noqa: E402
from opd_sft_h1.geometry_metrics import (  # noqa: E402
    effective_rank,
    log_spectrum_drift,
    spectral_gap,
    xs_log_spectrum_gap,
)
from opd_sft_h1.geometry_reader import read_geometry_rows  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    source: str
    size: str
    role: str
    model_dir: Path
    checkpoint_path: Path

    @property
    def eval_task(self) -> dict[str, str]:
        return {"dataset": self.source, "max_samples": self.size}

    @property
    def checkpoint_id(self) -> str:
        return f"{self.source}_{self.size}"


def run(cmd: list[str | Path], *, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None) -> None:
    display = " ".join(str(part) for part in cmd)
    print(f"\n[RUN] {display}", flush=True)
    merged_env = os.environ.copy()
    merged_env.update(
        {
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "TMPDIR": "/root/autodl-tmp/pip-tmp",
            "HF_HOME": "/root/.cache/huggingface",
            "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
        }
    )
    if env:
        merged_env.update(env)
    result = subprocess.run([str(part) for part in cmd], cwd=str(cwd), env=merged_env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with rc={result.returncode}: {display}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def symlink_model(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src, target_is_directory=True)


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def assert_checkpoint(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    if not (path / "adapter_config.json").exists() and not (path / "config.json").exists():
        raise FileNotFoundError(f"{label} is not a loadable checkpoint/model directory: {path}")


def merge_lora_adapter(base_model_dir: Path, adapter_dir: Path, merged_dir: Path) -> Path:
    if (merged_dir / "config.json").exists() and (
        (merged_dir / "model.safetensors").exists() or (merged_dir / "model.safetensors.index.json").exists()
    ):
        print(f"[SKIP] merged model exists: {merged_dir}", flush=True)
        return merged_dir

    print(f"[MERGE] base={base_model_dir} adapter={adapter_dir} -> {merged_dir}", flush=True)
    ensure_dir(merged_dir.parent)
    if merged_dir.exists():
        shutil.rmtree(merged_dir)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_dir))
    del merged, model, base
    cleanup_cuda()
    return merged_dir


def prepare_supervised_sft_data(exp_root: Path, num_samples: int = 1024) -> Path:
    out_dir = ensure_dir(exp_root / "data" / f"continued_sft_{num_samples}")
    train_jsonl = out_dir / "train.jsonl"
    if train_jsonl.exists():
        return train_jsonl

    df = pd.read_parquet(NUMINA_PARQUET)
    df = df[df["problem"].notna() & df["solution"].notna()].sample(n=num_samples, random_state=43)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        problem = str(row["problem"])
        solution = str(row["solution"])
        answer = str(row.get("answer") or "").strip()
        if answer and answer not in solution:
            solution = f"{solution}\n\nFinal answer: {answer}"
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ]
            }
        )
    write_jsonl(train_jsonl, rows)
    print(f"[DATA] continued SFT rows={len(rows)} -> {train_jsonl}", flush=True)
    return train_jsonl


def _sft_config_kwargs(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    accepted = set(signature.parameters)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in accepted}


def train_continued_sft(
    *,
    exp_root: Path,
    label: str,
    theta0_merged: Path,
    train_jsonl: Path,
    learning_rate: float,
    num_train_epochs: float = 3.0,
) -> ModelSpec:
    output_root = ensure_dir(exp_root / "step4_sft_controls" / label)
    adapter_dir = output_root / "checkpoint_output"
    merged_dir = output_root / "merged_model"
    if (merged_dir / "config.json").exists():
        print(f"[SKIP] SFT {label} merged model exists: {merged_dir}", flush=True)
        return ModelSpec(label, "1024", "sft", merged_dir, adapter_dir)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(str(theta0_merged), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=str(train_jsonl), split="train")

    def to_text(record: dict[str, Any]) -> dict[str, str]:
        return {
            "text": tokenizer.apply_chat_template(
                record["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    dataset = dataset.map(to_text, remove_columns=dataset.column_names)

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
        "save_strategy": "steps",
        "save_steps": 32,
        "save_total_limit": 3,
        "logging_steps": 8,
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
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    trainer = SFTTrainer(
        model=str(theta0_merged),
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    del trainer
    cleanup_cuda()

    merge_lora_adapter(theta0_merged, adapter_dir, merged_dir)
    return ModelSpec(label, "1024", "sft", merged_dir, adapter_dir)


def materialize_existing_models(exp_root: Path) -> tuple[ModelSpec, ModelSpec]:
    cold_adapter = exp_root / "step2_cold_start/checkpoint_output"
    opd_adapter = exp_root / "step3_opd_distill/checkpoint_output"
    assert_checkpoint(cold_adapter, "theta0 adapter")
    assert_checkpoint(opd_adapter, "OPD adapter")

    theta0_merged = cold_adapter / "merged_model"
    merge_lora_adapter(BASE_MODEL, cold_adapter, theta0_merged)

    opd_merged = opd_adapter / "merged_model"
    merge_lora_adapter(theta0_merged, opd_adapter, opd_merged)

    theta0 = ModelSpec("theta0", "20", "theta0", theta0_merged, cold_adapter)
    opd = ModelSpec("opd_lmbda1", "50", "opd", opd_merged, opd_adapter)
    return theta0, opd


TARGET_METRICS = [
    {"record_name": "GSM8K", "json_task": "gsm8k", "field": "results", "task_key": "gsm8k", "metric": "exact_match,flexible-extract"},
    {"record_name": "MATH500", "json_task": "hendrycks_math500", "field": "results", "task_key": "hendrycks_math500", "metric": "exact_match,none"},
    {"record_name": "MMLU", "json_task": "mmlu", "field": "results", "task_key": "mmlu", "metric": "acc,none"},
    {"record_name": "MMLU-STEM", "json_task": "mmlu", "field": "groups", "task_key": "mmlu_stem", "metric": "acc,none"},
    {"record_name": "MMLU-Humanities", "json_task": "mmlu", "field": "groups", "task_key": "mmlu_humanities", "metric": "acc,none"},
    {"record_name": "MMLU-Social Sciences", "json_task": "mmlu", "field": "groups", "task_key": "mmlu_social_sciences", "metric": "acc,none"},
    {"record_name": "MMLU-Other", "json_task": "mmlu", "field": "groups", "task_key": "mmlu_other", "metric": "acc,none"},
    {"record_name": "TruthfulQA-MC1", "json_task": "truthfulqa_mc1", "field": "results", "task_key": "truthfulqa_mc1", "metric": "acc,none"},
    {"record_name": "TruthfulQA-MC2", "json_task": "truthfulqa_mc2", "field": "results", "task_key": "truthfulqa_mc2", "metric": "acc,none"},
    {"record_name": "WinoGrande", "json_task": "winogrande", "field": "results", "task_key": "winogrande", "metric": "acc,none"},
]


def link_eval_models(exp_root: Path, specs: list[ModelSpec]) -> Path:
    model_root = ensure_dir(exp_root / "model_outputs")
    for spec in specs:
        symlink_model(spec.model_dir, model_root / spec.source / spec.size)
    return model_root


def run_full_eval(exp_root: Path, specs: list[ModelSpec]) -> Path:
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


OOD_BENCHMARKS = [
    "MMLU",
    "MMLU-STEM",
    "MMLU-Humanities",
    "MMLU-Social Sciences",
    "MMLU-Other",
    "TruthfulQA-MC1",
    "TruthfulQA-MC2",
    "WinoGrande",
]


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def build_eval_tables(exp_root: Path, specs: list[ModelSpec], metrics_csv: Path) -> dict[str, Path | dict[str, Any]]:
    spec_by_source = {spec.source: spec for spec in specs}
    df = pd.read_csv(metrics_csv, encoding="utf-8-sig")
    baseline = df[df["Source"].astype(str) == "theta0"]
    if baseline.empty:
        raise ValueError("No theta0 baseline row found in target metrics")
    base = baseline.iloc[0].to_dict()

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
        row["GSM8K_gain"] = None if _as_float(row.get("GSM8K")) is None else _as_float(row.get("GSM8K")) - _as_float(base.get("GSM8K"))
        row["MATH500_gain"] = None if _as_float(row.get("MATH500")) is None else _as_float(row.get("MATH500")) - _as_float(base.get("MATH500"))
        drops = []
        for benchmark in OOD_BENCHMARKS:
            score = _as_float(row.get(benchmark))
            base_score = _as_float(base.get(benchmark))
            drop = None if score is None or base_score is None else max(0.0, base_score - score)
            ood_rows.append(
                {
                    "checkpoint_id": row["checkpoint_id"],
                    "Source": source,
                    "benchmark": benchmark,
                    "score": score,
                    "baseline_score": base_score,
                    "drop": drop,
                }
            )
            if drop is not None:
                drops.append(drop)
        if drops:
            row["OOD_lite_avg"] = sum(_as_float(row.get(b)) for b in OOD_BENCHMARKS if _as_float(row.get(b)) is not None) / len(
                [b for b in OOD_BENCHMARKS if _as_float(row.get(b)) is not None]
            )
            row["OOD_lite_penalty_p2"] = sum(drop**2 for drop in drops) ** 0.5
            row["OOD_lite_penalty_p3"] = sum(drop**3 for drop in drops) ** (1.0 / 3.0)
            row["Worst_OOD_lite_drop"] = max(drops)
        else:
            row["OOD_lite_avg"] = None
            row["OOD_lite_penalty_p2"] = None
            row["OOD_lite_penalty_p3"] = None
            row["Worst_OOD_lite_drop"] = None
        eval_rows.append(row)

    tables_dir = ensure_dir(exp_root / "tables")
    eval_path = tables_dir / "eval_trajectory.csv"
    ood_path = tables_dir / "ood_penalty.csv"
    write_csv(eval_path, eval_rows)
    write_csv(ood_path, ood_rows)

    opd_rows = [row for row in eval_rows if row["role"] == "opd"]
    sft_rows = [row for row in eval_rows if row["role"] == "sft"]
    if not opd_rows:
        raise ValueError("No OPD rows available for selection")
    selected_opd = max(opd_rows, key=lambda row: _as_float(row.get("GSM8K")) if _as_float(row.get("GSM8K")) is not None else -1)
    opd_gain = _as_float(selected_opd.get("GSM8K_gain"))
    selected_sft = None
    best_gap = None
    for row in sft_rows:
        gain = _as_float(row.get("GSM8K_gain"))
        if gain is None or opd_gain is None:
            continue
        gap = abs(gain - opd_gain)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            selected_sft = row
    threshold = 0.02
    matched_path = tables_dir / "matched_gsm8k_pairs.csv"
    match_row = {
        "opd_run_id": selected_opd["Source"],
        "opd_checkpoint_id": selected_opd["checkpoint_id"],
        "opd_GSM8K_gain": selected_opd.get("GSM8K_gain"),
        "sft_run_id": selected_sft["Source"] if selected_sft else None,
        "sft_checkpoint_id": selected_sft["checkpoint_id"] if selected_sft else None,
        "sft_GSM8K_gain": selected_sft.get("GSM8K_gain") if selected_sft else None,
        "GSM8K_gain_gap": best_gap,
        "match_status": "valid_match" if best_gap is not None and best_gap <= threshold else ("nearest_match" if selected_sft else "no_sft_candidate"),
        "OOD_lite_penalty_p2_delta": (
            _as_float(selected_opd.get("OOD_lite_penalty_p2")) - _as_float(selected_sft.get("OOD_lite_penalty_p2"))
            if selected_sft and _as_float(selected_opd.get("OOD_lite_penalty_p2")) is not None and _as_float(selected_sft.get("OOD_lite_penalty_p2")) is not None
            else None
        ),
        "Worst_OOD_lite_drop_delta": (
            _as_float(selected_opd.get("Worst_OOD_lite_drop")) - _as_float(selected_sft.get("Worst_OOD_lite_drop"))
            if selected_sft and _as_float(selected_opd.get("Worst_OOD_lite_drop")) is not None and _as_float(selected_sft.get("Worst_OOD_lite_drop")) is not None
            else None
        ),
    }
    write_csv(matched_path, [match_row])

    main_result_path = tables_dir / "main_matched_result.csv"
    write_csv(main_result_path, [{**match_row, "selection_note": "best OPD and nearest SFT selected by GSM8K_gain"}])
    selection = {
        "opd": selected_opd,
        "sft": selected_sft,
        "match": match_row,
        "baseline": base,
    }
    write_json(exp_root / "selection.json", selection)
    return {
        "eval_trajectory": eval_path,
        "ood_penalty": ood_path,
        "matched_gsm8k_pairs": matched_path,
        "main_matched_result": main_result_path,
        "selection": selection,
    }


def prepare_getslice_inputs(exp_root: Path, num_samples: int = 96) -> tuple[Path, Path]:
    inputs_root = ensure_dir(exp_root / "getslice" / "inputs")
    s_dir = ensure_dir(inputs_root / "S" / "numina_math_probe")
    s_jsonl = s_dir / "gamma_s.jsonl"
    if not s_jsonl.exists():
        df = pd.read_parquet(NUMINA_PARQUET)
        df = df[df["problem"].notna() & df["solution"].notna()].sample(n=num_samples, random_state=44)
        rows = [{"question": str(row["problem"]), "answer": str(row["solution"])} for _, row in df.iterrows()]
        write_jsonl(s_jsonl, rows)
    if not REFERENCE_X_JSONL.exists():
        raise FileNotFoundError(f"Reference X JSONL missing: {REFERENCE_X_JSONL}")
    return s_dir.parent, REFERENCE_X_JSONL


def run_getslice_for_selected(exp_root: Path, selected_specs: list[ModelSpec]) -> Path:
    s_root, x_jsonl = prepare_getslice_inputs(exp_root)
    output_root = ensure_dir(exp_root / "getslice" / "outputs")
    config_root = ensure_dir(exp_root / "getslice" / "configs")
    for spec in selected_specs:
        save_path = output_root / spec.source / f"step_{spec.size}"
        expected_s = save_path / "numina_math_probe" / "sMat_numina_math_probe.json"
        expected_x = save_path / "X" / "xMat_X.json"
        if expected_s.exists() and expected_x.exists():
            print(f"[SKIP] GetSlice exists for {spec.source}: {save_path}", flush=True)
            continue
        cfg = {
            "model": str(spec.model_dir),
            "save_path": str(save_path),
            "tasks": ["numina_math_probe"],
            "mode": "split_whitened_svd",
            "DEV": "cuda",
            "model_seq_len": 1024,
            "seed": 3,
            "target_layer": 14,
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
            "s_nsamples": 32,
            "s_jsonl_path": str(s_root),
            "s_jsonl_file": "gamma_s.jsonl",
            "s_batch_size": 1,
            "x_nsamples": 32,
            "x_jsonl_path": str(x_jsonl),
            "x_batch_size": 1,
            "save_s_json_path": "sMat_{task}.json",
            "save_x_json_path": "xMat_{task}.json",
            "save_s_pt_path": None,
            "save_x_pt_path": None,
            "save_s_uv_path": None,
            "save_x_uv_path": None,
            "save_metrics_pt_path": None,
            "save_metrics_json_path": None,
        }
        cfg_path = config_root / f"{spec.source}.json"
        write_json(cfg_path, cfg)
        run([sys.executable, REPO_ROOT / "GetSlice/slice.py", "--config", cfg_path], cwd=REPO_ROOT / "GetSlice")
    return output_root


def build_geometry_tables(exp_root: Path, getslice_root: Path) -> Path:
    rows = read_geometry_rows(getslice_root)
    tables_dir = ensure_dir(exp_root / "tables")
    long_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    baseline_by_key: dict[tuple[str, str, str], list[float]] = {}
    xs_by_source_layer_module: dict[tuple[str, str, str], list[float]] = {}

    for row in rows:
        source = str(row["source"]).split("/", 1)[0]
        key = (row["probe_distribution"], row["layer"], row["module"])
        if source == "theta0":
            baseline_by_key[key] = row["singular_values"]
        if row["probe_distribution"] == "X":
            xs_by_source_layer_module[(source, row["layer"], row["module"])] = row["singular_values"]

    for row in rows:
        source = str(row["source"]).split("/", 1)[0]
        sigma = row["singular_values"]
        long_rows.append({**row, "source": source, "singular_values": json.dumps(sigma)})
        base_sigma = baseline_by_key.get((row["probe_distribution"], row["layer"], row["module"]))
        x_sigma = xs_by_source_layer_module.get((source, row["layer"], row["module"]))
        metric_rows.append(
            {
                "source": source,
                "step": row["step"],
                "probe_distribution": row["probe_distribution"],
                "probe_source": row["probe_source"],
                "layer": row["layer"],
                "module": row["module"],
                "effective_rank": effective_rank(sigma),
                "spectral_gap_k": 1,
                "spectral_gap": spectral_gap(sigma, 1),
                "spectral_gap_abs_delta_from_start": (
                    None
                    if base_sigma is None or spectral_gap(sigma, 1) is None or spectral_gap(base_sigma, 1) is None
                    else abs(spectral_gap(sigma, 1) - spectral_gap(base_sigma, 1))
                ),
                "spectral_drift_from_start": None if base_sigma is None else log_spectrum_drift(sigma, base_sigma),
                "X_S_spectrum_level_gap": (
                    xs_log_spectrum_gap(x_sigma, sigma)
                    if row["probe_distribution"] == "S" and x_sigma is not None
                    else None
                ),
                "principal_angle": None,
                "principal_angle_status": "unavailable_no_uv",
                "singular_json_path": row["singular_json_path"],
            }
        )

    geometry_long = tables_dir / "geometry_long.csv"
    geometry_metrics = tables_dir / "geometry_metrics.csv"
    write_csv(geometry_long, long_rows)
    write_csv(geometry_metrics, metric_rows)
    return geometry_metrics


def build_figures(exp_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = ensure_dir(exp_root / "figures")
    eval_df = pd.read_csv(exp_root / "tables/eval_trajectory.csv", encoding="utf-8-sig")
    geom_df = pd.read_csv(exp_root / "tables/geometry_metrics.csv", encoding="utf-8-sig")

    s_geom = geom_df[geom_df["probe_distribution"] == "S"].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    for source, group in s_geom.groupby("source"):
        ax.plot(group["module"].astype(str), group["spectral_gap"], marker="o", label=source)
    ax.set_ylabel("spectral_gap")
    ax.set_xlabel("module")
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures_dir / "main_geometry_trajectory.png", dpi=160)
    plt.close(fig)

    merged = eval_df[["Source", "OOD_lite_penalty_p2"]].merge(
        s_geom.groupby("source", as_index=False)["spectral_drift_from_start"].mean(),
        left_on="Source",
        right_on="source",
        how="left",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(merged["spectral_drift_from_start"], merged["OOD_lite_penalty_p2"])
    for _, row in merged.iterrows():
        ax.annotate(str(row["Source"]), (row["spectral_drift_from_start"], row["OOD_lite_penalty_p2"]), fontsize=8)
    ax.set_xlabel("mean spectral drift")
    ax.set_ylabel("OOD_lite_penalty_p2")
    fig.tight_layout()
    fig.savefig(figures_dir / "main_ood_vs_geometry.png", dpi=160)
    plt.close(fig)

    pivot = s_geom.pivot_table(index="source", columns="module", values="effective_rank", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    fig.colorbar(im, ax=ax, label="effective_rank")
    fig.tight_layout()
    fig.savefig(figures_dir / "appendix_all_layer_heatmap.png", dpi=160)
    plt.close(fig)


def write_registries_and_summary(exp_root: Path, specs: list[ModelSpec], artifacts: dict[str, Any]) -> None:
    registry_dir = ensure_dir(exp_root / "registry")
    run_rows = []
    checkpoint_rows = []
    for spec in specs:
        record = {
            "run_id": spec.checkpoint_id,
            "checkpoint_id": spec.checkpoint_id,
            "trajectory_group_id": "opd_minimal_03",
            "method": spec.role,
            "role_label": spec.source,
            "checkpoint_path": str(spec.checkpoint_path),
            "model_dir": str(spec.model_dir),
            "status": "completed",
            "artifacts": {
                "eval_trajectory_csv": str(exp_root / "tables/eval_trajectory.csv"),
                "geometry_metrics_csv": str(exp_root / "tables/geometry_metrics.csv"),
            },
        }
        run_rows.append(record)
        checkpoint_rows.append(record)
    write_jsonl(registry_dir / "run_registry.jsonl", run_rows)
    write_jsonl(registry_dir / "checkpoints.jsonl", checkpoint_rows)
    summary = {
        "experiment": "opd_minimal_03_closure",
        "status": "completed",
        "models": [spec.__dict__ | {"model_dir": str(spec.model_dir), "checkpoint_path": str(spec.checkpoint_path)} for spec in specs],
        "artifacts": {key: str(value) if isinstance(value, Path) else value for key, value in artifacts.items()},
    }
    write_json(exp_root / "summary.json", summary)


def shutdown_now() -> None:
    subprocess.run(["/usr/bin/shutdown", "-h", "now"], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finish OPD minimal experiment 03 closure")
    parser.add_argument("--exp-root", default=str(EXP_ROOT_DEFAULT))
    parser.add_argument("--shutdown-on-exit", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-getslice", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TMPDIR", "/root/autodl-tmp/pip-tmp")
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    ensure_dir(Path(os.environ["TMPDIR"]))

    exp_root = Path(args.exp_root).expanduser().resolve()
    artifacts: dict[str, Any] = {}
    try:
        theta0, opd = materialize_existing_models(exp_root)
        train_jsonl = prepare_supervised_sft_data(exp_root)
        if args.skip_train:
            sft_mid = ModelSpec("sft_lr_mid", "1024", "sft", exp_root / "step4_sft_controls/sft_lr_mid/merged_model", exp_root / "step4_sft_controls/sft_lr_mid/checkpoint_output")
            sft_low = ModelSpec("sft_lr_low", "1024", "sft", exp_root / "step4_sft_controls/sft_lr_low/merged_model", exp_root / "step4_sft_controls/sft_lr_low/checkpoint_output")
            for spec in (sft_mid, sft_low):
                if not (spec.model_dir / "config.json").exists():
                    raise FileNotFoundError(f"--skip-train requested but missing {spec.model_dir}")
        else:
            sft_mid = train_continued_sft(exp_root=exp_root, label="sft_lr_mid", theta0_merged=theta0.model_dir, train_jsonl=train_jsonl, learning_rate=3.0e-5)
            sft_low = train_continued_sft(exp_root=exp_root, label="sft_lr_low", theta0_merged=theta0.model_dir, train_jsonl=train_jsonl, learning_rate=1.0e-5)

        specs = [theta0, opd, sft_mid, sft_low]
        if args.skip_eval:
            metrics_csv = exp_root / "eval/csv_results/target_metrics_results.csv"
            if not metrics_csv.exists():
                raise FileNotFoundError(f"--skip-eval requested but missing {metrics_csv}")
        else:
            metrics_csv = run_full_eval(exp_root, specs)
        artifacts["target_metrics_csv"] = metrics_csv
        artifacts.update(build_eval_tables(exp_root, specs, metrics_csv))

        selection = artifacts["selection"]
        selected_sources = {"theta0", selection["opd"]["Source"]}
        if selection.get("sft"):
            selected_sources.add(selection["sft"]["Source"])
        selected_specs = [spec for spec in specs if spec.source in selected_sources]

        if args.skip_getslice:
            geometry_metrics = exp_root / "tables/geometry_metrics.csv"
            if not geometry_metrics.exists():
                raise FileNotFoundError(f"--skip-getslice requested but missing {geometry_metrics}")
        else:
            getslice_root = run_getslice_for_selected(exp_root, selected_specs)
            geometry_metrics = build_geometry_tables(exp_root, getslice_root)
        artifacts["geometry_metrics"] = geometry_metrics
        build_figures(exp_root)
        write_registries_and_summary(exp_root, specs, artifacts)
        print(f"[DONE] OPD minimal closure completed: {exp_root}", flush=True)
    except Exception as exc:
        write_json(exp_root / "closure_failed.json", {"error": repr(exc)})
        print(f"[FAILED] {exc!r}", flush=True)
        if args.shutdown_on_exit:
            shutdown_now()
        raise
    if args.shutdown_on_exit:
        shutdown_now()


if __name__ == "__main__":
    main()
