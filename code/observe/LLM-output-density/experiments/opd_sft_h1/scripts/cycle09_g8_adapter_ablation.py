#!/usr/bin/env python3
"""Cycle 09 block 2 G8: off-KD@624 adapter layer-group ablation."""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import cycle09_s1_12_mmlupro as mmlu_audit


REPO = Path("/root/LLM-output-density")
BASE = Path("/root/autodl-tmp/model/Qwen/Qwen3-4B-Base")
ADAPTER = Path("/root/autodl-tmp/cycle09_offkd/checkpoints/checkpoint-000624")
ALL_OPEN = Path("/root/autodl-tmp/cycle09_offkd/_merged_models/step_624")
ROOT = Path("/root/autodl-tmp/cycle09_g8")
BLOCK_ROOT = Path("/root/autodl-tmp/cycle09_block2")
MERGED = ROOT / "merged_models"
RESULTS = ROOT / "results"
SMOKE_RESULTS = ROOT / "smoke" / "results"
MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
S1_LOG_MANIFEST = MINI / "S1_mmlupro_log_manifest.json"
OUTPUT = MINI / "G8_adapter_ablation.csv"
MANIFEST = MINI / "G8_adapter_ablation_manifest.json"
LAYER_GROUPS = (
    ("close_00_05", tuple(range(0, 6))),
    ("close_06_11", tuple(range(6, 12))),
    ("close_12_17", tuple(range(12, 18))),
    ("close_18_23", tuple(range(18, 24))),
    ("close_24_29", tuple(range(24, 30))),
    ("close_30_35", tuple(range(30, 36))),
)
CONFIGS = ("all_open", *(name for name, _ in LAYER_GROUPS), "all_closed")
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker-only", action="store_true")
    parser.add_argument("--worker-id", default="gpu_worker")
    args = parser.parse_args()
    args.configs = tuple(item.strip() for item in args.configs.split(",") if item.strip())
    unknown = sorted(set(args.configs) - set(CONFIGS))
    if not args.configs or unknown:
        parser.error(f"invalid configs; unknown={unknown}")
    if args.smoke:
        args.configs = ("all_closed",)
    return args


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def acquire_config_lock(config: str):
    path = ROOT / "locks" / f"{config}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def write_worker_status(
    args: argparse.Namespace,
    status: str,
    *,
    completed_configs: list[str],
    current_config: str | None = None,
) -> None:
    if not args.worker_only:
        return
    atomic_json(
        BLOCK_ROOT / f"g8_{args.worker_id}_status.json",
        {
            "status": status,
            "worker_id": args.worker_id,
            "pid": os.getpid(),
            "configs": list(args.configs),
            "completed_configs": completed_configs,
            "current_config": current_config,
            "shared_outputs_written": False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def closed_layers(config: str) -> tuple[int, ...]:
    if config == "all_open":
        return ()
    if config == "all_closed":
        return tuple(range(36))
    return dict(LAYER_GROUPS)[config]


def preflight() -> None:
    required = (
        BASE / "config.json",
        ADAPTER / "adapter_config.json",
        ADAPTER / "adapter_model.safetensors",
        ALL_OPEN / "config.json",
        S1_LOG_MANIFEST,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    ROOT.mkdir(parents=True, exist_ok=True)
    MERGED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def model_path(config: str) -> Path:
    if config == "all_open":
        return ALL_OPEN
    if config == "all_closed":
        return BASE
    return MERGED / config


def build_selective_merged(config: str) -> Path:
    target = model_path(config)
    if config in {"all_open", "all_closed"}:
        return target
    if (target / "config.json").is_file() and list(target.glob("model*.safetensors")):
        print(f"[G8 merged cached] {config}", flush=True)
        return target

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    layers = set(closed_layers(config))
    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, ADAPTER, is_trainable=False)
    zeroed = 0
    seen_layers = set()
    pattern = re.compile(r"\.layers\.(\d+)\.")
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        match = pattern.search(name)
        if match and int(match.group(1)) in layers:
            parameter.data.zero_()
            zeroed += parameter.numel()
            seen_layers.add(int(match.group(1)))
    if seen_layers != layers or zeroed == 0:
        raise RuntimeError(
            f"selective zero mismatch {config}: seen={sorted(seen_layers)} zeroed={zeroed}"
        )
    merged = model.merge_and_unload(safe_merge=True)
    target.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(target, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(BASE).save_pretrained(target)
    atomic_json(
        target / "g8_ablation_provenance.json",
        {
            "config": config,
            "closed_layers": sorted(layers),
            "adapter": str(ADAPTER),
            "adapter_sha256": sha256_file(ADAPTER / "adapter_model.safetensors"),
            "zeroed_lora_parameters": zeroed,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    del merged, model, base
    gc.collect()
    print(f"[G8 merged] {config} layers={sorted(layers)}", flush=True)
    return target


def mmlu_pool() -> list[dict[str, Any]]:
    manifest = json.loads(S1_LOG_MANIFEST.read_text(encoding="utf-8"))
    base_cell = next(
        cell for cell in manifest["cells"] if cell["arm"] == "opd" and int(cell["step"]) == 0
    )
    rows = []
    for file_info in sorted(base_cell["sample_files"], key=lambda item: item["subject"]):
        with Path(file_info["path"]).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                rows.append(
                    {
                        "subject": file_info["subject"],
                        "doc_id": int(record["doc_id"]),
                        "prompt": record["arguments"]["gen_args_0"]["arg_0"],
                        "target": str(record["target"]).upper(),
                        "doc_hash": record.get("doc_hash"),
                    }
                )
    if len(rows) != 1400:
        raise RuntimeError(f"MMLU pool rows={len(rows)}/1400")
    rows.sort(key=lambda row: (row["subject"], row["doc_id"]))
    return rows


def fixed_mmlu_subset(n: int) -> tuple[list[dict[str, Any]], str]:
    pool = mmlu_pool()
    rng = np.random.default_rng(SEED)
    indices = sorted(int(value) for value in rng.choice(len(pool), size=n, replace=False))
    selected = [pool[index] for index in indices]
    fingerprint = sha256_bytes(
        json.dumps(
            [
                {
                    "subject": row["subject"],
                    "doc_id": row["doc_id"],
                    "doc_hash": row["doc_hash"],
                    "target": row["target"],
                }
                for row in selected
            ],
            sort_keys=True,
        ).encode()
    )
    return selected, fingerprint


def run_config(
    config: str,
    path: Path,
    *,
    gpu_mem: float,
    math_n: int,
    mmlu_rows: list[dict[str, Any]],
    subset_fingerprint: str,
    results_root: Path,
) -> dict[str, Any]:
    output_dir = results_root / config
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("mmlu_subset_fingerprint") != subset_fingerprint:
            raise RuntimeError(f"stale G8 subset cache: {summary_path}")
        print(f"[G8 cached] {config}", flush=True)
        return summary
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPO / "Eval/component/think_math"))

    import runner_think
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    math_rows = runner_think._load_math500(math_n)
    math_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["problem"] + runner_think.INSTR}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in math_rows
    ]
    llm = LLM(
        model=str(path),
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=10240,
        trust_remote_code=True,
    )
    math_out = llm.generate(
        math_prompts,
        SamplingParams(temperature=0.6, top_p=0.9, max_tokens=8192, seed=SEED),
    )
    mmlu_out = llm.generate(
        [row["prompt"] for row in mmlu_rows],
        SamplingParams(temperature=0.0, max_tokens=2048, stop=["Question:"], seed=SEED),
    )
    runner_think._shutdown_llm(llm)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    math_scores, math_score_statuses = runner_think._score_generations(
        [
            {"gen": generated.outputs[0].text, "gold": row["answer"]}
            for row, generated in zip(math_rows, math_out)
        ]
    )
    math_correct = math_trunc = 0
    math_records = []
    for row, generated, ok, score_status in zip(
        math_rows, math_out, math_scores, math_score_statuses
    ):
        candidate = generated.outputs[0]
        math_correct += int(ok)
        math_trunc += int(candidate.finish_reason == "length")
        math_records.append(
            {
                "gold": row["answer"],
                "pred": runner_think.extract_pred(candidate.text),
                "ok": ok,
                "score_status": score_status,
                "finish_reason": candidate.finish_reason,
                "n_tokens": len(candidate.token_ids),
                "generation": candidate.text,
            }
        )

    strict_fail = strict_correct = flexible_correct = 0
    mmlu_records = []
    for row, generated in zip(mmlu_rows, mmlu_out):
        candidate = generated.outputs[0]
        strict_match = mmlu_audit.STRICT_RE.search(candidate.text)
        strict_prediction = strict_match.group(1) if strict_match else None
        flexible_prediction, tier = mmlu_audit.flexible_extract(candidate.text)
        strict_fail += int(strict_prediction is None)
        strict_correct += int(strict_prediction == row["target"])
        flexible_correct += int(flexible_prediction == row["target"])
        mmlu_records.append(
            {
                **row,
                "strict_prediction": strict_prediction,
                "flexible_prediction": flexible_prediction,
                "flexible_tier": tier,
                "finish_reason": candidate.finish_reason,
                "n_tokens": len(candidate.token_ids),
                "generation": candidate.text,
            }
        )

    with (output_dir / "math500_samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in math_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "mmlupro_samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in mmlu_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "config": config,
        "closed_layers": list(closed_layers(config)),
        "model_path": str(path),
        "math_n": len(math_rows),
        "math_acc": math_correct / len(math_rows),
        "math_trunc_rate": math_trunc / len(math_rows),
        "math_score_timeout_count": sum(
            status == "timeout" for status in math_score_statuses
        ),
        "math_score_error_count": sum(
            status != "ok" and status != "timeout"
            for status in math_score_statuses
        ),
        "mmlu_n": len(mmlu_rows),
        "mmlu_strict_fail_rate": strict_fail / len(mmlu_rows),
        "mmlu_strict_acc": strict_correct / len(mmlu_rows),
        "mmlu_flexible_acc": flexible_correct / len(mmlu_rows),
        "mmlu_subset_seed": SEED,
        "mmlu_subset_fingerprint": subset_fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    preflight()
    math_n = 2 if args.smoke else 200
    mmlu_n = 5 if args.smoke else 500
    subset, fingerprint = fixed_mmlu_subset(mmlu_n)
    results_root = SMOKE_RESULTS if args.smoke else RESULTS
    rows = []
    completed_configs: list[str] = []
    write_worker_status(
        args,
        "running",
        completed_configs=completed_configs,
        current_config=args.configs[0],
    )
    for config in args.configs:
        write_worker_status(
            args,
            "running",
            completed_configs=completed_configs,
            current_config=config,
        )
        lock_handle = acquire_config_lock(config)
        try:
            path = build_selective_merged(config)
            rows.append(
                run_config(
                    config,
                    path,
                    gpu_mem=args.gpu_mem,
                    math_n=math_n,
                    mmlu_rows=subset,
                    subset_fingerprint=fingerprint,
                    results_root=results_root,
                )
            )
        finally:
            lock_handle.close()
        completed_configs.append(config)
    frame = pd.DataFrame(rows)
    if args.smoke:
        write_worker_status(
            args,
            "complete",
            completed_configs=completed_configs,
        )
        print(frame.to_string(index=False), flush=True)
        return
    if args.worker_only:
        write_worker_status(
            args,
            "complete",
            completed_configs=completed_configs,
        )
        print(f"[G8 worker] complete configs={completed_configs}", flush=True)
        return
    expected = set(CONFIGS)
    observed = set(frame["config"])
    if observed != expected:
        raise RuntimeError(f"G8 incomplete configs: missing={sorted(expected-observed)}")
    fields = (
        "config", "closed_layers", "math_n", "math_acc", "math_trunc_rate",
        "mmlu_n", "mmlu_strict_fail_rate", "mmlu_strict_acc", "mmlu_flexible_acc",
        "mmlu_subset_seed", "mmlu_subset_fingerprint", "model_path",
    )
    atomic_csv(OUTPUT, frame[list(fields)])
    atomic_json(
        MANIFEST,
        {
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": "Cycle 09 block 2 G8",
            "adapter": str(ADAPTER),
            "adapter_sha256": sha256_file(ADAPTER / "adapter_model.safetensors"),
            "configs": list(CONFIGS),
            "mmlu_pool_source": str(S1_LOG_MANIFEST),
            "mmlu_subset_seed": SEED,
            "mmlu_subset_fingerprint": fingerprint,
            "output": str(OUTPUT),
            "output_sha256": sha256_file(OUTPUT),
        },
    )


if __name__ == "__main__":
    main()
