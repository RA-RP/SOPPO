#!/usr/bin/env python3
"""Complete FAT-R1-v2 MATH C-region exact KL.

This is a reuse/correction runner for the FAT-R1-v2 output-link campaign.
Round 1 stored MATH NLL on P/C/B/T and exact KL on B/T, but deliberately did
not store C logits.  This script recomputes teacher-forced full-vocabulary
KL on the token-clean MATH C region without overwriting the original FAT files.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


REPO = Path("/root/LLM-output-density")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cycle09_fat_outlink_round1 as fat  # noqa: E402


SCHEMA = "cycle09_fat_math_kl_c_completion_v1"
TASK = "FAT-R1-v2-MATH-KL-C"
AUTODL = Path("/root/autodl-tmp")
MINI = REPO / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"
ORIGINAL = MINI / "fat_outlink_round1_v2"
OUT = MINI / "fat_outlink_round1_v2_math_kl_c"
SCRATCH = AUTODL / "cycle09_fat_math_kl_c"
CELL_ROOT = SCRATCH / "cells"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(value)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(tmp, path)


def append_code_evolution(text: str) -> None:
    path = REPO / "mypaper/code/code_evolution.md"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + text.strip() + "\n")


def states_for(models: Iterable[str], smoke: bool) -> list[tuple[str, str, int]]:
    if not smoke:
        return fat.states_for_run(models, smoke=False)
    states: list[tuple[str, str, int]] = []
    for model in models:
        terminal = 624 if model == "qwen" else 320
        states.extend([(model, "base", 0), (model, "opd", terminal)])
    return states


def status_path(model: str, arm: str, step: int) -> Path:
    report_arm = "base" if step == 0 else arm
    return CELL_ROOT / model / report_arm / fat.step_label(step) / "status.json"


def sample_path(model: str, arm: str, step: int) -> Path:
    report_arm = "base" if step == 0 else arm
    return CELL_ROOT / model / report_arm / fat.step_label(step) / "math_kl_c_samples.csv"


def done(model: str, arm: str, step: int, expected_rows: int | None = None) -> bool:
    p = status_path(model, arm, step)
    if not p.is_file() or not sample_path(model, arm, step).is_file():
        return False
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            return False
        if expected_rows is not None and int(payload.get("rows", -1)) != int(expected_rows):
            return False
        return True
    except Exception:
        return False


def math_records(tok: Any, samples: list[Any]) -> list[dict[str, Any]]:
    records = []
    for sample in samples:
        encoded = fat.encode_with_regions(tok, sample)
        c_positions = sorted(pos for pos in encoded["regions"].get("c", []) if pos > 0)
        if not c_positions:
            raise RuntimeError(f"empty C positions for sample {sample.sample_id}")
        records.append(
            {
                "sample": sample,
                "sample_id": sample.sample_id,
                "category": sample.category,
                "metadata": sample.metadata,
                "input_ids": encoded["input_ids"],
                "length": len(encoded["input_ids"]),
                "c_positions": c_positions,
                "predict_positions": sorted({pos - 1 for pos in c_positions}),
            }
        )
    return records


def pad_batch(batch: list[dict[str, Any]], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(row["length"] for row in batch)
    ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, row in enumerate(batch):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[i, : len(values)] = values
        mask[i, : len(values)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def batch_kl_c(
    base_model: Any,
    current_model: Any,
    tok: Any,
    batch: list[dict[str, Any]],
    device: str,
) -> list[dict[str, Any]]:
    input_ids, attention = pad_batch(batch, tok.pad_token_id, device)
    union_keep = sorted({pos for row in batch for pos in row["predict_positions"]})
    keep_tensor = torch.tensor(union_keep, dtype=torch.long, device=device)
    keep_index = {pos: idx for idx, pos in enumerate(union_keep)}
    base_logits = base_model(
        input_ids=input_ids,
        attention_mask=attention,
        use_cache=False,
        logits_to_keep=keep_tensor,
    ).logits.detach().float()
    current_logits = current_model(
        input_ids=input_ids,
        attention_mask=attention,
        use_cache=False,
        logits_to_keep=keep_tensor,
    ).logits.detach().float()
    rows: list[dict[str, Any]] = []
    for bi, record in enumerate(batch):
        kls = []
        for target_pos in record["c_positions"]:
            idx = keep_index[target_pos - 1]
            logp0 = torch.log_softmax(base_logits[bi, idx], dim=-1)
            logpt = torch.log_softmax(current_logits[bi, idx], dim=-1)
            kl = float((logp0.exp() * (logp0 - logpt)).sum().item())
            kls.append(max(0.0, kl))
        rows.append(
            {
                "sample_id": record["sample_id"],
                "category": record["category"],
                "unique_id": record["sample_id"],
                "subject": record["metadata"].get("subject", ""),
                "level": record["metadata"].get("level", ""),
                "n_tokens_c": len(record["c_positions"]),
                "kl_c": float(np.mean(kls)),
            }
        )
    del base_logits, current_logits, input_ids, attention
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def batch_kl_c_resilient(
    base_model: Any,
    current_model: Any,
    tok: Any,
    batch: list[dict[str, Any]],
    device: str,
) -> list[dict[str, Any]]:
    try:
        return batch_kl_c(base_model, current_model, tok, batch, device)
    except RuntimeError as error:
        message = str(error).lower()
        if "out of memory" not in message or len(batch) <= 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        mid = len(batch) // 2
        return batch_kl_c_resilient(base_model, current_model, tok, batch[:mid], device) + batch_kl_c_resilient(
            base_model, current_model, tok, batch[mid:], device
        )


def load_pair(model: str, arm: str, step: int, device: str) -> tuple[Any, Any, Any]:
    base_path = fat.direct_model_path(model, "opd", 0)
    base_model = fat.load_model(base_path, device)
    if step == 0:
        return base_model, base_model, nullcontext(base_path)
    context = fat.materialized_model(model, arm, step)
    current_path = context.__enter__()
    try:
        current_model = fat.load_model(current_path, device)
    except Exception:
        context.__exit__(*sys.exc_info())
        fat.unload_model(base_model)
        raise
    return base_model, current_model, context


def unload_pair(base_model: Any, current_model: Any, context: Any, step: int) -> None:
    try:
        if step == 0:
            fat.unload_model(base_model)
        else:
            fat.unload_model(current_model)
            fat.unload_model(base_model)
    finally:
        try:
            context.__exit__(None, None, None)
        except AttributeError:
            pass


def run_cell(model: str, arm: str, step: int, tok: Any, samples: list[Any], args: argparse.Namespace) -> None:
    report_arm = "base" if step == 0 else arm
    out_path = sample_path(model, report_arm, step)
    if done(model, report_arm, step, expected_rows=len(samples)):
        return
    records = math_records(tok, samples)
    batches = fat.make_batches(records, args.max_batch_tokens)
    start = time.time()
    print(f"[KL_C] {model}/{report_arm}/{step} samples={len(samples)} batches={len(batches)}", flush=True)
    base_model = current_model = context = None
    try:
        base_model, current_model, context = load_pair(model, arm, step, args.device)
        rows = []
        if step == 0:
            for record in records:
                rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "category": record["category"],
                        "unique_id": record["sample_id"],
                        "subject": record["metadata"].get("subject", ""),
                        "level": record["metadata"].get("level", ""),
                        "n_tokens_c": len(record["c_positions"]),
                        "kl_c": 0.0,
                    }
                )
        else:
            for batch in batches:
                rows.extend(batch_kl_c_resilient(base_model, current_model, tok, batch, args.device))
        frame = pd.DataFrame(rows)
        frame.insert(0, "model", model)
        frame.insert(1, "arm", report_arm)
        frame.insert(2, "checkpoint", step)
        frame.insert(3, "domain", "math")
        frame["aggregation_unit"] = "sample_mean_over_C_tokens"
        frame["kl_direction"] = "base_to_checkpoint_D_KL_p0_parallel_pt"
        frame["full_vocabulary"] = True
        frame["forward_dtype"] = "bf16"
        frame["log_softmax_kl_dtype"] = "fp32"
        atomic_csv(out_path, frame)
        atomic_json(
            status_path(model, report_arm, step),
            {
                "schema_version": SCHEMA,
                "task": TASK,
                "status": "complete",
                "model": model,
                "arm": report_arm,
                "checkpoint": step,
                "samples": len(frame),
                "rows": len(frame),
                "wall_seconds": round(time.time() - start, 3),
                "device": args.device,
                "max_batch_tokens": args.max_batch_tokens,
                "created_utc": utc_now(),
            },
        )
    finally:
        if base_model is not None:
            unload_pair(base_model, current_model, context, step)


def aggregate_and_write(smoke: bool = False) -> None:
    frames = [pd.read_csv(p) for p in sorted(CELL_ROOT.glob("*/*/step_*/math_kl_c_samples.csv"))]
    samples = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if samples.empty:
        raise RuntimeError("no KL_C sample rows found")
    atomic_csv(OUT / "fat_r1_v2_math_kl_c_samples.csv", samples)
    cells = (
        samples.groupby(["model", "arm", "checkpoint", "domain"], dropna=False)
        [["kl_c", "n_tokens_c"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    cells["aggregation"] = "sample_macro"
    atomic_csv(OUT / "fat_r1_v2_math_kl_c_cells.csv", cells)
    old_math = pd.read_csv(ORIGINAL / "fat_r1_v2_math_cells.csv")
    old_sample = old_math[old_math["aggregation"].eq("sample_macro")].copy()
    merged = old_sample.merge(
        cells[["model", "arm", "checkpoint", "kl_c"]],
        on=["model", "arm", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    merged["kl_b_minus_c"] = merged["kl_b"] - merged["kl_c"]
    merged["delta_nll_b_minus_c"] = merged["delta_nll_b"] - merged["delta_nll_c"]
    contrasts = merged[
        [
            "model",
            "arm",
            "checkpoint",
            "domain",
            "aggregation",
            "kl_b",
            "kl_c",
            "kl_b_minus_c",
            "delta_nll_b",
            "delta_nll_c",
            "delta_nll_b_minus_c",
        ]
    ].copy()
    contrasts["notes"] = "MATH C exact full-vocabulary KL completion; B/C use token-clean FAT-R1-v2 spans"
    atomic_csv(OUT / "fat_r1_v2_math_kl_c_contrasts.csv", contrasts)
    task_rows = []
    for model, arm, step in states_for(fat.MODELS, smoke=False):
        report_arm = "base" if step == 0 else arm
        status = json.loads(status_path(model, report_arm, step).read_text(encoding="utf-8")) if status_path(model, report_arm, step).is_file() else {}
        task_rows.append(
            {
                "model": model,
                "arm": report_arm,
                "checkpoint": step,
                "status": status.get("status", "missing"),
                "rows": status.get("rows", 0),
                "wall_seconds": status.get("wall_seconds", math.nan),
            }
        )
    task = pd.DataFrame(task_rows)
    atomic_csv(OUT / "fat_r1_v2_math_kl_c_task_status.csv", task)
    outputs = {}
    for name in [
        "fat_r1_v2_math_kl_c_task_status.csv",
        "fat_r1_v2_math_kl_c_samples.csv",
        "fat_r1_v2_math_kl_c_cells.csv",
        "fat_r1_v2_math_kl_c_contrasts.csv",
    ]:
        path = OUT / name
        outputs[name] = {
            "path": str(path),
            "rows": max(0, sum(1 for _ in path.open(encoding="utf-8")) - 1) if path.is_file() else 0,
            "sha256": fat.sha256_file(path) if path.is_file() else None,
        }
    complete = bool((task["status"] == "complete").all()) if not task.empty else False
    manifest = {
        "schema_version": SCHEMA,
        "task": TASK,
        "status": "complete" if complete else "partial",
        "created_utc": utc_now(),
        "scope": "MATH500 token-clean C region exact full-vocabulary KL completion for FAT-R1-v2 formal grid",
        "does_not_modify": [
            str(ORIGINAL / "fat_r1_v2_math_cells.csv"),
            str(ORIGINAL / "fat_r1_v2_math_samples.csv"),
            str(ORIGINAL / "fat_r1_v2_region_contrasts.csv"),
        ],
        "numeric_protocol": {
            "forward_dtype": "bf16",
            "log_softmax_kl_dtype": "fp32",
            "kl_direction": "base_to_checkpoint_D_KL_p0_parallel_pt",
            "aggregation": "per-token exact KL mean within sample C span, then sample macro mean",
            "stored_logits": "none; batch-local only",
        },
        "outputs": outputs,
    }
    atomic_json(OUT / "fat_r1_v2_math_kl_c_manifest.json", manifest)
    lines = [
        "# FAT-R1-v2 MATH KL_C completion handoff",
        "",
        f"- created_utc: `{manifest['created_utc']}`",
        f"- status: `{manifest['status']}`",
        "- scope: MATH500 `C` region exact full-vocabulary KL only; no rollout/training/new geometry.",
        "- old FAT-R1-v2 files were not overwritten.",
        "- numeric: BF16 forward; FP32 log-softmax/KL; direction `D_KL(p0 || pt)`.",
        "- aggregation: token mean within each sample C span, then sample macro cell mean.",
        "",
        "## Outputs",
        "",
        "| file | rows | sha256 |",
        "|---|---:|---|",
    ]
    for name, info in outputs.items():
        lines.append(f"| `{name}` | {info['rows']} | `{info['sha256']}` |")
    lines.extend(
        [
            "",
            "## Completion Status",
            "",
            task.to_markdown(index=False),
        ]
    )
    atomic_text(OUT / "fat_r1_v2_math_kl_c_handoff.md", "\n".join(lines) + "\n")
    manifest["outputs"]["fat_r1_v2_math_kl_c_handoff.md"] = {
        "path": str(OUT / "fat_r1_v2_math_kl_c_handoff.md"),
        "sha256": fat.sha256_file(OUT / "fat_r1_v2_math_kl_c_handoff.md"),
    }
    atomic_json(OUT / "fat_r1_v2_math_kl_c_manifest.json", manifest)
    if complete and not smoke:
        append_code_evolution(
            f"""
## {utc_now()} FAT-R1-v2 MATH KL_C completion

- Script: `experiments/opd_sft_h1/scripts/cycle09_fat_math_kl_c_completion.py`
- Output root: `{OUT}`
- Manifest: `{OUT / 'fat_r1_v2_math_kl_c_manifest.json'}`
- Scope: completed exact full-vocabulary `KL_C` for MATH500 C-region on the FAT-R1-v2 formal grid; no rollout/training/new geometry; original FAT-R1-v2 artifacts left intact.
- Numeric protocol: BF16 checkpoint forward, FP32 log-softmax/KL, `D_KL(p0 || pt)`, batch-local logits only.
"""
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal", "finalize", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", default="all")
    parser.add_argument("--max-batch-tokens", type=int, default=384)
    parser.add_argument("--smoke-math-n", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    models = fat.MODELS if args.models == "all" else tuple(x.strip() for x in args.models.split(",") if x.strip())
    if args.mode in ("smoke", "formal", "all"):
        bundle = fat.build_token_manifest()
        math_samples = bundle["samples"]["math"]
        if args.mode == "smoke":
            math_samples = math_samples[: args.smoke_math_n]
        for model in models:
            tok = fat.load_tokenizer(model, fat.direct_model_path(model, "opd", 0))
            for _, arm, step in states_for([model], smoke=args.mode == "smoke"):
                run_cell(model, arm, step, tok, math_samples, args)
            del tok
            gc.collect()
    if args.mode in ("finalize", "formal", "all"):
        aggregate_and_write(smoke=args.mode == "smoke")


if __name__ == "__main__":
    main()
