#!/usr/bin/env python3
"""Inventory Stage 1 machine-migration inputs without copying any data."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


MINI = Path(
    "/root/LLM-output-density/mypaper/local_experiment_results/"
    "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
STEPS = (5, 10, 20, 40, 80, 160, 320, 480, 624)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(tmp, path)


def tree_stat(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return path.stat().st_size, 1
    size = 0
    files = 0
    stack = [path]
    while stack:
        root = stack.pop()
        with os.scandir(root) as entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        size += entry.stat(follow_symlinks=False).st_size
                        files += 1
                    elif entry.is_symlink():
                        size += entry.stat(follow_symlinks=False).st_size
                        files += 1
                except FileNotFoundError:
                    continue
    return size, files


def required_items() -> list[dict]:
    rows: list[dict] = []

    def add(category: str, item: str, path: str, notes: str = "") -> None:
        rows.append(
            {
                "category": category,
                "item": item,
                "path": path,
                "required": True,
                "notes": notes,
            }
        )

    rollout = "/root/autodl-tmp/cycle09_offkd/rollout"
    add("teacher_data", "teacher_rollout_jsonl", f"{rollout}/teacher_rollout.jsonl")
    add("teacher_data", "top32_token_ids_memmap", f"{rollout}/pass2_stream/top32_ids.npy")
    add(
        "teacher_data",
        "top32_logprob_memmap",
        f"{rollout}/pass2_stream/top32_logprob.npy",
    )
    add("teacher_data", "top32_row_offsets", f"{rollout}/pass2_stream/row_offsets.npy")
    add("teacher_data", "rollout_manifest", f"{rollout}/rollout_manifest.json")
    add("teacher_data", "pass2_validation", f"{rollout}/pass2_validation.json")

    add(
        "weights",
        "base_model_step_000",
        "/root/autodl-tmp/model/Qwen/Qwen3-4B-Base",
        "shared step-0 model for all arms",
    )
    for step in STEPS:
        label = f"step_{step:03d}"
        add(
            "weights",
            f"opd_merged_{label}",
            f"/root/autodl-tmp/cycle08_opd_trajectory/_merged_models/{label}",
        )
        add(
            "weights",
            f"sft_adapter_{label}",
            f"/root/autodl-tmp/cycle07_base_sft_trajectory/checkpoints/{label}",
        )
        add(
            "weights",
            f"offkd_merged_{label}",
            f"/root/autodl-tmp/cycle09_offkd/_merged_models/{label}",
            "numerical backfill" if step in (80, 320, 480) else "formal landmark",
        )
        if step in (5, 10, 20, 40, 160, 624):
            adapter = f"/root/autodl-tmp/cycle09_offkd/checkpoints/checkpoint-{step:06d}"
        elif step == 80:
            adapter = (
                "/root/autodl-tmp/cycle09_offkd/checkpoint_backfill/"
                "from_040/checkpoint-000080"
            )
        else:
            adapter = (
                "/root/autodl-tmp/cycle09_offkd/checkpoint_backfill/"
                f"from_160/checkpoint-{step:06d}"
            )
        add(
            "weights",
            f"offkd_adapter_{label}",
            adapter,
            "numerical backfill" if step in (80, 320, 480) else "formal landmark",
        )

    add(
        "probe_assets",
        "round4_probe_corpora",
        "/root/autodl-tmp/cycle09_r4/corpora",
    )
    add(
        "probe_assets",
        "base_whitening_references",
        "/root/autodl-tmp/cycle09_r4/scratch/references",
    )

    datasets = {
        "MATH500": "/root/.cache/huggingface/datasets/HuggingFaceH4___math-500",
        "GPQA": "/root/.cache/huggingface/datasets/Idavidrein___gpqa",
        "MMLU-Pro": "/root/.cache/huggingface/datasets/TIGER-Lab___mmlu-pro",
        "IFEval": "/root/.cache/huggingface/datasets/google___if_eval",
        "TruthfulQA": "/root/.cache/huggingface/datasets/truthfulqa___truthful_qa",
        "Numina": "/root/autodl-tmp/dataset/NuminaMath-1___5",
    }
    for name, path in datasets.items():
        add("eval_data", name, path, "processed local dataset cache")
    add(
        "eval_data",
        "huggingface_dataset_download_cache",
        "/root/.cache/huggingface/datasets/downloads",
        "raw download cache backing processed datasets",
    )

    add("software", "repository", "/root/LLM-output-density")
    add(
        "software",
        "density_conda_environment",
        "/root/miniconda3/envs/density",
        "lm-eval, vLLM, HF and PEFT runtime used by current pipeline",
    )
    return rows


def main() -> None:
    rows = required_items()
    for row in rows:
        path = Path(row["path"])
        size, files = tree_stat(path)
        row.update(
            {
                "exists": path.exists(),
                "object_type": (
                    "directory" if path.is_dir() else "file" if path.is_file() else "missing"
                ),
                "size_bytes": size,
                "size_gib": size / (1024**3),
                "file_count": files,
                "status": "READY" if path.exists() else "MISSING",
            }
        )
    frame = pd.DataFrame(rows).sort_values(["category", "item"]).reset_index(drop=True)
    output = MINI / "S1_machine_migration_inventory.csv"
    atomic_csv(frame, output)
    missing = frame.loc[~frame["exists"], ["category", "item", "path"]].to_dict(
        orient="records"
    )
    atomic_json(
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "operation": "inventory_only_no_sync",
            "source_spec": "/root/LLM-output-density/mypaper/theory/stage_plan_handoff.md#cutover-sync-checklist",
            "output": str(output),
            "n_items": len(frame),
            "n_ready": int(frame["exists"].sum()),
            "n_missing": int((~frame["exists"]).sum()),
            "missing": missing,
            "sum_item_bytes_non_deduplicated": int(frame["size_bytes"].sum()),
            "notes": "No files were copied, moved, compressed, or deleted.",
        },
        MINI / "S1_machine_migration_inventory_manifest.json",
    )
    print(
        f"[S1 inventory] items={len(frame)} ready={int(frame['exists'].sum())} "
        f"missing={int((~frame['exists']).sum())} -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
