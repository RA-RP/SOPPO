#!/usr/bin/env python3
"""Export four Llama LoRA trajectories to auditable adapters and merged eval models."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cycle09_block3_common as c


EXPORT_ROOT = c.RUN_ROOT / "llama_models"
ADAPTER_ROOT = EXPORT_ROOT / "adapters"
MERGED_ROOT = EXPORT_ROOT / "merged"
MANIFEST = EXPORT_ROOT / "export_manifest.json"


def parse_names(value: str, allowed: tuple[str, ...]) -> list[str]:
    if value.strip().lower() == "all":
        return list(allowed)
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names).difference(allowed))
    if not names or unknown:
        raise ValueError(f"invalid names={names}; unknown={unknown}")
    return names


def parse_steps(value: str) -> list[int]:
    steps = [int(item.strip()) for item in value.split(",") if item.strip()]
    unknown = sorted(set(steps).difference(c.MEASURED_CHECKPOINTS))
    if not steps or unknown:
        raise ValueError(f"invalid steps={steps}; unknown={unknown}")
    return steps


def adapter_target(arm: str, step: int) -> Path:
    return ADAPTER_ROOT / arm / f"checkpoint-{step:06d}"


def merged_target(arm: str, step: int) -> Path:
    return c.LLAMA_STUDENT_RUNTIME if step == 0 else MERGED_ROOT / arm / f"step_{step:03d}"


def adapter_complete(path: Path) -> bool:
    return (
        (path / "adapter_config.json").is_file()
        and (path / "adapter_model.safetensors").is_file()
        and (path / "adapter_model.safetensors").stat().st_size > 0
    )


def merged_complete(path: Path) -> bool:
    return c.model_check(path)["complete"]


def validate_adapter(path: Path, arm: str, step: int) -> dict[str, Any]:
    if not adapter_complete(path):
        raise FileNotFoundError(f"incomplete PEFT adapter {arm}/{step}: {path}")
    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if int(rank) != 32 or int(alpha) != 64:
        raise RuntimeError(f"LoRA contract drift {arm}/{step}: r={rank} alpha={alpha}")
    return {
        "path": str(path),
        "adapter_config_sha256": c.sha256_file(path / "adapter_config.json"),
        "adapter_weights_sha256": c.sha256_file(path / "adapter_model.safetensors"),
        "adapter_bytes": (path / "adapter_model.safetensors").stat().st_size,
        "rank": int(rank),
        "alpha": int(alpha),
    }


def find_adapter(root: Path) -> Path:
    candidates = sorted(
        path.parent for path in root.rglob("adapter_config.json") if adapter_complete(path.parent)
    )
    if len(candidates) != 1:
        raise RuntimeError(f"verl merger adapter candidates={list(map(str, candidates))}")
    return candidates[0]


def copy_adapter(source: Path, target: Path) -> None:
    if adapter_complete(target):
        return
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    if not adapter_complete(temporary):
        raise RuntimeError(f"copied adapter is incomplete: {temporary}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, target)


def export_opd_adapter(step: int) -> Path:
    target = adapter_target("opd", step)
    if adapter_complete(target):
        return target
    actor = c.adapter_path("opd", step) / "actor"
    if not actor.is_dir() or not any(actor.glob("model_world_size_*_rank_*.pt")):
        raise FileNotFoundError(f"incomplete verl actor checkpoint: {actor}")
    work = EXPORT_ROOT / "work" / f"opd_step_{step:03d}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    command = [
        str(c.VERL_PYTHON),
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor),
        "--target_dir",
        str(work),
    ]
    result = subprocess.run(command, cwd=c.REPO)
    if result.returncode != 0:
        raise RuntimeError(f"verl model_merger failed rc={result.returncode} for step {step}")
    source = find_adapter(work)
    copy_adapter(source, target)
    shutil.rmtree(work)
    return target


def source_adapter(arm: str, step: int) -> Path:
    if arm == "opd":
        return export_opd_adapter(step)
    source = c.adapter_path(arm, step)
    validate_adapter(source, arm, step)
    target = adapter_target(arm, step)
    copy_adapter(source, target)
    return target


def merge(arm: str, step: int, adapter: Path) -> Path:
    target = merged_target(arm, step)
    if merged_complete(target):
        c.install_llama_chat_template(target)
        return target
    # This module is invoked by absolute path from the detached supervisor.
    # Import from its own directory rather than assuming a top-level `scripts`
    # package is present on sys.path.
    from run_opd_minimal_closure import merge_lora_adapter

    result = merge_lora_adapter(c.LLAMA_STUDENT, adapter, target)
    c.install_llama_chat_template(result)
    gc.collect()
    return result


def export_cell(arm: str, step: int) -> dict[str, Any]:
    if step == 0:
        runtime = c.ensure_llama_runtime_model()
        return {
            "arm": "base",
            "step": 0,
            "adapter": None,
            "merged": runtime["model"],
            "runtime_model": runtime,
            "status": "complete",
        }
    adapter = source_adapter(arm, step)
    adapter_info = validate_adapter(adapter, arm, step)
    merged = merge(arm, step, adapter)
    merged_info = c.model_check(merged)
    if not merged_info["complete"]:
        raise RuntimeError(f"merged model incomplete {arm}/{step}: {merged_info['error']}")
    return {
        "arm": arm,
        "step": step,
        "adapter": adapter_info,
        "merged": merged_info,
        "delta_w_source": "PEFT adapter BA fp32; never merged-minus-base",
        "status": "complete",
    }


def update_manifest(new_cells: list[dict[str, Any]]) -> dict[str, Any]:
    payload = c.read_json(
        MANIFEST,
        {
            "schema_version": 1,
            "task": "Cycle09 block3 Llama four-arm model export",
            "base_model": str(c.LLAMA_STUDENT),
            "base_runtime_model": str(c.LLAMA_STUDENT_RUNTIME),
            "cells": [],
        },
    )
    keyed = {(str(row["arm"]), int(row["step"])): row for row in payload["cells"]}
    keyed.update({(str(row["arm"]), int(row["step"])): row for row in new_cells})
    payload["cells"] = [keyed[key] for key in sorted(keyed)]
    payload["status"] = "partial"
    expected = {(arm, step) for arm in c.ARMS for step in c.MEASURED_CHECKPOINTS if step}
    observed = {(str(row["arm"]), int(row["step"])) for row in payload["cells"] if row["arm"] != "base"}
    if expected.issubset(observed):
        payload["status"] = "complete"
    payload["updated_utc"] = c.utc_now()
    c.atomic_json(MANIFEST, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="all")
    parser.add_argument("--steps", default=",".join(map(str, c.MEASURED_CHECKPOINTS)))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    arms = parse_names(args.arms, c.ARMS)
    steps = parse_steps(args.steps)
    cells = [(arm, step) for arm in arms for step in steps if step != 0]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "cells": cells,
                    "base": str(c.LLAMA_STUDENT),
                    "base_runtime": str(c.LLAMA_STUDENT_RUNTIME),
                    "adapter_root": str(ADAPTER_ROOT),
                    "merged_root": str(MERGED_ROOT),
                },
                indent=2,
            )
        )
        return
    results = [export_cell(arm, step) for arm, step in cells]
    if 0 in steps:
        results.append(export_cell("opd", 0))
    print(json.dumps(update_manifest(results), indent=2))


if __name__ == "__main__":
    main()
