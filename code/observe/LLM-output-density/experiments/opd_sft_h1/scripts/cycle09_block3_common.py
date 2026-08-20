#!/usr/bin/env python3
"""Shared paths, provenance helpers, and GPU budget ledger for Cycle 09 block 3."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


REPO = Path("/root/LLM-output-density")
AUTODL = Path("/root/autodl-tmp")
SCRIPT_DIR = REPO / "experiments/opd_sft_h1/scripts"
HANDOFF = REPO / "mypaper/theory/stage_plan_handoff.md"
MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)

RUN_ROOT = AUTODL / "cycle09_block3"
L1_ROOT = RUN_ROOT / "llama_opd"
L1_DATA = L1_ROOT / "data"
L1_CHECKPOINTS = L1_ROOT / "checkpoints"
L1_RAW_ROLLOUTS = L1_ROOT / "rollouts/raw"
L1_CANONICAL_ROLLOUTS = L1_ROOT / "rollouts/canonical"
L1_LOGS = L1_ROOT / "logs"

Q1_ROOT = RUN_ROOT / "qwen_alpha05"
Q1_DATA = Q1_ROOT / "data"
Q1_CHECKPOINTS = Q1_ROOT / "checkpoints"
Q1_LOGS = Q1_ROOT / "logs"
Q1_FROZEN = Q1_ROOT / "frozen_external"

LLAMA_STUDENT = AUTODL / "model/Meta/modelscope/Llama-3.2-3B"
LLAMA_TEACHER = AUTODL / "model/Meta/modelscope/Meta-Llama-3.1-8B-Instruct"
LLAMA_STUDENT_RUNTIME = L1_ROOT / "model/student_runtime"
SOURCE_PROMPTS = AUTODL / "cycle08_opd_trajectory/data/opd_prompts_5k.parquet"
VERL_ROOT = AUTODL / "verl"
VERL_PYTHON = AUTODL / "envs/verl/bin/python"
DENSITY_PYTHON = Path("/root/miniconda3/envs/density/bin/python")

QWEN_STUDENT = AUTODL / "model/Qwen/Qwen3-4B-Base"
QWEN_TEACHER = AUTODL / "model/Qwen/Qwen3-8B"
QWEN_OFFKD_ROOT = AUTODL / "cycle09_offkd"

LLAMA_OFFLINE_ROOT = AUTODL / "cycle09_block2/model2_llama/g6"
LLAMA_ARM_ROOTS = {
    "opd": L1_CHECKPOINTS,
    "sft": LLAMA_OFFLINE_ROOT / "sft/checkpoints",
    "offkd": LLAMA_OFFLINE_ROOT / "offkd/checkpoints",
    "seqkd": LLAMA_OFFLINE_ROOT / "seqkd/checkpoints",
}

ARMS = ("opd", "sft", "offkd", "seqkd")
TRAINING_CHECKPOINTS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
MEASURED_CHECKPOINTS = (0, 5, 20, 40, 80, 160, 320, 624)
CRITICAL_CHECKPOINTS = (0, 5, 20, 40, 624)
GEOMETRY_LANDMARKS = (0, 5, 20, 40, 160, 624)
LLAMA_LAYERS = (7, 14, 21)
MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

SEED = 42
TRAIN_BATCH_SIZE = 16
TOTAL_STEPS = 624
# L1 is intentionally split into separately authorized deliveries.  The
# existing offline arms retain their native 624-step schedules above.
L1_STAGE_A_FINAL_STEP = 160
L1_STAGE_B_FINAL_STEP = 320
L1_FINAL_STEP = L1_STAGE_A_FINAL_STEP
L1_CHECKPOINT_GRID = tuple(step for step in TRAINING_CHECKPOINTS if step <= L1_FINAL_STEP)
MAX_PROMPT_TOKENS = 1024
MAX_RESPONSE_TOKENS = 10240
GPU_BUDGET_HOURS = 72.0
BUDGET_LEDGER = RUN_ROOT / "gpu_budget_ledger.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(
    path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=1)
def llama_template_contract() -> dict[str, Any]:
    student_tokenizer = LLAMA_STUDENT / "tokenizer.json"
    teacher_tokenizer = LLAMA_TEACHER / "tokenizer.json"
    student_config = LLAMA_STUDENT / "tokenizer_config.json"
    teacher_config = LLAMA_TEACHER / "tokenizer_config.json"
    for path in (student_tokenizer, teacher_tokenizer, student_config, teacher_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    student_hash = sha256_file(student_tokenizer)
    teacher_hash = sha256_file(teacher_tokenizer)
    if student_hash != teacher_hash:
        raise RuntimeError("Llama student/teacher tokenizer.json mismatch")
    teacher_payload = read_json(teacher_config, {})
    template = teacher_payload.get("chat_template")
    if not isinstance(template, str) or not template.strip():
        raise RuntimeError(f"teacher chat_template missing: {teacher_config}")
    return {
        "chat_template": template,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "chat_template_source": str(teacher_config),
        "student_tokenizer_sha256": student_hash,
        "teacher_tokenizer_sha256": teacher_hash,
    }


def load_llama_tokenizer(model_path: Path = LLAMA_STUDENT):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    tokenizer.chat_template = llama_template_contract()["chat_template"]
    return tokenizer


def install_llama_chat_template(model_path: Path) -> dict[str, Any]:
    if model_path.resolve() == LLAMA_STUDENT.resolve():
        raise RuntimeError("refusing to modify the immutable Llama base directory")
    config_path = model_path / "tokenizer_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = read_json(config_path, {})
    contract = llama_template_contract()
    payload["chat_template"] = contract["chat_template"]
    atomic_json(config_path, payload)
    observed = hashlib.sha256(
        str(read_json(config_path, {}).get("chat_template", "")).encode("utf-8")
    ).hexdigest()
    if observed != contract["chat_template_sha256"]:
        raise RuntimeError(f"chat-template installation failed: {model_path}")
    return {
        "tokenizer_config": artifact(config_path),
        "chat_template_sha256": observed,
        "chat_template_source": contract["chat_template_source"],
    }


def ensure_llama_runtime_model() -> dict[str, Any]:
    contract = llama_template_contract()
    if LLAMA_STUDENT_RUNTIME.exists():
        check = model_check(LLAMA_STUDENT_RUNTIME)
        config = read_json(LLAMA_STUDENT_RUNTIME / "tokenizer_config.json", {})
        observed = hashlib.sha256(
            str(config.get("chat_template", "")).encode("utf-8")
        ).hexdigest()
        if not check["complete"] or observed != contract["chat_template_sha256"]:
            raise RuntimeError(
                f"stale/incomplete Llama runtime model: {LLAMA_STUDENT_RUNTIME}"
            )
    else:
        LLAMA_STUDENT_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{LLAMA_STUDENT_RUNTIME.name}.",
                dir=LLAMA_STUDENT_RUNTIME.parent,
            )
        )
        try:
            for source in LLAMA_STUDENT.iterdir():
                if source.name == "tokenizer_config.json":
                    continue
                (temporary / source.name).symlink_to(
                    source, target_is_directory=source.is_dir()
                )
            tokenizer_config = read_json(LLAMA_STUDENT / "tokenizer_config.json", {})
            tokenizer_config["chat_template"] = contract["chat_template"]
            atomic_json(temporary / "tokenizer_config.json", tokenizer_config)
            check = model_check(temporary)
            if not check["complete"]:
                raise RuntimeError(f"runtime model build failed: {check['error']}")
            os.replace(temporary, LLAMA_STUDENT_RUNTIME)
            temporary = None
        finally:
            if temporary is not None and temporary.exists():
                shutil.rmtree(temporary)
    check = model_check(LLAMA_STUDENT_RUNTIME)
    return {
        "model": check,
        "runtime_path": str(LLAMA_STUDENT_RUNTIME),
        "immutable_weight_source": str(LLAMA_STUDENT),
        "tokenizer_config": artifact(LLAMA_STUDENT_RUNTIME / "tokenizer_config.json"),
        "chat_template_sha256": contract["chat_template_sha256"],
        "chat_template_source": contract["chat_template_source"],
        "student_tokenizer_sha256": contract["student_tokenizer_sha256"],
        "teacher_tokenizer_sha256": contract["teacher_tokenizer_sha256"],
    }


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def file_check(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    return {"path": str(path), "complete": bool(exists and size > 0), "bytes": size}


def model_check(path: Path) -> dict[str, Any]:
    config = path / "config.json"
    weights = sorted(path.glob("*.safetensors")) or sorted(path.glob("pytorch_model*.bin"))
    index = path / "model.safetensors.index.json"
    error = None
    if not config.is_file() or config.stat().st_size == 0:
        error = f"missing/empty {config}"
    elif index.is_file():
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            names = sorted(set(payload.get("weight_map", {}).values()))
            weights = [path / name for name in names]
            missing = [item for item in weights if not item.is_file() or item.stat().st_size == 0]
            if not names or missing:
                error = f"incomplete model index; missing={list(map(str, missing))}"
        except (OSError, json.JSONDecodeError) as caught:
            error = str(caught)
    elif not weights or any(item.stat().st_size == 0 for item in weights):
        error = f"missing/empty model weights under {path}"
    return {
        "path": str(path),
        "complete": error is None,
        "error": error,
        "config_bytes": config.stat().st_size if config.is_file() else 0,
        "weight_files": len(weights),
        "weight_bytes": sum(item.stat().st_size for item in weights if item.is_file()),
    }


def adapter_path(arm: str, step: int) -> Path:
    if arm not in ARMS:
        raise ValueError(f"unknown Llama arm: {arm}")
    if step == 0:
        return LLAMA_STUDENT
    if arm == "opd":
        return L1_CHECKPOINTS / f"global_step_{step}"
    return LLAMA_ARM_ROOTS[arm] / f"checkpoint-{step:06d}"


def gpu_inventory() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, name, memory = [part.strip() for part in line.split(",", 2)]
        rows.append({"index": int(index), "name": name, "memory_total_mib": int(memory)})
    return rows


def l1_preflight() -> dict[str, Any]:
    models = [model_check(LLAMA_STUDENT), model_check(LLAMA_TEACHER)]
    files = [file_check(SOURCE_PROMPTS)]
    runtime = [
        {"path": str(VERL_ROOT), "complete": (VERL_ROOT / "verl").is_dir()},
        {"path": str(VERL_PYTHON), "complete": VERL_PYTHON.is_file()},
    ]
    try:
        gpus = gpu_inventory()
        gpu_complete = len(gpus) >= 2 and all(
            item["memory_total_mib"] >= 90000 for item in gpus[:2]
        )
        gpu_error = None
    except (OSError, subprocess.SubprocessError, ValueError) as caught:
        gpus, gpu_complete, gpu_error = [], False, str(caught)
    disk = shutil.disk_usage(AUTODL)
    payload = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "models": models,
        "files": files,
        "runtime": runtime,
        "gpus": gpus,
        "gpu_error": gpu_error,
        "gpu_complete": gpu_complete,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }
    payload["complete"] = bool(
        all(item["complete"] for item in models + files + runtime)
        and gpu_complete
        and disk.free >= 120 * 2**30
    )
    return payload


def _new_ledger() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "budget_gpu_hours": GPU_BUDGET_HOURS,
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "runs": [],
        "consumed_gpu_hours": 0.0,
        "remaining_gpu_hours": GPU_BUDGET_HOURS,
    }


def load_ledger(path: Path = BUDGET_LEDGER) -> dict[str, Any]:
    payload = read_json(path, _new_ledger())
    if float(payload.get("budget_gpu_hours", -1)) != GPU_BUDGET_HOURS:
        raise RuntimeError(f"GPU budget drift in {path}")
    return payload


@contextmanager
def ledger_lock(path: Path = BUDGET_LEDGER):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w", encoding="utf-8")
    fcntl.flock(handle, fcntl.LOCK_EX)
    try:
        yield
    finally:
        handle.close()


def _refresh_ledger(payload: dict[str, Any], now: float | None = None) -> None:
    now = time.time() if now is None else now
    consumed = 0.0
    for run in payload["runs"]:
        end = float(run.get("ended_at_unix", now))
        wall = max(0.0, end - float(run["started_at_unix"])) / 3600.0
        run["wall_hours"] = wall
        run["gpu_hours"] = wall * int(run["gpu_count"])
        consumed += run["gpu_hours"]
    payload["consumed_gpu_hours"] = consumed
    payload["remaining_gpu_hours"] = max(0.0, GPU_BUDGET_HOURS - consumed)
    payload["updated_utc"] = utc_now()


def budget_start(
    task: str,
    *,
    gpu_count: int,
    planned_upper_gpu_hours: float,
    path: Path = BUDGET_LEDGER,
) -> str:
    with ledger_lock(path):
        payload = load_ledger(path)
        _refresh_ledger(payload)
        active = [run for run in payload["runs"] if "ended_at_unix" not in run]
        committed = sum(
            max(
                0.0,
                float(run["planned_upper_gpu_hours"]) - float(run.get("gpu_hours", 0.0)),
            )
            for run in active
        )
        available = max(0.0, payload["remaining_gpu_hours"] - committed)
        if planned_upper_gpu_hours > available + 1e-9:
            raise RuntimeError(
                f"budget gate: {task} needs <= {planned_upper_gpu_hours:.3f} GPUh, "
                f"uncommitted_remaining={available:.3f} GPUh"
            )
        now = time.time()
        run_id = f"{task}:{time.time_ns()}"
        payload["runs"].append(
            {
                "run_id": run_id,
                "task": task,
                "gpu_count": int(gpu_count),
                "planned_upper_gpu_hours": float(planned_upper_gpu_hours),
                "started_at_unix": now,
                "started_utc": utc_now(),
                "status": "running",
            }
        )
        _refresh_ledger(payload, now)
        atomic_json(path, payload)
        return run_id


def budget_finish(
    run_id: str,
    *,
    status: str,
    detail: str = "",
    path: Path = BUDGET_LEDGER,
) -> dict[str, Any]:
    with ledger_lock(path):
        payload = load_ledger(path)
        matches = [run for run in payload["runs"] if run["run_id"] == run_id]
        if len(matches) != 1:
            raise RuntimeError(f"unknown/non-unique GPU ledger run: {run_id}")
        run = matches[0]
        if "ended_at_unix" not in run:
            run["ended_at_unix"] = time.time()
            run["ended_utc"] = utc_now()
        run["status"] = status
        run["detail"] = detail
        _refresh_ledger(payload)
        atomic_json(path, payload)
        return payload


def checkpoint_inventory(
    steps: Iterable[int] = TRAINING_CHECKPOINTS,
) -> list[dict[str, Any]]:
    rows = []
    for step in steps:
        path = adapter_path("opd", step)
        if step == 0:
            complete = model_check(path)["complete"]
        else:
            actor = path / "actor"
            complete = path.is_dir() and actor.is_dir() and (
                any(actor.glob("model_world_size_*_rank_*.pt"))
                or any(actor.rglob("*.safetensors"))
            )
        rows.append(
            {
                "arm": "opd",
                "step": step,
                "path": str(path),
                "complete": complete,
                "bytes": sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
                if path.is_dir()
                else 0,
            }
        )
    return rows
