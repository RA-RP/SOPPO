#!/usr/bin/env python3
"""Run the Stage-1 format-immune MMLU-Pro conditional-LL grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cycle09_r4_common as c4


STEPS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
ARMS = ("opd", "sft", "offkd")
SUBJECTS = (
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)
REPO = Path("/root/LLM-output-density")
MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
RUN_ROOT = Path("/root/autodl-tmp/cycle09_s1/mmlupro_loglik")
INPUT = RUN_ROOT / "input/mmlupro_1400.jsonl"
PROGRESS = RUN_ROOT / "S1_8_PROGRESS.json"
OUTPUT = MINI / "S1_mmlupro_loglik.csv"
MANIFEST = MINI / "S1_mmlupro_loglik_manifest.json"
PROMPT_FILE = MINI / "S1_mmlupro_loglik_prompt_template.txt"
SOURCE_MANIFEST = MINI / "S1_mmlupro_log_manifest.json"
FLEXIBLE = MINI / "S1_mmlupro_flexible.csv"
TASK_DIR = REPO / "experiments/opd_sft_h1/tasks/s1_mmlupro_loglik"
TASK_YAML = TASK_DIR / "s1_mmlupro_loglik.yaml"
TASK_UTILS = TASK_DIR / "utils.py"
TASK = "s1_mmlupro_loglik"
OFFKD_MODELS = Path("/root/autodl-tmp/cycle09_offkd/_merged_models")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        handle.write(text)
    os.replace(tmp, path)


def atomic_csv(rows: list[dict], fields: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def atomic_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    os.replace(tmp, path)


def model_path(arm: str, step: int) -> Path:
    if step == 0:
        path = c4.BASE_MODEL
    elif arm == "offkd":
        path = OFFKD_MODELS / c4.step_label(step)
    else:
        path = c4.model_path(arm, step)
    if not (path / "config.json").is_file():
        raise FileNotFoundError(path)
    return path


def load_task_utils():
    spec = importlib.util.spec_from_file_location("s1_mmlupro_task_utils", TASK_UTILS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TASK_UTILS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_sample_paths() -> list[Path]:
    payload = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    candidates = [
        cell
        for cell in payload.get("cells", [])
        if cell.get("arm") == "opd" and int(cell.get("step", -1)) == 0
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one canonical base cell, found {len(candidates)}")
    paths = [Path(item["path"]) for item in candidates[0].get("sample_files", [])]
    if len(paths) != 14 or any(not path.is_file() for path in paths):
        raise RuntimeError("canonical MMLU-Pro base cell does not contain 14 sample files")
    return sorted(paths)


def prepare_input() -> dict:
    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(SOURCE_MANIFEST)
    source_paths = canonical_sample_paths()
    rows = []
    source_files = []
    for source_path in source_paths:
        count = 0
        with source_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                doc = sample["doc"]
                options = [str(option).strip() for option in doc["options"]]
                answer_index = int(doc["answer_index"])
                if not 0 <= answer_index < len(options):
                    raise ValueError(f"invalid answer index for question {doc['question_id']}")
                rows.append(
                    {
                        "question_id": int(doc["question_id"]),
                        "category": str(doc["category"]),
                        "question": str(doc["question"]).strip(),
                        "options": options,
                        "answer_index": answer_index,
                        "answer": str(doc["answer"]),
                        "source_doc_hash": sample.get("doc_hash", ""),
                    }
                )
                count += 1
        source_files.append(
            {
                "path": str(source_path),
                "n_rows": count,
                "size_bytes": source_path.stat().st_size,
                "sha256": sha256_file(source_path),
            }
        )
    subject_order = {subject: index for index, subject in enumerate(SUBJECTS)}
    rows.sort(key=lambda row: (subject_order[row["category"]], row["question_id"]))
    categories = Counter(row["category"] for row in rows)
    option_counts = Counter(len(row["options"]) for row in rows)
    if len(rows) != 1400 or len({row["question_id"] for row in rows}) != 1400:
        raise RuntimeError("canonical MMLU-Pro subset must contain 1400 unique questions")
    if categories != Counter({subject: 100 for subject in SUBJECTS}):
        raise RuntimeError(f"subject balance mismatch: {categories}")
    if any(not option for row in rows for option in row["options"]):
        raise RuntimeError("empty choice continuation found")
    atomic_jsonl(rows, INPUT)
    qid_digest = hashlib.sha256(
        json.dumps([row["question_id"] for row in rows]).encode("ascii")
    ).hexdigest()
    return {
        "path": str(INPUT),
        "size_bytes": INPUT.stat().st_size,
        "sha256": sha256_file(INPUT),
        "question_id_sha256": qid_digest,
        "n_questions": len(rows),
        "subjects": dict(sorted(categories.items())),
        "option_count_distribution": {
            str(key): value for key, value in sorted(option_counts.items())
        },
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "source_files": source_files,
    }


def result_metric(task: dict, name: str) -> float:
    for key in (f"{name},none", name):
        if key in task:
            return float(task[key])
    raise KeyError(f"missing metric {name}: {sorted(task)}")


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def validate_attempt(root: Path, expected_n: int) -> dict | None:
    results = sorted(root.rglob("results_*.json")) if root.is_dir() else []
    samples = sorted(root.rglob(f"samples_{TASK}_*.jsonl")) if root.is_dir() else []
    if len(results) != 1 or len(samples) != 1:
        return None
    try:
        payload = json.loads(results[0].read_text(encoding="utf-8"))
        task = payload["results"][TASK]
        n_rows = count_lines(samples[0])
        acc = result_metric(task, "acc")
        acc_norm = result_metric(task, "acc_norm")
    except (KeyError, ValueError, json.JSONDecodeError):
        return None
    if n_rows != expected_n or not (0.0 <= acc <= 1.0 and 0.0 <= acc_norm <= 1.0):
        return None
    return {
        "attempt_root": str(root),
        "result_path": str(results[0]),
        "result_sha256": sha256_file(results[0]),
        "sample_path": str(samples[0]),
        "sample_sha256": sha256_file(samples[0]),
        "n_questions": n_rows,
        "acc_ll": acc,
        "acc_ll_norm": acc_norm,
    }


def cached_attempt(cell_root: Path, expected_n: int) -> dict | None:
    attempts = sorted(cell_root.glob("attempt_*"), reverse=True) if cell_root.is_dir() else []
    for attempt in attempts:
        complete = validate_attempt(attempt, expected_n)
        if complete is not None:
            return complete
    return None


def run_one(
    *,
    key: str,
    arm: str,
    step: int,
    expected_n: int,
    limit: int | None,
    gpu_memory: float,
    max_model_len: int,
    seed: int,
    smoke: bool,
) -> dict:
    cell_root = RUN_ROOT / ("smoke" if smoke else "formal") / key
    complete = cached_attempt(cell_root, expected_n)
    if complete is not None:
        print(f"[S1-8 cached] {key}", flush=True)
        return complete
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt = cell_root / f"attempt_{timestamp}_{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=False)
    model = model_path(arm, step)
    model_args = (
        f"pretrained={model},dtype=bfloat16,gpu_memory_utilization={gpu_memory},"
        f"max_model_len={max_model_len}"
    )
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        TASK,
        "--include_path",
        str(TASK_DIR),
        "--num_fewshot",
        "0",
        "--batch_size",
        "auto",
        "--seed",
        str(seed),
        "--output_path",
        str(attempt),
        "--log_samples",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    env = dict(os.environ)
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    print(
        f"[S1-8 run] {key} model={model} n={expected_n} attempt={attempt.name}",
        flush=True,
    )
    result = subprocess.run(command, cwd=str(REPO), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"S1-8 cell failed key={key} rc={result.returncode}")
    complete = validate_attempt(attempt, expected_n)
    if complete is None:
        raise RuntimeError(f"S1-8 cell output failed validation: {attempt}")
    return complete


def unique_grid() -> list[tuple[str, str, int]]:
    return [("base_step_000", "opd", 0)] + [
        (f"{arm}_step_{step:03d}", arm, step)
        for arm in ARMS
        for step in STEPS
        if step != 0
    ]


def write_progress(completed: dict[str, dict], input_info: dict, state: str) -> None:
    atomic_json(
        {
            "status": state,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "n_complete_unique_cells": len(completed),
            "n_total_unique_cells": 28,
            "input": input_info,
            "cells": completed,
        },
        PROGRESS,
    )


def read_flexible() -> dict[tuple[str, int], dict]:
    rows = {}
    with FLEXIBLE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[(row["arm"], int(row["step"]))] = row
    expected = {(arm, step) for arm in ARMS for step in STEPS}
    if set(rows) != expected:
        raise RuntimeError("S1 flexible table does not contain the full 30-cell grid")
    return rows


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def finalize(completed: dict[str, dict], input_info: dict, args) -> None:
    flexible = read_flexible()
    rows = []
    for arm in ARMS:
        for step in STEPS:
            key = "base_step_000" if step == 0 else f"{arm}_step_{step:03d}"
            ll = completed[key]
            previous = flexible[(arm, step)]
            rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "n": ll["n_questions"],
                    "strict_exact_match": float(previous["exact_match"]),
                    "flexible_exact_match": float(previous["mmlu_pro_flexible"]),
                    "acc_ll": ll["acc_ll"],
                    "acc_ll_norm": ll["acc_ll_norm"],
                }
            )
    if len(rows) != 30:
        raise RuntimeError("S1-8 final grid must contain 30 rows")
    fields = [
        "arm",
        "step",
        "n",
        "strict_exact_match",
        "flexible_exact_match",
        "acc_ll",
        "acc_ll_norm",
    ]
    atomic_csv(rows, fields, OUTPUT)
    task_utils = load_task_utils()
    prompt_text = (
        "PROMPT_TEMPLATE (context; rendered independently for every question):\n"
        + task_utils.PROMPT_TEMPLATE
        + "\n\nCHOICE_TEMPLATE (each candidate continuation):\n"
        + task_utils.CHOICE_TEMPLATE
        + "\n"
    )
    atomic_text(prompt_text, PROMPT_FILE)
    manifest_cells = []
    for key, arm, step in unique_grid():
        manifest_cells.append(
            {
                "key": key,
                "arm": "shared_base" if step == 0 else arm,
                "step": step,
                "model_path": str(model_path(arm, step)),
                **completed[key],
            }
        )
    atomic_json(
        {
            "schema_version": 1,
            "task": "S1-8 MMLU-Pro conditional loglikelihood reevaluation",
            "status": "COMPLETE",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": {
                "backend": "lm-eval multiple_choice with vLLM",
                "task": TASK,
                "non_cot": True,
                "prompt_template": task_utils.PROMPT_TEMPLATE,
                "choice_template": task_utils.CHOICE_TEMPLATE,
                "target_delimiter": "one ASCII space (explicit task setting)",
                "choice_scoring": "conditional loglikelihood of each full answer-option text",
                "acc_ll": "argmax of summed conditional loglikelihood",
                "acc_ll_norm": (
                    "argmax of summed conditional loglikelihood divided by the "
                    "character length of the choice string (lm-eval acc_norm)"
                ),
                "choices_in_prompt_context": False,
                "num_fewshot": 0,
                "seed": args.seed,
                "dtype": "bfloat16",
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory,
                "chat_template": False,
                "log_samples": True,
                "shared_base_evaluated_once": True,
            },
            "input": input_info,
            "arms": list(ARMS),
            "steps": list(STEPS),
            "n_grid_rows": len(rows),
            "n_unique_model_cells": len(manifest_cells),
            "cells": manifest_cells,
            "four_metric_source": {
                "strict_and_flexible": str(FLEXIBLE),
                "strict_and_flexible_sha256": sha256_file(FLEXIBLE),
            },
            "software": {
                "python": sys.version,
                "lm_eval": package_version("lm_eval"),
                "vllm": package_version("vllm"),
                "transformers": package_version("transformers"),
                "torch": package_version("torch"),
            },
            "artifacts": {
                "output": str(OUTPUT),
                "output_sha256": sha256_file(OUTPUT),
                "prompt_template": str(PROMPT_FILE),
                "prompt_template_sha256": sha256_file(PROMPT_FILE),
                "task_yaml": str(TASK_YAML),
                "task_yaml_sha256": sha256_file(TASK_YAML),
                "task_utils": str(TASK_UTILS),
                "task_utils_sha256": sha256_file(TASK_UTILS),
                "script": str(Path(__file__).resolve()),
                "script_sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        MANIFEST,
    )
    print(f"[S1-8] complete grid_rows={len(rows)} unique_cells={len(completed)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    if not TASK_YAML.is_file() or not TASK_UTILS.is_file():
        raise FileNotFoundError(TASK_DIR)
    input_info = prepare_input()
    print(
        f"[S1-8 input] n={input_info['n_questions']} sha256={input_info['sha256']}",
        flush=True,
    )
    if args.prepare_only:
        write_progress({}, input_info, "INPUT_PREPARED")
        return
    if args.smoke:
        if not 1 <= args.smoke_limit <= 10:
            raise ValueError("--smoke-limit must be in [1, 10]")
        result = run_one(
            key="base_step_000",
            arm="opd",
            step=0,
            expected_n=args.smoke_limit,
            limit=args.smoke_limit,
            gpu_memory=args.gpu_memory,
            max_model_len=args.max_model_len,
            seed=args.seed,
            smoke=True,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    completed = {}
    write_progress(completed, input_info, "RUNNING")
    for key, arm, step in unique_grid():
        completed[key] = run_one(
            key=key,
            arm=arm,
            step=step,
            expected_n=1400,
            limit=None,
            gpu_memory=args.gpu_memory,
            max_model_len=args.max_model_len,
            seed=args.seed,
            smoke=False,
        )
        write_progress(completed, input_info, "RUNNING")
    if len(completed) != 28:
        raise RuntimeError(f"incomplete S1-8 unique grid: {len(completed)}")
    finalize(completed, input_info, args)
    write_progress(completed, input_info, "COMPLETE")


if __name__ == "__main__":
    main()
