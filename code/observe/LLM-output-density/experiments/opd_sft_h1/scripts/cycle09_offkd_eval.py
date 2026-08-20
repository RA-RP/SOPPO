#!/usr/bin/env python3
"""Resumable evaluation campaign for the Cycle 09 off-policy KD control.

The checkpoint grid is matched point-for-point to the OPD/SFT trajectories.
Large raw artifacts remain under autodl-tmp; only protocol and summary files are
copied into the paper result tree. This runner never retries a failed command.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

REPO = Path("/root/LLM-output-density")
SIDE = REPO / "experiments/opd_sft_h1"
EVAL_DIR = REPO / "Eval"
COMPONENT = EVAL_DIR / "component"
for item in (SIDE, COMPONENT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

BASE_MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen3-4B-Base")
DEFAULT_RUN_ROOT = Path("/root/autodl-tmp/cycle09_offkd")
DEFAULT_COPYBACK = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/offkd"
)
THINK_RUNNER = EVAL_DIR / "component/think_math/runner_think.py"
NUMINA_PATH = Path("/root/autodl-tmp/prepared/NuminaMath-1___5/test.jsonl")
AIME_PATH = EVAL_DIR / "tasks/data/aime24/train.jsonl"
NUMINA_CAP_SELECTION = (
    DEFAULT_COPYBACK.parent / "mini/R3_numina_cap_selection.json"
)

NATIVE_CHECKPOINT_STEPS = (0, 5, 10, 20, 40, 160, 624)
CHECKPOINT_STEPS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
BACKFILL_STEPS = (80, 320, 480)
NUMINA_STEPS = (40, 160, 624)
FINAL_STEP = 624
SEED = 42
NUMINA_N = 200
NUMINA_CAP = 12288
AIME_N = 30
AIME_CAP = 24576
AIME_SEEDS = tuple(range(42, 52))


def step_label(step: int) -> str:
    return f"step_{int(step):03d}"


def checkpoint_label(step: int) -> str:
    return f"checkpoint-{int(step):06d}"


def parse_steps(value: str) -> list[int]:
    steps = [int(item.strip()) for item in value.split(",") if item.strip()]
    unknown = set(steps).difference(CHECKPOINT_STEPS)
    if unknown:
        raise ValueError(f"Unsupported checkpoint steps: {sorted(unknown)}")
    return steps


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def result_json(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.rglob("results_*.json")) if output_dir.exists() else []
    return candidates[-1] if candidates else None


def offline_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("TMPDIR", "/root/autodl-tmp/pip-tmp")
    return env


def run_command(cmd: list[str], *, cwd: Path) -> None:
    print("[CMD] " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(cwd), env=offline_env())
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def training_manifest(run_root: Path) -> Path:
    return run_root / "checkpoints/training_manifest.json"


def adapter_path(run_root: Path, step: int) -> Path:
    if step == 80:
        return run_root / "checkpoint_backfill/from_040" / checkpoint_label(step)
    if step in (320, 480):
        return run_root / "checkpoint_backfill/from_160" / checkpoint_label(step)
    return run_root / "checkpoints" / checkpoint_label(step)


def merged_path(run_root: Path, step: int) -> Path:
    return run_root / "_merged_models" / step_label(step)


def model_path(run_root: Path, step: int) -> Path:
    return BASE_MODEL if step == 0 else merged_path(run_root, step)


def protocol_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "control": "off_policy_kd",
        "checkpoint_grid": list(args.steps),
        "base_model": str(BASE_MODEL),
        "checkpoint_sources": {
            str(step): str(adapter_path(args.run_root, step))
            for step in args.steps
            if step != 0
        },
        "backfill_validation": str(
            args.run_root / "checkpoint_backfill/backfill_validation.json"
        ),
        "backfill_caveat": (
            "steps 80/320/480 are numerically equivalent replays from the nearest "
            "landmark, not bitwise-identical uninterrupted updates"
        ),
        "merged_model_root": str(args.run_root / "_merged_models"),
        "large_artifact_root": str(args.eval_root),
        "copyback_root": str(args.copyback_root),
        "model_retention": "merged checkpoints retained for later geometry",
        "math500": {
            "n": 500,
            "seed": SEED,
            "temperature": 0.6,
            "top_p": 0.9,
            "early_steps_0_to_20": {"max_tokens": 4096, "max_model_len": 6144},
            "late_steps_40_to_624": {"max_tokens": 16384, "max_model_len": 17408},
            "scorer": "scorer_v2.score plus scorer.extract_pred",
        },
        "numina": {
            "steps": list(NUMINA_STEPS),
            "n": NUMINA_N,
            "max_tokens": NUMINA_CAP,
            "max_model_len": 32768,
            "seed": SEED,
            "temperature": 0.6,
            "top_p": 0.9,
            "cap_source": str(NUMINA_CAP_SELECTION),
        },
        "aime24": {
            "steps": "offKD MATH500 peak plus final checkpoint",
            "n": AIME_N,
            "max_tokens": AIME_CAP,
            "max_model_len": 32768,
            "seeds": list(AIME_SEEDS),
            "temperature": 0.6,
            "top_p": 0.9,
            "status": "secondary",
        },
        "gpqa_diamond": {
            "task": "gpqa_diamond_zeroshot",
            "n": 198,
            "num_fewshot": 0,
            "batch_size": "auto",
            "max_model_len": 4096,
            "chat_template": False,
            "seed": SEED,
        },
        "mmlu_pro": {
            "task": "mmlu_pro",
            "limit": "100 per category, 14 categories, n=1400",
            "num_fewshot": 0,
            "batch_size": "auto",
            "max_model_len": 4096,
            "chat_template": False,
            "seed": SEED,
        },
        "ood_expansion": {
            "tasks": ["ifeval", "truthfulqa_mc1"],
            "num_fewshot": 0,
            "batch_size": "auto",
            "max_model_len": 4096,
            "apply_chat_template": True,
            "enable_thinking": False,
            "log_samples": True,
        },
        "protocol_sources": [
            "scripts/run_cycle07.py",
            "scripts/cap_unified_retest.py",
            "scripts/cycle09_r3_id.py",
            "scripts/cycle09_r3_ood.py",
            "mypaper/code/QA_cycle09.md",
            "mypaper/theory/offkd_rollout_handoff.md",
        ],
    }


def update_manifest(
    args: argparse.Namespace,
    *,
    status: str,
    stage: str,
    detail: str = "",
) -> None:
    path = args.eval_root / "evaluation_manifest.json"
    data = read_json(path) if path.exists() else protocol_payload(args)
    now = time.time()
    data.setdefault("started_at_unix", now)
    data.update(
        {
            "status": status,
            "stage": stage,
            "detail": detail,
            "updated_at_unix": now,
        }
    )
    if status == "complete":
        data["completed_at_unix"] = now
    write_json_atomic(path, data)


def require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"Required Python module is missing: {name}")


def validate_preflight(args: argparse.Namespace, *, require_complete: bool) -> None:
    required_paths = [
        BASE_MODEL / "config.json",
        THINK_RUNNER,
        NUMINA_PATH,
        AIME_PATH,
        EVAL_DIR / "tasks",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required eval input is missing: {path}")
    for module in ("torch", "transformers", "peft", "vllm", "lm_eval"):
        require_module(module)

    import nltk

    require_module("immutabledict")
    require_module("langdetect")
    nltk.data.find("tokenizers/punkt_tab")

    if NUMINA_CAP_SELECTION.exists():
        selected = int(read_json(NUMINA_CAP_SELECTION)["selected_cap"])
        if selected != NUMINA_CAP:
            raise ValueError(
                f"Numina cap drift: latest selection is {selected}, runner expects {NUMINA_CAP}"
            )
    else:
        raise FileNotFoundError(f"Missing Numina cap decision: {NUMINA_CAP_SELECTION}")

    manifest_path = training_manifest(args.run_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing training manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    expected_grid = list(NATIVE_CHECKPOINT_STEPS)
    if list(manifest.get("checkpoint_grid", [])) != expected_grid:
        raise ValueError(
            f"Training checkpoint grid drift: {manifest.get('checkpoint_grid')} != {expected_grid}"
        )
    if require_complete and manifest.get("status") != "complete":
        raise RuntimeError(f"Training is not complete: status={manifest.get('status')}")

    if require_complete:
        if set(args.steps).intersection(BACKFILL_STEPS):
            validation_path = (
                args.run_root / "checkpoint_backfill/backfill_validation.json"
            )
            if not validation_path.exists():
                raise FileNotFoundError(f"Missing backfill validation: {validation_path}")
            validation = read_json(validation_path)
            if validation.get("status") != "pass":
                raise RuntimeError(f"Backfill validation did not pass: {validation}")
        for step in args.steps:
            if step == 0:
                continue
            adapter = adapter_path(args.run_root, step)
            for name in ("adapter_config.json", "adapter_model.safetensors", "complete.json"):
                if not (adapter / name).exists():
                    raise FileNotFoundError(f"Incomplete checkpoint {step}: missing {adapter / name}")

    free_gib = shutil.disk_usage(args.run_root).free / (1024**3)
    if free_gib < 64:
        raise RuntimeError(f"Only {free_gib:.1f} GiB free; at least 64 GiB is required")
    print(
        f"[PREFLIGHT] ok; training_status={manifest.get('status')} "
        f"completed_steps={manifest.get('completed_steps')}/{manifest.get('total_steps')} "
        f"free={free_gib:.1f}GiB python={sys.executable}",
        flush=True,
    )


def ensure_merged_model(run_root: Path, step: int) -> Path:
    if step == 0:
        return BASE_MODEL
    adapter = adapter_path(run_root, step)
    if not (adapter / "complete.json").exists():
        raise FileNotFoundError(f"Checkpoint is not complete: {adapter}")
    from scripts.run_opd_minimal_closure import merge_lora_adapter

    target = merged_path(run_root, step)
    merged = merge_lora_adapter(BASE_MODEL, adapter, target)
    gc.collect()
    return merged


def run_think_eval(
    args: argparse.Namespace,
    *,
    task: str,
    step: int,
    n: int,
    max_tokens: int,
    max_model_len: int,
) -> Path:
    label = step_label(step)
    output = args.eval_root / "generative" / label / task
    summary = output / f"{label}.json"
    if summary.exists():
        print(f"[SKIP] {task} {label}: {summary}", flush=True)
        return summary
    output.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(THINK_RUNNER),
        "--task",
        task,
        "--model",
        str(model_path(args.run_root, step)),
        "--label",
        label,
        "--n",
        str(n),
        "--max-tokens",
        str(max_tokens),
        "--max-model-len",
        str(max_model_len),
        "--gpu-mem",
        str(args.gpu_mem),
        "--temperature",
        "0.6",
        "--top-p",
        "0.9",
        "--seed",
        str(SEED),
        "--outdir",
        str(output),
    ]
    run_command(cmd, cwd=SIDE)
    if not summary.exists():
        raise RuntimeError(f"Think eval returned without summary: {summary}")
    return summary


def math500_budget(step: int) -> tuple[int, int]:
    return (4096, 6144) if step <= 20 else (16384, 17408)


def run_math500(args: argparse.Namespace, *, smoke: bool = False) -> None:
    for step in args.steps:
        if smoke:
            n, max_tokens, max_model_len = 2, 256, 2048
        else:
            max_tokens, max_model_len = math500_budget(step)
            n = 500
        run_think_eval(
            args,
            task="math500",
            step=step,
            n=n,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
        )


def configure_id_runner(args: argparse.Namespace):
    import cycle09_r3_id as r3_id

    r3_id.r2.model_path_for = lambda _arm, step: model_path(args.run_root, int(step))
    runner_args = SimpleNamespace(
        run_root=args.eval_root,
        gpu_mem=args.gpu_mem,
        max_model_len=32768,
    )
    return r3_id, runner_args


def run_numina(args: argparse.Namespace) -> None:
    r3_id, runner_args = configure_id_runner(args)
    rows = r3_id.load_numina(NUMINA_N)
    for step in NUMINA_STEPS:
        if step not in args.steps:
            continue
        r3_id.run_model_cases(
            runner_args,
            "numina",
            "offkd",
            step,
            rows,
            [NUMINA_CAP],
            [SEED],
        )


def math500_peak(args: argparse.Namespace) -> int:
    scored: list[tuple[int, float]] = []
    for step in args.steps:
        path = (
            args.eval_root
            / "generative"
            / step_label(step)
            / "math500"
            / f"{step_label(step)}.json"
        )
        if not path.exists():
            raise FileNotFoundError(f"Cannot select MATH500 peak; missing {path}")
        scored.append((step, float(read_json(path)["acc"])))
    peak, accuracy = max(scored, key=lambda item: item[1])
    print(f"[AIME] offKD MATH500 peak={step_label(peak)} acc={accuracy:.6f}", flush=True)
    return peak


def run_aime(args: argparse.Namespace) -> list[int]:
    r3_id, runner_args = configure_id_runner(args)
    rows = r3_id.load_aime(AIME_N)
    steps = sorted({FINAL_STEP, math500_peak(args)}.intersection(args.steps))
    for step in steps:
        r3_id.run_model_cases(
            runner_args,
            "aime24",
            "offkd",
            step,
            rows,
            [AIME_CAP],
            list(AIME_SEEDS),
        )
    return steps


def lm_eval_binary() -> str:
    candidate = Path(sys.executable).resolve().with_name("lm_eval")
    return str(candidate) if candidate.exists() else sys.executable


def lm_eval_prefix() -> list[str]:
    binary = lm_eval_binary()
    return [binary] if binary != sys.executable else [sys.executable, "-m", "lm_eval"]


def run_knowledge_eval(args: argparse.Namespace, *, step: int, task: str) -> None:
    label = step_label(step)
    output = args.eval_root / "lm_eval" / label / task
    existing = result_json(output)
    if existing is not None:
        print(f"[SKIP] {task} {label}: {existing}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    model_args = (
        f"pretrained={model_path(args.run_root, step)},dtype=bfloat16,"
        f"gpu_memory_utilization={args.gpu_mem},max_model_len=4096"
    )
    cmd = lm_eval_prefix() + [
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        task,
        "--num_fewshot",
        "0",
        "--batch_size",
        "auto",
        "--seed",
        str(SEED),
        "--output_path",
        str(output),
    ]
    if task == "mmlu_pro":
        cmd.extend(["--limit", "100"])
    run_command(cmd, cwd=REPO)
    if result_json(output) is None:
        raise RuntimeError(f"lm_eval returned without results for {task}/{label}")


def run_ood_eval(
    args: argparse.Namespace,
    *,
    step: int,
    tasks: Iterable[str] = ("ifeval", "truthfulqa_mc1"),
    limit: int | None = None,
) -> None:
    task_list = list(tasks)
    label = step_label(step)
    output = args.eval_root / "ood_expansion" / label
    existing = result_json(output)
    if existing is not None:
        print(f"[SKIP] OOD {label}: {existing}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    model_args = (
        f"pretrained={model_path(args.run_root, step)},dtype=bfloat16,max_model_len=4096,"
        f"gpu_memory_utilization={args.gpu_mem},enable_thinking=false"
    )
    cmd = lm_eval_prefix() + [
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(task_list),
        "--num_fewshot",
        "0",
        "--batch_size",
        "auto",
        "--output_path",
        str(output),
        "--include_path",
        str(EVAL_DIR / "tasks"),
        "--log_samples",
        "--apply_chat_template",
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    run_command(cmd, cwd=EVAL_DIR)
    if result_json(output) is None:
        raise RuntimeError(f"lm_eval returned without OOD results for {label}")


def run_lm_eval_suite(args: argparse.Namespace) -> None:
    for step in args.steps:
        run_knowledge_eval(args, step=step, task="gpqa_diamond_zeroshot")
        run_knowledge_eval(args, step=step, task="mmlu_pro")
        run_ood_eval(args, step=step)


def get_metric(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return ""


def parse_lm_results(args: argparse.Namespace, step: int) -> dict[str, Any]:
    row: dict[str, Any] = {}
    gpqa_path = result_json(
        args.eval_root / "lm_eval" / step_label(step) / "gpqa_diamond_zeroshot"
    )
    mmlu_path = result_json(args.eval_root / "lm_eval" / step_label(step) / "mmlu_pro")
    ood_path = result_json(args.eval_root / "ood_expansion" / step_label(step))
    if gpqa_path:
        metrics = read_json(gpqa_path).get("results", {}).get("gpqa_diamond_zeroshot", {})
        row["gpqa_diamond_acc"] = get_metric(metrics, "acc,none", "acc")
        row["gpqa_diamond_n"] = metrics.get("sample_len", "")
    if mmlu_path:
        metrics = read_json(mmlu_path).get("results", {}).get("mmlu_pro", {})
        row["mmlu_pro_exact_match"] = get_metric(
            metrics, "exact_match,custom-extract", "exact_match"
        )
        row["mmlu_pro_n"] = metrics.get("sample_len", "")
    if ood_path:
        results = read_json(ood_path).get("results", {})
        ifeval = results.get("ifeval", {})
        truthful = results.get("truthfulqa_mc1", {})
        row.update(
            {
                "ifeval_prompt_strict": get_metric(
                    ifeval, "prompt_level_strict_acc,none", "prompt_level_strict_acc"
                ),
                "ifeval_instruction_strict": get_metric(
                    ifeval, "inst_level_strict_acc,none", "inst_level_strict_acc"
                ),
                "ifeval_prompt_loose": get_metric(
                    ifeval, "prompt_level_loose_acc,none", "prompt_level_loose_acc"
                ),
                "ifeval_instruction_loose": get_metric(
                    ifeval, "inst_level_loose_acc,none", "inst_level_loose_acc"
                ),
                "ifeval_n": ifeval.get("sample_len", ""),
                "truthfulqa_mc1_acc": get_metric(truthful, "acc,none", "acc"),
                "truthfulqa_mc1_n": truthful.get("sample_len", ""),
            }
        )
    return row


def id_summary_path(args: argparse.Namespace, task: str, step: int, cap: int, seed: int) -> Path:
    return (
        args.eval_root
        / "id_completion"
        / task
        / "offkd"
        / step_label(step)
        / f"cap_{cap}"
        / f"seed_{seed}.json"
    )


def aggregate(args: argparse.Namespace, aime_steps: list[int]) -> None:
    rows: list[dict[str, Any]] = []
    for step in args.steps:
        label = step_label(step)
        math_path = args.eval_root / "generative" / label / "math500" / f"{label}.json"
        if not math_path.exists():
            raise FileNotFoundError(f"Missing formal MATH500 result: {math_path}")
        math = read_json(math_path)
        row: dict[str, Any] = {
            "arm": "offkd",
            "step": step,
            "math500_n": math.get("n", ""),
            "math500_cap": math.get("max_tokens", ""),
            "math500_acc": math.get("acc", ""),
            "math500_trunc_rate": math.get("trunc_rate", ""),
            "math500_mean_response_len": math.get("mean_response_len", ""),
        }
        numina_path = id_summary_path(args, "numina", step, NUMINA_CAP, SEED)
        if step in NUMINA_STEPS and step in args.steps:
            if not numina_path.exists():
                raise FileNotFoundError(f"Missing formal Numina result: {numina_path}")
            numina = read_json(numina_path)
            row.update(
                {
                    "numina_n": numina.get("n", ""),
                    "numina_cap": numina.get("cap", ""),
                    "numina_acc": numina.get("acc", ""),
                    "numina_trunc_rate": numina.get("trunc_rate", ""),
                    "numina_mean_response_len": numina.get("mean_response_len", ""),
                }
            )
        if step in aime_steps:
            values = []
            for seed in AIME_SEEDS:
                path = id_summary_path(args, "aime24", step, AIME_CAP, seed)
                if not path.exists():
                    raise FileNotFoundError(f"Missing formal AIME24 seed result: {path}")
                values.append(read_json(path))
            row.update(
                {
                    "aime24_n": AIME_N,
                    "aime24_cap": AIME_CAP,
                    "aime24_seed_count": len(values),
                    "aime24_acc_seed_mean": sum(float(v["acc"]) for v in values) / len(values),
                    "aime24_trunc_rate_seed_mean": sum(float(v["trunc_rate"]) for v in values)
                    / len(values),
                    "aime24_status": "secondary",
                }
            )
        row.update(parse_lm_results(args, step))
        required = (
            "gpqa_diamond_acc",
            "mmlu_pro_exact_match",
            "ifeval_prompt_strict",
            "truthfulqa_mc1_acc",
        )
        missing = [key for key in required if row.get(key, "") == ""]
        if missing:
            raise RuntimeError(f"Missing lm_eval metrics at {label}: {missing}")
        rows.append(row)

    fields = [
        "arm",
        "step",
        "math500_n",
        "math500_cap",
        "math500_acc",
        "math500_trunc_rate",
        "math500_mean_response_len",
        "numina_n",
        "numina_cap",
        "numina_acc",
        "numina_trunc_rate",
        "numina_mean_response_len",
        "aime24_n",
        "aime24_cap",
        "aime24_seed_count",
        "aime24_acc_seed_mean",
        "aime24_trunc_rate_seed_mean",
        "aime24_status",
        "gpqa_diamond_n",
        "gpqa_diamond_acc",
        "mmlu_pro_n",
        "mmlu_pro_exact_match",
        "ifeval_n",
        "ifeval_prompt_strict",
        "ifeval_instruction_strict",
        "ifeval_prompt_loose",
        "ifeval_instruction_loose",
        "truthfulqa_mc1_n",
        "truthfulqa_mc1_acc",
    ]
    trajectory = args.eval_root / "offkd_eval_trajectory.csv"
    with trajectory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    aime_rows = []
    for step in aime_steps:
        for seed in AIME_SEEDS:
            data = read_json(id_summary_path(args, "aime24", step, AIME_CAP, seed))
            aime_rows.append(
                {
                    "arm": "offkd",
                    "step": step,
                    "seed": seed,
                    "n": data["n"],
                    "cap": data["cap"],
                    "acc": data["acc"],
                    "trunc_rate": data["trunc_rate"],
                    "boxed_before_trunc_rate": data["boxed_before_trunc_rate"],
                    "status": "secondary",
                }
            )
    aime_csv = args.eval_root / "offkd_aime24_seeds.csv"
    with aime_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aime_rows[0]) if aime_rows else [])
        if aime_rows:
            writer.writeheader()
            writer.writerows(aime_rows)

    args.copyback_root.mkdir(parents=True, exist_ok=True)
    for source in (trajectory, aime_csv, args.eval_root / "evaluation_manifest.json"):
        shutil.copy2(source, args.copyback_root / source.name)
    print(f"[SUMMARY] {trajectory} rows={len(rows)}", flush=True)
    print(f"[COPYBACK] {args.copyback_root}", flush=True)


def run_smoke(args: argparse.Namespace) -> None:
    ensure_merged_model(args.run_root, FINAL_STEP)
    run_math500(args, smoke=True)
    run_ood_eval(
        args,
        step=FINAL_STEP,
        tasks=("ifeval", "truthfulqa_mc1"),
        limit=1,
    )


def run_formal(args: argparse.Namespace) -> None:
    update_manifest(args, status="running", stage="merge", detail="retaining merged checkpoints")
    for step in args.steps:
        ensure_merged_model(args.run_root, step)

    update_manifest(args, status="running", stage="math500")
    run_math500(args)

    update_manifest(args, status="running", stage="numina")
    run_numina(args)

    update_manifest(args, status="running", stage="lm_eval")
    run_lm_eval_suite(args)

    update_manifest(args, status="running", stage="aime24", detail="secondary; ten seeds")
    aime_steps = run_aime(args)

    update_manifest(args, status="running", stage="aggregate")
    aggregate(args, aime_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--copyback-root", type=Path, default=DEFAULT_COPYBACK)
    parser.add_argument("--steps", default=",".join(map(str, CHECKPOINT_STEPS)))
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.steps = parse_steps(args.steps)
    if args.smoke:
        args.steps = [FINAL_STEP]
        args.eval_root = args.run_root / "smoke_eval"
    else:
        args.eval_root = args.run_root / "eval"
    args.eval_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        validate_preflight(args, require_complete=False)
        print(json.dumps(protocol_payload(args), indent=2, ensure_ascii=False), flush=True)
        return

    validate_preflight(args, require_complete=True)
    update_manifest(args, status="running", stage="initializing")
    try:
        if args.smoke:
            update_manifest(args, status="running", stage="smoke")
            run_smoke(args)
        else:
            run_formal(args)
    except Exception as exc:
        update_manifest(args, status="failed", stage="failed", detail=repr(exc))
        raise
    update_manifest(args, status="complete", stage="complete")
    if not args.smoke:
        shutil.copy2(
            args.eval_root / "evaluation_manifest.json",
            args.copyback_root / "evaluation_manifest.json",
        )
    print("[DONE] offKD evaluation complete", flush=True)


if __name__ == "__main__":
    main()
