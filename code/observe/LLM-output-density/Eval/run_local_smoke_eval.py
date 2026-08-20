#!/usr/bin/env python3
"""Run a minimal lm_eval smoke test with either a local dataset or real HF-backed tasks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "model" / "Qwen"
DEFAULT_DATASET_PATH = REPO_ROOT / "Eval" / "local_smoke" / "data" / "local_qwen_smoke_mc.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "Eval" / "local_smoke" / "results" / "local_qwen_smoke_result.json"
DEFAULT_TASK_NAME = "local_qwen_smoke_mc"
DEFAULT_SHARED_CONFIG_PATH = REPO_ROOT / "MultiTasks" / "Density_base.yaml"
KNOWN_LM_EVAL_BINARIES = [
    Path(sys.executable).resolve().with_name("lm_eval"),
    Path("/root/miniconda3/envs/density/bin/lm_eval"),
]

HF_ENV_KEY_MAPPING = {
    "hf_home": "HF_HOME",
    "hf_datasets_cache": "HF_DATASETS_CACHE",
    "hf_hub_cache": "HF_HUB_CACHE",
    "hf_hub_offline": "HF_HUB_OFFLINE",
    "hf_datasets_offline": "HF_DATASETS_OFFLINE",
    "transformers_offline": "TRANSFORMERS_OFFLINE",
}

ETHICS_SUBTASKS = [
    "ethics_deontology",
    "ethics_utilitarianism",
    "ethics_virtue",
    "ethics_justice",
    "ethics_cm",
]


def resolve_lm_eval_binary() -> str:
    for candidate in KNOWN_LM_EVAL_BINARIES:
        if candidate.exists():
            return str(candidate)

    path_bin = shutil.which("lm_eval")
    if path_bin:
        return path_bin

    raise FileNotFoundError(
        "Could not find lm_eval. Activate the density environment or install lm_eval first."
    )


def default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass

    return "cpu"


def default_dtype(device: str) -> str:
    return "float16" if device.startswith("cuda") else "float32"


def build_task_yaml(task_name: str, dataset_path: Path) -> str:
    return textwrap.dedent(
        f"""
        task: {task_name}
        dataset_path: json
        dataset_name: null
        dataset_kwargs:
          data_files:
            test: \"{dataset_path}\"
        output_type: multiple_choice
        training_split: null
        validation_split: null
        test_split: test
        doc_to_text: |
          Question: {{{{question}}}}
          A. {{{{A}}}}
          B. {{{{B}}}}
          C. {{{{C}}}}
          D. {{{{D}}}}
          Answer:
        doc_to_choice: [\"A\", \"B\", \"C\", \"D\"]
        doc_to_target: answer
        metric_list:
          - metric: acc
            aggregation: mean
            higher_is_better: true
        metadata:
          version: 1.0
        """
    ).lstrip()


def expand_lm_tasks(logical_tasks: list[str]) -> list[str]:
    expanded_tasks: list[str] = []
    for logical_task in logical_tasks:
        if logical_task == "ethics":
            expanded_tasks.extend(ETHICS_SUBTASKS)
        else:
            expanded_tasks.append(logical_task)
    return expanded_tasks


def flatten_loaded_task_names(loaded: object) -> list[str]:
    if isinstance(loaded, dict):
        flattened: list[str] = []
        for key, value in loaded.items():
            if isinstance(value, dict):
                flattened.extend(flatten_loaded_task_names(value))
            else:
                flattened.append(str(key))
        return flattened
    return []


def parse_task_csv(raw_tasks: str) -> list[str]:
    tasks = [task.strip() for task in raw_tasks.split(",") if task.strip()]
    if not tasks:
        raise ValueError("At least one lm_eval task is required.")
    return tasks


def resolve_include_path(include_path: str | None) -> str | None:
    if not include_path:
        return None
    resolved = Path(include_path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return str(resolved.resolve())


def load_shared_eval_config(config_path: Path) -> dict:
    if not config_path.exists():
        print(f"[WARN] Shared config not found, using current environment defaults: {config_path}")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config.get("eval", {}) or {}


def resolve_logical_tasks(hf_tasks_arg: str | None, shared_eval_cfg: dict) -> list[str]:
    if hf_tasks_arg:
        return parse_task_csv(hf_tasks_arg)

    shared_tasks = shared_eval_cfg.get("lm_tasks")
    if shared_tasks:
        return [str(task) for task in shared_tasks]

    raise ValueError(
        "No hub tasks were provided and shared config does not contain eval.lm_tasks."
    )


def build_hf_eval_env(eval_cfg: dict, offline_override: bool | None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    effective_cfg = dict(eval_cfg)
    if offline_override is not None:
        effective_cfg["hf_hub_offline"] = offline_override
        effective_cfg["hf_datasets_offline"] = offline_override
        effective_cfg["transformers_offline"] = offline_override

    for config_key, env_key in HF_ENV_KEY_MAPPING.items():
        value = effective_cfg.get(config_key)
        if value in (None, ""):
            continue

        normalized_value = "1" if isinstance(value, bool) and value else "0" if isinstance(value, bool) else str(value)
        env[env_key] = normalized_value
        if env_key == "HF_HUB_CACHE":
            env["HUGGINGFACE_HUB_CACHE"] = normalized_value

    for path_key in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        cache_path = env.get(path_key)
        if cache_path:
            Path(cache_path).mkdir(parents=True, exist_ok=True)

    return env


def print_hf_env(env: dict[str, str]) -> None:
    tracked_keys = [
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ]
    effective_items = [(key, env.get(key)) for key in tracked_keys if env.get(key)]
    if not effective_items:
        print("[INFO] Hugging Face cache settings: using current environment defaults")
        return

    formatted = ", ".join(f"{key}={value}" for key, value in effective_items)
    print(f"[INFO] Hugging Face cache settings: {formatted}")


def prefetch_hub_datasets(task_names: list[str], include_path: str | None = None) -> list[str]:
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager(include_path=include_path)
    loaded = task_manager.load_task_or_group(task_names)
    leaf_tasks = sorted(set(flatten_loaded_task_names(loaded)))
    return leaf_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-mode", choices=["local", "hub"], default="local")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--task-name", type=str, default=DEFAULT_TASK_NAME)
    parser.add_argument(
        "--hf-tasks",
        type=str,
        default=None,
        help="Comma-separated built-in lm_eval task names, e.g. ethics_cm or gsm8k,winogrande.",
    )
    parser.add_argument(
        "--shared-config",
        type=Path,
        default=DEFAULT_SHARED_CONFIG_PATH,
        help="Shared config whose eval.hf_* settings should be reused so cache paths stay aligned with MultiDensity.",
    )
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--verbosity", type=str, default="INFO")
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override HF offline mode. If omitted, reuse MultiTasks/Density_base.yaml.",
    )
    parser.add_argument(
        "--prepare-cache-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only preload/cache datasets for the selected hub tasks. Do not run model evaluation.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def resolve_written_result_path(output_path: Path) -> Path | None:
    if output_path.exists():
        return output_path

    parent = output_path.parent
    pattern = f"{output_path.stem}_*.json"
    candidates = sorted(parent.glob(pattern))
    if candidates:
        return candidates[-1]

    return None


def summarize_results(output_path: Path, task_names: list[str]) -> None:
    resolved_output = resolve_written_result_path(output_path)
    if resolved_output is None:
        print(f"[WARN] Result file not found near: {output_path}")
        return

    with open(resolved_output, "r", encoding="utf-8") as f:
        result = json.load(f)

    print("[OK] Smoke test finished.")
    print(f"[OK] Result file: {resolved_output}")
    for task_name in task_names:
        task_result = result.get("results", {}).get(task_name)
        if not task_result:
            print(f"[WARN] Could not find task results for {task_name} in {resolved_output}")
            continue
        print(f"[OK] Task metrics for {task_name}: {task_result}")


def main() -> int:
    args = parse_args()
    model_path = args.model_path.resolve()
    output_path = args.output_path.resolve()
    shared_config_path = args.shared_config.resolve()

    ensure_exists(model_path, "Model path")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = args.dtype or default_dtype(args.device)
    model_args = f"pretrained={model_path},dtype={dtype},trust_remote_code=True"
    lm_eval_bin = resolve_lm_eval_binary()
    shared_eval_cfg = load_shared_eval_config(shared_config_path)
    env = build_hf_eval_env(shared_eval_cfg, args.offline)

    task_names: list[str]
    dataset_path: Path | None = None
    task_file: Path | None = None
    task_dir: Path | None = None
    logical_tasks: list[str] | None = None

    if args.task_mode == "local":
        dataset_path = args.dataset_path.resolve()
        ensure_exists(dataset_path, "Dataset path")

        task_dir = output_path.parent / "task_config"
        task_file = task_dir / f"{args.task_name}.yaml"
        task_dir.mkdir(parents=True, exist_ok=True)
        task_file.write_text(build_task_yaml(args.task_name, dataset_path), encoding="utf-8")
        task_names = [args.task_name]
    else:
        logical_tasks = resolve_logical_tasks(args.hf_tasks, shared_eval_cfg)
        task_names = expand_lm_tasks(logical_tasks)

        if args.prepare_cache_only:
            print("[INFO] prepare-cache-only mode enabled")
            print("[INFO] logical tasks:", logical_tasks)
            print("[INFO] expanded tasks:", task_names)
            print("[INFO] shared config:", shared_config_path)
            print_hf_env(env)

            leaf_tasks = prefetch_hub_datasets(task_names, resolve_include_path(shared_eval_cfg.get("include_path")))
            print("[OK] Dataset cache warm-up finished.")
            print(f"[OK] Loaded leaf tasks ({len(leaf_tasks)}): {leaf_tasks}")
            return 0

    cmd = [
        lm_eval_bin,
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(task_names),
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--num_fewshot",
        "0",
        "--limit",
        str(args.limit),
        "--output_path",
        str(output_path),
        "--verbosity",
        args.verbosity,
    ]

    if args.task_mode == "local" and task_dir is not None:
        cmd.extend(["--include_path", str(task_dir)])
    elif args.task_mode == "hub":
        include_path = resolve_include_path(shared_eval_cfg.get("include_path"))
        if include_path:
            cmd.extend(["--include_path", include_path])

    print("[INFO] lm_eval binary:", lm_eval_bin)
    print("[INFO] model path:", model_path)
    print("[INFO] task mode:", args.task_mode)
    print("[INFO] shared config:", shared_config_path)
    print_hf_env(env)
    if dataset_path is not None:
        print("[INFO] dataset path:", dataset_path)
    else:
        print("[INFO] logical tasks:", logical_tasks)
        print("[INFO] lm_eval tasks:", task_names)
    if task_file is not None:
        print("[INFO] task yaml:", task_file)
    print("[INFO] output path:", output_path)
    print("[INFO] offline mode:", args.offline)
    print("[INFO] running command:")
    print(" ".join(cmd))

    process = subprocess.run(cmd, env=env, text=True, cwd=REPO_ROOT)
    if process.returncode != 0:
        print(f"[ERROR] lm_eval exited with code {process.returncode}")
        return process.returncode

    summarize_results(output_path, task_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
