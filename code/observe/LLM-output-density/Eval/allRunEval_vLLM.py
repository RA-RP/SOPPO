import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from tqdm import tqdm

from Eval.component.Eval2Res import eval2res
from Eval.component.result_files import iter_visible_json_files, split_model_size


REPO_ROOT = Path(__file__).resolve().parents[1]

ETHICS_SUBTASKS = [
    "ethics_deontology",
    "ethics_utilitarianism",
    "ethics_virtue",
    "ethics_justice",
    "ethics_cm",
]

TASK_KEYED_FIELDS = {
    "results",
    "groups",
    "group_subtasks",
    "configs",
    "versions",
    "n-shot",
    "higher_is_better",
    "n-samples",
    "samples",
}

HF_ENV_KEY_MAPPING = {
    "hf_home": "HF_HOME",
    "hf_datasets_cache": "HF_DATASETS_CACHE",
    "hf_hub_cache": "HF_HUB_CACHE",
    "hf_hub_offline": "HF_HUB_OFFLINE",
    "hf_datasets_offline": "HF_DATASETS_OFFLINE",
    "transformers_offline": "TRANSFORMERS_OFFLINE",
}


def load_existing_eval_result(eval_json: Path) -> Optional[Dict[str, Any]]:
    if not eval_json.exists():
        return None

    try:
        with open(eval_json, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    except Exception as e:
        print(f"[WARN] 已存在评测结果但无法读取，将重新评测: {eval_json} | {e}")
        return None

    if not isinstance(eval_data, dict) or not eval_data:
        print(f"[WARN] 已存在评测结果为空或格式异常，将重新评测: {eval_json}")
        return None

    return eval_data


def split_task_results_exist(
    eval_root: str,
    logical_tasks: List[str],
    dataset_name: str,
    max_samples: str,
) -> bool:
    for logical_task in logical_tasks:
        task_json = Path(eval_root) / logical_task / f"{dataset_name}_{max_samples}.json"
        if load_existing_eval_result(task_json) is None:
            return False

    return True


def resolve_lm_eval_binary() -> str:
    """优先使用当前 Python 环境旁边的 lm_eval，避免 PATH 指向错误环境。"""
    local_bin = Path(sys.executable).resolve().with_name("lm_eval")
    if local_bin.exists():
        return str(local_bin)

    path_bin = shutil.which("lm_eval")
    if path_bin:
        return path_bin

    raise FileNotFoundError(
        "未找到 lm_eval，可先激活包含 lm_eval/vllm 的环境，或使用对应环境的 python 运行本脚本。"
    )


def build_hf_eval_env(eval_cfg: Dict[str, Any]) -> Dict[str, str]:
    """构造传给 lm_eval 子进程的 Hugging Face 缓存/离线环境变量。"""
    env = os.environ.copy()
    if eval_cfg.get("allow_code_eval", False):
        env["HF_ALLOW_CODE_EVAL"] = "1"
    else:
        env.pop("HF_ALLOW_CODE_EVAL", None)

    for config_key, env_key in HF_ENV_KEY_MAPPING.items():
        value = eval_cfg.get(config_key)
        if value in (None, ""):
            continue

        if isinstance(value, bool):
            normalized_value = "1" if value else "0"
        else:
            normalized_value = str(value)

        env[env_key] = normalized_value
        # 兼容部分 huggingface_hub 旧环境变量名。
        if env_key == "HF_HUB_CACHE":
            env["HUGGINGFACE_HUB_CACHE"] = normalized_value

    for path_key in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        cache_path = env.get(path_key)
        if cache_path:
            Path(cache_path).mkdir(parents=True, exist_ok=True)

    return env


def print_hf_eval_env(env: Dict[str, str]) -> None:
    tracked_keys = [
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_ALLOW_CODE_EVAL",
    ]
    effective_items = [(key, env.get(key)) for key in tracked_keys if env.get(key)]
    if not effective_items:
        print("📦 Hugging Face 缓存/离线设置: 使用当前环境默认值")
        return

    formatted = ", ".join(f"{key}={value}" for key, value in effective_items)
    print(f"📦 Hugging Face 缓存/离线设置: {formatted}")

    if env.get("HF_HUB_OFFLINE") == "1" or env.get("HF_DATASETS_OFFLINE") == "1":
        print("💡 已启用离线模式；只有已缓存过的数据集才能被 lm_eval 正常加载。")


def expand_lm_tasks(logical_tasks: List[str]) -> List[str]:
    """将逻辑任务展开为 lm_eval 真正执行的 tasks 列表。"""
    expanded_tasks: List[str] = []
    for logical_task in logical_tasks:
        if logical_task == "ethics":
            expanded_tasks.extend(ETHICS_SUBTASKS)
        else:
            expanded_tasks.append(logical_task)
    return expanded_tasks


def contains_unsafe_code_task(tasks: List[str]) -> bool:
    """HumanEval and its variants are marked unsafe by lm-eval."""
    return any(task == "humaneval" or task.startswith("humaneval_") for task in tasks)


def resolve_include_path(include_path: Optional[str]) -> Optional[str]:
    if not include_path:
        return None
    resolved = Path(include_path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return str(resolved.resolve())


def task_key_matches(logical_task: str, task_key: str) -> bool:
    """判断 lm_eval 输出中的 task key 是否属于某个逻辑任务。"""
    if logical_task == "ethics":
        return task_key == "ethics" or task_key.startswith("ethics_")
    if logical_task == "truthfulqa":
        return task_key == "truthfulqa" or task_key.startswith("truthfulqa_")
    if logical_task == "mmlu":
        return task_key == "mmlu" or task_key.startswith("mmlu_")
    return task_key == logical_task or task_key.startswith(f"{logical_task}_")


def filter_task_mapping(data: Dict[str, Any], logical_task: str) -> Dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if task_key_matches(logical_task, key)
    }


def build_task_specific_result(eval_data: Dict[str, Any], logical_task: str) -> Dict[str, Any]:
    """从一次 combined lm_eval 结果中，切出单个逻辑任务需要的部分。"""
    task_result: Dict[str, Any] = {}

    for key, value in eval_data.items():
        if key in TASK_KEYED_FIELDS and isinstance(value, dict):
            filtered_value = filter_task_mapping(value, logical_task)
            if filtered_value:
                task_result[key] = filtered_value
        else:
            task_result[key] = value

    return task_result


def write_split_results(
    eval_data: Dict[str, Any],
    logical_tasks: List[str],
    eval_root: str,
    dataset_name: str,
    max_samples: str,
) -> None:
    """把一次 combined 评测结果拆回原先按任务分目录的结构。"""
    for logical_task in logical_tasks:
        task_result = build_task_specific_result(eval_data, logical_task)
        results = task_result.get("results", {})
        groups = task_result.get("groups", {})

        if not results and not groups:
            print(f"[⚠️ Warning] {dataset_name}-{max_samples} 未拆出 {logical_task} 的结果，已跳过写入。")
            continue

        task_dir = Path(eval_root) / logical_task
        task_dir.mkdir(parents=True, exist_ok=True)
        task_json = task_dir / f"{dataset_name}_{max_samples}.json"

        with open(task_json, "w", encoding="utf-8") as f:
            json.dump(task_result, f, indent=2, ensure_ascii=False)


def get_existing_combined_candidates(
    eval_root: str,
    dataset_name: str,
    max_samples: str,
) -> List[Path]:
    combined_dir = Path(eval_root) / "_combined"
    if not combined_dir.exists():
        return []

    candidates: List[Path] = []
    for combined_path in iter_visible_json_files(str(combined_dir)):
        model, size, _ = split_model_size(combined_path.stem)
        if model == dataset_name and size == str(max_samples):
            candidates.append(combined_path)

    return sorted(
        candidates,
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )


def load_existing_combined_result(
    eval_root: str,
    dataset_name: str,
    max_samples: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Path]]:
    for combined_path in get_existing_combined_candidates(eval_root, dataset_name, max_samples):
        existing_eval = load_existing_eval_result(combined_path)
        if existing_eval is not None:
            return existing_eval, combined_path

    return None, None


def split_available_combined_results(eval_root: str, logical_tasks: List[str]) -> None:
    combined_dir = Path(eval_root) / "_combined"
    if not combined_dir.exists():
        print(f"[INFO] 未找到 combined 目录，跳过拆分: {combined_dir}")
        return

    latest_by_run: Dict[tuple[str, str], Path] = {}
    for combined_path in iter_visible_json_files(str(combined_dir)):
        dataset_name, max_samples, _ = split_model_size(combined_path.stem)
        if not dataset_name or not max_samples:
            print(f"[WARN] combined 文件名无法解析，已跳过: {combined_path.name}")
            continue

        key = (dataset_name, max_samples)
        current = latest_by_run.get(key)
        if current is None or (
            combined_path.stat().st_mtime,
            combined_path.name,
        ) > (
            current.stat().st_mtime,
            current.name,
        ):
            latest_by_run[key] = combined_path

    if not latest_by_run:
        print(f"[INFO] combined 目录中没有可拆分的 JSON: {combined_dir}")
        return

    for (dataset_name, max_samples), combined_path in sorted(latest_by_run.items()):
        existing_eval = load_existing_eval_result(combined_path)
        if existing_eval is None:
            continue

        print(f"[SPLIT] {combined_path} -> {Path(eval_root).resolve()}/<task>/{dataset_name}_{max_samples}.json")
        write_split_results(existing_eval, logical_tasks, eval_root, dataset_name, max_samples)


def run_eval_task(
    task: Dict[str, Any],
    model_output_root: str,
    eval_root: str,
    logical_tasks: List[str],
    device: str,
    batch_size: int,
    vllm_model_len: int,
    gpu_memory_utilization_rate: float,
    lm_eval_bin: str,
    hf_env: Dict[str, str],
    skip_existing: bool,
    include_path: Optional[str],
    fail_on_error: bool = True,
    confirm_run_unsafe_code: bool = False,
    eval_limit: Optional[int] = None,
    apply_chat_template: bool = True,
    max_gen_toks: Optional[int] = None,
    enable_thinking: bool = False,
):
    """对单个 merge 后模型只启动一次 lm_eval/vLLM，再拆分为各 benchmark 结果。"""
    dataset_name, max_samples = task["dataset"], str(task["max_samples"])
    model_dir = Path(model_output_root) / dataset_name / max_samples

    expanded_tasks = expand_lm_tasks(logical_tasks)
    has_unsafe_code_task = contains_unsafe_code_task(expanded_tasks)
    if has_unsafe_code_task and not confirm_run_unsafe_code:
        raise RuntimeError(
            "当前 eval.lm_tasks 包含 humaneval，它在 lm-eval 中标记为 unsafe_code。"
            "请在 configs/stages/eval.yaml 中设置 confirm_run_unsafe_code: true，"
            "或从 eval.lm_tasks 与 eval.target_metrics 中移除 humaneval。"
        )

    combined_dir = Path(eval_root) / "_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_json = combined_dir / f"{dataset_name}_{max_samples}.json"

    if skip_existing:
        existing_eval, existing_path = load_existing_combined_result(eval_root, dataset_name, max_samples)
        if existing_eval is not None:
            print(f"[SKIP] 复用已有 combined 评测结果: {dataset_name}-{max_samples} -> {existing_path}")
            write_split_results(existing_eval, logical_tasks, eval_root, dataset_name, max_samples)
            return existing_eval

        if split_task_results_exist(eval_root, logical_tasks, dataset_name, max_samples):
            print(
                f"[SKIP] 复用已有拆分评测结果: {dataset_name}-{max_samples} -> {Path(eval_root).resolve()}"
            )
            return {"dataset": dataset_name, "max_samples": max_samples, "reused_existing": True}

    if not model_dir.exists():
        raise FileNotFoundError(f"未找到统一模型输出目录: {model_dir}")

    model_args = (
        f"pretrained={model_dir},"
        f"tensor_parallel_size=1,"
        f"max_model_len={vllm_model_len},"
        f"gpu_memory_utilization={gpu_memory_utilization_rate},"
        f"enable_thinking={str(enable_thinking).lower()}"
    )
    tasks_arg = ",".join(expanded_tasks)

    cmd = [
        lm_eval_bin,
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        tasks_arg,
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(combined_json),
        "--num_fewshot",
        "0",
    ]
    if apply_chat_template:
        # Without this, generate_until tasks (gsm8k, hendrycks_math500, ...)
        # get a raw completion-style prompt on an instruction-tuned model,
        # which depresses scores independently of model capability — see
        # FINDING_05_gsm8k_chat_template_mismatch.md. The ID/NuminaMath eval
        # script already applies the chat template; this brings the lm_eval
        # tasks in line with it.
        cmd.append("--apply_chat_template")
    if max_gen_toks is not None:
        # lm_eval's vLLM model defaults to max_gen_toks=256, which truncates
        # Qwen3 <think> reasoning before it reaches a final answer on
        # generate_until tasks (gsm8k, hendrycks_math500, ...) — see
        # FINDING_05_gsm8k_chat_template_mismatch.md.
        cmd.extend(["--gen_kwargs", f"max_gen_toks={max_gen_toks}"])
    if has_unsafe_code_task and confirm_run_unsafe_code:
        cmd.append("--confirm_run_unsafe_code")

    if eval_limit is not None:
        cmd.extend(["--limit", str(eval_limit)])

    if include_path:
        cmd.extend(["--include_path", include_path])

    print(
        f"🚀 Running combined eval on {dataset_name}-{max_samples} ({device}) | "
        f"logical_tasks={logical_tasks} | expanded_tasks={expanded_tasks}"
    )
    process = subprocess.run(cmd, text=True, env=hf_env)
    if process.returncode != 0:
        message = f"{dataset_name}-{max_samples}: lm_eval 退出码 {process.returncode}"
        print(f"[❌ Error] {message}")
        if has_unsafe_code_task:
            print(
                "[💡 Hint] 当前 combined 任务包含 HumanEval，unsafe 确认已按配置处理；"
                "如果 traceback 指向其它 benchmark，请优先按 traceback 中的具体 task 处理。"
            )
        elif hf_env.get("HF_HUB_OFFLINE") == "1" or hf_env.get("HF_DATASETS_OFFLINE") == "1":
            print(
                "[💡 Hint] 当前启用了 Hugging Face 离线模式；"
                "如果 benchmark 数据还没有缓存到本地，首次评测会失败。"
            )
        else:
            print(
                "[💡 Hint] 这些 benchmark 默认通过 Hugging Face datasets 从 Hub 加载；"
                "如果当前环境不能访问 Hub，请先联网缓存一次，或改用已缓存的本地目录。"
            )
        if fail_on_error:
            raise RuntimeError(message)
        return None

    eval_data = load_existing_eval_result(combined_json)
    if eval_data is None:
        print(f"[⚠️ Warning] Missing combined eval file: {combined_json}")
        return None

    write_split_results(eval_data, logical_tasks, eval_root, dataset_name, max_samples)
    return eval_data


def run_eval_vllm(config):
    eval_cfg = config["eval"]
    model_output_root = eval_cfg.get("model_output_root") or eval_cfg.get("merge_root")
    eval_root = eval_cfg["output_origin_root"]
    batch_size = eval_cfg.get("batch_size", 8)
    lm_tasks = eval_cfg.get("lm_tasks", ["mmlu"])
    dataset_tasks = eval_cfg["tasks"]
    enable_eval = eval_cfg.get("enable_eval", True)
    enable_result = eval_cfg.get("enable_result", False)
    skip_existing = eval_cfg.get("skip_existing", True)
    fail_on_error = eval_cfg.get("fail_on_eval_error", True)
    confirm_run_unsafe_code = eval_cfg.get("confirm_run_unsafe_code", False)
    gpu_memory_utilization_rate = eval_cfg.get("gpu_memory_utilization", 0.7)
    vllm_model_len = eval_cfg.get("vllm_model_len", 4096)
    eval_limit = eval_cfg.get("eval_limit")
    apply_chat_template = eval_cfg.get("apply_chat_template", True)
    max_gen_toks = eval_cfg.get("max_gen_toks")
    enable_thinking = eval_cfg.get("enable_thinking", False)
    include_path = resolve_include_path(eval_cfg.get("include_path"))
    num_gpus = torch.cuda.device_count()
    device = "cuda:0" if num_gpus > 0 else "cpu"
    lm_eval_bin = resolve_lm_eval_binary()
    hf_env = build_hf_eval_env(eval_cfg)

    print(f"🧠 Detected {num_gpus} GPUs, using device: {device}")
    print(f"🛠️ Using lm_eval binary: {lm_eval_bin}")
    print(f"💬 apply_chat_template: {apply_chat_template}")
    print(f"🧠 enable_thinking: {enable_thinking}")
    print(f"📁 Model output root: {model_output_root}")
    if include_path:
        print(f"📎 lm_eval include_path: {include_path}")
    print_hf_eval_env(hf_env)

    # ==== Step 1: 执行评估 ====
    if enable_eval:
        print("\n===== 🚀 Step 1: Running combined model evaluations =====")
        print(f"🧩 每个模型只启动一次 vLLM，统一评测这些任务: {lm_tasks}")
        print(f"♻️ 评测热重启: {'开启' if skip_existing else '关闭'}")

        for task in tqdm(dataset_tasks, desc="Evaluating datasets"):
            run_eval_task(
                task=task,
                model_output_root=model_output_root,
                eval_root=eval_root,
                logical_tasks=lm_tasks,
                device=device,
                batch_size=batch_size,
                vllm_model_len=vllm_model_len,
                gpu_memory_utilization_rate=gpu_memory_utilization_rate,
                lm_eval_bin=lm_eval_bin,
                hf_env=hf_env,
                skip_existing=skip_existing,
                include_path=include_path,
                fail_on_error=fail_on_error,
                confirm_run_unsafe_code=confirm_run_unsafe_code,
                eval_limit=eval_limit,
                apply_chat_template=apply_chat_template,
                max_gen_toks=max_gen_toks,
                enable_thinking=enable_thinking,
            )

        print("\n🎯 所有评估任务完成，结果已按 benchmark 拆分保存")
    else:
        print("\n⏭️ 跳过 Step 1: 评估阶段 (enable_eval=False)")

    # ==== Step 2: 评估结果转化 ====
    if enable_result:
        print("\n===== 🧮 Step 2: Splitting JSON and building target metrics CSV =====")
        try:
            split_available_combined_results(eval_root, lm_tasks)
            eval2res(config)
            print("✅ Step 2 完成：结果已转化为 target_metrics CSV。")
        except Exception as e:
            print(f"❌ Step 2 出错: {e}")
    else:
        print("\n⏭️ 跳过 Step 2: 结果转化阶段 (enable_result=False)")


if __name__ == "__main__":
    config_path = "./Eval/eval_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    run_eval_vllm(config)
