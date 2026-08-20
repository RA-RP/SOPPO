from __future__ import annotations

import inspect
import importlib.metadata
from pathlib import Path
from typing import Any

import torch

from .paths import ensure_dir, resolve_repo_path
from .registry import append_jsonl, update_status, validate_checkpoint_record
from .run_builder import build_run_record, load_yaml_config


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _load_trl_distillation(*, allow_vllm: bool = False):
    if not allow_vllm:
        try:
            import trl.import_utils as trl_import_utils

            trl_import_utils.is_vllm_available = lambda min_version=None: False
        except Exception:
            pass
    try:
        from trl.experimental.distillation import DistillationConfig, DistillationTrainer
    except Exception as exc:
        version = _package_version("trl")
        raise RuntimeError(
            "Could not import trl.experimental.distillation. "
            f"Detected trl version: {version}. Install or activate the OPD baseline TRL fork, "
            "for example with `pip install -e C:\\Users\\Administrator\\Desktop\\paper\\mycode\\OPD-Baseline\\trl`."
        ) from exc
    return DistillationConfig, DistillationTrainer


def _jsonl_to_messages_dataset(
    jsonl_path: str | Path,
    *,
    prompt_field: str = "problem",
    completion_field: str | None = None,
    max_samples: int | None = None,
):
    """读取一个 prompt(或 prompt+completion) JSONL，转成含 `messages` 列的 dataset。"""
    from datasets import load_dataset

    resolved = resolve_repo_path(jsonl_path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"JSONL not found: {resolved}")

    dataset = load_dataset("json", data_files=str(resolved), split="train")
    if max_samples:
        dataset = dataset.select(range(min(int(max_samples), len(dataset))))

    if "messages" in dataset.column_names:
        return dataset

    def to_messages(example):
        prompt = str(example.get(prompt_field) or "")
        messages = [{"role": "user", "content": prompt}]
        if completion_field and example.get(completion_field) is not None:
            messages.append({"role": "assistant", "content": str(example[completion_field])})
        return {"messages": messages}

    return dataset.map(to_messages, remove_columns=dataset.column_names)


def _load_prompt_dataset(data_cfg: dict[str, Any]):
    try:
        import datasets  # noqa: F401
    except Exception as exc:
        raise RuntimeError("The `datasets` package is required for TRL distillation smoke runs.") from exc

    prompt_jsonl = data_cfg.get("prompt_jsonl")
    if not prompt_jsonl:
        raise ValueError("data.prompt_jsonl must be set")

    return _jsonl_to_messages_dataset(
        prompt_jsonl,
        prompt_field=str(data_cfg.get("prompt_text_field") or "problem"),
        completion_field=data_cfg.get("completion_text_field"),
        max_samples=data_cfg.get("max_samples"),
    )


def _load_eval_dataset(data_cfg: dict[str, Any]):
    """加载 held-out 验证集（可选）。data.eval_jsonl 指定路径，无则返回 None。"""
    eval_jsonl = data_cfg.get("eval_jsonl")
    if not eval_jsonl:
        return None
    return _jsonl_to_messages_dataset(
        eval_jsonl,
        prompt_field=str(data_cfg.get("prompt_text_field") or "problem"),
        completion_field=data_cfg.get("completion_text_field"),
        max_samples=data_cfg.get("eval_max_samples"),
    )


def _accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return keyword in signature.parameters


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if _accepts_keyword(callable_obj, key)}


def _resolve_student_model_path(
    student_path: str | Path | None,
    base_model: str | None,
) -> str:
    """
    解析 student model 路径。

    如果 student_path 是 LoRA adapter（含 adapter_config.json 但不含 config.json），
    则自动合并 LoRA adapter 与 base_model，返回合并后的模型路径。

    如果 student_path 是完整模型（含 config.json），直接返回。
    如果 student_path 为 None，返回 base_model。
    """
    if student_path is None:
        return base_model or ""

    student_path = Path(student_path)
    if not student_path.exists():
        raise FileNotFoundError(f"Student path not found: {student_path}")

    # 完整模型（有 config.json）
    if (student_path / "config.json").exists():
        return str(student_path)

    # LoRA adapter（有 adapter_config.json）
    if (student_path / "adapter_config.json").exists():
        if not base_model:
            raise ValueError(
                f"Student path {student_path} is a LoRA adapter, but no base_model provided to merge with."
            )
        merged_dir = student_path / "merged_model"
        if (merged_dir / "config.json").exists():
            print(f"  [SKIP] Merged model already exists at {merged_dir}")
            return str(merged_dir)

        print(f"  [MERGE] Merging LoRA adapter {student_path} with base_model {base_model}...")
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            str(student_path),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        merged = model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)

        # 保存 tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(student_path))
        tokenizer.save_pretrained(str(merged_dir))

        print(f"  [MERGE] Saved merged model to {merged_dir}")
        return str(merged_dir)

    raise FileNotFoundError(f"Student path {student_path} is not a model or LoRA adapter")


def _build_peft_config(trl_cfg: dict[str, Any]):
    peft_cfg = trl_cfg.get("peft") or {}
    if not peft_cfg or not bool(peft_cfg.get("enabled", False)):
        return None

    try:
        from peft import LoraConfig, TaskType
    except Exception as exc:
        raise RuntimeError("PEFT LoRA is enabled, but the `peft` package is not importable.") from exc

    task_type_name = str(peft_cfg.get("task_type") or "CAUSAL_LM")
    task_type = getattr(TaskType, task_type_name, task_type_name)
    target_modules = peft_cfg.get("target_modules")
    if isinstance(target_modules, str):
        target_modules = target_modules.strip() or None

    return LoraConfig(
        r=int(peft_cfg.get("r", 16)),
        lora_alpha=int(peft_cfg.get("lora_alpha", 32)),
        lora_dropout=float(peft_cfg.get("lora_dropout", 0.05)),
        bias=str(peft_cfg.get("bias", "none")),
        task_type=task_type,
        target_modules=target_modules,
    )


def _distillation_args(config: dict[str, Any], output_dir: Path):
    trl = config.get("trl") or {}
    DistillationConfig, _ = _load_trl_distillation(allow_vllm=bool(trl.get("use_vllm", False)))
    experiment = config.get("experiment") or {}
    model = config.get("model") or {}

    teacher_server_url = trl.get("teacher_model_server_url")
    kwargs = {
        "output_dir": str(output_dir),
        "seed": int(experiment.get("seed", 42)),
        "model_init_kwargs": trl.get("model_init_kwargs"),
        "teacher_model_name_or_path": model.get("teacher_model"),
        "teacher_model_init_kwargs": trl.get("teacher_model_init_kwargs"),
        "lmbda": float(trl.get("lmbda", 1.0)),
        "beta": float(trl.get("beta", 0.5)),
        "loss_top_k": int(trl.get("loss_top_k", 1)),
        "max_length": int(trl.get("max_length", 4096)),
        "max_completion_length": int(trl.get("max_completion_length", 512)),
        "per_device_train_batch_size": int(trl.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(trl.get("gradient_accumulation_steps", 16)),
        "learning_rate": float(trl.get("learning_rate", 3.0e-5)),
        "max_steps": int(trl.get("max_steps", 100)),
        "save_steps": int(trl.get("save_steps", 25)),
        "save_total_limit": trl.get("save_total_limit"),
        "logging_steps": trl.get("logging_steps"),
        "gradient_checkpointing": bool(trl.get("gradient_checkpointing", False)),
        "optim": trl.get("optim"),
        "bf16": trl.get("bf16"),
        "use_vllm": bool(trl.get("use_vllm", False)),
        "use_teacher_server": bool(trl.get("use_teacher_server", False)),
        "teacher_model_server_url": teacher_server_url,
        "report_to": trl.get("report_to", []),
    }

    # vLLM colocate 提速：student rollout 用同进程内 vLLM 引擎（continuous batching）
    # 仅当 use_vllm 时透传这些字段；None 值在末尾被过滤
    if bool(trl.get("use_vllm", False)):
        kwargs.update(
            {
                "vllm_mode": trl.get("vllm_mode", "colocate"),
                "vllm_gpu_memory_utilization": trl.get("vllm_gpu_memory_utilization"),
                "vllm_max_model_length": trl.get("vllm_max_model_length"),
                "vllm_enable_sleep_mode": trl.get("vllm_enable_sleep_mode"),
                "vllm_tensor_parallel_size": trl.get("vllm_tensor_parallel_size"),
                "vllm_sync_frequency": trl.get("vllm_sync_frequency"),
            }
        )

    # eval / best-checkpoint 选择：当 data.eval_jsonl 存在时启用 held-out eval loss
    has_eval = bool((config.get("data") or {}).get("eval_jsonl"))
    if has_eval:
        eval_steps = int(trl.get("eval_steps", trl.get("save_steps", 25)))
        eval_kwargs = {
            "eval_strategy": "steps",
            "eval_steps": eval_steps,
            "save_steps": eval_steps,  # save 与 eval 对齐
            "per_device_eval_batch_size": int(trl.get("per_device_eval_batch_size", 1)),
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        }
        # load_best_model_at_end 在 vLLM colocate 下不可用：
        # colocate 会初始化 torch.distributed(NCCL)，使 peft 的 _load_best_model
        # 走入 _maybe_shard_state_dict_for_tp（TP 分片）分支，撞上 peft0.19.1 与
        # transformers4.57.6 的版本不兼容（ImportError: EmbeddingParallel）。
        # 故 use_vllm 时仅保留 eval 监控（eval_loss 入日志），保存末态 checkpoint，不回载 best。
        if not bool(trl.get("use_vllm", False)):
            eval_kwargs["load_best_model_at_end"] = True
        kwargs.update(eval_kwargs)

    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return DistillationConfig(**_filter_kwargs(DistillationConfig.__init__, kwargs))


def run_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    experiment = config.get("experiment") or {}
    model_cfg = config.get("model") or {}
    trl_cfg = config.get("trl") or {}

    student_model_raw = (
        model_cfg.get("student_start_checkpoint")
        or model_cfg.get("cold_start_checkpoint")
        or model_cfg.get("base_model")
    )
    if not student_model_raw:
        raise ValueError("Set model.student_start_checkpoint, model.cold_start_checkpoint, or model.base_model")

    # 自动检测并合并 LoRA adapter
    base_model = model_cfg.get("base_model")
    student_model = _resolve_student_model_path(student_model_raw, base_model)

    use_teacher_server = bool(trl_cfg.get("use_teacher_server", False))
    teacher_model = model_cfg.get("teacher_model")
    if not use_teacher_server and not teacher_model:
        raise ValueError("Set model.teacher_model or trl.use_teacher_server=true with teacher_model_server_url")

    output_root = resolve_repo_path(experiment.get("output_root") or "/root/autodl-tmp/exp0609/trl_first_minimal")
    assert output_root is not None
    ensure_dir(output_root)
    output_dir = ensure_dir(output_root / "checkpoint_output")
    registry_path = output_root / "registry" / "run_registry.jsonl"
    checkpoint_registry_path = output_root / "registry" / "checkpoints.jsonl"

    run_record = build_run_record(config, status="running")
    append_jsonl(registry_path, run_record)

    try:
        _, DistillationTrainer = _load_trl_distillation(allow_vllm=bool(trl_cfg.get("use_vllm", False)))
        dataset = _load_prompt_dataset(config.get("data") or {})
        eval_dataset = _load_eval_dataset(config.get("data") or {})
        args = _distillation_args(config, output_dir)
        peft_config = _build_peft_config(trl_cfg)

        trainer_kwargs = {
            "model": student_model,
            "teacher_model": None if use_teacher_server else teacher_model,
            "args": args,
            "train_dataset": dataset,
        }
        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset
        if peft_config is not None:
            if not _accepts_keyword(DistillationTrainer.__init__, "peft_config"):
                raise RuntimeError(
                    "PEFT LoRA is enabled, but this DistillationTrainer does not accept `peft_config`. "
                    "Use the OPD TRL fork with PEFT support or disable trl.peft.enabled."
                )
            trainer_kwargs["peft_config"] = peft_config

        trainer = DistillationTrainer(**trainer_kwargs)
        trainer.train()
        trainer.save_model(str(output_dir))
        # 注意：vLLM colocate 会初始化 torch.distributed(NCCL)。本函数只保存 adapter，
        # 不在此进程内做 merge —— merge（peft adapter 加载）必须在无 distributed 的独立进程做，
        # 否则会走 peft 的 TP 分片分支撞 EmbeddingParallel(peft0.19.1 vs transformers4.57.6)。
        # cycle04 通过 subprocess 隔离每个训练，子进程退出即清理进程组，merge 在父进程进行。
    except Exception:
        update_status(run_record["run_id"], "failed", registry_path)
        raise

    update_status(run_record["run_id"], "completed", registry_path)
    checkpoint_record = {
        **run_record,
        "checkpoint_id": f"{run_record['run_id']}__final",
        "checkpoint_path": str(output_dir),
        "status": "checkpoint_saved",
        "artifacts": {**run_record.get("artifacts", {}), "checkpoint_path": str(output_dir)},
    }
    validate_checkpoint_record(checkpoint_record)
    append_jsonl(checkpoint_registry_path, checkpoint_record)
    return {"run": run_record, "checkpoint": checkpoint_record}
