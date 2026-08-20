# coding: utf8
import argparse
import gc
import json
import math
import os

import torch

from utils.config_utils import load_config, validate_config_keys
from utils.data_utils import get_token_data_from_jsonl
from utils.model_utils import get_model_from_huggingface
from utils.profiling_utils import (
    ensure_svd_sqrt_profile,
    profile_svdllm_low_resource,
    profile_svdllm_single_layer_group,
    whitening,
)


REQUIRED_CONFIG_KEYS = {
    "model",
    "save_path",
    "DEV",
    "model_seq_len",
    "seed",
    "mode",
}

OPTIONAL_CONFIG_KEYS = {
    "epsilon",
    "model_dtype",
    "trust_remote_code",
    "hf_cache_dir",
    "s_nsamples",
    "x_nsamples",
    "s_jsonl_path",
    "x_jsonl_path",
    "x_batch_size",
    "s_batch_size",
    "x_data_cache_path",
    "s_data_cache_path",
    "profiling_mat_path_s",
    "save_profile_s_path",
    "profiling_mat_path_x",
    "save_profile_x_path",
    "save_metrics_pt_path",
    "save_metrics_json_path",
    "save_s_json_path",
    "save_x_json_path",
    "save_s_pt_path",
    "save_x_pt_path",
    "save_s_uv_path",
    "save_x_uv_path",
    "tasks",
    "s_jsonl_file",
    "activation_cache_device",
    "uv_dtype",
    "cleanup_intermediate",
    "skip_existing_outputs",
    "svd_singular_floor",
    "cholesky_jitter",
    "whitening_nsamples",
    "X_nsamples",
    "task",
    "target_layer",
    "layer_gpu_chunk_size",
    "single_layer_task_group_size",
}

DEFAULT_TASK_JSONL_FILE = "gsm8k.jsonl"

MODE_TO_SIDES = {
    "s_only_svd": ("s",),
    "x_only_svd": ("x",),
    "split_whitened_svd": ("s", "x"),
}

MODE_ALIASES = {
    "svdllm_step1": "s_only_svd",
    "s_only": "s_only_svd",
    "x_only": "x_only_svd",
    "xs_whitened_svd": "split_whitened_svd",
    "split_svd": "split_whitened_svd",
}


def _safe_json_value(value):
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {k: _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json_value(v) for v in value]
    return value


def _cfg_or_default(cfg: dict, key: str, default):
    value = cfg.get(key, None)
    return default if value is None else value


def _parse_target_layer(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
        if value.startswith("layer_"):
            value = value.split("_", 1)[1]
    try:
        layer_idx = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError("target_layer must be null, an integer, or a string like 'layer_5'") from e
    if layer_idx < 0:
        raise ValueError("target_layer must be >= 0")
    return layer_idx


def _layer_label(layer_idx: int) -> str:
    return f"layer_{int(layer_idx)}"


def _parse_layer_gpu_chunk_size(value) -> int:
    if value is None:
        return 1
    try:
        chunk_size = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError("layer_gpu_chunk_size must be a positive integer") from e
    if chunk_size <= 0:
        raise ValueError("layer_gpu_chunk_size must be > 0")
    return chunk_size


def _parse_task_group_size(value, pending_count: int) -> int:
    if value is None:
        return 4
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped == "all":
            return max(1, int(pending_count))
        value = stripped
    try:
        group_size = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError("single_layer_task_group_size must be a positive integer or 'all'") from e
    if group_size <= 0:
        raise ValueError("single_layer_task_group_size must be > 0")
    return group_size


def _normalize_legacy_config(cfg: dict) -> dict:
    # Backward compatibility with historical key names.
    if "s_nsamples" not in cfg:
        if "X_nsamples" in cfg:
            cfg["s_nsamples"] = cfg["X_nsamples"]
            print("[Warn] 'X_nsamples' is deprecated, please use 's_nsamples'.")
        elif "whitening_nsamples" in cfg:
            cfg["s_nsamples"] = cfg["whitening_nsamples"]
            print("[Warn] 'whitening_nsamples' is deprecated, please use 's_nsamples'.")
    if "mode" in cfg:
        raw_mode = str(cfg["mode"]).strip()
        cfg["mode"] = MODE_ALIASES.get(raw_mode, raw_mode)
    return cfg


def _path_tag(path: str) -> str:
    base = os.path.basename(str(path))
    stem, _ = os.path.splitext(base)
    if not stem:
        stem = "data"
    return stem.replace("-", "_").replace(".", "_")


def _derive_task_from_s_jsonl(s_jsonl_path: str) -> str:
    parent = os.path.basename(os.path.dirname(os.path.normpath(str(s_jsonl_path))))
    if not parent:
        return "task"
    return parent.replace("-", "_").replace(".", "_")


def _jsonl_filename(cfg: dict) -> str:
    filename = str(_cfg_or_default(cfg, "s_jsonl_file", DEFAULT_TASK_JSONL_FILE)).strip()
    if not filename:
        raise ValueError("'s_jsonl_file' must not be empty")
    return filename


def _is_empty_tasks_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple)):
        return len(value) == 0 or all(_is_empty_tasks_value(v) for v in value)
    return False


def _coerce_tasks(value) -> list:
    if _is_empty_tasks_value(value):
        return []
    raw_tasks = value if isinstance(value, (list, tuple)) else [value]
    tasks = []
    seen = set()
    for raw_task in raw_tasks:
        if _is_empty_tasks_value(raw_task):
            continue
        task = str(raw_task).strip()
        if task in seen:
            continue
        seen.add(task)
        tasks.append(task)
    return tasks


def _discover_tasks_from_s_root(s_root: str, jsonl_file: str) -> list:
    if os.path.isfile(s_root):
        task = os.path.basename(os.path.dirname(os.path.normpath(s_root)))
        if not task:
            raise ValueError(f"Cannot derive task from s_jsonl_path file: {s_root}")
        print(f"[Info] s_jsonl_path is a file. Use derived single task: {task}")
        return [task]

    if not os.path.isdir(s_root):
        raise FileNotFoundError(f"[S] s_jsonl_path root not found: {s_root}")

    tasks = []
    for entry in sorted(os.listdir(s_root)):
        task_dir = os.path.join(s_root, entry)
        jsonl_path = os.path.join(task_dir, jsonl_file)
        if os.path.isdir(task_dir) and os.path.isfile(jsonl_path):
            tasks.append(entry)

    if not tasks:
        raise FileNotFoundError(
            f"No task folders found under {s_root}. Expected files like: "
            f"{os.path.join(s_root, '{task}', jsonl_file)}"
        )
    return tasks


def _resolve_tasks(cfg: dict) -> list:
    if "tasks" in cfg:
        tasks = _coerce_tasks(cfg.get("tasks", None))
        if tasks:
            return tasks
        return _discover_tasks_from_s_root(cfg["s_jsonl_path"], _jsonl_filename(cfg))

    if "task" in cfg:
        tasks = _coerce_tasks(cfg.get("task", None))
        if tasks:
            print("[Warn] 'task' is deprecated, please use 'tasks'.")
            return tasks
        print("[Warn] Empty 'task' is deprecated, please use empty/null 'tasks'.")
        return _discover_tasks_from_s_root(cfg["s_jsonl_path"], _jsonl_filename(cfg))

    return _discover_tasks_from_s_root(cfg["s_jsonl_path"], _jsonl_filename(cfg))


def _resolve_s_jsonl_for_task(cfg: dict, task: str) -> str:
    s_root = cfg["s_jsonl_path"]
    if os.path.isfile(s_root):
        return s_root
    return os.path.join(s_root, task, _jsonl_filename(cfg))


def _format_task_template(value, task: str):
    if value is None:
        return None
    value = str(value)
    if "{task}" not in value:
        return value
    try:
        return value.format(task=task)
    except Exception as e:
        print(f"[Warn] Failed to format '{{task}}' in '{value}': {e}")
        return value


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _output_root_for_task(cfg: dict, task: str) -> str:
    output_root = os.path.join(cfg["save_path"], task)
    target_layer = cfg.get("target_layer", None)
    if target_layer is not None:
        output_root = os.path.join(output_root, _layer_label(target_layer))
    return output_root


def _build_default_output_paths(cfg: dict, output_root: str) -> dict:
    model_name_safe = os.path.basename(cfg["model"]).replace("-", "_")
    x_tag = _path_tag(cfg.get("x_jsonl_path", "x"))
    s_tag = _path_tag(cfg.get("s_jsonl_path", "s"))
    s_json_name = _format_task_template(cfg.get("save_s_json_path", "sMat_{task}.json"), cfg["task"])
    x_json_name = _format_task_template(cfg.get("save_x_json_path", "xMat_{task}.json"), cfg["task"])
    s_uv_name = _format_task_template(cfg.get("save_s_uv_path", "sUV_{task}.pt"), cfg["task"])
    x_uv_name = _format_task_template(cfg.get("save_x_uv_path", "xUV_{task}.pt"), cfg["task"])

    return {
        "x_profile": os.path.join(
            output_root,
            f"{model_name_safe}_profiling_X_{x_tag}_{cfg.get('x_nsamples', 'na')}_{cfg['seed']}.pt",
        ),
        "s_profile": os.path.join(
            output_root,
            f"{model_name_safe}_profiling_S_{s_tag}_{cfg.get('s_nsamples', 'na')}_{cfg['seed']}.pt",
        ),
        "s_json": os.path.join(output_root, s_json_name),
        "x_json": os.path.join(output_root, x_json_name),
        "s_uv": os.path.join(output_root, s_uv_name) if s_uv_name is not None else None,
        "x_uv": os.path.join(output_root, x_uv_name) if x_uv_name is not None else None,
        "x_data_cache": os.path.join(
            output_root,
            f"{model_name_safe}_x_data_{x_tag}_{cfg.get('x_nsamples', 'na')}_{cfg['model_seq_len']}_{cfg['seed']}.pt",
        ),
        "s_data_cache": os.path.join(
            output_root,
            f"{model_name_safe}_s_data_{s_tag}_{cfg.get('s_nsamples', 'na')}_{cfg['model_seq_len']}_{cfg['seed']}.pt",
        ),
    }


def _set_global_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_mode(cfg: dict) -> str:
    mode = str(cfg["mode"]).strip()
    mode = MODE_ALIASES.get(mode, mode)
    if mode not in MODE_TO_SIDES:
        raise ValueError(f"Unsupported mode '{cfg['mode']}'. Choose one of: {sorted(MODE_TO_SIDES)}")
    return mode


def _require_side_config(cfg: dict, sides):
    missing = []
    for side in sides:
        for key in (f"{side}_nsamples", f"{side}_jsonl_path"):
            if key not in cfg:
                missing.append(key)
    if missing:
        raise KeyError(f"Missing required side config keys for mode '{cfg['mode']}': {sorted(missing)}")


def _validate_side_paths(cfg: dict, sides):
    for side in sides:
        path_key = f"{side}_jsonl_path"
        if not os.path.exists(cfg[path_key]):
            raise FileNotFoundError(f"[{side.upper()}] {path_key} not found: {cfg[path_key]}")


def _model_layer_count(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return len(model.model.decoder.layers)
    raise AttributeError("Cannot resolve transformer layers from model")


def _validate_target_layer_for_model(target_layer, model):
    if target_layer is None:
        return
    num_layers = _model_layer_count(model)
    if target_layer >= num_layers:
        raise ValueError(
            f"target_layer={target_layer} out of range. "
            f"Valid range: 0..{num_layers - 1}"
        )


def _profile_key_to_layer_index(layer_key) -> int:
    if isinstance(layer_key, str) and layer_key.startswith("layer_"):
        return int(layer_key.split("_", 1)[1])
    return int(layer_key)


def _filter_profile_to_target_layer(profiling_mat, target_layer: int):
    for layer_key, layer_profile in profiling_mat.items():
        if _profile_key_to_layer_index(layer_key) == int(target_layer):
            return {layer_key: layer_profile}
    available = ", ".join(str(k) for k in sorted(profiling_mat.keys(), key=_profile_key_to_layer_index))
    raise KeyError(f"target_layer={target_layer} not found in profiling_mat. Available layers: {available}")


def _side_output_path(cfg: dict, defaults: dict, side: str, kind: str):
    key = f"save_{side}_{kind}_path"
    default = defaults.get(f"{side}_{kind}")
    value = cfg.get(key, None)
    if value is None:
        return _format_task_template(default, cfg["task"])

    path = _format_task_template(value, cfg["task"])
    if path is None or os.path.isabs(path):
        return path

    base_default = default or defaults.get(f"{side}_json") or defaults.get(f"{side}_uv")
    if base_default is None:
        return path
    return os.path.join(os.path.dirname(base_default), path)


def _final_output_paths(cfg: dict, defaults: dict, side: str) -> list:
    paths = [_side_output_path(cfg, defaults, side, "json")]
    for kind in ("pt", "uv"):
        path = _side_output_path(cfg, defaults, side, kind)
        if path is not None:
            paths.append(path)
    return paths


def _outputs_complete(paths) -> bool:
    return bool(paths) and all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in paths)


def _should_skip_existing_outputs(cfg: dict, defaults: dict, side: str, label: str) -> bool:
    if not bool(_cfg_or_default(cfg, "skip_existing_outputs", True)):
        return False

    output_paths = _final_output_paths(cfg, defaults, side)
    if not _outputs_complete(output_paths):
        return False

    print(f"[Skip] {label} already has complete outputs:")
    for path in output_paths:
        print(f"  {path}")
    return True


def _cleanup_intermediate_files(intermediate_files_to_cleanup, protected_outputs):
    for temp_path in sorted(set(intermediate_files_to_cleanup)):
        if temp_path is None:
            continue
        temp_abs = os.path.abspath(temp_path)
        if temp_abs in protected_outputs:
            continue
        if os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
                print(f"[Cleanup] Removed intermediate file: {temp_path}")
            except OSError as e:
                print(f"[Warn] Failed to remove intermediate file '{temp_path}': {e}")


def _run_whitening_and_save(
    side: str,
    cfg: dict,
    defaults: dict,
    model,
    dev,
    profiling_mat,
    protected_outputs,
):
    side_upper = side.upper()
    uv_dtype = _cfg_or_default(cfg, "uv_dtype", "float32")
    uv_path = _side_output_path(cfg, defaults, side, "uv")

    print(f"[{side_upper}] Running whitening SVD ...")
    sigma_dict, uv_dict = whitening(
        model_name=cfg["model"],
        model=model,
        profiling_mat=profiling_mat,
        dev=dev,
        uv_dtype=uv_dtype,
        return_uv=uv_path is not None,
    )

    json_path = _side_output_path(cfg, defaults, side, "json")
    _ensure_parent_dir(json_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_safe_json_value(sigma_dict), f, indent=2)
    protected_outputs.add(os.path.abspath(json_path))
    print(f"[{side_upper}] Singular values json saved to: {json_path}")

    pt_path = _side_output_path(cfg, defaults, side, "pt")
    if pt_path is not None:
        _ensure_parent_dir(pt_path)
        torch.save(sigma_dict, pt_path)
        protected_outputs.add(os.path.abspath(pt_path))
        print(f"[{side_upper}] Singular values pt saved to: {pt_path}")

    if uv_path is not None:
        _ensure_parent_dir(uv_path)
        torch.save(uv_dict, uv_path)
        protected_outputs.add(os.path.abspath(uv_path))
        print(f"[{side_upper}] U/S/VT pt saved to: {uv_path}")

    del sigma_dict, uv_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return json_path


def _run_svd_side(
    side: str,
    cfg: dict,
    defaults: dict,
    model,
    tokenizer,
    dev,
    intermediate_files_to_cleanup,
    protected_outputs,
):
    side_upper = side.upper()
    singular_floor = float(_cfg_or_default(cfg, "svd_singular_floor", 0.0))
    cholesky_jitter = float(_cfg_or_default(cfg, "cholesky_jitter", 1e-4))
    cleanup_intermediate = bool(_cfg_or_default(cfg, "cleanup_intermediate", True))
    activation_cache_device = _cfg_or_default(cfg, "activation_cache_device", "cuda")

    profile_load_path = _format_task_template(
        _cfg_or_default(cfg, f"profiling_mat_path_{side}", None),
        cfg["task"],
    )
    if profile_load_path is not None and os.path.exists(profile_load_path):
        print(f"[{side_upper}] Loading existing profiling_mat: {profile_load_path}")
        profiling_mat = torch.load(profile_load_path, map_location="cpu")
        profiling_mat = ensure_svd_sqrt_profile(
            profiling_mat,
            singular_floor=singular_floor,
            cholesky_jitter=cholesky_jitter,
        )
    else:
        data_cache_path = _format_task_template(
            _cfg_or_default(cfg, f"{side}_data_cache_path", defaults[f"{side}_data_cache"]),
            cfg["task"],
        )

        print(f"[{side_upper}] Building token data from existing jsonl: {cfg[f'{side}_jsonl_path']}")
        side_data = get_token_data_from_jsonl(
            jsonl_path=cfg[f"{side}_jsonl_path"],
            tokenizer=tokenizer,
            nsamples=int(cfg[f"{side}_nsamples"]),
            seqlen=int(cfg["model_seq_len"]),
            seed=int(cfg["seed"]),
            batch_size=int(_cfg_or_default(cfg, f"{side}_batch_size", 1)),
            cache_file=data_cache_path,
            mode=side,
        )
        if cleanup_intermediate and cfg.get(f"{side}_data_cache_path", None) is None:
            intermediate_files_to_cleanup.append(data_cache_path)

        print(f"[{side_upper}] Running SVDLLM-style profiling ...")
        profiling_mat = profile_svdllm_low_resource(
            model_name=cfg["model"],
            model=model,
            calib_loader=side_data,
            dev=dev,
            singular_floor=singular_floor,
            activation_cache_device=activation_cache_device,
            cholesky_jitter=cholesky_jitter,
        )
        del side_data

        profile_save_path = _format_task_template(
            _cfg_or_default(cfg, f"save_profile_{side}_path", defaults[f"{side}_profile"]),
            cfg["task"],
        )
        _ensure_parent_dir(profile_save_path)
        torch.save(profiling_mat, profile_save_path)
        if cleanup_intermediate and cfg.get(f"save_profile_{side}_path", None) is None:
            intermediate_files_to_cleanup.append(profile_save_path)
        print(f"[{side_upper}] profiling_mat saved to: {profile_save_path}")

    json_path = _run_whitening_and_save(
        side=side,
        cfg=cfg,
        defaults=defaults,
        model=model,
        dev=dev,
        profiling_mat=profiling_mat,
        protected_outputs=protected_outputs,
    )

    del profiling_mat
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return json_path


def _load_existing_single_layer_profile(side: str, cfg: dict, target_layer: int):
    side_upper = side.upper()
    singular_floor = float(_cfg_or_default(cfg, "svd_singular_floor", 0.0))
    cholesky_jitter = float(_cfg_or_default(cfg, "cholesky_jitter", 1e-4))
    profile_load_path = _format_task_template(
        _cfg_or_default(cfg, f"profiling_mat_path_{side}", None),
        cfg["task"],
    )
    if profile_load_path is None or not os.path.exists(profile_load_path):
        return None

    print(f"[{side_upper}] Loading existing profiling_mat: {profile_load_path}")
    profiling_mat = torch.load(profile_load_path, map_location="cpu")
    profiling_mat = ensure_svd_sqrt_profile(
        profiling_mat,
        singular_floor=singular_floor,
        cholesky_jitter=cholesky_jitter,
    )
    return _filter_profile_to_target_layer(profiling_mat, target_layer)


def _build_side_data(
    side: str,
    cfg: dict,
    defaults: dict,
    tokenizer,
    intermediate_files_to_cleanup,
):
    cleanup_intermediate = bool(_cfg_or_default(cfg, "cleanup_intermediate", True))
    data_cache_path = _format_task_template(
        _cfg_or_default(cfg, f"{side}_data_cache_path", defaults[f"{side}_data_cache"]),
        cfg["task"],
    )

    print(f"[{side.upper()}] Building token data from existing jsonl: {cfg[f'{side}_jsonl_path']}")
    side_data = get_token_data_from_jsonl(
        jsonl_path=cfg[f"{side}_jsonl_path"],
        tokenizer=tokenizer,
        nsamples=int(cfg[f"{side}_nsamples"]),
        seqlen=int(cfg["model_seq_len"]),
        seed=int(cfg["seed"]),
        batch_size=int(_cfg_or_default(cfg, f"{side}_batch_size", 1)),
        cache_file=data_cache_path,
        mode=side,
    )
    if cleanup_intermediate and cfg.get(f"{side}_data_cache_path", None) is None:
        intermediate_files_to_cleanup.append(data_cache_path)
    return side_data


def _save_profile_for_side(side: str, cfg: dict, defaults: dict, profiling_mat, intermediate_files_to_cleanup):
    cleanup_intermediate = bool(_cfg_or_default(cfg, "cleanup_intermediate", True))
    profile_save_path = _format_task_template(
        _cfg_or_default(cfg, f"save_profile_{side}_path", defaults[f"{side}_profile"]),
        cfg["task"],
    )
    _ensure_parent_dir(profile_save_path)
    torch.save(profiling_mat, profile_save_path)
    if cleanup_intermediate and cfg.get(f"save_profile_{side}_path", None) is None:
        intermediate_files_to_cleanup.append(profile_save_path)
    print(f"[{side.upper()}] profiling_mat saved to: {profile_save_path}")


def _single_layer_task_cfg_and_defaults(cfg: dict, task: str) -> tuple:
    task_cfg = dict(cfg)
    task_cfg["task"] = task
    if task != "X":
        task_cfg["s_jsonl_path"] = _resolve_s_jsonl_for_task(cfg, task)
    output_root = _output_root_for_task(task_cfg, task)
    defaults = _build_default_output_paths(task_cfg, output_root=output_root)
    return task_cfg, defaults, output_root


def _run_single_layer_s_task_group(
    task_group,
    cfg: dict,
    model,
    tokenizer,
    dev,
) -> list:
    target_layer = cfg["target_layer"]
    singular_floor = float(_cfg_or_default(cfg, "svd_singular_floor", 0.0))
    cholesky_jitter = float(_cfg_or_default(cfg, "cholesky_jitter", 1e-4))
    activation_cache_device = _cfg_or_default(cfg, "activation_cache_device", "cuda")
    layer_gpu_chunk_size = int(_cfg_or_default(cfg, "layer_gpu_chunk_size", 1))

    output_json_paths = []
    pending_entries = {}

    print("\n" + "=" * 60)
    print(f"[SingleLayer S] {', '.join(task_group)} -> {_layer_label(target_layer)}")
    print("=" * 60)

    for task in task_group:
        task_cfg, defaults, output_root = _single_layer_task_cfg_and_defaults(cfg, task)
        _validate_side_paths(task_cfg, ("s",))
        os.makedirs(output_root, exist_ok=True)

        intermediate_files_to_cleanup = []
        protected_outputs = set()

        profiling_mat = _load_existing_single_layer_profile("s", task_cfg, target_layer)
        if profiling_mat is not None:
            output_json_paths.append(
                _run_whitening_and_save(
                    side="s",
                    cfg=task_cfg,
                    defaults=defaults,
                    model=model,
                    dev=dev,
                    profiling_mat=profiling_mat,
                    protected_outputs=protected_outputs,
                )
            )
            _cleanup_intermediate_files(intermediate_files_to_cleanup, protected_outputs)
            del profiling_mat
            continue

        side_data = _build_side_data(
            side="s",
            cfg=task_cfg,
            defaults=defaults,
            tokenizer=tokenizer,
            intermediate_files_to_cleanup=intermediate_files_to_cleanup,
        )
        pending_entries[task] = {
            "cfg": task_cfg,
            "defaults": defaults,
            "side_data": side_data,
            "intermediate_files_to_cleanup": intermediate_files_to_cleanup,
            "protected_outputs": protected_outputs,
        }

    if pending_entries:
        print(
            f"[SingleLayer S] Running grouped profiling for {len(pending_entries)} task(s), "
            f"layer_gpu_chunk_size={layer_gpu_chunk_size}"
        )
        profiling_by_task = profile_svdllm_single_layer_group(
            model_name=cfg["model"],
            model=model,
            calib_loaders_by_task={task: entry["side_data"] for task, entry in pending_entries.items()},
            dev=dev,
            target_layer=target_layer,
            layer_gpu_chunk_size=layer_gpu_chunk_size,
            singular_floor=singular_floor,
            activation_cache_device=activation_cache_device,
            cholesky_jitter=cholesky_jitter,
        )

        for task, entry in pending_entries.items():
            profiling_mat = profiling_by_task[task]
            del entry["side_data"]

            _save_profile_for_side(
                side="s",
                cfg=entry["cfg"],
                defaults=entry["defaults"],
                profiling_mat=profiling_mat,
                intermediate_files_to_cleanup=entry["intermediate_files_to_cleanup"],
            )
            output_json_paths.append(
                _run_whitening_and_save(
                    side="s",
                    cfg=entry["cfg"],
                    defaults=entry["defaults"],
                    model=model,
                    dev=dev,
                    profiling_mat=profiling_mat,
                    protected_outputs=entry["protected_outputs"],
                )
            )
            _cleanup_intermediate_files(
                entry["intermediate_files_to_cleanup"],
                entry["protected_outputs"],
            )
            del profiling_mat

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for path in output_json_paths:
        print(f"[Done] Singular values json saved to: {path}")
    return output_json_paths


def _run_single_layer_x(
    cfg: dict,
    model,
    tokenizer,
    dev,
) -> list:
    target_layer = cfg["target_layer"]
    singular_floor = float(_cfg_or_default(cfg, "svd_singular_floor", 0.0))
    cholesky_jitter = float(_cfg_or_default(cfg, "cholesky_jitter", 1e-4))
    activation_cache_device = _cfg_or_default(cfg, "activation_cache_device", "cuda")
    layer_gpu_chunk_size = int(_cfg_or_default(cfg, "layer_gpu_chunk_size", 1))

    x_cfg, defaults, output_root = _single_layer_task_cfg_and_defaults(cfg, "X")
    _validate_side_paths(x_cfg, ("x",))
    os.makedirs(output_root, exist_ok=True)

    intermediate_files_to_cleanup = []
    protected_outputs = set()
    output_json_paths = []

    print("\n" + "=" * 60)
    print(f"[SingleLayer X] Global baseline -> {_layer_label(target_layer)}")
    print("=" * 60)

    profiling_mat = _load_existing_single_layer_profile("x", x_cfg, target_layer)
    if profiling_mat is None:
        side_data = _build_side_data(
            side="x",
            cfg=x_cfg,
            defaults=defaults,
            tokenizer=tokenizer,
            intermediate_files_to_cleanup=intermediate_files_to_cleanup,
        )
        profiling_by_task = profile_svdllm_single_layer_group(
            model_name=cfg["model"],
            model=model,
            calib_loaders_by_task={"X": side_data},
            dev=dev,
            target_layer=target_layer,
            layer_gpu_chunk_size=layer_gpu_chunk_size,
            singular_floor=singular_floor,
            activation_cache_device=activation_cache_device,
            cholesky_jitter=cholesky_jitter,
        )
        profiling_mat = profiling_by_task["X"]
        del side_data
        _save_profile_for_side(
            side="x",
            cfg=x_cfg,
            defaults=defaults,
            profiling_mat=profiling_mat,
            intermediate_files_to_cleanup=intermediate_files_to_cleanup,
        )

    output_json_paths.append(
        _run_whitening_and_save(
            side="x",
            cfg=x_cfg,
            defaults=defaults,
            model=model,
            dev=dev,
            profiling_mat=profiling_mat,
            protected_outputs=protected_outputs,
        )
    )
    _cleanup_intermediate_files(intermediate_files_to_cleanup, protected_outputs)

    del profiling_mat
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for path in output_json_paths:
        print(f"[Done] Singular values json saved to: {path}")
    return output_json_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = _normalize_legacy_config(cfg)

    missing, unknown = validate_config_keys(
        cfg,
        required_keys=REQUIRED_CONFIG_KEYS,
        optional_keys=OPTIONAL_CONFIG_KEYS,
    )
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")
    if unknown:
        print(f"[Warn] Unknown/unused config keys (ignored): {unknown}")

    cfg["target_layer"] = _parse_target_layer(cfg.get("target_layer", None))
    cfg["layer_gpu_chunk_size"] = _parse_layer_gpu_chunk_size(
        _cfg_or_default(cfg, "layer_gpu_chunk_size", 1)
    )
    cfg["mode"] = _resolve_mode(cfg)
    sides = MODE_TO_SIDES[cfg["mode"]]
    _require_side_config(cfg, sides)
    tasks = _resolve_tasks(cfg) if "s" in sides else []
    if "s" in sides and not tasks:
        raise ValueError("No tasks resolved to run")

    print(f"[Mode] {cfg['mode']} -> sides: {', '.join(side.upper() for side in sides)}")
    if cfg["target_layer"] is not None:
        print(
            f"[SingleLayer] {_layer_label(cfg['target_layer'])}, "
            f"layer_gpu_chunk_size={cfg['layer_gpu_chunk_size']}"
        )
    if "s" in sides:
        print(f"[Tasks] {len(tasks)} task(s): {', '.join(tasks)}")

    all_output_json_paths = []
    pending_s_tasks = []
    if "s" in sides:
        for task in tasks:
            task_cfg = dict(cfg)
            task_cfg["task"] = task
            task_cfg["s_jsonl_path"] = _resolve_s_jsonl_for_task(cfg, task)

            output_root = _output_root_for_task(task_cfg, task)
            defaults = _build_default_output_paths(task_cfg, output_root=output_root)
            if _should_skip_existing_outputs(task_cfg, defaults, "s", f"Task {task}"):
                all_output_json_paths.append(_side_output_path(task_cfg, defaults, "s", "json"))
            else:
                pending_s_tasks.append(task)

    x_needs_run = "x" in sides
    if "x" in sides:
        x_cfg = dict(cfg)
        x_cfg["task"] = "X"
        output_root = _output_root_for_task(x_cfg, x_cfg["task"])
        defaults = _build_default_output_paths(x_cfg, output_root=output_root)
        if _should_skip_existing_outputs(x_cfg, defaults, "x", "Global X"):
            all_output_json_paths.append(_side_output_path(x_cfg, defaults, "x", "json"))
            x_needs_run = False

    if not pending_s_tasks and not x_needs_run:
        print("\n[Done] All requested outputs already exist:")
        for path in all_output_json_paths:
            print(f"  {path}")
        return

    _set_global_seed(int(cfg["seed"]))

    dev = cfg["DEV"]
    if str(dev).startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA unavailable. Fallback to CPU.")
        dev = "cpu"

    model, tokenizer = get_model_from_huggingface(
        model_id=cfg["model"],
        torch_dtype=_cfg_or_default(cfg, "model_dtype", "float16"),
        trust_remote_code=bool(_cfg_or_default(cfg, "trust_remote_code", True)),
        cache_dir=_cfg_or_default(cfg, "hf_cache_dir", None),
    )
    model = model.eval()
    model.seqlen = int(cfg["model_seq_len"])
    _validate_target_layer_for_model(cfg["target_layer"], model)

    if "s" in sides and cfg["target_layer"] is not None:
        group_size = _parse_task_group_size(
            _cfg_or_default(cfg, "single_layer_task_group_size", 4),
            pending_count=len(pending_s_tasks),
        )
        print(f"[SingleLayer S] single_layer_task_group_size={group_size}")
        for start in range(0, len(pending_s_tasks), group_size):
            task_group = pending_s_tasks[start:start + group_size]
            all_output_json_paths.extend(
                _run_single_layer_s_task_group(
                    task_group=task_group,
                    cfg=cfg,
                    model=model,
                    tokenizer=tokenizer,
                    dev=dev,
                )
            )

    elif "s" in sides:
        for task in pending_s_tasks:
            task_cfg = dict(cfg)
            task_cfg["task"] = task
            task_cfg["s_jsonl_path"] = _resolve_s_jsonl_for_task(cfg, task)

            _validate_side_paths(task_cfg, ("s",))

            output_root = _output_root_for_task(task_cfg, task)
            defaults = _build_default_output_paths(task_cfg, output_root=output_root)

            os.makedirs(output_root, exist_ok=True)

            intermediate_files_to_cleanup = []
            protected_outputs = set()
            output_json_paths = []

            print("\n" + "=" * 60)
            print(f"[Task] {task}")
            print("=" * 60)
            output_json_paths.append(
                _run_svd_side(
                    side="s",
                    cfg=task_cfg,
                    defaults=defaults,
                    model=model,
                    tokenizer=tokenizer,
                    dev=dev,
                    intermediate_files_to_cleanup=intermediate_files_to_cleanup,
                    protected_outputs=protected_outputs,
                )
            )
            all_output_json_paths.extend(output_json_paths)

            _cleanup_intermediate_files(intermediate_files_to_cleanup, protected_outputs)

            for path in output_json_paths:
                print(f"[Done] Singular values json saved to: {path}")

    if x_needs_run and cfg["target_layer"] is not None:
        all_output_json_paths.extend(
            _run_single_layer_x(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                dev=dev,
            )
        )

    elif x_needs_run:
        x_cfg = dict(cfg)
        x_cfg["task"] = "X"
        _validate_side_paths(x_cfg, ("x",))

        output_root = _output_root_for_task(x_cfg, x_cfg["task"])
        defaults = _build_default_output_paths(x_cfg, output_root=output_root)
        os.makedirs(output_root, exist_ok=True)

        intermediate_files_to_cleanup = []
        protected_outputs = set()
        output_json_paths = []

        print("\n" + "=" * 60)
        print("[X] Global baseline")
        print("=" * 60)
        output_json_paths.append(
            _run_svd_side(
                side="x",
                cfg=x_cfg,
                defaults=defaults,
                model=model,
                tokenizer=tokenizer,
                dev=dev,
                intermediate_files_to_cleanup=intermediate_files_to_cleanup,
                protected_outputs=protected_outputs,
            )
        )
        all_output_json_paths.extend(output_json_paths)

        _cleanup_intermediate_files(intermediate_files_to_cleanup, protected_outputs)

        for path in output_json_paths:
            print(f"[Done] Singular values json saved to: {path}")

    print("\n[Done] All task outputs:")
    for path in all_output_json_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
