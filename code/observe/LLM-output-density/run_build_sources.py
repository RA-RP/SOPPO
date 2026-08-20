#!/usr/bin/env python3
"""
run_build_sources.py — 数据构造统一入口（阶段 A）

从 global.yaml 读取 data_construction 配置，按 method 派发到
baseline / offPolicyData / GetData 的内部 API；
输出统一到 {home}/sources/ 并写 manifest.json。

用法：
    python run_build_sources.py --config configs/global.yaml
    python run_build_sources.py --config configs/global.yaml --methods baseline
    python run_build_sources.py --config configs/global.yaml --baseline-methods random --sizes 200
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.loader import load_layered_config, expand_for_run


def _merge_base_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """从 global 构造一个 baseline 可读的 config dict。"""
    base = {
        "seed": cfg.get("seed", 42),
        "device": cfg.get("device", "cpu"),
        "base_model": cfg.get("base_model", ""),
        "train": copy.deepcopy(cfg.get("train", {})),
        "prepare": {"dataset": {"dataset_alias": "sft_train"}},
    }
    if cfg.get("prepare"):
        base["prepare"] = copy.deepcopy(cfg["prepare"])
    return base


def _params_hash(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def _as_list(value: Any) -> List[Any]:
    if value in (None, "", "null"):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize_base_datasets(cfg: Dict[str, Any], selected: List[str] | None = None) -> List[Dict[str, str]]:
    selected_set = {str(item) for item in selected or []}
    base_datasets = cfg.get("baseDatasets") or []
    if not isinstance(base_datasets, list) or not base_datasets:
        raise ValueError("global.yaml 缺少 baseDatasets: [{name, source_prefix, train_path}]")

    normalized: List[Dict[str, str]] = []
    for idx, item in enumerate(base_datasets):
        if not isinstance(item, dict):
            raise ValueError(f"baseDatasets[{idx}] 必须是 object")
        name = str(item.get("name", "")).strip()
        source_prefix = str(item.get("source_prefix", "")).strip()
        train_path = str(item.get("train_path", "")).strip()
        if not name or not source_prefix or not train_path:
            raise ValueError(f"baseDatasets[{idx}] 必须包含 name/source_prefix/train_path")
        if selected_set and name not in selected_set and source_prefix not in selected_set:
            continue
        normalized.append({
            "name": name,
            "source_prefix": source_prefix,
            "train_path": train_path,
        })
    if not normalized:
        raise ValueError(f"未找到匹配的 baseDatasets: {selected or []}")
    return normalized


def _resolve_run_sizes(cfg_in: Dict[str, Any], override: List[int] | None = None) -> List[int]:
    raw = override if override is not None else cfg_in.get("sizes")
    sizes = [int(size) for size in _as_list(raw)]
    if not sizes:
        raise ValueError("global.yaml 缺少 sizes，或 CLI --sizes 为空")
    if any(size <= 0 for size in sizes):
        raise ValueError(f"sizes 必须全部 > 0: {sizes}")
    return sizes


def _enabled_method_names(methods_cfg: Dict[str, Any]) -> List[str]:
    names = []
    for name, method_cfg in methods_cfg.items():
        if isinstance(method_cfg, dict) and method_cfg.get("enable", False):
            names.append(name)
    return names


def _resolve_sizes(method_cfg: Dict[str, Any], cfg_in: Dict[str, Any], default: List[int] | None = None) -> List[int]:
    raw = method_cfg.get("sizes")
    if raw in (None, "", "null", []):
        raw = cfg_in.get("sizes") or default or [200]
    return [int(size) for size in _as_list(raw)]


def _resolve_repo_local_path(value: Any) -> str:
    raw = str(value)
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    repo_path = (REPO_ROOT / path).resolve()
    if raw.startswith(".") or repo_path.exists():
        return str(repo_path)
    return raw


def _update_manifest(manifest_path: Path, entries: List[Dict[str, Any]]) -> None:
    existing = []
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            existing = json.load(f)
    # 按统一 task key 去重，新的覆盖旧的
    key_set = {_manifest_key(e) for e in entries}
    existing = [e for e in existing if _manifest_key(e) not in key_set]
    existing.extend(entries)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2, default=str)
    print(f"[build_sources] manifest 已更新: {manifest_path} ({len(existing)} entries)")


def _manifest_key(entry: Dict[str, Any]) -> tuple:
    metadata = entry.get("metadata") or {}
    return (
        entry.get("task_id") or entry.get("name"),
        entry.get("method"),
        str(entry.get("size", entry.get("max_samples", ""))),
        metadata.get("model", ""),
    )


def _task_manifest_entry(
    *,
    task: Dict[str, Any],
    method: str,
    base_source: str,
    model_name: str,
    input_path: str,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    if model_name:
        metadata.setdefault("model", model_name)
    task_id = str(task["task_id"])
    size = task.get("max_samples")
    source = str(task.get("dataset") or task_id)
    train_sample_count = task.get("train_sample_count")
    if train_sample_count is None:
        train_sample_count = size
    return {
        "task_id": task_id,
        "name": task_id,
        "source": source,
        "base_source": base_source,
        "dataset": source,
        "size": size,
        "train_sample_count": int(train_sample_count) if train_sample_count not in (None, "") else None,
        "method": method,
        "input_paths": list(task.get("input_paths") or ([input_path] if input_path else [])),
        "path": input_path,
        "dataset_alias": task.get("dataset_alias", "sft_train"),
        "baseline_meta_path": task.get("baseline_meta_path"),
        "train_args_overrides": task.get("train_args_overrides", {}),
        "metadata": metadata,
        "params_hash": _params_hash({
            "task_id": task_id,
            "source": source,
            "base_source": base_source,
            "size": size,
            "train_sample_count": train_sample_count,
            "method": method,
            "input_paths": task.get("input_paths", []),
            "metadata": metadata,
        }),
    }


def run_baseline_method(
    cfg_in: Dict[str, Any],
    baseline_cfg: Dict[str, Any],
    output_root: Path,
    overwrite: bool,
    model_name: str = "",
) -> List[Dict[str, Any]]:
    """调用 Baseline/select_baseline.py 的内部 API 运行 baseline 选数。"""
    # 构造一个兼容 run_baseline(config) 的 config dict
    cfg = _merge_base_config(cfg_in)
    cfg["baseline"] = {
        "enable": True,
        "input_path": baseline_cfg.get("input_path", ""),
        "dataset_name": baseline_cfg.get("dataset_name", ""),
        "source_prefix": baseline_cfg.get("source_prefix", baseline_cfg.get("dataset_name", "")),
        "output_root": baseline_cfg.get("output_root", str(output_root / "baseline")),
        "methods": baseline_cfg.get("methods", ["random"]),
        "sizes": _resolve_run_sizes(cfg_in, baseline_cfg.get("sizes")),
        "overwrite": bool(overwrite or baseline_cfg.get("overwrite", False)),
        "dry_run": baseline_cfg.get("dry_run", False),
        "ppl": baseline_cfg.get("ppl", {}),
        "diversity": baseline_cfg.get("diversity", {}),
        "cfs": baseline_cfg.get("cfs", {}),
        "seed": cfg.get("seed", 42),
    }

    from Baseline.select_baseline import run_baseline
    tasks = run_baseline(cfg)

    # 将每个 task 注册为 manifest entry
    entries = []
    for task in tasks:
        method = str((task.get("meta") or {}).get("method") or task.get("method") or "baseline")
        entries.append(_task_manifest_entry(
            task=task,
            method=f"baseline_{method}",
            base_source=str(baseline_cfg.get("dataset_name", "")),
            model_name=model_name,
            input_path=task["input_paths"][0] if task.get("input_paths") else "",
            metadata={
                "stage": "baseline",
                "selection_method": method,
                "source_prefix": baseline_cfg.get("source_prefix"),
                "input_path": cfg["baseline"]["input_path"],
                "baseline_meta_path": task.get("baseline_meta_path"),
            },
        ))
    return entries


def run_offpolicy_method(
    cfg_in: Dict[str, Any],
    op_cfg: Dict[str, Any],
    output_root: Path,
    model_name: str = "",
    dry_run: bool = False,
    overwrite: bool = False,
    baseline_cfg: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Run/register offPolicyData outputs as build_sources manifest entries."""
    from offPolicyData.components.common import load_json_config, run_root, save_json
    from offPolicyData.components import cluster_kl

    pipelines = {
        "cluster_kl": cluster_kl,
    }

    config_path = Path(str(op_cfg.get("config_path", "offPolicyData/config.json"))).expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    op_base = load_json_config(config_path)
    sizes = [size for size in _resolve_run_sizes(cfg_in, op_cfg.get("sizes")) if size > 0]
    if not sizes:
        return []

    entries: List[Dict[str, Any]] = []
    base_source = str(op_cfg.get("dataset_name") or "")
    source_prefix = str(op_cfg.get("source_prefix") or base_source)
    train_path = str(op_cfg.get("train_path") or "")
    if not base_source or not source_prefix or not train_path:
        raise ValueError("offPolicyData 需要 dataset_name/source_prefix/train_path，由 baseDatasets 注入")
    requested_stages = op_cfg.get("stages", ["all"])
    if isinstance(requested_stages, str):
        requested_stages = [requested_stages]

    op_run_cfg = copy.deepcopy(op_base)
    op_run_cfg["pipeline"] = str(op_cfg.get("pipeline") or op_run_cfg.get("pipeline", "cluster_kl"))
    op_run_cfg["seed"] = int(op_cfg.get("seed", cfg_in.get("seed", op_run_cfg.get("seed", 42))))
    op_run_cfg["overwrite_outputs"] = bool(overwrite or op_cfg.get("overwrite_outputs", op_run_cfg.get("overwrite_outputs", False)))
    op_run_cfg["output_root"] = str(Path(op_cfg.get("output_root", output_root / "offPolicyData" / "runs")).expanduser())
    op_run_cfg["base_dataset"] = {
        "name": base_source,
        "source_prefix": source_prefix,
        "train_path": train_path,
    }
    if cfg_in.get("base_model"):
        op_run_cfg.setdefault("getslice", {})
        op_run_cfg["getslice"]["model"] = _resolve_repo_local_path(cfg_in["base_model"])
    op_run_cfg.setdefault("aggregation", {})
    base_run_name = str(op_cfg.get("run_name") or op_run_cfg.get("run_name") or "offpolicy")
    op_run_cfg["run_name"] = f"{source_prefix}_{base_run_name}"

    pipeline_name = str(op_run_cfg.get("pipeline", "cluster_kl")).strip()
    if pipeline_name not in pipelines:
        raise ValueError(f"unsupported offPolicyData pipeline: {pipeline_name}")
    pipeline_module = pipelines[pipeline_name]
    root = run_root(op_run_cfg)
    if "all" in requested_stages:
        stages = list(pipeline_module.ALL_STAGES)
    else:
        unknown = [stage for stage in requested_stages if stage not in pipeline_module.STAGE_FUNCS]
        if unknown:
            raise ValueError(f"unsupported offPolicyData stages: {unknown}")
        stages = list(requested_stages)
    base_stages = [stage for stage in stages if stage != "aggregate"]
    run_aggregate = "aggregate" in stages

    if dry_run:
        print(f"[build_sources] offPolicyData dry-run: pipeline={pipeline_name}, root={root}, sizes={sizes}")
    else:
        from Baseline.select_baseline import ensure_shared_diversity_cache_for_config

        cache_cfg = copy.deepcopy(baseline_cfg or {})
        cache_cfg.setdefault("input_path", train_path)
        cache_cfg.setdefault("dataset_name", base_source)
        cache_cfg.setdefault("source_prefix", source_prefix)
        cache_cfg.setdefault("output_root", str(Path(output_root).parent / "baseline"))
        cache_size = int(op_cfg.get("cache_size") or sizes[0])
        cache_dir = ensure_shared_diversity_cache_for_config(cfg_in, cache_cfg, cache_size)
        op_run_cfg["cluster_cache_dir"] = str(cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        save_json(root / "run_config.resolved.json", op_run_cfg)
        kl_csv = root / op_run_cfg.get("kl", {}).get("csv_name", "cluster_kl.csv")
        if kl_csv.exists() and not op_run_cfg["overwrite_outputs"] and set(base_stages).issuperset({"sample", "getslice_config", "getslice", "kl"}):
            print(f"[build_sources] 复用 offPolicyData KL 指标: {kl_csv}")
        else:
            print(f"[build_sources] 运行 offPolicyData 指标阶段: pipeline={pipeline_name}, stages={base_stages}")
            for stage in base_stages:
                pipeline_module.STAGE_FUNCS[stage](op_run_cfg, root)

    for size in sizes:
        size_run_cfg = copy.deepcopy(op_run_cfg)
        size_run_cfg.setdefault("aggregation", {})
        size_run_cfg["aggregation"]["dataset_size"] = int(size)
        if not dry_run and run_aggregate:
            print(f"[build_sources] 运行 offPolicyData 聚合阶段: pipeline={pipeline_name}, size={size}")
            pipeline_module.STAGE_FUNCS["aggregate"](size_run_cfg, root)

        ranked_dir_name = "ranked_datasets"
        ranked_root = root / ranked_dir_name / f"size{size}"
        if ranked_root.exists():
            band_order = {f"kl_{name}": idx for idx, name in enumerate(size_run_cfg.get("aggregation", {}).get("bands", ["high", "mid", "low"]))}
            dataset_dirs = sorted(
                (path for path in ranked_root.iterdir() if path.is_dir()),
                key=lambda path: (band_order.get(path.name, 999), path.name),
            )
        else:
            dataset_dirs = []
        if dry_run and not dataset_dirs:
            # Advertise planned dataset ids so manifest shape can be inspected without running GetSlice.
            dataset_dirs = [ranked_root / f"kl_{name}" for name in size_run_cfg.get("aggregation", {}).get("bands", ["high", "mid", "low"])]

        for dataset_dir in dataset_dirs:
            task_source = f"{source_prefix}-offPolicyData_{dataset_dir.name}"
            task_id = f"{task_source}__{size}"
            task = {
                "task_id": task_id,
                "dataset": task_source,
                "max_samples": size,
                "train_sample_count": size,
                "input_paths": [str(dataset_dir / size_run_cfg.get("task_file", "dataset.jsonl"))],
                "dataset_alias": size_run_cfg.get("dataset_alias", "sft_train"),
            }
            entries.append(_task_manifest_entry(
                task=task,
                method=f"offPolicyData_{pipeline_name}",
                base_source=base_source,
                model_name=model_name,
                input_path=task["input_paths"][0],
                metadata={
                    "stage": "offPolicyData",
                    "pipeline": pipeline_name,
                    "source_prefix": source_prefix,
                    "run_root": str(root),
                    "dataset_id": dataset_dir.name,
                    "dataset_size": size,
                },
            ))
    return entries


def parse_args():
    parser = argparse.ArgumentParser(description="数据构造统一入口")
    parser.add_argument("--config", default="configs/global.yaml", help="分层配置入口")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="要运行的构造方法（baseline / offPolicyData / getdata）")
    parser.add_argument("--baseline-methods", nargs="*", default=None,
                        help="覆盖 baseline.methods（如 random ppl_cond_middle）")
    parser.add_argument("--baseline-sizes", nargs="*", type=int, default=None,
                        help="兼容旧参数；等同于 --sizes")
    parser.add_argument("--sizes", nargs="*", type=int, default=None,
                        help="覆盖 global.yaml sizes")
    parser.add_argument("--models", nargs="*", default=None, help="模型列表（覆盖 global.yaml）")
    parser.add_argument("--base-datasets", nargs="*", default=None, help="baseDataset name/source_prefix 列表（覆盖 global.yaml）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划不执行")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有产物")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_layered_config(args.config)

    dc_cfg = cfg.get("data_construction", {})
    run_cfg = dc_cfg.get("run", {})
    output_root = Path(dc_cfg.get("output_root", f"{cfg.get('home', '.')}/sources")).expanduser().resolve()
    manifest_path = Path(dc_cfg.get("manifest_path", str(output_root / "manifest.json"))).expanduser().resolve()
    methods_cfg = dc_cfg.get("methods", {})
    run_dry_run = bool(args.dry_run or run_cfg.get("dry_run", False))
    run_overwrite = bool(args.overwrite or run_cfg.get("overwrite", False))

    all_entries: List[Dict[str, Any]] = []
    requested = args.methods or _as_list(run_cfg.get("methods")) or _enabled_method_names(methods_cfg)
    models = args.models or _as_list(run_cfg.get("models")) or [m["name"] for m in cfg.get("models", [])] or [""]
    sizes = _resolve_run_sizes(cfg, args.sizes if args.sizes is not None else args.baseline_sizes)
    base_datasets = _normalize_base_datasets(cfg, args.base_datasets)

    for model_name in models:
        eff = expand_for_run(cfg, model_name) if model_name else cfg
        eff["sizes"] = sizes

        for base_dataset in base_datasets:
            baseline_cfg_for_base = copy.deepcopy(methods_cfg.get("baseline", {}))
            baseline_cfg_for_base["dataset_name"] = base_dataset["name"]
            baseline_cfg_for_base["source_prefix"] = base_dataset["source_prefix"]
            baseline_cfg_for_base["input_path"] = base_dataset["train_path"]
            baseline_cfg_for_base["sizes"] = sizes

            # ---- baseline ----
            if "baseline" in requested:
                baseline_cfg = copy.deepcopy(baseline_cfg_for_base)
                if not baseline_cfg.get("enable", True):
                    print("[build_sources] baseline 已禁用，跳过")
                else:
                    if args.baseline_methods:
                        baseline_cfg["methods"] = args.baseline_methods
                    if run_dry_run:
                        baseline_cfg["dry_run"] = True
                    print(
                        f"[build_sources] 运行 baseline: model={model_name}, "
                        f"baseDataset={base_dataset['name']}, methods={baseline_cfg.get('methods')}, sizes={sizes}"
                    )
                    entries = run_baseline_method(eff, baseline_cfg, output_root, run_overwrite, model_name=model_name)
                    all_entries.extend(entries)

            # ---- offPolicyData ----
            if "offPolicyData" in requested:
                op_cfg = copy.deepcopy(methods_cfg.get("offPolicyData", {}))
                if not op_cfg.get("enable", False):
                    print("[build_sources] offPolicyData 已禁用或未配置，跳过")
                else:
                    op_cfg["dataset_name"] = base_dataset["name"]
                    op_cfg["source_prefix"] = base_dataset["source_prefix"]
                    op_cfg["train_path"] = base_dataset["train_path"]
                    op_cfg["sizes"] = sizes
                    entries = run_offpolicy_method(
                        eff,
                        op_cfg,
                        output_root,
                        model_name=model_name,
                        dry_run=run_dry_run,
                        overwrite=run_overwrite,
                        baseline_cfg=baseline_cfg_for_base,
                    )
                    all_entries.extend(entries)

    # ---- getdata（占位，后续接入） ----
    if "getdata" in requested:
        gd_cfg = methods_cfg.get("getdata", {})
        if not gd_cfg.get("enable", False):
            print("[build_sources] getdata 已禁用或未配置，跳过")
        else:
            print("[build_sources] getdata 后端尚未接入，跳过")

    if all_entries and not run_dry_run:
        _update_manifest(manifest_path, all_entries)

    print(f"\n[build_sources] 完成，共 {len(all_entries)} 个 entry")


if __name__ == "__main__":
    main()
