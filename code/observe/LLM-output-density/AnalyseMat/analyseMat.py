# AnalyseMat/analyseMat.py
"""
Step 1: safetensor → npy 提取（PE 需要的共享基础）
Step 2: principalEvidence（target_layer 单层主成分证据）
旧版 analyze_model_differences（逐层 SVD JSON）已删除。
"""

import os
import re
import yaml
import numpy as np
import torch
import math
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file
import copy

try:
    from AnalyseMat.principalEvidence import run_principal_evidence
except ImportError:
    from principalEvidence import run_principal_evidence


# -------------------------------
# 工具函数
# -------------------------------
def safe_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]', '_', name)


def _task_filter(config):
    tasks = config.get("analyse", {}).get("tasks") or config.get("analyseMat", {}).get("tasks") or []
    allowed = set()
    for task in tasks:
        if isinstance(task, (list, tuple)) and len(task) >= 2:
            allowed.add((str(task[0]), str(task[1])))
        elif isinstance(task, dict):
            source = task.get("source") or task.get("dataset") or task.get("name")
            size = task.get("size") or task.get("max_samples") or task.get("DataSize")
            if source is not None and size is not None:
                allowed.add((str(source), str(size)))
    return allowed


def _rel_matches_task(rel_dir: Path, allowed_tasks) -> bool:
    if not allowed_tasks:
        return True
    parts = rel_dir.parts
    if len(parts) < 2:
        return False
    return (str(parts[0]), str(parts[1])) in allowed_tasks


# =====================================================
# Step 1: safetensors → npy 提取
# =====================================================
def extract_safetensors_to_npy(config):
    input_root = Path(config["analyse"]["input_model_root"]).resolve()
    output_root = Path(config["analyse"]["npy_output_root"]).resolve()
    overwrite = config["analyse"].get("overwrite", False)
    keep_keywords = config["analyse"].get("keep_layer_keywords", None)
    prefix_with_stem = config["analyse"].get("prefix_with_stem", False)
    allowed_tasks = _task_filter(config)

    # 底模 npy 不存在时自动从 safetensors 提取
    base_model_npy_dir = Path(config["analyse"].get("base_model_npy_dir", "")).resolve()
    if not base_model_npy_dir.exists():
        base_model_path = Path(config["base_model"]).resolve()
        if base_model_path.exists():
            print(f"[analyseMat] 底模 npy 不存在，从 safetensors 提取: {base_model_path} -> {base_model_npy_dir}")
            base_model_npy_dir.mkdir(parents=True, exist_ok=True)
            for st_path in sorted(base_model_path.rglob("*.safetensors")):
                try:
                    rel_dir = st_path.parent.relative_to(base_model_path)
                except Exception:
                    rel_dir = Path("unknown")
                out_dir = base_model_npy_dir / rel_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                try:
                    tensors = load_file(str(st_path))
                except Exception as e:
                    print(f"[analyseMat] 无法加载 {st_path}: {e}")
                    continue
                n_saved = 0
                for name, tensor in tensors.items():
                    arr = tensor.cpu().to(torch.float32).numpy()
                    out_path = out_dir / f"{safe_filename(name)}.npy"
                    if out_path.exists():
                        continue
                    np.save(out_path, arr)
                    n_saved += 1
                print(f"[analyseMat] 底模 {st_path.name}: 已保存 {n_saved} 层到 {out_dir}")

    print(f"[analyseMat] 提取 safetensors → npy")
    print(f"  输入目录: {input_root}")
    print(f"  输出目录: {output_root}")

    st_paths = sorted(input_root.rglob("*.safetensors"))
    if not st_paths:
        print("[analyseMat] 未找到任何 .safetensors 文件。")
        return

    for st_path in tqdm(st_paths, desc="Converting safetensors"):
        try:
            rel_dir = st_path.parent.relative_to(input_root)
        except Exception:
            rel_dir = Path("unknown")
        if not _rel_matches_task(rel_dir, allowed_tasks):
            continue

        out_dir = output_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        st_stem = st_path.stem
        try:
            tensors = load_file(str(st_path))
        except Exception as e:
            print(f"[analyseMat] 无法加载 {st_path}: {e}")
            continue

        n_saved = 0
        for name, tensor in tensors.items():
            if keep_keywords:
                lk = name.lower()
                if not any(kw.lower() in lk for kw in keep_keywords):
                    continue

            arr = tensor.cpu().to(torch.float32).numpy()
            safe_name = safe_filename(name)
            out_fname = (
                f"{safe_filename(st_stem)}_{safe_name}.npy"
                if prefix_with_stem else f"{safe_name}.npy"
            )
            out_path = out_dir / out_fname

            if out_path.exists() and not overwrite:
                continue

            np.save(out_path, arr)
            n_saved += 1

        print(f"[analyseMat] {st_path.name}: 已保存 {n_saved} 层到 {out_dir}")


# =====================================================
# 外部接口：Step1 总是跑 + 按 related_work.enable 跑 PE
# =====================================================
def run_analyse(config):
    """GPU pipeline 调用入口。Step1 总是执行；Step2 (PE) 按配置决定。"""
    if config.get("analyse", {}).get("skip_extract_npy", False):
        print("[analyseMat] skip_extract_npy=true，跳过 safetensors → npy，仅执行后续分析")
    else:
        extract_safetensors_to_npy(config)
    if config.get("analyse", {}).get("related_work", {}).get("enable", False):
        run_principal_evidence(config)


# =====================================================
# 独立入口
# =====================================================
def main():
    with open("Density.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_analyse(config)


if __name__ == "__main__":
    main()
