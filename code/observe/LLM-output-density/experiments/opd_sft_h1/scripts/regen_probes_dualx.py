#!/usr/bin/env python3
"""
regen_probes_dualx.py — 复用已训练模型 + 已有 eval，仅重做探针/几何。

用途：在 opd_minimal_03_v2 已完成（6 模型 merged + eval CSV 已存在）的基础上，
  1. 清理旧探针/旧 getslice 输出（旧版被 256 截断、单 X）
  2. 重新生成双 X 探针(X_prompt / X_bos)与全部 S 探针，rollout 生成到自然 EOS 不截断
  3. 重跑 S×model 交叉矩阵 + 双 X 的 GetSlice
  4. 重建 geometry_metrics / geometry_long / 图表

不重训、不重评估。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path("/root/LLM-output-density")
SIDECAR_ROOT = REPO_ROOT / "experiments/opd_sft_h1"
EXP_ROOT = Path("/root/autodl-tmp/exp0609/opd_minimal_03_v2")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from scripts.run_opd_minimal_closure import ModelSpec  # noqa: E402
from scripts.run_opd_minimal_closure_v2 import (  # noqa: E402
    build_probes,
    run_getslice_cross,
    build_geometry_tables_cross,
    build_figures_cross,
    build_unified_pool,
)

# 中等规模 cfg（与 main 一致）
N_PROBE = 32
MAX_NEW_TOKENS = 2048
TARGET_LAYER = 14
GS_SEQLEN = 512
GS_NSAMPLES = 16
SFT_SIZES = [256, 512, 1024, 2048]


def main() -> None:
    print(f"[REGEN] exp_root = {EXP_ROOT}", flush=True)

    # ---- 重建 pool（已存在则复用）----
    pool = build_unified_pool(
        EXP_ROOT, n_cold=512, n_opd=200, n_sft_max=max(SFT_SIZES),
        n_heldout=64, n_probe=N_PROBE, seed=42,
    )

    # ---- 从已有 merged 模型重建 ModelSpec ----
    def merged(path: str) -> Path:
        d = EXP_ROOT / path
        if (d / "merged_model" / "config.json").exists():
            return d / "merged_model"
        if (d / "checkpoint_output" / "merged_model" / "config.json").exists():
            return d / "checkpoint_output" / "merged_model"
        raise FileNotFoundError(f"merged model not found under {d}")

    theta0 = ModelSpec("theta0", "512", "theta0",
                       merged("step2_cold_start"), EXP_ROOT / "step2_cold_start/checkpoint_output")
    opd = ModelSpec("opd_lmbda1", "800", "opd",
                    merged("step3_opd_distill"), EXP_ROOT / "step3_opd_distill/checkpoint_output")
    sft_specs = [
        ModelSpec(f"sft_n{s}", str(s), "sft",
                  merged(f"step4_sft_controls/sft_n{s}"),
                  EXP_ROOT / f"step4_sft_controls/sft_n{s}/checkpoint_output")
        for s in SFT_SIZES
    ]
    specs = [theta0, opd, *sft_specs]
    print(f"[REGEN] {len(specs)} models: {[s.source for s in specs]}", flush=True)

    # ---- 清理旧探针 + 旧 getslice 输出（旧版截断/单 X，必须重做）----
    gs = EXP_ROOT / "getslice"
    for sub in ["inputs", "outputs", "configs"]:
        p = gs / sub
        if p.exists():
            shutil.rmtree(p)
            print(f"[REGEN] cleared {p}", flush=True)

    # ---- 重新生成双 X + S 探针（不截断），跑交叉 GetSlice，重建几何 ----
    probes = build_probes(EXP_ROOT, pool, theta0, opd, sft_specs,
                          n_probe=N_PROBE, max_new_tokens=MAX_NEW_TOKENS)
    getslice_root = run_getslice_cross(
        EXP_ROOT, specs, probes, target_layer=TARGET_LAYER,
        seqlen=GS_SEQLEN, s_nsamples=GS_NSAMPLES, x_nsamples=GS_NSAMPLES,
    )
    build_geometry_tables_cross(EXP_ROOT, getslice_root)
    build_figures_cross(EXP_ROOT)
    print(f"[REGEN-DONE] dual-X probes + geometry rebuilt at {EXP_ROOT}", flush=True)


if __name__ == "__main__":
    main()
