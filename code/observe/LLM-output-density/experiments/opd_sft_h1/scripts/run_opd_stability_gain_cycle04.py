#!/usr/bin/env python3
"""
run_opd_stability_gain_cycle04.py — Cycle 04: OPD Stability-to-Gain Gate

相对 Cycle 03 (run_opd_minimal_closure_v2.py) 的改进：
  1. theta0=256（更不饱和，留 gain 检测空间）
  2. 两个 OPD arms：opd_lmbda1_seed42 / opd_lmbda05_seed42
  3. SFT 控制扫 128/256/512/1024（matched 新 theta0）
  4. GSM8K full eval（eval_limit=None），降噪以支撑 gain claim
  5. geometry 三层 early/mid/late (layer 6/14/22)
  6. vLLM colocate 提速 student rollout（OPD/cold-start），48G 单卡

复用 v2 工具函数；OPD arm 训练自包含（支持 lmbda/seed/vLLM colocate）。
验收门 Gate A/B/C/D 见 acceptance_criteria_04。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path("/root/LLM-output-density")
SIDECAR_ROOT = REPO_ROOT / "experiments/opd_sft_h1"
EXP_ROOT_DEFAULT = Path("/root/autodl-tmp/cycle04_opd_stability_gain")
BASE_MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen3-1.7B")
TEACHER_MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen3-4B")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from scripts.run_opd_minimal_closure import ModelSpec, merge_lora_adapter  # noqa: E402
from scripts.run_opd_minimal_closure_v2 import (  # noqa: E402
    build_unified_pool,
    train_sft_control,
    run_full_eval_v2,
    rollout_completions,
    rollout_freeform_bos,
    run_getslice_cross,
    build_geometry_tables_cross,
    build_figures_cross,
    ensure_dir,
    write_json,
    write_jsonl,
    write_csv,
    _read_jsonl,
    _as_float,
    OOD_BENCHMARKS,
)

# 3 层 geometry：Qwen3-1.7B 28 层 → early/mid/late
GEOMETRY_LAYERS = [6, 14, 22]


# =============================================================================
# OPD arm 训练（自包含：lmbda / seed / vLLM colocate）
# =============================================================================
def _write_opd_arm_config(
    cfg_path: Path,
    *,
    name: str,
    output_root: Path,
    student_checkpoint: str | None,
    prompt_jsonl: str,
    eval_jsonl: str,
    max_samples: int,
    max_steps: int,
    eval_steps: int,
    grad_accum: int,
    lmbda: float,
    seed: int,
    role_label: str,
    registry_method: str,
    use_vllm: bool,
    vllm_gpu_mem: float,
) -> None:
    trl: dict[str, Any] = {
        "lmbda": lmbda,
        "beta": 0.5,
        "loss_top_k": 1,
        "max_length": 4096,
        "max_completion_length": 512,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": grad_accum,
        "per_device_eval_batch_size": 1,
        "learning_rate": 3.0e-5,
        "max_steps": max_steps,
        "save_steps": eval_steps,
        "eval_steps": eval_steps,
        "save_total_limit": 2,
        "logging_steps": max(eval_steps // 2, 1),
        "gradient_checkpointing": False,
        "optim": "adamw_torch",
        "bf16": True,
        "model_init_kwargs": {"torch_dtype": "bfloat16"},
        "teacher_model_init_kwargs": {"torch_dtype": "bfloat16"},
        "peft": {
            "enabled": True, "r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "use_vllm": use_vllm,
        "use_teacher_server": False,
        "teacher_model_server_url": None,
        "report_to": [],
    }
    if use_vllm:
        # colocate：同进程内 vLLM 引擎做 student rollout（continuous batching 并行）
        trl.update({
            "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": vllm_gpu_mem,
            "vllm_max_model_length": 4096,
            "vllm_enable_sleep_mode": True,   # 训练步让 vLLM 休眠释放显存给 teacher+反向
            "vllm_tensor_parallel_size": 1,
            "vllm_sync_frequency": 1,         # 每个 optimizer step 同步一次 LoRA 权重
        })
    cfg = {
        "experiment": {"name": name, "output_root": str(output_root), "seed": seed},
        "model": {
            "base_model": str(BASE_MODEL),
            "cold_start_checkpoint": student_checkpoint,
            "student_start_checkpoint": student_checkpoint,
            "teacher_model": str(TEACHER_MODEL),
        },
        "data": {
            "prompt_jsonl": prompt_jsonl, "eval_jsonl": eval_jsonl,
            "prompt_text_field": "problem", "max_samples": max_samples, "eval_max_samples": 64,
        },
        "trl": trl,
        "registry": {"method": registry_method, "role_label": role_label, "pi_mix_lambda": lmbda},
    }
    ensure_dir(cfg_path.parent)
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def train_opd_arm(
    *,
    exp_root: Path,
    label: str,
    student_checkpoint: str | None,
    prompt_jsonl: str,
    eval_jsonl: str,
    max_samples: int,
    max_steps: int,
    grad_accum: int,
    eval_steps: int,
    lmbda: float,
    seed: int,
    role_label: str,
    registry_method: str,
    model_role: str,
    use_vllm: bool,
    vllm_gpu_mem: float,
) -> ModelSpec:
    """训练一个 OPD-like arm（cold-start 或 OPD distill）。DataSize=实际消耗 prompt 数。

    训练在**独立子进程**运行（subprocess 调 run_trl_distill_smoke.py）：vLLM colocate 会初始化
    torch.distributed(NCCL)，同进程内无法串行启动多个 colocate 训练（第二个会撞已销毁/复用的进程组），
    且 distributed initialized 时 peft adapter 加载会撞 EmbeddingParallel。子进程隔离后，训练子进程
    退出即清理进程组，merge（peft 加载）在本父进程做，二者都干净。
    """
    import subprocess

    out_root = ensure_dir(exp_root / label)
    adapter_dir = out_root / "checkpoint_output"
    merged_dir = adapter_dir / "merged_model"
    consumed = max_steps * 1 * grad_accum
    if (merged_dir / "config.json").exists():
        print(f"[SKIP] {label} merged exists", flush=True)
        return ModelSpec(label, str(consumed), model_role, merged_dir, adapter_dir)

    cfg_path = out_root / f"config_{label}.yaml"
    _write_opd_arm_config(
        cfg_path, name=label, output_root=out_root, student_checkpoint=student_checkpoint,
        prompt_jsonl=prompt_jsonl, eval_jsonl=eval_jsonl, max_samples=max_samples,
        max_steps=max_steps, eval_steps=eval_steps, grad_accum=grad_accum, lmbda=lmbda, seed=seed,
        role_label=role_label, registry_method=registry_method, use_vllm=use_vllm, vllm_gpu_mem=vllm_gpu_mem,
    )
    # 子进程训练（隔离 distributed）
    env = dict(os.environ, TRL_EXPERIMENTAL_SILENCE="1", TMPDIR="/root/autodl-tmp/pip-tmp")
    runner = str(SIDECAR_ROOT / "scripts" / "run_trl_distill_smoke.py")
    print(f"[TRAIN-SUBPROC] {label} (use_vllm={use_vllm}) -> {cfg_path}", flush=True)
    r = subprocess.run([sys.executable, runner, "--config", str(cfg_path)], env=env, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"OPD arm training failed (rc={r.returncode}): {label}")
    # merge 在父进程（无 distributed）
    if not (merged_dir / "config.json").exists():
        base_for_merge = Path(student_checkpoint) if student_checkpoint else BASE_MODEL
        merge_lora_adapter(base_for_merge, adapter_dir, merged_dir)
    return ModelSpec(label, str(consumed), model_role, merged_dir, adapter_dir)


# =============================================================================
# 多 arm 探针（X 双探针 + 每个 OPD arm / theta0 / sft 各自 S）
# =============================================================================
def build_probes_multi(
    exp_root: Path,
    pool: dict[str, Any],
    theta0: ModelSpec,
    opd_arms: list[ModelSpec],
    sft_specs: list[ModelSpec],
    *,
    n_probe: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """X：theta0 的 X_prompt / X_bos（冻结共用，不截断）。
    S：theta0=teacher rollout；每个 opd arm=自己 rollout；每个 sft=训练数据切片。
    """
    gs = ensure_dir(exp_root / "getslice")
    inputs = ensure_dir(gs / "inputs")
    probe_rows = _read_jsonl(Path(pool["probe_prompts"]))[:n_probe]
    prompts = [str(r["problem"]) for r in probe_rows]

    # X 双探针（theta0 生成，冻结）
    x_probes: dict[str, Path] = {}
    xp = ensure_dir(inputs / "X_prompt") / "x_probe.jsonl"
    if not xp.exists():
        print(f"[PROBE-X_prompt] theta0 rollout {len(prompts)} (EOS)", flush=True)
        comps = rollout_completions(theta0.model_dir, prompts, max_new_tokens=max_new_tokens)
        write_jsonl(xp, [{"output": {"text": f"{p}\n{c}"}} for p, c in zip(prompts, comps)])
    x_probes["prompt"] = xp
    xb = ensure_dir(inputs / "X_bos") / "x_probe.jsonl"
    if not xb.exists():
        print(f"[PROBE-X_bos] theta0 freeform x{len(prompts)} (EOS)", flush=True)
        comps = rollout_freeform_bos(theta0.model_dir, len(prompts), max_new_tokens=max_new_tokens)
        write_jsonl(xb, [{"output": {"text": c}} for c in comps])
    x_probes["bos"] = xb

    s_roots: dict[str, Path] = {}
    # theta0-S：teacher rollout
    s_t = ensure_dir(inputs / "S" / "theta0" / "numina_math_probe") / "gamma_s.jsonl"
    if not s_t.exists():
        print(f"[PROBE-S theta0] teacher rollout {len(prompts)} (EOS)", flush=True)
        comps = rollout_completions(TEACHER_MODEL, prompts, max_new_tokens=max_new_tokens)
        write_jsonl(s_t, [{"question": p, "answer": c} for p, c in zip(prompts, comps)])
    s_roots["theta0"] = inputs / "S" / "theta0"
    # 每个 OPD arm：自己 rollout
    for arm in opd_arms:
        s_dir = ensure_dir(inputs / "S" / arm.source / "numina_math_probe") / "gamma_s.jsonl"
        if not s_dir.exists():
            print(f"[PROBE-S {arm.source}] student rollout {len(prompts)} (EOS)", flush=True)
            comps = rollout_completions(arm.model_dir, prompts, max_new_tokens=max_new_tokens)
            write_jsonl(s_dir, [{"question": p, "answer": c} for p, c in zip(prompts, comps)])
        s_roots[arm.source] = inputs / "S" / arm.source
    # 每个 SFT：训练数据切片
    sft_rows = _read_jsonl(Path(pool["train_sft"]))
    for spec in sft_specs:
        s_dir = ensure_dir(inputs / "S" / spec.source / "numina_math_probe") / "gamma_s.jsonl"
        if not s_dir.exists():
            sel = sft_rows[: int(spec.size)][:n_probe]
            rows = []
            for r in sel:
                msgs = r["messages"]
                q = next((m["content"] for m in msgs if m["role"] == "user"), "")
                a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                rows.append({"question": q, "answer": a})
            write_jsonl(s_dir, rows)
        s_roots[spec.source] = inputs / "S" / spec.source

    return {"x_probes": x_probes, "s_roots": s_roots}


# =============================================================================
# eval + selection + Gate 评估
# =============================================================================
def build_eval_selection_gates(exp_root: Path, specs: list[ModelSpec], metrics_csv: Path) -> dict[str, Any]:
    """构建 eval_trajectory / ood_penalty / 每个 OPD arm 的 matched pair / Gate 初判。"""
    spec_by_source = {s.source: s for s in specs}
    df = pd.read_csv(metrics_csv, encoding="utf-8-sig")
    base_rows = df[df["Source"].astype(str) == "theta0"]
    if base_rows.empty:
        raise ValueError("No theta0 baseline in metrics CSV")
    base = base_rows.iloc[0].to_dict()

    eval_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []
    for _, series in df.iterrows():
        row = series.to_dict()
        source = str(row["Source"])
        spec = spec_by_source.get(source)
        row["checkpoint_id"] = f"{source}_{row.get('DataSize')}"
        row["role"] = spec.role if spec else "unknown"
        row["checkpoint_path"] = str(spec.checkpoint_path) if spec else ""
        row["model_dir"] = str(spec.model_dir) if spec else ""
        gsm = _as_float(row.get("GSM8K"))
        row["GSM8K_gain"] = None if gsm is None else gsm - _as_float(base.get("GSM8K"))
        drops = []
        for bench in OOD_BENCHMARKS:
            score = _as_float(row.get(bench))
            base_score = _as_float(base.get(bench))
            drop = None if score is None or base_score is None else max(0.0, base_score - score)
            ood_rows.append({"checkpoint_id": row["checkpoint_id"], "Source": source, "benchmark": bench,
                             "score": score, "baseline_score": base_score, "drop": drop})
            if drop is not None:
                drops.append(drop)
        if drops:
            valid = [_as_float(row.get(b)) for b in OOD_BENCHMARKS if _as_float(row.get(b)) is not None]
            row["OOD_lite_avg"] = sum(valid) / len(valid)
            row["OOD_lite_penalty_p2"] = sum(d**2 for d in drops) ** 0.5
            row["Worst_OOD_lite_drop"] = max(drops)
        else:
            row["OOD_lite_avg"] = row["OOD_lite_penalty_p2"] = row["Worst_OOD_lite_drop"] = None
        eval_rows.append(row)

    tables = ensure_dir(exp_root / "tables")
    write_csv(tables / "eval_trajectory.csv", eval_rows)
    write_csv(tables / "ood_penalty.csv", ood_rows)

    opd_rows = [r for r in eval_rows if r["role"] == "opd"]
    sft_rows = [r for r in eval_rows if r["role"] == "sft"]
    theta0_gsm = _as_float(base.get("GSM8K"))

    # 每个 OPD arm 找最近 SFT match
    pairs = []
    for opd in opd_rows:
        og = _as_float(opd.get("GSM8K_gain"))
        best = None
        best_gap = None
        for s in sft_rows:
            sg = _as_float(s.get("GSM8K_gain"))
            if sg is None or og is None:
                continue
            gap = abs(og - sg)
            if best_gap is None or gap < best_gap:
                best_gap, best = gap, s
        pair = {
            "opd_arm": opd["Source"], "opd_GSM8K": opd.get("GSM8K"), "opd_GSM8K_gain": og,
            "opd_OOD_lite_penalty_p2": opd.get("OOD_lite_penalty_p2"),
            "sft_match": best["Source"] if best else None,
            "sft_GSM8K_gain": best.get("GSM8K_gain") if best else None,
            "sft_OOD_lite_penalty_p2": best.get("OOD_lite_penalty_p2") if best else None,
            "GSM8K_gain_gap": best_gap,
            "match_status": "valid_match" if (best_gap is not None and best_gap <= 0.02) else "nearest_match",
        }
        pairs.append(pair)
    write_csv(tables / "matched_gsm8k_pairs.csv", pairs)

    return {"eval_rows": eval_rows, "pairs": pairs, "theta0_gsm": theta0_gsm, "base": base}


def main() -> None:
    ap = argparse.ArgumentParser(description="Cycle 04 OPD Stability-to-Gain Gate")
    ap.add_argument("--exp-root", default=str(EXP_ROOT_DEFAULT))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-vllm", action="store_true", help="禁用 vLLM colocate（回退 HF generate）")
    ap.add_argument("--skip-getslice", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("TMPDIR", "/root/autodl-tmp/pip-tmp")
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    ensure_dir(Path(os.environ["TMPDIR"]))

    exp_root = Path(args.exp_root).expanduser().resolve()
    ensure_dir(exp_root)
    use_vllm = not args.no_vllm

    if args.smoke:
        cfg = dict(n_cold=8, cold_steps_div=2, opd_steps=2, opd_grad=2,
                   sft_sizes=[8, 16], n_heldout=8, n_probe=4, max_new_tokens=256,
                   eval_limit=8, gs_seqlen=64, gs_nsamples=4, vllm_gpu_mem=0.30,
                   arms=[("opd_lmbda1", 1.0, 42)])
    else:
        cfg = dict(n_cold=256, cold_steps_div=4, opd_steps=200, opd_grad=4,
                   sft_sizes=[128, 256, 512, 1024], n_heldout=64, n_probe=32, max_new_tokens=2048,
                   eval_limit=None, gs_seqlen=512, gs_nsamples=16, vllm_gpu_mem=0.30,
                   arms=[("opd_lmbda1", 1.0, 42), ("opd_lmbda05", 0.5, 42)])

    artifacts: dict[str, Any] = {}
    try:
        n_sft_max = max(cfg["sft_sizes"])
        pool = build_unified_pool(exp_root, n_cold=cfg["n_cold"], n_opd=cfg["opd_steps"],
                                  n_sft_max=n_sft_max, n_heldout=cfg["n_heldout"], n_probe=cfg["n_probe"], seed=42)

        # theta0 cold-start（OPD-like, lmbda=1）
        cold_steps = max(cfg["n_cold"] // cfg["cold_steps_div"], 1)
        theta0 = train_opd_arm(
            exp_root=exp_root, label="theta0_cold_start", student_checkpoint=None,
            prompt_jsonl=pool["train_prompts"], eval_jsonl=pool["heldout_eval"],
            max_samples=cfg["n_cold"], max_steps=cold_steps, grad_accum=cfg["cold_steps_div"],
            eval_steps=max(cold_steps // 2, 1), lmbda=1.0, seed=42, role_label="theta0",
            registry_method="cold_start", model_role="theta0", use_vllm=use_vllm, vllm_gpu_mem=cfg["vllm_gpu_mem"],
        )
        theta0 = ModelSpec("theta0", str(cfg["n_cold"]), "theta0", theta0.model_dir, theta0.checkpoint_path)

        # OPD arms
        opd_arms: list[ModelSpec] = []
        for arm_label, lmbda, seed in cfg["arms"]:
            consumed = cfg["opd_steps"] * cfg["opd_grad"]
            arm = train_opd_arm(
                exp_root=exp_root, label=arm_label, student_checkpoint=str(theta0.model_dir),
                prompt_jsonl=pool["train_prompts"], eval_jsonl=pool["heldout_eval"],
                max_samples=n_sft_max, max_steps=cfg["opd_steps"], grad_accum=cfg["opd_grad"],
                eval_steps=max(cfg["opd_steps"] // 4, 1), lmbda=lmbda, seed=seed, role_label=arm_label,
                registry_method="trl_opd_like", model_role="opd", use_vllm=use_vllm, vllm_gpu_mem=cfg["vllm_gpu_mem"],
            )
            opd_arms.append(ModelSpec(arm_label, str(consumed), "opd", arm.model_dir, arm.checkpoint_path))

        # SFT controls
        sft_specs: list[ModelSpec] = []
        for size in cfg["sft_sizes"]:
            spec = train_sft_control(
                exp_root=exp_root, label=f"sft_n{size}", theta0_merged=theta0.model_dir,
                train_sft_jsonl=Path(pool["train_sft"]), heldout_jsonl=Path(pool["heldout_eval"]),
                num_samples=size, learning_rate=1.0e-5,
            )
            sft_specs.append(spec)

        specs = [theta0, *opd_arms, *sft_specs]

        # full GSM8K eval + OOD-lite
        metrics_csv = run_full_eval_v2(exp_root, specs, eval_limit=cfg["eval_limit"])
        artifacts["target_metrics_csv"] = str(metrics_csv)
        sel = build_eval_selection_gates(exp_root, specs, metrics_csv)

        # 探针 + 三层交叉 GetSlice + 几何
        if not args.skip_getslice:
            probes = build_probes_multi(exp_root, pool, theta0, opd_arms, sft_specs,
                                        n_probe=cfg["n_probe"], max_new_tokens=cfg["max_new_tokens"])
            gs_root = None
            for layer in GEOMETRY_LAYERS:
                gs_root = run_getslice_cross(
                    exp_root, specs, probes, target_layer=layer,
                    seqlen=cfg["gs_seqlen"], s_nsamples=cfg["gs_nsamples"], x_nsamples=cfg["gs_nsamples"],
                )
            artifacts["geometry_metrics"] = str(build_geometry_tables_cross(exp_root, gs_root))
            build_figures_cross(exp_root)

        write_json(exp_root / "selection.json", {"pairs": sel["pairs"], "theta0_gsm": sel["theta0_gsm"]})
        write_json(exp_root / "summary.json", {
            "experiment": "cycle04_opd_stability_gain", "status": "completed",
            "use_vllm_colocate": use_vllm, "geometry_layers": GEOMETRY_LAYERS,
            "models": [s.source for s in specs], "artifacts": artifacts,
        })
        print(f"[DONE] cycle04 completed: {exp_root}", flush=True)
    except Exception as exc:
        write_json(exp_root / "closure_failed.json", {"error": repr(exc)})
        print(f"[FAILED] {exc!r}", flush=True)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
