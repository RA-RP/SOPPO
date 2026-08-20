#!/usr/bin/env python3
"""Render the complete, manuscript-relevant table layer for human_read-ch.md.

The renderer deliberately stops at the declared analysis unit: dataset-level
behavior and layer-level equal means over the seven projection modules. Raw
per-sample rows, per-module spectra, and full tail curves remain in their CSV
artifacts and are not duplicated into the human-readable report.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
MINI = PAPER / "local_experiment_results" / "cycle_09_aaai_competitiveness_completion" / "run_01" / "mini"
STEPS = [0, 5, 10, 20, 40, 80, 160, 320, 480, 624]
ALPHA_STEPS = [0, 5, 20, 40, 80, 160, 320]
LLAMA_STEPS = [0, 5, 20, 40, 80, 160, 320]
ARMS = ["opd", "sft", "offkd", "seqkd"]
ARM_LABEL = {"opd": "OPD", "sft": "SFT", "offkd": "off-KD", "seqkd": "seqKD"}


def read_csv(name: str) -> list[dict[str, str]]:
    with (MINI / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def number(value: str | float | int | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def avg(values) -> float | None:
    xs = [x for x in (number(v) for v in values) if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "—"
    if signed:
        return f"{value:+.{digits}f}"
    return f"{value:.{digits}f}"


def sci(value: float | None) -> str:
    return "—" if value is None else f"{value:.2e}"


OUT: list[str] = []


def emit(line: str = "") -> None:
    OUT.append(line)


def heading(level: int, title: str) -> None:
    emit("#" * level + " " + title)
    emit()


def table(headers: list[str], rows: list[list[str]]) -> None:
    emit("| " + " | ".join(headers) + " |")
    emit("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        emit("| " + " | ".join(str(x) for x in row) + " |")
    emit()


def canonical_probe(name: str) -> str:
    aliases = {
        "legacy_S_math": "$D_{\\mathrm{mathCoTtrain}}$",
        "E_ood": "$E_{\\mathrm{mmluPro}}$",
        "E_general": "$E_{\\mathrm{general}}$",
        "E_math_hard": "$E_{\\mathrm{aime24}}$",
        "E_if": "$E_{\\mathrm{ifeval}}$",
        "S_bos": "legacy BOS control",
        "E_math": "$E_{\\mathrm{mathHeld}}$",
        "E_math_hard_v2": "$E_{\\mathrm{aime25}}$",
    }
    return aliases.get(name, name)


def alpha_probe(name: str) -> str:
    return "$D_{\\mathrm{mathCoTtrain}}$" if name == "S_math" else canonical_probe(name)


def llama_probe(name: str) -> str:
    return "$X_{\\mathrm{mathHeld}}$" if name == "S_math" else canonical_probe(name)


def r5_probe(name: str) -> str:
    return "$X_{\\mathrm{mathCoTtrain}}$" if name == "S_math" else canonical_probe(name)


heading(2, "B.1A 完整表的覆盖契约")
emit("本自动生成块恢复所有曾用于 human_read 判断、且当前协议仍有效的论文级聚合表。完整性的分析单位是：行为按数据集/类别聚合；几何按指定层、七个 projection module 等权聚合。逐样本行、逐模块奇异值、完整 tail curve 和 bootstrap draws 仍由正式 CSV/NPZ 保存，因为它们是原始数据而不是人类可读结果表。")
emit()
emit("表中“当前值（相对 step0 变化）”同时保留绝对标尺与论文使用的 $\\Delta r_\\varepsilon$，从而不再需要在两个重复大表之间切换。符号 — 表示该协议没有运行，而不是零。")
emit()
emit("旧版中三类表不按“有效表”恢复：以已废弃 whitened entropy ER 为核心的 dose-response/有限样本表；误筛 `frozen_base` track 的旧 G3 compact 表；把不配对的 4k/24k cap 运行解释为逐题因果效应的表述。淘汰原因只在附录 E.4 登记，不能继续充当论文证据。")
emit()


# Qwen r_epsilon, six fixed probes.
heading(2, "B.1B Qwen 六固定探针的完整 $r_\\varepsilon$ 轨迹")
emit("口径：Qwen、L18、per-checkpoint、$\\varepsilon=.05$、七模块等权均值。单元格为当前值（相对本臂 step0 的变化）。")
emit()

rank: dict[tuple[str, str, int], tuple[float, float]] = {}
for row in read_csv("C14_per_checkpoint_layer_sensitivity.csv"):
    if row["track"] != "per_checkpoint" or row["layer"] != "18" or row["epsilon"] != "0.05":
        continue
    probe = row["probe_family"]
    if probe not in {"legacy_S_math", "E_ood", "E_general", "E_math_hard", "S_bos"}:
        continue
    rank[(probe, row["arm"], int(row["step"]))] = (
        float(row["r_epsilon_current_mean"]),
        float(row["r_epsilon_delta_mean"]),
    )

eif_groups: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
for row in read_csv("C5_eif_m1_geometry.csv"):
    if row["track"] != "per_checkpoint" or row["layer"] != "18" or row["epsilon"] != "0.05":
        continue
    key = (row["arm"], int(row["step"]))
    eif_groups[key]["current"].append(float(row["r_epsilon"]))
    eif_groups[key]["delta"].append(float(row["r_epsilon_delta"]))
for (arm, step), values in eif_groups.items():
    rank[("E_if", arm, step)] = (avg(values["current"]), avg(values["delta"]))

for probe in ["legacy_S_math", "E_ood", "E_general", "E_math_hard", "S_bos", "E_if"]:
    heading(3, canonical_probe(probe))
    body = []
    for step in STEPS:
        row = [str(step)]
        for arm in ARMS:
            pair = rank.get((probe, arm, step))
            row.append("—" if pair is None else f"{fmt(pair[0])} ({fmt(pair[1], signed=True)})")
        body.append(row)
    table(["step"] + [ARM_LABEL[a] for a in ARMS], body)


# Qwen terminal layer sensitivity.
heading(3, "六探针终态的三层敏感性")
emit("单元格为 $\\Delta r_\\varepsilon$；L18 是正文层，L9/L27 是边界检查。")
emit()
layer_rank: dict[tuple[str, str, int, int], float] = {}
for row in read_csv("C14_per_checkpoint_layer_sensitivity.csv"):
    if row["track"] == "per_checkpoint" and row["epsilon"] == "0.05":
        probe = row["probe_family"]
        if probe in {"legacy_S_math", "E_ood", "E_general", "E_math_hard", "S_bos"}:
            layer_rank[(probe, row["arm"], int(row["step"]), int(row["layer"]))] = float(row["r_epsilon_delta_mean"])
eif_layer: dict[tuple[str, int, int], list[float]] = defaultdict(list)
for row in read_csv("C5_eif_m1_geometry.csv"):
    if row["track"] == "per_checkpoint" and row["epsilon"] == "0.05":
        eif_layer[(row["arm"], int(row["step"]), int(row["layer"]))].append(float(row["r_epsilon_delta"]))
for key, values in eif_layer.items():
    arm, step, layer = key
    layer_rank[("E_if", arm, step, layer)] = avg(values)
body = []
for probe in ["legacy_S_math", "E_ood", "E_general", "E_math_hard", "S_bos", "E_if"]:
    for arm in ARMS:
        body.append([
            canonical_probe(probe),
            ARM_LABEL[arm],
            fmt(layer_rank.get((probe, arm, 624, 9)), signed=True),
            fmt(layer_rank.get((probe, arm, 624, 18)), signed=True),
            fmt(layer_rank.get((probe, arm, 624, 27)), signed=True),
        ])
table(["probe", "arm", "L9", "L18", "L27"], body)


# Qwen behavior.
heading(2, "B.1C Qwen 四臂完整行为轨迹")
behavior = {(r["arm"], int(r["step"])): r for r in read_csv("three_arm_full_trajectory.csv")}
flex = {(r["arm"], int(r["step"])): r for r in read_csv("S1_mmlupro_flexible.csv")}
extract = {(r["arm"], int(r["step"])): r for r in read_csv("S1_mmlupro_extract_audit.csv")}

heading(3, "MATH500：accuracy / cap-hit / mean response length")
emit("每个单元格依次为 accuracy / truncation-or-cap-hit rate / mean response tokens。")
emit()
body = []
for step in STEPS:
    line = [str(step)]
    for arm in ARMS:
        r = behavior.get((arm, step))
        if not r:
            line.append("—")
        else:
            line.append(f"{fmt(number(r['math500_acc']))} / {fmt(number(r['math500_trunc_rate']))} / {fmt(number(r['math500_mean_response_len']), 0)}")
    body.append(line)
table(["step"] + [ARM_LABEL[a] for a in ARMS], body)

heading(3, "MATH500：各 checkpoint 的实际 generation cap")
emit("该表把生成预算本身显式列出，避免把 cap 改变与训练效应混淆；数值单位为 response tokens。")
emit()
body = []
for step in STEPS:
    body.append([str(step)] + [fmt(number(behavior.get((arm, step), {}).get("math500_cap")), 0) for arm in ARMS])
table(["step"] + [ARM_LABEL[a] for a in ARMS], body)

heading(3, "MMLU-Pro：strict / flexible / extract-fail")
body = []
for step in STEPS:
    line = [str(step)]
    for arm in ARMS:
        r = flex.get((arm, step))
        line.append("—" if not r else f"{fmt(number(r['exact_match']))} / {fmt(number(r['mmlu_pro_flexible']))} / {fmt(number(r['strict_extract_fail_rate']))}")
    body.append(line)
table(["step"] + [ARM_LABEL[a] for a in ARMS], body)

heading(3, "IFEval：prompt-strict / instruction-strict")
body = []
for step in STEPS:
    line = [str(step)]
    for arm in ARMS:
        r = behavior.get((arm, step))
        line.append("—" if not r else f"{fmt(number(r['ifeval_prompt_strict']))} / {fmt(number(r['ifeval_instruction_strict']))}")
    body.append(line)
table(["step"] + [ARM_LABEL[a] for a in ARMS], body)

heading(3, "GPQA-Diamond / TruthfulQA-MC1")
body = []
for step in STEPS:
    line = [str(step)]
    for arm in ARMS:
        r = behavior.get((arm, step))
        line.append("—" if not r else f"{fmt(number(r['gpqa_diamond_acc']))} / {fmt(number(r['truthfulqa_mc1_acc']))}")
    body.append(line)
table(["step"] + [ARM_LABEL[a] for a in ARMS], body)

ifeval_rows = read_csv("S1_ifeval_breakdown.csv")
ifeval_map = {(r["arm"], int(r["step"]), r["instruction_category"]): number(r["pass_rate"]) for r in ifeval_rows}
categories = sorted({r["instruction_category"] for r in ifeval_rows})
heading(3, "IFEval 九类别完整 pass-rate 轨迹")
for arm in ARMS:
    emit(f"**{ARM_LABEL[arm]}**")
    emit()
    body = []
    for step in STEPS:
        body.append([str(step)] + [fmt(ifeval_map.get((arm, step, c))) for c in categories])
    table(["step"] + categories, body)

heading(3, "MMLU-Pro extract failure 的完整构成")
emit("各失败子类均以全体 1,400 道题为分母；它们用于区分 bad-format 与 truncation，不作为知识分数。")
emit()
for arm in ARMS:
    emit(f"**{ARM_LABEL[arm]}**")
    emit()
    body = []
    for step in STEPS:
        r = extract.get((arm, step))
        body.append([
            str(step),
            fmt(number(r["extract_fail_rate"]) if r else None),
            fmt(number(r["failure_no_uppercase_standalone_A_to_J_rate_all_samples"]) if r else None),
            fmt(number(r["failure_letter_bad_format_rate_all_samples"]) if r else None),
            fmt(number(r["failure_truncated_rate_all_samples"]) if r else None),
        ])
    table(["step", "all extract-fail", "no standalone A–J", "bad format", "truncated"], body)


# PPL, ablation, entropy.
heading(3, "三个冻结训练文本 corpus 的完整 response-only PPL")
ppl_groups = {(r["corpus"], r["arm"], int(r["step"])): number(r["ppl"]) for r in read_csv("C8_training_corpus_ppl.csv")}
for corpus in ["X_OPD_reconstructed", "X_SFT_dataset", "X_teacher"]:
    emit(f"**{corpus}**")
    emit()
    body = [[str(step)] + [fmt(ppl_groups.get((corpus, arm, step)), 4) for arm in ARMS] for step in STEPS]
    table(["step"] + [ARM_LABEL[a] for a in ARMS], body)

heading(3, "off-KD@624 的完整 adapter 层组消融")
ablation = []
for r in read_csv("G8_adapter_ablation.csv"):
    ablation.append([
        r["config"],
        r["closed_layers"],
        fmt(number(r["math_acc"])),
        fmt(number(r["math_trunc_rate"])),
        fmt(number(r["mmlu_strict_fail_rate"])),
        fmt(number(r["mmlu_strict_acc"])),
        fmt(number(r["mmlu_flexible_acc"])),
    ])
table(["config", "closed layers", "MATH acc", "MATH trunc", "strict fail", "strict acc", "flexible"], ablation)

heading(3, "MMLU-Pro 答案位熵的终点类别分解")
emit("单元格为 full-vocabulary entropy / legal-option mass。为控制篇幅，这里保留正文机制所需的 14 类终点；完整 560 行十点表仍在 C11_mmlupro_answer_token_entropy_by_category.csv。")
emit()
entropy = read_csv("C11_mmlupro_answer_token_entropy_by_category.csv")
entropy_map = {
    (r["category"], r["arm"]): (
        number(r["full_vocab_entropy_nats_mean"]),
        number(r["option_first_token_mass_mean"]),
    )
    for r in entropy
    if int(r["step"]) == 624
}
body = []
for category in sorted({r["category"] for r in entropy}):
    line = [category]
    for arm in ARMS:
        pair = entropy_map.get((category, arm))
        line.append("—" if pair is None else f"{fmt(pair[0])} / {fmt(pair[1])}")
    body.append(line)
table(["category"] + [ARM_LABEL[a] for a in ARMS], body)


# M2 and theta U.
heading(2, "B.1D Qwen 次级功能几何的完整轨迹")
fixed_probes = ["legacy_S_math", "E_ood", "E_general", "E_math_hard", "S_bos"]

m2_groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
for r in read_csv("R4_m2_output_drift.csv"):
    if r["layer"] != "18" or r["reference"] != "X0_primary":
        continue
    probe = r["task_id"]
    if probe.startswith("S_bos__"):
        probe = "S_bos"
    if probe not in fixed_probes:
        continue
    m2_groups[(probe, r["arm"], int(r["step"]))].append(float(r["m2_output_drift"]))

heading(3, "M2：$\\|\\Delta W S_D\\|_F/\\|W_0S_D\\|_F$")
emit("口径：L18、X0_primary、七模块等权；BOS control 另对三个冻结 generation seeds 等权。")
emit()
for probe in fixed_probes:
    emit(f"**{canonical_probe(probe)}**")
    emit()
    body = [[str(step)] + [fmt(avg(m2_groups.get((probe, arm, step), [])), 4) for arm in ARMS] for step in STEPS]
    table(["step"] + [ARM_LABEL[a] for a in ARMS], body)

theta_groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
for r in read_csv("R5_theta_reps.csv"):
    if r["layer"] != "18" or r["track"] != "frozen_base" or r["epsilon"] != "0.05":
        continue
    probe = r["probe"]
    if probe.startswith("S_bos__"):
        probe = "S_bos"
    if probe not in fixed_probes:
        continue
    theta_groups[(probe, r["arm"], int(r["step"]))].append(float(r["theta_u_max_deg"]))

heading(3, "$\\theta_U$：base 与 checkpoint 左奇异子空间最大主夹角")
emit("口径：L18、frozen-base reference、$\\varepsilon=.05$、七模块等权；这是次级转角仪器，不替代 per-checkpoint $r_\\varepsilon$。")
emit()
for probe in fixed_probes:
    emit(f"**{canonical_probe(probe)}**")
    emit()
    body = [[str(step)] + [fmt(avg(theta_groups.get((probe, arm, step), [])), 2) for arm in ARMS] for step in STEPS if step != 0]
    table(["step"] + [ARM_LABEL[a] for a in ARMS], body)


# Related-work/native comparator panel.
heading(2, "B.1E 相关工作与 native-space 仪器完整表")

heading(3, "normalized raw ER 的逐步 bootstrap")
emit("单元格为 point delta [95% CI]；这是 raw activation 构念，量纲不能与 directions 直接比较。")
emit()
raw_rows = read_csv("C2_raw_er_bootstrap.csv")
raw_map = {
    (r["arm"], int(r["step"])): (
        number(r["point_delta"]),
        number(r["ci95_lo"]),
        number(r["ci95_hi"]),
    )
    for r in raw_rows
}
raw_steps = sorted({int(r["step"]) for r in raw_rows})
body = []
for step in raw_steps:
    line = [str(step)]
    for arm in ARMS:
        triple = raw_map.get((arm, step))
        line.append("—" if triple is None else f"{sci(triple[0])} [{sci(triple[1])},{sci(triple[2])}]")
    body.append(line)
table(["step"] + [ARM_LABEL[a] for a in ARMS], body)

heading(3, "R5 activation-suite：CKA、raw ER、PR 与 anisotropy")
emit("口径：L18；现有正式表只覆盖 OPD/SFT、四个旧协议 probes 和七个 landmarks。raw-ER 列按产物定义复刻，仅作构念审计。")
emit()
r5 = [r for r in read_csv("R5_paper_metrics_qwen3.csv") if r["layer"] == "18"]
probe_order = ["legacy_S_math", "S_math", "E_ood", "E_general"]
body = []
for probe in probe_order:
    for arm in ["opd", "sft"]:
        for r in sorted((x for x in r5 if x["probe"] == probe and x["arm"] == arm), key=lambda x: int(x["step"])):
            body.append([
                r5_probe(probe),
                ARM_LABEL[arm],
                r["step"],
                fmt(number(r["cka_vs_base"]), 4),
                sci(number(r["erank_covariance_verbatim"])),
                sci(number(r["pr_normalized"])),
                fmt(number(r["top1_variance_share"]), 4),
                fmt(number(r["anisotropy_centered"]), 4),
            ])
table(["probe", "arm", "step", "CKA", "raw ER norm", "PR norm", "top1 share", "centered anis."], body)

heading(3, "weight-only canonical cosine（rank32）")
emit("单元格为七模块平均 left-min-cos / right-min-cos。数值接近 1 表示 source weight 的 rank-32 canonical directions 转角小。")
emit()
weight_groups: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
for r in read_csv("R3_theta_w.csv"):
    if r["layer"] == "18" and r["rank"] == "32":
        key = (r["arm"], int(r["step"]))
        weight_groups[key]["left"].append(float(r["left_min_cosine"]))
        weight_groups[key]["right"].append(float(r["right_min_cosine"]))
body = []
for step in STEPS:
    line = [str(step)]
    for arm in ["opd", "sft", "offkd"]:
        g = weight_groups.get((arm, step))
        line.append("—" if not g else f"{fmt(avg(g['left']), 4)} / {fmt(avg(g['right']), 4)}")
    body.append(line)
table(["step", "OPD", "SFT", "off-KD"], body)

heading(3, "现有 left-only source-principal 投影（不是严格 $p_k$）")
emit("口径：L18、$k=32$、七模块等权。SFT/off-KD 使用 fp32 BA；OPD 使用 top32 approximation。单元格为 $\\rho_U^2$ / matched random-null mean。seqKD 与 joint right projection 缺失。")
emit()
track_for_arm = {"sft": "sft_fp32_BA", "offkd": "offkd_fp32_BA", "opd": "opd_top32_approx"}
rho_groups: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
for r in read_csv("T4_rho_dualtrack.csv"):
    if r["layer"] != "18" or r["k"] != "32":
        continue
    arm = next((a for a, track in track_for_arm.items() if r["track"] == track), None)
    if arm is None:
        continue
    key = (arm, int(r["step"]))
    rho_groups[key]["rho"].append(float(r["rho2_U"]))
    rho_groups[key]["null"].append(float(r["null_mean"]))
body = []
for step in [s for s in STEPS if s != 0]:
    line = [str(step)]
    for arm in ["opd", "sft", "offkd"]:
        g = rho_groups.get((arm, step))
        line.append("—" if not g else f"{fmt(avg(g['rho']), 4)} / {fmt(avg(g['null']), 4)}")
    body.append(line)
table(["step", "OPD approx", "SFT fp32", "off-KD fp32"], body)

heading(3, "大转动方向的 endpoint overlap")
emit("口径：L18、U-space、$\\varepsilon=.05$、step624。单元格为非空模块的 overlap-coefficient 均值（有效模块数/7）；空集合不当作零。")
emit()
overlap_groups: dict[tuple[str, tuple[str, str]], list[float]] = defaultdict(list)
for r in read_csv("S1_direction_overlap.csv"):
    if r["step"] != "624" or r["layer"] != "18" or r["space"] != "U" or r["epsilon"] != "0.05":
        continue
    pair = tuple(sorted((r["arm_a"], r["arm_b"])))
    val = number(r["overlap_coefficient"])
    if val is not None:
        overlap_groups[(r["task_id"], pair)].append(val)
pairs = [tuple(sorted(x)) for x in [("opd", "offkd"), ("opd", "sft"), ("sft", "offkd")]]
tasks = sorted({r["task_id"] for r in read_csv("S1_direction_overlap.csv")})
body = []
for task in tasks:
    line = [canonical_probe(task)]
    for pair in pairs:
        vals = overlap_groups.get((task, pair), [])
        line.append("—" if not vals else f"{fmt(avg(vals), 2)} ({len(vals)}/7)")
    body.append(line)
table(["probe/task", "OPD–off-KD", "OPD–SFT", "SFT–off-KD"], body)


# Alpha epsilon sensitivity.
heading(2, "B.1G Qwen alpha=.5 的完整 epsilon 敏感性")
emit("口径：L18、per-checkpoint、七模块等权。单元格均为相对 step0 的 $\\Delta r_\\varepsilon$；主文仍只使用 $\\varepsilon=.05$。")
emit()
alpha_groups: dict[tuple[str, int, float], list[float]] = defaultdict(list)
for r in read_csv("qwen_alpha05_r_epsilon.csv"):
    if r["layer"] == "18" and r["track"] == "per_checkpoint":
        alpha_groups[(r["probe"], int(r["step"]), float(r["epsilon"]))].append(float(r["r_epsilon_delta"]))
alpha_probes = ["S_math", "E_math", "E_math_hard_v2", "E_ood", "E_if", "E_general"]
epsilons = [0.01, 0.025, 0.05, 0.1]
for probe in alpha_probes:
    emit(f"**{alpha_probe(probe)}**")
    emit()
    body = []
    for step in ALPHA_STEPS:
        body.append([str(step)] + [fmt(avg(alpha_groups.get((probe, step, eps), [])), signed=True) for eps in epsilons])
    table(["step"] + [f"eps={eps:g}" for eps in epsilons], body)


# Llama four-arm geometry and behavior through step 320.
heading(2, "B.1H Llama-3.2-3B 四臂至 step320 的完整交接表")
emit("口径：OPD/SFT/off-KD/seqKD 四臂，checkpoint {0,5,20,40,80,160,320}；几何正文层为 L14，稳健性层为 L7/L21。step0 是四臂共享 base。")
emit()
llama_r = read_csv("llama_early_320_r_epsilon.csv")
llama_groups: dict[tuple[str, str, int, int, float], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
for r in llama_r:
    if r["track"] != "per_checkpoint":
        continue
    key = (r["probe"], r["arm"], int(r["step"]), int(r["layer"]), float(r["epsilon"]))
    llama_groups[key]["current"].append(float(r["r_epsilon"]))
    llama_groups[key]["delta"].append(float(r["delta_from_base"]))
llama_probes = ["S_math", "E_math", "E_math_hard_v2", "E_ood", "E_if", "E_general"]

heading(3, "L14、epsilon=.05 的六探针四臂轨迹")
for probe in llama_probes:
    emit(f"**{llama_probe(probe)}**")
    emit()
    body = []
    for step in LLAMA_STEPS:
        line = [str(step)]
        for arm in ARMS:
            source_arm = "base" if step == 0 else arm
            g = llama_groups.get((probe, source_arm, step, 14, 0.05))
            line.append("—" if not g else f"{fmt(avg(g['current']))} ({fmt(avg(g['delta']), signed=True)})")
        body.append(line)
    table(["step"] + [ARM_LABEL[a] for a in ARMS], body)

heading(3, "step160 的四臂 epsilon×layer 敏感性")
body = []
for probe in llama_probes:
    for arm in ARMS:
        for layer in [7, 14, 21]:
            line = [llama_probe(probe), ARM_LABEL[arm], str(layer)]
            for eps in epsilons:
                g = llama_groups.get((probe, arm, 160, layer, eps))
                line.append("—" if not g else fmt(avg(g["delta"]), signed=True))
            body.append(line)
table(["probe", "arm", "layer"] + [f"eps={eps:g}" for eps in epsilons], body)

heading(3, "Llama 四臂行为")
llama_b = {(r["arm"], r["task"], int(r["step"])): r for r in read_csv("llama_early_320_behavior.csv")}
body = []
for arm in ARMS:
    for step in LLAMA_STEPS:
        source_arm = "base" if step == 0 else arm
        math_r = llama_b.get((source_arm, "math500", step))
        mmlu_r = llama_b.get((source_arm, "mmlu_pro", step))
        if_r = llama_b.get((source_arm, "ifeval", step))
        body.append([
            ARM_LABEL[arm],
            str(step),
            "—" if not math_r else f"{fmt(number(math_r['accuracy']))} / {fmt(number(math_r['cap_hit_rate']))} / {fmt(number(math_r['response_length_mean']), 0)}",
            "—" if not mmlu_r else f"{fmt(number(mmlu_r['strict_accuracy']))} / {fmt(number(mmlu_r['flexible_accuracy']))} / {fmt(number(mmlu_r['extract_failure_rate']))}",
            "—" if not if_r else f"{fmt(number(if_r['prompt_strict_accuracy']))} / {fmt(number(if_r['instruction_strict_accuracy']))}",
        ])
table(["arm", "step", "MATH acc/cap-hit/len", "MMLU strict/flex/fail", "IFEval prompt/instruction"], body)

heading(3, "Llama L14 可用的 raw representation suite")
emit("这张表是 native activation-space 对照，按产物字段原样展示；其 entropy-ER 命名/尺度与 Qwen R5 的 normalized convention 不同，禁止直接横向比较数值。正式 suite 覆盖共享 base 与四臂的 5/20/40/160；没有 80/320 行，不作插值。")
emit()
llama_raw = [r for r in read_csv("llama_early_320_raw_representation_suite.csv") if r["layer"] == "14"]
body = []
for probe in llama_probes:
    for r in sorted(
        (x for x in llama_raw if x["probe"] == probe),
        key=lambda x: (0 if x["arm"] == "base" else ARMS.index(x["arm"]) + 1, int(x["step"])),
    ):
        body.append([
            llama_probe(probe),
            "base" if r["arm"] == "base" else ARM_LABEL[r["arm"]],
            r["step"],
            fmt(number(r["linear_cka_vs_step0"]), 4),
            fmt(number(r["normalized_entropy_effective_rank"]), 3),
            fmt(number(r["participation_ratio"]), 3),
            fmt(number(r["top1_explained_share"]), 4),
            fmt(number(r["centered_anisotropy"]), 4),
        ])
table(["probe", "arm", "step", "CKA", "entropy ER", "PR", "top1 share", "centered anis."], body)


rendered = "\n".join(OUT).rstrip() + "\n"

if "--update" in sys.argv:
    target = PAPER / "theory" / "human_read-ch.md"
    begin = "<!-- BEGIN AUTO-GENERATED FULL TABLES -->"
    end = "<!-- END AUTO-GENERATED FULL TABLES -->"
    block = begin + "\n\n" + rendered + "\n" + end
    text = target.read_text(encoding="utf-8")
    if begin in text and end in text:
        prefix, remainder = text.split(begin, 1)
        _, suffix = remainder.split(end, 1)
        updated = prefix + block + suffix
    else:
        anchor = "## B.2 raw ER 与其他表征谱指标：保留，但不冒充主权重对照"
        if anchor not in text:
            raise RuntimeError(f"Cannot find insertion anchor in {target}")
        updated = text.replace(anchor, block + "\n\n" + anchor, 1)
    target.write_text(updated, encoding="utf-8")
    print(f"updated {target} with {len(OUT)} generated lines")
else:
    print(rendered, end="")
