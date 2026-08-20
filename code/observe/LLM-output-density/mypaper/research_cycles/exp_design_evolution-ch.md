# 实验设计演进（Cycle 03 → 05，中文阅读版）

```yaml
view_type: cross_cycle_evolution_log_ch
source_file: exp_design_evolution.md
last_organized: 2026-06-16
```

> 本文件是 `exp_design_evolution.md` 的中文阅读版，内容与源文件保持一致。稳定的实现方法论（模型/数据/训练/评测/几何协议，不是每轮都变）现在单独放在 `../exp/experimental_methodology/`；本文件只记录每轮**设计和 gate** 变了什么。

## Cycle 03 —— 起点

全文来源：`cycle_03_trl_opd_like/03_experiment_design/initial_experiment_design_03.md`、`minimal_trl_opd_experiment_03.md`。

最初的设计（`initial_experiment_design_03.md`）规划了一个 8 个实验的宏大计划（GetSlice sanity check、SFT cold-start、LR 扫描的 continued SFT、`pi_mix` 扫描的 OPD、matched-MATH500 比较、几何验证、可选 calibration、可选 selection extension）。**实际跑的**是一个小得多的 TRL-first pilot：一个 `theta0` cold-start（512 条 prompt）、一个 OPD arm（`lmbda=1.0`）、四个 SFT 对照（256/512/1024/2048）、按 **GSM8K**（不是 MATH500）gain 匹配、单层（14 层）GetSlice 几何、`eval_limit=200`。这个"宏大设计 vs 最小 pilot"的差距是有意的（feasibility 优先），并且在 `00_cycle_index.md` 的"Next Gate"里被明确承认：提高/取消 `eval_limit`、加更多 OPD 设置/seed、用更不饱和的 `theta0`、把几何扩展到 14 层之外。

## Cycle 03 → Cycle 04：照单全收 Cycle 03 的 gate，并新增预注册 gate

全文来源：`cycle_04_opd_stability_gain/03_experiment_design/experiment_design_04.md`、`acceptance_criteria_04.md`。

Cycle 03 的四项"Next Gate"全部被采纳：`theta0` 降到 256 条（更不饱和）、两个 OPD arm（`lmbda` 1.0 和 0.5）、四个新的 SFT 数据量（128/256/512/1024）、GSM8K 用 `eval_limit=None`（全量 N=1319）、几何扩展到 3 层（6/14/22）并加入 vLLM colocate 提速。更大的变化是**流程**：Cycle 04 引入了预注册的 Gate A–D（`acceptance_criteria_04.md`），有硬性 pass/fail 阈值（`GSM8K_gain_gap <= 0.02`、OOD-lite penalty 对比、80% cells 的 drift 要求、Gate D 的 0.02 GSM8K-gain 阈值），取代了 Cycle 03 那种更松散的五档解读方案。结果：Gate A/B 通过，C/D 失败，根因是 matched-gain anchor 在更不饱和的 `theta0` 下坍缩到 `sft_n128`（见 `result_interpretation_evolution-ch.md`）。

## Cycle 04 → Cycle 05：围绕 Cycle 04 的失败诊断重新设计

全文来源：`cycle_05_matched_control_id_ood/03_experiment_design/experiment_design_05.md`、`acceptance_criteria_05.md`（= 当前 `../exp/current_experiment_design-ch.md`）。

新的 Gate A–F（`acceptance_criteria_05.md`）补上了 Cycle 04 缺的两项结构性要求：**强制的 base-ceiling row**（Gate B）和显式的**ID/OOD 轴分离**（Gate C：NuminaMath-test = ID，GSM8K = OOD/迁移，取代 GSM8K 之前的双重角色）。Equal-ID 和 equal-budget 两种 matching 规则被加进来，旧的 equal-GSM8K 规则降级为"仅作诊断"（Gate D）。

**Cycle 05 已关闭（2026-06-18）。** 实际执行情况：

Phase 1 reuse audit 最初把 Cycle 04 那个基于 GSM8K 的 base-ceiling 发现读作已确认（base 持平 theta0）→ 这几乎触发了切换到 non-instruct base model 的 pivot → 一项预注册的 pre-pivot capability check（对照 Qwen 官方基准数字）反而发现 Cycle 04 的 GSM8K eval 一直没加 chat template → 对 4/8 个模型的修正重测推翻了 headroom 判断 → base-swap pivot 被放弃。这是本项目至今最重要的一次设计自我纠错，而且是被一项预注册检查抓住的，不是靠运气。

**所有剩余设计任务均已完成（2026-06-17/18）：**
- `opd_lmbda1`、`sft_n256`、`sft_n512`、`sft_n1024` 已在修正协议下完成 GSM8K/MATH500 重测。完整 8 个模型的 Gate B–F 重分析已完成（见 `result_interpretation_evolution-ch.md` Cycle 05 已关闭条目）。
- OOD-lite 套件：MMLU 崩溃（FINDING_05_mmlu_chat_template_collapse.md）通过 base model 模式解决（无 chat template，5-shot，标准 loglikelihood——Qwen3 官方协议 arXiv 2505.09388）。TruthfulQA 和 WinoGrande 已确认。ARC-challenge 作为第 4 项 OOD-lite 任务加入（25-shot，无 chat template，acc_norm，N=1172）。
- "加 chat template" vs "不加"的区分现已完整确立：generate_until 类任务（GSM8K、MATH500、TruthfulQA）需要加；loglikelihood 类任务（MMLU、WinoGrande、ARC）加了反而有害。这条规则已固化到 `../exp/experimental_methodology/evaluation_protocol-ch.md` 的每任务协议表格中。

**Eval stack 重设计（2026-06-18）：** 全部 7 项评测任务统一到 `Eval/run_eval.py` 单一入口，`--task` 和 `--backend {custom,lm_eval}` 参数控制。自定义脚本移入 `Eval/component/{gsm8k,math500,numina}/runner.py`，各自暴露干净的 `run()` API。`Eval/run_cycle05_eval.py` 已更新，所有 subprocess 调用均通过 `run_eval.py` 路由。无用旧文件已删除。
