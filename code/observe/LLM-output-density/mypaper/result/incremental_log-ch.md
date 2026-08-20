# 结果增量日志

```yaml
view_type: incremental_log
status: active
maintenance_rule: append_only_per_cycle
last_updated: 2026-06-18
```

本文件按 cycle 追加，记录每轮相对上一轮新增了什么、改变了什么。历史条目不改写。详细数字和证据见 `local_experiment_results/<cycle_name>/`；当前全项目可用结论见 `current_picture.md`。

---

## Cycle 03：首次 OPD-like 实验（`cycle_03_trl_opd_like`）

**新增内容：** 首次在 Qwen3-1.7B + NuminaMath 上完成基于 TRL 的 OPD-like vs SFT 对比。GSM8K 作为主要评测轴。

**结果说明了什么：** OPD-like 训练在 matched GSM8K gain 下看起来比 SFT 更稳定：OPD 维持或微升 GSM8K，而较大的 SFT 变体退化。视为初步正向信号。

**本轮不能声称的：** 信号受 `theta0` 饱和度混淆。matched SFT 对照是相对退化的大 SFT，比较本身容易通过。无 ID/OOD 轴分离；无几何；GSM8K 是唯一轴。

**遗留局限：** GSM8K 同时作为训练域和评测域指标；小样本高噪声；无未训练基座参照。

---

## Cycle 04：更严格的复现实验（`cycle_04_opd_stability_gain`）

**新增内容：** 更严格的 matched-control 设计；`theta0` 从不饱和起点（256 prompts）初始化；全样本 GSM8K（N=1319）；OOD-lite 套件（MMLU/TruthfulQA/WinoGrande）；多层 GetSlice 几何；引入 NuminaMath-test 作为独立 ID 轴；增加未训练 `Qwen3-1.7B` instruct checkpoint 作为参照行。

**结果说明了什么：**
- Gate A/B 通过：实验跑通，两个 OPD arm 都找到了有效的 GSM8K gain match。
- Gate C/D 失败：OPD 未在 OOD-lite penalty 或谱漂移上击败 matched SFT。matched SFT 退化到 `sft_n128`（几乎未离开 `theta0`），OPD 对比的是一个几乎不动的对照——比较先天不公平。
- 当时看似存在 base ceiling：未训练基座在 NuminaMath ID 和 GSM8K 上都与 theta0 近似持平，且 ID 高于两个 OPD arm，说明任何训练方式都没有产生相对 base 的净增益。
- 几何：混合信号——OPD 降低 effective rank（谱更集中），SFT 升高；但 OPD 未能一致降低相对 theta0 的漂移。
- MATH500 全部 = 0.0：发现是评分假象（`FINDING_04_eval_extraction_artifact.md`），不是无能力。

**本轮两个重大发现：**
- `FINDING_04_eval_extraction_artifact.md`：MATH500 = 0.0 是 `$` span 抽取假象；NuminaMath-test 作为 ID 轴引入。
- `FINDING_04_base_ceiling.md`：未训练基座与 theta0 持平，当时认为能力 headroom 耗尽（后被 Cycle 05 修正推翻）。

详细结果：`local_experiment_results/cycle_04_opd_stability_gain/RESULTS_04.md`。

---

## Cycle 05：评测协议审计（`cycle_05_matched_control_id_ood`）——已关闭（2026-06-18）

**新增内容：** 预 pivot 能力核查发现 Cycle 04 测量栈中三个叠加的评测 bug。无新训练；所有发现均为对已有 Cycle 04 checkpoint 的协议修正。全部 8 个模型在修正协议下完成重测。

### Finding 05a：GSM8K 缺少 chat template

Cycle 04 每一条 GSM8K 分数都是用 `lm_eval` 默认的原始补全 prompt 在 instruct chat 模型上跑出的（无 chat template，无 `enable_thinking=False`）。修正协议（chat template + `enable_thinking=False` + 3072 tokens + `\boxed{}` 感知抽取）让每个模型 GSM8K 提升约 +0.28-0.30 绝对值。原来的 ~0.41 区间几乎全是测量地板。详见 `local_experiment_results/cycle_05_matched_control_id_ood/FINDING_05_gsm8k_chat_template_mismatch.md`。

### Finding 05b：MATH500 dollar-span 抽取 bug

`lm_eval` 的 `hendrycks_math500` 评分器通过截取整个生成文本中第一个和最后一个 `$` 之间的内容来抽取"答案"——对于任何包含中间 LaTeX 的详细 CoT 生成都是灾难性错误。修复方案（专用脚本，`last_boxed` 抽取 + `math_verify` 符号等价评分）将每个模型提升到 ~0.54-0.73。详见 `local_experiment_results/cycle_05_matched_control_id_ood/FINDING_05_math500_dollar_span_extraction_bug.md`。

### Finding 05c：MMLU 崩溃 → 通过 base model 模式解决

对 MMLU（loglikelihood/多选任务）加 `--apply_chat_template` 而不加 `enable_thinking=False`，让所有模型都崩到恰好 0.2295（≈ 4 选项随机猜测）。解决方案：采用 Qwen3 官方评测协议（arXiv 2505.09388）—— base model 模式（无 chat template，5-shot，标准 loglikelihood）。这使得全部 8 个模型的 MMLU 得分落在 0.60-0.61 范围内，与已知的 Qwen3-1.7B 能力一致。社区独立确认：lm-eval-harness Issue #3405/#3576/#3322。详见 `FINDING_05_mmlu_chat_template_collapse.md`。

**全部 8 个模型修正后结果（2026-06-17）：**

| 模型 | NuminaMath ID | GSM8K | ±SE | MATH500 | ±SE | MMLU | TQA-MC1 | WinoGrande |
|---|---|---|---|---|---|---|---|---|
| base | 0.5516 | 0.7005 | 0.0126 | 0.7140 | 0.0202 | 0.6061 | 0.3341 | 0.6172 |
| theta0 | 0.5527 | 0.7180 | 0.0124 | 0.7260 | 0.0199 | 0.6019 | 0.3415 | 0.6235 |
| opd_lmbda05 | 0.5392 | 0.7180 | 0.0124 | 0.7080 | 0.0203 | 0.6004 | 0.3390 | 0.6251 |
| opd_lmbda1 | 0.5348 | 0.7346 | 0.0122 | 0.7140 | 0.0202 | 0.6061 | 0.3427 | 0.6235 |
| sft_n128 | 0.5370 | 0.7096 | 0.0125 | 0.7260 | 0.0199 | 0.6049 | 0.3439 | 0.6290 |
| sft_n256 | 0.5516 | 0.6839 | 0.0128 | 0.7080 | 0.0203 | 0.6033 | 0.3415 | 0.6235 |
| sft_n512 | 0.5179 | 0.6732 | 0.0129 | 0.7140 | 0.0202 | 0.6086 | 0.3378 | 0.6322 |
| sft_n1024 | 0.3890 | 0.6141 | 0.0134 | 0.5400 | 0.0223 | 0.6065 | 0.3317 | 0.6322 |

**评测协议（全 8 个模型）：**
- NuminaMath-test ID：chat template，`enable_thinking=False`，open-answer 子集
- GSM8K：chat template + `enable_thinking=False` + 3072 tokens + `\boxed{}` 感知抽取
- MATH500：chat template + `enable_thinking=False` + 3072 tokens + `math_verify` + `last_boxed` 抽取
- MMLU：base model 模式（无 chat template，5-shot，标准 loglikelihood）——`mmlu_protocol_version=2`
- TruthfulQA-MC1：chat template，loglikelihood，0-shot
- WinoGrande：无 chat template，5-shot，loglikelihood

**Cycle 05 Gates：**
- Gate A05：✅ PASS — 全 8 个模型在修正协议下完成全部任务重测
- Gate B05：✅ PASS — MMLU 通过 base model 模式解决（Qwen3 官方协议）
- Gate C05：✅ PASS — 数字自洽，无新评分假象

**Cycle 04 Gate 用修正数字重分析：**

*Gate B04（matched control anchor）：* 匹配基准是 NuminaMath-test ID，不是 GSM8K。在此基准下，sft_n128（0.5370）仍然是 opd_lmbda05（0.5392，Δ=+0.0022）和 opd_lmbda1（0.5348，Δ=−0.0022）的最近 SFT 锚点。**Gate B04：确认 PASS。**

*Gate C04/D04（OOD-lite 稳定性）：* 全 8 个模型的 OOD-lite 都有有效数字。全部 8 个模型 spread：MMLU ±0.0041（range 0.0082），TruthfulQA ±0.0061（range 0.0122），WinoGrande ±0.0075（range 0.0150）。OPD arm 在任何 OOD-lite 指标上都没有击败 sft_n128。**Gate C04：确认 FAIL。** 然而失败是非信号：全部 8 个模型包括 base 在每个指标上都聚集在 <1.5% 内；该轴在当前设置下判别力不足。**Gate D04：确认 FAIL（依赖 Gate C04）。**

**值得注意的发现 — sft_n1024 数学退化：** sft_n1024 表现出严重的数学退化（NuminaMath ID −0.163，GSM8K −0.086，MATH500 −0.174 vs base），OOD-lite 表现稳定。训练/评测格式对齐已验证（Qwen3 tokenizer 在训练和评测路径中注入完全相同的空 think block）。这种格局与 1.7B 模型在 n=1024 SFT 样本下的灾难性遗忘一致。sft_n1024 是异常值；sft_n128/256/512 是相关的 SFT 对照。

### ARC-Challenge OOD-lite（增量，2026-06-17）

ARC-challenge 在 Cycle 05 初始结果写入后完成，作为 OOD-lite 套件的第 4 项任务。协议：无 chat template，`enable_thinking=False`，25-shot（lm-eval arc_challenge 任务默认值，日志确认"Overwriting default num_fewshot of arc_challenge from None to 25"），`acc_norm`，N=1172。与 MMLU 和 WinoGrande 相同的无 chat template loglikelihood 协议。

| 模型 | ARC acc_norm | vs base |
|---|---|---|
| base | 0.4565 | — |
| theta0 | 0.4590 | +0.0026 |
| opd_lmbda05 | 0.4650 | +0.0086 |
| opd_lmbda1 | 0.4616 | +0.0051 |
| sft_n128 | 0.4599 | +0.0034 |
| sft_n256 | 0.4667 | +0.0103 |
| sft_n512 | 0.4710 | +0.0145 |
| sft_n1024 | 0.4667 | +0.0103 |

所有 SE = ±0.0146。Spread range = 0.0145。确认 OOD-lite 非判别性格局：全 8 个模型聚集在 <1.5% 内——与 MMLU（0.0082）、TruthfulQA（0.0122）、WinoGrande（0.0150）结论完全一致。sft_n1024 在 ARC 上无退化（+0.0103 over base），与 OOD-lite 能力保持格局一致。

详细结果：`local_experiment_results/cycle_05_matched_control_id_ood/run_01/RESULTS_05.md`。

**Cycle 05 状态：已关闭（2026-06-18）。** 全 8 个模型在全 7 项任务的修正协议下完成评测。Eval stack 统一至 `Eval/run_eval.py`。参见 `research_cycles/result_interpretation_evolution-ch.md` 和 `research_cycles/exp_design_evolution-ch.md` 的已关闭 cycle 条目。
