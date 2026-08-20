# RESULTS 04 — OPD Stability-to-Gain Gate（Cycle 04）

```yaml
artifact_type: results
cycle: cycle_04_opd_stability_gain
date: 2026-06-15
run_root: /root/autodl-tmp/cycle04_opd_stability_gain
status: completed
v2_finalized: 2026-06-16  # clean ID re-run(3072+采样)+ copyback + 自动关机均成功
gate_verdict: "A pass / B pass / C FAIL / D fail"
allowed_conclusion: "Cycle 03 的稳定性没有在 Cycle 04 条件下复现"
```

> 配套阅读:[FINDING_04_eval_extraction_artifact.md](FINDING_04_eval_extraction_artifact.md)（本轮重大发现:评测抽取假象 + NuminaMath ID 指标）。

## 0. 结论速览

| Gate | 结果 | 一句话 |
|---|---|---|
| **A 可行性** | ✅ PASS | 全部脚本跑通,5 表 3 图非空,colocate 提速,3 层几何齐 |
| **B matched** | ✅ PASS | 两个 OPD arm 均匹配 sft_n128,`GSM8K_gain_gap ≤ 0.02` |
| **C 稳定性** | ❌ **FAIL** | OOD-lite penalty 与谱漂移均**未**优于 matched SFT |
| **D 增益** | ❌ FAIL | 需 C 通过;且 OPD gain < 0.02 |

**主结论(预注册允许表述)**:**Cycle 03 的 OPD-like 稳定性信号没有在 Cycle 04 更严格条件下复现。** 根因不是 OPD 变差,而是 **matched-gain 对照锚点被 theta0 饱和度主导**(详见 §7)。

**ID×OOD 旁证(v2 clean,§5)**:在新增的 NuminaMath-test(ID)上,theta0=256 已是峰值,OPD 无增益、与小 SFT 持平;**OPD 与 SFT 唯一分离处是 GSM8K**(OPD 升 / SFT 随数据降)。即 OPD 优势仅限单一 OOD 任务且噪声邻近——与 Gate C 失败自洽。

**未训练基座对照(§5,本轮最强负面证据)**:基座 Qwen3-1.7B 在 ID(0.552)与 GSM8K(0.416)上**与 theta0 持平、且 ID 高于两个 OPD arm**。⇒ **cold-start / OPD / SFT 全程未在已经很强的基座之上产生净增益**;OPD 在 GSM8K 上对 base 仅 +0.008~+0.015(< 1.1σ)。基座已达此数据量(≤1024 条)/ 模型规模(1.7B)的能力天花板——这是"为何看不到 improvement"的根本原因,比 Gate C/D 失败更上游。

## 1. 实验配置

| 项 | 取值 |
|---|---|
| 起点 theta0 | cold-start OPD-like on **256** prompts(不饱和,留 gain 空间) |
| OPD arms | **opd_lmbda1**(λ=1.0, seed42)、**opd_lmbda05**(λ=0.5, seed42),各 800 步 |
| SFT 对照 | **sft_n128 / n256 / n512 / n1024**(同 theta0 起点,增数据非增 epoch) |
| 主指标 | GSM8K full,**N=1319**,flexible-extract(strict 因格式恒为 0,见 §4) |
| OOD-lite | MMLU(全 STEM/Hum/Soc/Other)、TruthfulQA-MC1/MC2、WinoGrande |
| 几何 | GetSlice,7 模块 ×(layer_6 / 14 / 22)× 双 X(prompt/bos)× S×model 交叉 |
| 加速 | vLLM colocate(student rollout 并行),48G GPU,无 OOM |
| 模型 | teacher Qwen3-4B → student Qwen3-1.7B,LoRA r16 |

## 2. 训练/评测轨迹

GSM8K(flexible, N=1319)与 OOD-lite penalty(越小越稳):

| 模型 | DataSize | GSM8K | GSM8K_gain | OOD_lite_penalty_p2 | Worst_OOD_drop |
|---|---|---|---|---|---|
| **theta0** | 256 | 0.4132 | — | 0.0000 | 0.0000 |
| opd_lmbda05 | 800 | **0.4306** | **+0.0174** | 0.0062 | 0.0037 |
| opd_lmbda1 | 800 | 0.4238 | +0.0106 | 0.0013 | 0.0013 |
| sft_n128 | 128 | 0.4132 | 0.0000 | 0.0011 | 0.0009 |
| sft_n256 | 256 | 0.3912 | −0.0220 | 0.0049 | 0.0045 |
| sft_n512 | 512 | 0.3677 | −0.0455 | 0.0084 | 0.0071 |
| sft_n1024 | 1024 | 0.3306 | −0.0826 | 0.0202 | 0.0107 |

**读法**:OPD 维持/微升 GSM8K;SFT 随数据量单调降(经典退化斜率)。但 OOD-lite penalty 全部 < 0.011(MMLU/TQA/WinoGrande 噪声远大于此),属噪声邻近量级。GSM8K stderr≈0.0136,故 OPD 的 +0.011~0.017 不足 ~1.3σ。

## 3. Gate 逐项评估

**Gate A — 可行性:✅ PASS。** 主脚本 `[DONE] cycle04 completed`;`tables/` 含 eval_trajectory / geometry_long(45MB)/ geometry_metrics / matched_gsm8k_pairs / ood_penalty;`figures/` 三图;`summary.json status=completed`。

**Gate B — matched control:✅ PASS。**

| OPD arm | gain | matched SFT | sft gain | gap | 判定 |
|---|---|---|---|---|---|
| opd_lmbda05 | +0.0174 | sft_n128 | 0.0 | 0.0174 | valid_match |
| opd_lmbda1 | +0.0106 | sft_n128 | 0.0 | 0.0106 | valid_match |

两 arm 均 `gap ≤ 0.02`。注意:由于 n256+ 的 SFT 全为负 gain,唯一能匹配 OPD 正 gain 的就是最小的 **sft_n128**(几乎未离开 theta0)。这是 Gate C 失败的伏笔。

**Gate C — 稳定性:❌ FAIL(两条件皆不满足)。**

- 条件 1(OOD-lite penalty 要 OPD ≤ matched SFT):
  - opd_lmbda05 `0.0062` **>** sft_n128 `0.0011` ✗
  - opd_lmbda1 `0.0013` **>** sft_n128 `0.0011` ✗
- 条件 2(谱漂移 OPD < SFT 需占 ≥80% 的 module×layer cells,S-对角 drift_from_theta0,共 21 cells):
  - opd_lmbda1 vs sft_n128:**3/21 = 14%** ✗
  - opd_lmbda05 vs sft_n128:**13/21 = 62%** ✗(< 80%)

**Gate D — 增益:❌ FAIL。** 需 Gate C 通过(未通过);且 OPD gain(+0.0106 / +0.0174)< 0.02 阈值,且不足 1.3σ。

## 4. 重大发现(详见 FINDING_04)

1. **评测抽取假象**:`MATH500 = 0.0`(所有模型)是 lm_eval 严格抽取在我们输出格式上系统性失败造成的 scoring artifact,**非能力**。铁证:GSM8K `strict=0.000` vs `flexible=0.413`(同模型同生成)。MATH500 行在分析中按 **uninformative** 处理,不计入 OOD 判断。
2. **NuminaMath-1.5 留出集可作 ID 主指标**:`test.jsonl`(1024)与训练 prompt 严格不相交(`train_row_ids ∩ test_row_ids = 0`,直接重叠 0)。可让 GSM8K 重定位为泛化/OOD 指标。
3. **scorer 硬化有实质影响**:换 math_verify + 修 MCQ/区间后,teacher open-answer 由 0.376 → 0.556(同生成)。

## 5. ID(NuminaMath)× OOD(GSM8K)双轴

> 以下表格由 v2 clean re-run(`max_tokens=3072`, 采样 temp0.7/top_p0.8/top_k20)自动注入;硬化 scorer(math_verify)。GSM8K 取自 cycle04 full eval。

<!-- BEGIN_ID_OOD -->
**ID × OOD 双轴**（v2 clean: max_tokens=3072, 采样 temp0.7/top_p0.8/top_k20;硬化 scorer / math_verify）

| 模型 | ID open(NuminaMath) | ID overall | 截断率 | OOD GSM8K |
|---|---|---|---|---|
| teacher 4B | 0.6188 | 0.5865 | 0.089 | — |
| **base(未训练)** | **0.5516** | 0.5308 | 0.070 | **0.4155** |
| theta0 | 0.5527 | 0.5308 | 0.069 | 0.4132 |
| opd_lmbda1 | 0.5348 | 0.5200 | 0.079 | 0.4238 |
| opd_lmbda05 | 0.5392 | 0.5210 | 0.065 | 0.4306 |
| sft_n128 | 0.5370 | 0.5230 | 0.068 | 0.4132 |
| sft_n256 | 0.5516 | 0.5279 | 0.058 | 0.3912 |
| sft_n512 | 0.5179 | 0.4966 | 0.057 | 0.3677 |
| sft_n1024 | 0.3890 | 0.3832 | 0.145 | 0.3306 |

**自动判读(由实际数值计算)**:
- ID open 最高:**teacher 4B** (0.6188)。
- opd_lmbda1:ID vs theta0 = -0.0179;ID vs matched sft_n128 = -0.0022;GSM8K vs theta0 = +0.0106
- opd_lmbda05:ID vs theta0 = -0.0135;ID vs matched sft_n128 = +0.0022;GSM8K vs theta0 = +0.0174
- 任一 OPD 在 ID 上超过 theta0 ≥ 0.02:**否** ⇒ OPD 无 ID 增益,优势仍仅在 GSM8K。
- theta0 是否为(训练后模型中)ID 峰值:**是**。
- **未训练基座(Qwen3-1.7B):ID open 0.5516、GSM8K 0.4155**。与 theta0 两轴持平(ID −0.0011 / GSM8K +0.0023,均 < 0.2σ),且 **base 的 ID 高于两个 OPD arm**(+0.012~+0.017)。OPD 对 base 的 GSM8K 增益仅 +0.008(λ1)/ +0.015(λ05),< 1.1σ。
- **⇒ cold-start / OPD / SFT 全程未在基座之上产生净增益**;base 已达此数据量/规模的能力天花板。
<!-- END_ID_OOD -->

**最终读数(v2 clean,截断已降至 6–9%、boxed 率 0.92–0.96,绝对值可信;ID open N=892,stderr≈0.017)**:
- **ID 上 theta0(0.553)与 sft_n256(0.552)并列最高**;两个 OPD arm(0.535 / 0.539)与 sft_n128(0.537)同处中段,均略低于 theta0,但**差距 < 1.1σ(不显著)**。sft_n512(0.518)略降,sft_n1024(0.389)在 ID/OOD **双崩**。
- **OPD 在 ID 上无增益**:opd 对 theta0 为 −0.013~−0.018(< 1.1σ),对其 matched 对照 sft_n128 仅 ±0.002(持平)。若有方向,小 SFT 反而略胜。
- **OPD 与 SFT 唯一分离处仍是 GSM8K**(OPD 升、SFT 随数据降);在 ID、OOD-lite、谱漂移三处 OPD 均不优于、甚至略逊。
- 参照:1.7B 学生 ID open ~0.53–0.55,达 4B teacher(0.619)的 ~89%。
- 结论:**OPD 的"优势"很窄——仅限 GSM8K 单一 OOD 任务、且噪声邻近量级**,与 Gate C 失败完全自洽。**v2 清洗版(截断 25%→7%)确认 v1 的全部定性结论。**

> **基座 OOD-lite(0-shot,补充)**:base MMLU 0.557 / TQA-MC1 0.293 / TQA-MC2 0.459 / WinoGrande 0.608,均较 theta0(0.564 / 0.294 / 0.470 / 0.619)**低 0.006~0.011(噪声级)**。即 cold-start 在通用 OOD-lite 上给了**微小正向**,但在数学两轴(ID NuminaMath + GSM8K)上对 base **零增益**——进一步坐实"基座在该数学任务上已饱和"。

## 6. 几何(GetSlice)

layer_14、S-对角、7 模块均值:

| 模型 | effective_rank(L14) | mean\|drift_from_theta0\|(L14) |
|---|---|---|
| theta0 | 909.4 | 0.0000(基线) |
| opd_lmbda1 | 889.0 | 0.0433（最高） |
| opd_lmbda05 | 893.8 | 0.0287 |
| sft_n128 | 928.1 | 0.0295 |
| sft_n256 | 928.5 | 0.0301 |
| sft_n512 | 930.4 | 0.0336 |
| sft_n1024 | 932.8 | 0.0391 |

**几何观察**:OPD **压低** effective_rank(889/894 < theta0 909),SFT **抬高**(928–933)——OPD 让激活谱更集中、SFT 更分散。但**漂移幅度上 OPD 并不更小**:opd_lmbda1(0.0433)比所有 SFT 都漂得多,opd_lmbda05(0.0287)≈ sft_n128(0.0295)。这是 Gate C 谱漂移条件失败的直接来源。(X_S_spectrum_level_gap、双 X、layer_6/22 详见 `geometry_metrics.csv`;UV/principal-angle 未保存,`principal_angle_status=unavailable_no_uv`。)

## 7. 方法学结论(本轮真正的收获)

Gate C 没复现,根因是 **matched-gain 对照锚点被 theta0 饱和度主导**:

- theta0 换成不饱和的 256 后,**所有更大的 SFT(n256/512/1024)GSM8K 都在掉**,唯一能匹配 OPD 正 gain 的只剩 **sft_n128**。
- 而 sft_n128 仅训 128 条、gain=0、几乎未离开 theta0 ⇒ 其 OOD 掉点(0.0011)与谱漂移(0.0295)都近乎平凡地"稳定"。
- OPD 训了 800 步、动得更多,于是"和一个几乎没动的 SFT 比稳定性"时**反而输**。

**对比 Cycle 03**:那里用饱和 theta0=512,matched 到的是已退化的大 SFT,OPD 显得稳;Cycle 04 用不饱和 theta0,matched 被逼到最小 SFT,OPD 显得不稳。**两轮的稳定性结论都被对照锚点主导**——这是比"OPD 稳不稳"更重要的发现。

**给下一轮的设计修正**:
1. 对照应按**等训练量**(step/样本数)配对,而非只按 GSM8K 等增益——否则锚点恒为最小 SFT。
2. 或改用 **ID(NuminaMath)等性能**配对,再比 OOD,这样 OPD/SFT 在"学到同样多"的前提下比稳定性。
3. theta0 的饱和度应作为显式自变量(扫 128/256/512),报告稳定性结论随它如何变化。

## 8. 局限与下一轮种子

- OOD-lite penalty 全在噪声内;OPD 的 GSM8K gain 不足 1.3σ ⇒ 不能声称 improvement。
- 谱漂移无 UV ⇒ 无 principal-angle 证据(遵守不允许结论)。
- ID eval 截断已解决:v1(1280, greedy)截断率 ~25% 压低绝对分;**v2(3072 + 采样)截断降至 6–9%(sft_n1024 因退化偏高 14%)、boxed 率 0.92–0.96,绝对值可信**;两版定性结论一致。
- **基座天花板(关键)**:未训练 base 在 ID/GSM8K 上 ≈ theta0、≥ 所有 OPD/SFT。说明在"1.7B 强 instruct 基座 + ≤1024 条 NuminaMath"这个组合里**没有可供 OPD 体现优势的提升空间**——base 已饱和。任何"OPD vs SFT 稳定性/增益"的对比都建立在近乎零的有效信号上。
- **种子**:① 选一个 base **未饱和**的设置(更难的下游任务、或更弱/更小的 base、或基座本身在该任务上明显低分)——否则一切训练都是噪声;② 等训练量/等 ID-性能 的 matched 设计;③ theta0 饱和度扫描;④ NuminaMath-test 作 ID 主指标 + 硬化 scorer 固化进主流程。

## 9. Artifacts

- 运行根:`/root/autodl-tmp/cycle04_opd_stability_gain/`(tables/ figures/ eval/ getslice/ summary.json)
- ID eval:`/root/autodl-tmp/floor_probe/`(scorer_v2.py + math_verify;`full_results/` v1、`full_results_v2/` clean;`*_samples.jsonl` 含全文生成)
- 发现:[FINDING_04_eval_extraction_artifact.md](FINDING_04_eval_extraction_artifact.md)
- copyback:`research_cycles/cycle_04_opd_stability_gain/04_results_import/`、`local_experiment_results/cycle_04_opd_stability_gain/`
