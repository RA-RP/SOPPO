# Cycle 09 意图

```yaml
artifact_type: next_cycle_intent_ch
cycle: cycle_09_aaai_competitiveness_completion
status: ready_for_theory_and_experiment_design
created: 2026-07-07
source_qa: next_cycle/QA.md (cycle_09)
source_file: current_cycle09_intent.md
prior_draft: 2026-07-03（已从 draft_seed_pending_cycle08_results 更新）
write_owner_next: Related Work（先做抢发审查），然后 Theory，然后 Experiment Design
```

本文件是 Next Cycle Seed → Theory / Experiment Design 的 Cycle 09 交接文件。

---

## Cycle 08 结果汇总

Cycle 08 训练了一个 OPD arm（Qwen3-4B-Base ← Qwen3-8B teacher，verl on-policy 蒸馏，
LoRA r=32，lambda=1.0），与 Cycle 07 SFT 参考轨迹进行比较。

**Gate 结论：**

| Gate | 结论 |
|---|---|
| B08——数学能力 | **STRONG PASS**：OPD final 0.848 vs SFT 0.752（+0.096）；每个 step OPD 均优于 SFT（+0.10 至 +0.24），unified cap 16384。保守估计——OPD 截断更严重。 |
| C08——MMLU-Pro 保留 | **PASS**：OPD net +0.016 vs SFT net −0.029；delta margin +0.045。轨迹非单调：step_40/80 大幅下跌（−0.12），step_160 恢复，final 净正。 |
| D08——Geometry 区分 | **正面**：SFT 出现 L18 step_20 rank bump；OPD ER 单调收缩无 bump。ER 是唯一清晰区分两个 arm 并追踪 OOD 结果的指标。 |
| A08——权重空间方向 | **不确定**：OverlapLift retracted（≈ random-LoRA null）。ρ 读 OPD on-principal / SFT 早期 off-principal；外部全参数文献（2606.13657/07082）说 OPD off-principal。矛盾未解决。 |

**新发现（未预注册）：**
- **OPD 早期下陷 step_5**（MATH500 −0.100，numina −0.070）：比 SFT step_20 下陷（−0.064）
  更早更深。OPD 到 step_20 恢复；SFT 到 step_160。两者均与 ER uptick 共定位。核心机制故事候选。
- **LoRA ρ vs 全参数矛盾**：LoRA ρ 读 OPD on-principal；外部全参数文献说 OPD off-principal。
  LoRA 结构可能改变了几何制度。
- **Eval-cap 修正**：OPD 长蒸馏 CoT 在 4096 下截断。MATH500 比较使用 unified cap 16384；
  晚期"下降"是截断伪影。

---

## Seed 决策

Cycle 09 是 **AAAI 竞争力补全 cycle**，但需要在跑更多实验前先解决一个战略问题：

> **首要优先级：鉴于外部文献抢发，先明确并收窄创新贡献。**
> 然后：跑 fp32 retrain + P0（统计 + geometry + 输出控制）+ P2（理论 + 论文改写）。
> P1（控制实验 + robustness）推迟至 Cycle 10。

外部全参数文献（arXiv 2606.13657/07082）已发表 OPD 权重空间 off-principal 结论。我们的
activation-space ER 证据未被该工作覆盖，但我们的 LoRA ρ 读数（OPD on-principal under LoRA）
与之矛盾。创新护城河在 activation-space 观测，而非权重空间方向。

---

## 贡献框架（三个候选——Cycle 09 Experiment Design 启动前必须选择）

**(a) LoRA 补充：**
LoRA adapter 改变了主方向几何规律，不同于全参数。我们的 LoRA ρ（OPD on-principal under
LoRA）是外部全参数工作未覆盖的 LoRA 制度发现。若 fp32 retrain 也显示 on-principal，
强化该论证。若 fp32 显示 off-principal（符合外部文献），则 LoRA adapter 本身改变方向
——这也是发现。

**(b) 新观测空间：**
Activation-ER 是权重空间分析未覆盖的、具有区分力的可观测量。我们提出 ER 作为连接训练
动态与 OOD 结果的机制可观测量。此框架**不依赖**权重空间方向结果，对外部文献抢发鲁棒。

**(c) 几何-过程-结果链（最大野心，可能需要全参数 + 多任务）：**
ER 状态（activation 压缩程度）→ dip-recovery 过程（时序和深度）→ OOD 结果（MMLU-Pro
保留 vs 退化）。将 activation geometry、训练轨迹和泛化整合成因果叙事框架。

**Cycle 09 Theory 对话必须在 Experiment Design 启动前做出选择。**
答案可能决定论文是机制论文（b, c）还是测量论文（a），以及所需计算量。

---

## 必做工作流

### Tier 0：贡献定位（无实验——Theory 对话）

基于以下内容确定创新贡献和论文范围：
- 仔细阅读外部文献（arXiv 2606.13657/07082）：他们精确主张什么？未覆盖什么（LoRA、
  activation 空间、训练轨迹）？
- 在框架 (a)、(b)、(c) 或组合中做出选择。
- 写作前更新 `result/claims_allowed.md` 中选定的主张边界。

**这是所有新代码运行前的阻塞任务。**

---

### Gate A09：权重空间方向（全参数/fp32 Retrain）

运行全参数或 fp32-master retrain 以解决 LoRA ρ 与外部文献的矛盾。

必须：
- 同 student（Qwen3-4B-Base），同数据（Math-CoT-20k），全参数而非 LoRA。
- 用与 Cycle 08 相同的协议测量 ρ（尺度不变方向对齐）。
- 比较 OPD 全参数 ρ vs SFT 全参数 ρ vs Cycle 08 的 LoRA ρ。

决策树：
- 若全参数 OPD 也是 on-principal：LoRA 读数是真实的，不是 LoRA 混淆。→ 强化框架 (a)。
- 若全参数 OPD 是 off-principal（符合外部文献）：LoRA adapter 改变了方向 → LoRA-vs-全参数
  几何差异本身是发现。→ 以不同方式启用框架 (a)。

**A09 是论文中任何权重空间方向主张的前提条件。**

---

### Gate B09：统计可靠性（P0）

对所有核心 OPD-vs-SFT 比较运行 paired bootstrap 95% CI。所有 main-text claim 必须有 CI
或明确标为 qualitative/diagnostic。

必须的 CI：
- OPD final − SFT final（unified cap 16384）
- OPD peak − SFT peak
- OPD dip depth − SFT dip depth（step_5 vs step_20）
- OPD 轨迹 AUC − SFT 轨迹 AUC
- OPD 非终止峰值 − SFT 非终止峰值

**没有 CI，OPD 早期下陷机制故事（Q3=a）不能作为主要主张。**

---

### OPD 早期下陷机制（核心贡献）

用证据形式化"更尖锐更短暂瞬态"故事：

1. 确认 OPD step_5 处的 ER uptick（activation geometry 时序）。
2. 比较 OPD vs SFT ER 轨迹形状：OPD ER 是否更早峰值并更快压缩？
3. 对下陷 checkpoint 的 finisher acc 和恢复速度（恢复步数）添加 CI。
4. 用非因果措辞写机制假说：
   > "On-policy 蒸馏触发了更早更尖锐的压缩瞬态（step_5，−0.100），在一个 checkpoint 间隔
   > 内解消。SFT 触发了更浅但持续更长的瞬态（step_20，−0.064，step_160 解消）。两个瞬态
   > 均与 activation 空间的 effective rank uptick 共定位。"

---

### OPD Geometry 机制证据包（P0，诊断性）

将 Cycle 08 geometry 转化为与 Cycle 07 SFT 可直接比较的论文级证据：

- Effective rank 轨迹：OPD vs SFT，同 checkpoint grid，同 layer/module。
- xs_log_spectrum_gap：OPD vs SFT。
- Drift from base checkpoint：OPD vs SFT（负对照）。
- OOD 领域输入上的 ER（MMLU-Pro prompts，如可行）：强化压缩→OOD 故事。

**Gate B09：OPD 和 SFT geometry 表格必须可直接比较，才能写入论文章节。**

---

### Gate C09：输出控制分析（P0）

报告并分类 OPD 准确率优势：

- as-run accuracy（math500 unified cap 16384；早期 step 保持 4096）
- finisher accuracy（correct / finisher count）
- response length 轨迹
- 非终止率（与 Cycle 07 SFT 表格平行）
- truncation / boxed-answer-before-truncation audit

分类：OPD 相对 SFT 的提升来自能力、输出控制，还是两者兼有？

注意：OPD 产生非常长的回答（step_624 3974 token）。step_40/80 的 MMLU-Pro 大幅下跌
应检查是否为 response length 或分布偏移伪影。

---

### 理论更新与论文改写（P2）

围绕选定框架更新理论和 LaTeX 草稿，例如：

> "Token-level imitation 下的 SFT 放大了较浅但持续较长的表示扩展瞬态。OPD 的 on-policy
> 对齐触发了更尖锐、更短暂的瞬态，迅速恢复，维持了表示压缩和 OOD 相关方向。"

这是机制框架，不是定理。严格因果主张不被允许。

论文主叙事中不能保留"Cycle N"语言。

---

## Cycle 09 不应做的事

- 在贡献框架决策前（Theory 对话先）不运行 Cycle 09 实验。
- 在 A09 fp32 结果出来前不主张 OPD on/off-principal 权重空间结论。
- 不主张压缩→OOD 链是因果的——它是相关性（n=2 arm）。
- 不启动 off-policy KD 控制或 robustness 轴——推迟至 Cycle 10。
- 不使用 Qwen3-1.7B 作为 robustness 模型（prior 证据表明 headroom 不足）。
- 不引用 Cycle 08 OverlapLift 作为信号——已 retract。

---

## 推迟至 Cycle 10

| 项目 | Gate | 推迟原因 |
|---|---|---|
| OPD 特异性控制（off-policy KD） | D09 | 核心贡献尚未确立。 |
| Robustness 轴（第二数据分布） | E09 | 同上。 |
| Teacher / theta0 / headroom baselines | F09 | 同上；核心主张固化后优先级提升。 |

---

## Cycle 09 Gate 汇总

| Gate | 状态 | 优先级 |
|---|---|---|
| 贡献定位 | 开放——实验前必须决定 | 阻塞（Tier 0） |
| A09：fp32 权重空间方向 | 开放——需跑 | P0 |
| B09：统计 CI | 开放——尚未计算 | P0 |
| OPD dip 机制（CI + ER 时序） | 开放 | P0 |
| C09：输出控制分解 | 开放——部分数据存在 | P0 |
| B09 geometry 证据包（OPD vs SFT 表格） | 开放——数据存在，需分析 | P0 |
| P2：理论 + 论文改写 | 开放 | P2（P0 之后） |
| D09：off-policy KD 控制 | 推迟 | Cycle 10 |
| E09：robustness 轴 | 推迟 | Cycle 10 |

---

## 下一步阅读顺序

### Related Work（第一步）

Related Work 应按顺序阅读：
1. `next_cycle/current_cycle09_intent.md`（英文版，主要参考）
2. `result/current_picture.md`
3. `result/claims_allowed.md`
4. `local_experiment_results/cycle_08_h_opd_vs_sft_comparison/run_01/RESULTS_08.md`

Related Work 的关键任务：
- 详细审查 arXiv 2606.13657 和 2606.07082：他们精确主张什么？
  未覆盖什么（LoRA 制度、activation-space ER、训练轨迹）？
- 识别三个贡献框架（a/b/c）各自的剩余新颖性空间。
- 更新 `related_work/current_related_work.md`，写入抢发审查报告和 gap map。

### Theory（第二步，在 Related Work 之后）

Theory 阅读 Related Work 输出，然后发展机制框架：
- 在贡献框架 (a) LoRA 补充 / (b) 新观测空间 / (c) 几何-过程-结果链 中做出选择
  或组合使用。
- 更新 `theory/current_theory_update.md`。

### Experiment Design（第三步，在 Theory 之后）

写 `exp/current_experiment_design.md` 和 `exp/current_acceptance_criteria.md`，
涵盖 Cycle 09 实验（A09 fp32 retrain、CI、geometry 证据包、输出控制）。
