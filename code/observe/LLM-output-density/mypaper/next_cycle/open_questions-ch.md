# 开放问题

```yaml
view_type: open_questions_ch
status: active_cycle09_seed
source_file: open_questions.md
maintenance_rule: dynamic_incremental; resolved_items_moved_to_history_questions.md; new_items_appended
last_updated: 2026-07-07
```

进入 Cycle 09 的所有当前开放问题。Cycle 08 已解决的问题已移入
`history_questions.md`（Cycle 08 Close 节）。

**Cycle 08 后的项目状态：** B08 STRONG PASS（OPD final 0.848 vs SFT 0.752，每个 step
+0.10 至 +0.24，unified cap 16384）。C08 PASS（OPD MMLU-Pro net +0.016 vs SFT −0.029）。
D08 正面（ER 清晰区分两个 arm；SFT step_20 rank bump；OPD 单调收缩）。A08 不确定
（OverlapLift retracted；ρ 与外部全参数文献矛盾）。OPD 下陷比 SFT 更早更深
（step_5，−0.100 vs step_20，−0.064），但恢复更快（step_20 vs step_160）。

**核心开放问题：** 鉴于外部全参数文献（arXiv 2606.13657/07082）已发表 OPD
权重空间 off-principal，我们的创新贡献是什么？候选护城河是 activation-space ER 观测 +
几何-过程-结果链，而非权重空间主方向。

**Cycle 09 Gate 优先级：** A09（贡献定位 + fp32 测量）> B09（统计 CI + geometry）>
C09（输出控制）> D09/E09 推迟至 Cycle 10。

---

## Cycle 09 主线：贡献定位（紧急——在新实验之前）

1. **鉴于外部文献抢发，我们的精确创新主张是什么？**
   外部文献已覆盖全参数下 OPD 权重空间 off-principal。我们论文的三个候选框架：
   - **(a) LoRA 补充：** LoRA adapter 改变了主方向几何规律；LoRA 下权重空间主方向与全参数不同。
     我们的 LoRA ρ 读数（OPD on-principal under LoRA）是真实的 LoRA 制度结论。
   - **(b) 新观测空间：** activation-ER 是权重空间分析未覆盖的、具有区分力的可观测量。
     我们提出 ER 作为连接训练动态与 OOD 结果的机制可观测量。
   - **(c) 几何-过程-结果链：** ER 状态（压缩程度）→ dip-recovery 过程 → OOD 结果。
     将 activation geometry、训练轨迹和泛化整合成一个叙事。该链可能需要全参数 + 多任务验证。
   **状态：紧急。在任何新实验前分配给 Theory 对话 + 用户决策。**

2. **在计算量约束下，哪个框架给出最强的论文贡献？**
   **状态：取决于 fp32 retrain 结果（Gate A09）和用户对范围的判断。**

---

## Gate A09 — 权重空间方向（全参数/fp32 Retrain）

1. **全参数或 fp32-master retrain 是否确认 LoRA ρ（OPD on-principal）还是符合外部文献（OPD off-principal）？**
   当前：LoRA ρ 读 OPD on-principal 全程（z +5.7 至 +7.3）；SFT 早期 off-principal 后翻转。
   外部全参数文献说 OPD off-principal。方向矛盾。
   **状态：必须跑。结果决定框架 (a)（LoRA 补充）是否可辩护。**

2. **LoRA-vs-全参数几何差异能否成为独立贡献？**
   若 fp32 retrain 显示 OPD off-principal（符合外部文献），则 LoRA-vs-全参数差异本身是发现：
   "LoRA 训练改变了权重空间主方向几何"。
   **状态：取决于 A09 fp32 结果。**

---

## Gate B09 — 统计可靠性

1. **是否已对所有 OPD-vs-SFT delta 计算 paired bootstrap 95% CI？**
   必须：OPD final − SFT final、OPD peak − SFT peak、OPD dip depth − SFT dip depth、
   轨迹 AUC 差、非终止峰值差。
   **状态：尚未运行。所有 main-text claim 必须有 CI 或明确标为 qualitative/diagnostic。
   分配给 Cycle 09 Code 对话。**

2. **OPD 早期下陷机制主张（step_5，−0.100）在统计上是否显著？**
   需要：下陷 checkpoint 的 finisher acc CI；恢复速度差异的 CI。将"更尖锐更短暂瞬态"
   提升为论文主张需要统计证据。
   **状态：开放。Cycle 09 必须解决才能使 dip 机制成为主要发现（Q3=a 决策）。**

---

## OPD 早期下陷机制（核心贡献候选）

1. **"更尖锐更短暂瞬态"故事能否被形式化？**
   OPD：step_5 下陷（−0.100），step_20 恢复（4 step 下陷期）。SFT：step_20 下陷（−0.064），
   step_160 恢复（7 step 下陷期）。两者下陷均与 ER uptick 共定位。
   假说：on-policy 蒸馏触发快速压缩断裂，然后在 teacher 引导下快速重新锚定。SFT 触发
   较慢的断裂，无实时引导，恢复缓慢。预测：OPD ER uptick 更早且持续更短。
   **状态：需要 ER 时序分析 + dip 指标 CI。分配给 Cycle 09 geometry + stats 任务。**

2. **ER 时序是否确认 activation 层面的"更尖锐更短暂"故事？**
   需要：OPD step_5 vs step_0 的 ER；OPD ER 轨迹形状。
   **状态：开放。必须从 Cycle 08 geometry artifacts 中提取。**

---

## 压缩 → OOD 因果链（诊断性，Cycle 09）

1. **能否在 OOD 领域输入（MMLU-Pro prompts）上测量 ER，以强化压缩→OOD 因果链？**
   当前 ER 仅在 math-probe activations 上测量。要建立因果链，需要显示 SFT 的 rank bump
   在 MMLU 领域 activations 上也存在，而 OPD 的单调收缩保留了 OOD 相关方向。
   **状态：开放。高价值，需要在保存的 checkpoints 上对 MMLU-Pro prompts 运行额外 ER 测量。**

2. **OPD MMLU-Pro 轨迹的大幅震荡（step_40/80 大跌，step_160 恢复）是否反映 OOD 上的第二次压缩瞬态？**
   step_40/80 OPD MMLU-Pro 大幅下跌（约 −0.12）。这可能是与 ID 下陷平行的 OOD 轴压缩瞬态。
   **状态：开放。需结合 ER 测量解释。**

---

## Gate C09 — 输出控制分析（P0）

1. **OPD 相对 SFT 的准确率优势来自能力、输出控制，还是两者兼有？**
   OPD 回答很长（final 3974 token），4096 cap 下截断严重。MATH500 比较使用 unified cap
   16384；但输出控制行为与 SFT 本质上不同。
   **状态：unified cap 下各 checkpoint 的 finisher acc + 非终止率分解尚未完成。
   分配给 Cycle 09 输出控制分析。**

2. **OPD 与 SFT 在匹配训练 step 下的非终止行为直接对比结果如何？**
   SFT 在 step_80 达到约 79% 非终止峰值。OPD 模式不同。需要直接对比。
   **状态：开放。分配给 Cycle 09 Code 对话。**

---

## 论文改写与理论（P2，Cycle 09）

1. **论文是否能以几何-过程-结果框架作为主叙事？**
   - 几何：activation ER 作为状态指标（压缩程度）
   - 过程：dip-recovery 瞬态（时序和深度作为机制信号）
   - 结果：OOD 结果（MMLU-Pro 保留 vs 退化）
   **状态：开放。Theory 对话必须先发展此框架，然后才能重组 LaTeX 草稿。全程非因果措辞。**

2. **如何相对于外部文献框架权重空间 ρ 结论？**
   取决于 A09 fp32 结果。若 OPD off-principal（符合外部文献）：聚焦 activation-space 护城河。
   若 OPD on-principal（符合 LoRA ρ）：LoRA 改变了几何规律，论证 LoRA 补充贡献。
   **状态：取决于 A09。必须在论文框架确定前解决。**

3. **如何从论文主叙事中删除 cycle 语言（"Cycle 07/08"）？**
   **状态：P2 论文改写任务。分配给论文写作对话。**

---

## 推迟至 Cycle 10

| 项目 | 推迟原因 |
|---|---|
| **Gate D09：OPD 特异性控制（off-policy KD）** | 主要贡献尚未确立。核心主张必须先定型。 |
| **Gate E09：Robustness 轴（第二数据分布）** | 同 D09。 |
| **Gate F09：Teacher / theta0 / headroom baselines** | 同上。核心主张固化后优先级提升。 |
