# QA — Next Cycle Seed · Cycle 09（中文版）

```yaml
phase: next_cycle
cycle: cycle_09
status: answered_internalized
created: 2026-07-07
source_file: QA.md
maintenance_rule: overwrite_each_cycle; do_not_append
internalize_to: next_cycle/open_questions.md, next_cycle/history_questions.md, next_cycle/current_cycle09_intent.md
```

**用途：** Cycle 08 结果已完整。本 QA 关闭 Cycle 08，确定 Cycle 09 工作范围。
请直接在每道题的"你的回答："后面写你的决策（中文即可）。写完后告知我。

历史 QA 内容存于 `history_questions.md`。

---

## 背景（来自 Cycle 08 结果 — 2026-07-07）

Cycle 08 训练了一个 OPD arm（Qwen3-4B-Base ← Qwen3-8B teacher，verl on-policy 蒸馏，
LoRA r=32），与 Cycle 07 SFT 参考轨迹进行比较。

**门控结论（以 acceptance_criteria.md 标签为准）：**

| Gate | 结论 |
|---|---|
| B08（数学能力 + 避免下陷） | **STRONG PASS**：OPD final 0.848 vs SFT 0.752（+0.096）；在每个 step OPD 均优于 SFT（+0.10 至 +0.24），unified cap 16384。 |
| C08（MMLU-Pro 保留） | **PASS**：OPD net +0.016 vs SFT net −0.029（delta margin +0.045）。轨迹中间有大跌（step_40/80）后恢复。 |
| D08（geometry 瞬态） | **正面**：SFT 出现 L18 step_20 rank bump；OPD ER 单调收缩，无 bump。ER 是唯一清晰区分两个 arm 并追踪 OOD 结果的 geometry 指标。 |
| A08（权重空间方向） | **不确定**：OverlapLift retracted（≈ random-LoRA null，量级/bf16 主导）。ρ（尺度不变）读数：OPD on-principal / SFT 早期 off-principal；但外部全参数文献（arXiv 2606.13657/07082）说 OPD 是 OFF-principal。方向矛盾，怀疑 bf16+LoRA 混淆。 |

**OPD 早期下陷（新发现）：** OPD 在 step_5 下陷（MATH500 0.552 ← base 0.652，−0.100；
numina −0.070），比 SFT 的 step_20 下陷更早、更深。OPD 到 step_20 已恢复；SFT 到 step_160
才恢复。两次下陷均与 ER uptick 共定位。此发现在 Cycle 09 草案中未预注册。

**Cycle 09 草案意图已存在**（`current_cycle09_intent.md`，状态：draft_seed_pending_cycle08_results，
写于 2026-07-03）。规划：P0（统计 CI + geometry + 输出控制），P1（OPD 特异性控制 + robustness
轴 + teacher baselines），P2（理论 + 论文改写）。现在需要激活并更新。

---

## 问题

### Q1 — Cycle 09 范围：原草案是否在实际结果下仍然成立？

Cycle 09 草案在 Cycle 08 运行前写成。现在 B08 STRONG PASS + C08 PASS 已确认，D08 提供了
正面 geometry 区分信号，原计划范围是否仍正确？

预注册 P0 工作流：
- 对所有 OPD-vs-SFT delta 跑 paired bootstrap CI（P0 统计可靠性）
- OPD geometry 机制证据包——与 Cycle 07 SFT 完全同协议，生成论文级比较（P0）
- 输出控制分析：as-run acc / finisher acc / response length / non-term rate 分解（P0）

预注册 P1 工作流：
- OPD 特异性控制：off-policy KD 或 teacher-forced distillation（P1）
- Robustness 轴：第二数据 / prompt 分布（P1）
- Teacher / theta0 / headroom baselines（P1）

P2：理论更新 + 论文改写。

选项：
- **(a) 按草案推进。** B08/C08 确认说明论文范围正确；立即启动 P0，P1 并行推进。
- **(b) 收窄 P1。** B08/C08 已足够强，OPD 特异性控制和/或 robustness 轴可以推迟；
  Cycle 09 专注 P0（CI + geometry + 输出控制）和 P2。
- **(c) 扩展。** ρ 与外部文献的矛盾足够重要，需在 P0 中增加一个权重空间方向 gate
  （fp32 retrain 或替代方案）。
- (d) 其他。

**你的回答：**
c，我现在在Lora下的PE等结论和已有论文中全参的矛盾，而且已经有全参数的论文，我现在需要先讨论我工作的创新点和贡献，同时将目标缩窄，从而在cycle09中得到一个比较接近终稿的结论。

---

### Q2 — ρ 与外部文献矛盾：如何处理权重空间方向的不一致？

我们的 ρ 指标（尺度不变方向对齐，在 bf16 LoRA adapter 上计算）读数：
- OPD 全程 on-principal（z +5.7 至 +7.3）
- SFT 早期 steps off-principal（step_5 z −6.7），到 step_20 翻转为 on-principal

外部全参数文献（arXiv 2606.13657, 2606.07082, 2026年6月）说 OPD 是 **OFF-principal**
（relaxed regime）。我们的 LoRA 读数方向相反。

可能的解释：LoRA 结构混淆、bf16 偏置、制度差异（全参数 vs LoRA）。
current_picture.md 和 claims_allowed.md 将此标记为"不是有效结论——推迟到全参数/fp32"。

选项：
- **(a) 跑全参数或 fp32-master retrain** 以解决测量歧义。昂贵，但可与外部文献直接比较。
- **(b) 接受 ρ 作为 LoRA 制度的特有测量；作为真实但不可外推到全参数的发现呈现。**
  为全参数制度引用外部文献，诚实呈现 LoRA 差异。A08 给出"真实但有限"的结论。
- **(c) 完全放弃权重空间主 geometry。** 只依赖 activation-space ER 作为机制指标。
  承认权重空间方向不确定，引用外部文献。
- (d) 其他。

**你的回答：**
a，我现在的想法是他们文章没有讨论Lora引起的问题，全参量在几何上的规律不复存在。我的文章最基础能够作为在Lora上的补充，其次我的观测空间上还是有区分度的，看看能不能通过我的观测空间进行解释，这样的贡献更大，提出一个新的观测空间，然而这个可能同样需要在全参量、全任务上进行测试。最后我的观测空间是出自OOD的想法，如果能够将我的观测空间，几何状态，dip-recover进行结合，得到一个几何-过程-结果的关系，这样的话我的论文的理论价值就会比较高。

---

### Q3 — OPD 早期下陷（step_5）：主要机制故事，还是次要发现？

OPD 在 step_5 下陷（MATH500 0.552 ← base 0.652，−0.100；numina −0.070），比 SFT 的
step_20 下陷更早、更深。OPD 到 step_20 恢复；SFT 到 step_160 才恢复。两者下陷均与
ER uptick 共定位。

"压缩瞬态"假说：OPD 强制触发更尖锐的早期压缩断裂（step_5 ER 突增），因 on-policy
蒸馏重新锚定轨迹而迅速恢复。SFT 触发较浅的断裂（step_20 ER 突增），因与模型生成分布
不对齐而恢复缓慢（至 step_160）。

此发现在 Cycle 09 草案中未预注册。问题是它是否值得在机制叙事中占据核心位置。

选项：
- **(a) 作为核心机制故事。** OPD 更尖锐但更短暂的瞬态 vs SFT 更浅但更长的瞬态，是
  on-policy trajectory alignment 作为因果机制的关键证据。在 Cycle 09 中作为第 4 节
  主要 claim 加入。
- **(b) 作为次要发现。** 在轨迹表格和机制节中记录时序差异；不作为主要证据。
  ER-vs-OOD 链接是主要机制。
- **(c) 其他。**

**你的回答：**
a，我认为这是一个可以追求的核心贡献，作为我的几个核心贡献中的一个，但是具体还是需要和最终的结果进行结合。

---

### Q4 — P1 优先级：OPD 特异性控制 vs Robustness 轴？

Cycle 09 草案将两者均列为 P1，但未严格排序。两者都需要相当的计算量。
如果 Cycle 09 无法全部完成，优先哪个？

- **OPD 特异性控制（off-policy KD 或 teacher-forced distillation，Gate D09）：**
  直接关闭"teacher logits 即充分条件；on-policy 无额外贡献"的反驳。若 OPD 在
  dip/AUC/non-termination 上优于该控制，成为论文重要贡献。若不，claim 必须收窄。

- **Robustness 轴（第二数据/prompt 分布，Gate E09）：**
  直接关闭"单一分布结果"的反驳。同 Qwen3-4B student、同 pipeline，在第二数据分布
  上验证。比换模型更容易实现。

选项：
- **(a) 优先 off-policy KD 控制（D09）。** "OPD 只是蒸馏"比单一分布更难驳回。
- **(b) 优先 Robustness 轴（E09）。** 单一分布反驳容易在 rebuttal 处理，但
  B08/C08 在第二分布下成立增加直接价值。
- **(c) 在 Cycle 09 中并行运行两者**（如果计算量允许）。
- **(d) 推迟两个 P1 项**——Cycle 09 完全专注 P0 和 P2；Cycle 10 再做。

**你的回答：**
d，现在的主要矛盾，主要贡献还没有得到，这个补充我认为需要放在之后的cycle10或者后面。

---

*请在上方每道题的"你的回答："后面直接写你的决策。完成后告知我。*
