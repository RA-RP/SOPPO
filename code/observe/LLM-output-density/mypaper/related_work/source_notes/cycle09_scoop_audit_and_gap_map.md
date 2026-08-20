# Cycle 09 转交文件：Scoop Audit + 护城河 Gap Map（Related Work → Theory）

```yaml
view_type: handoff_note
cycle: cycle_09_aaai_competitiveness_completion
from: Related Work
to: Theory (Tier-0 contribution positioning)
created: 2026-07-07
status: handoff_ready
purpose: 为 Theory 选定 novel contribution framing 提供 scoop 审计 + 新颖性缺口地图 + 数据筛选创新点评估 + AAAI 竞争力判断
sources_audited:
  - "arXiv 2606.07082 — On the Geometry of On-Policy Distillation (Shen et al.)"
  - "arXiv 2606.13657 — Dense Supervision, Sparse Updates (Guo Yu et al.)"
  - "arXiv 2605.30524 — Representation Collapse in Sequential Post-Training (Liu et al.)"
  - "SVD-LLM (Wang et al. 2024, arXiv 2403.07378) — 本项目白化 ER 的方法学来源"
  - "本项目早期设计：source_notes/ood_id_positioning_with_ppl.md（几何/PPL 数据筛选）"
```

本文件是 Cycle 09 Related Work 阶段的 scoop 审计结论，供 Theory 对话选 framing。**Theory 先读本文件，再读 result/claims_allowed.md。**

---

## 1. 三篇 scoop 论文精确审计

### 1a. 两篇 OPD weight-geometry 论文（对我们其实无威胁）

| | 2606.07082 (Shen et al.) | 2606.13657 (Guo Yu et al.) |
|---|---|---|
| 标题 | On the Geometry of On-Policy Distillation | Dense Supervision, Sparse Updates |
| 空间 | **参数/weight** | **参数/weight** |
| 核心 claim | OPD vs SFT vs RLVR 参数几何；OPD "relaxed off-principal" + "subspace locking"（更新早期锁进低维通道） | OPD 权重稀疏、谱集中、更新落在近零权重坐标 |
| 训练 | 未明确（分析为主） | full-param |
| 碰 activation space？ | ❌ | ❌ |
| 碰 dip-recover（能力/OOD 轴）？ | ❌（只做参数 subspace-locking 暂态） | ❌ |
| 链接 OOD 泛化？ | ❌ | ❌ |
| ⚠️关键 control | **rollout 转 off-policy 仍保持 rank 动态 → "on-policy 不是（weight-rank 的）驱动"** | — |

**结论**：它们坐实了"**OPD off-principal weight geometry 已被发表**"（framing a 的地基没了），但**不碰我们的三块 activation 护城河**。2606.07082 的 off-policy control 结论对我们有用（见 §5 数据筛选创新点）。

### 1b. 2605.30524 — Representation Collapse（**头号威胁，须精确 engage**）

**做什么**：LLM 经**顺序 post-training**（SFT→SFT→…，如 General→Math→Code→Safety，或 →Long-CoT→DPO）后，内部表示逐步塌缩成 low-rank/各向异性/同质化；用一套指标**预测**后续可塑性/OOD/校准退化，并给 mitigation。

**框架层级**：**预测/预后诊断（主）** → 机制（次，仅 Appendix A linearized theory）→ mitigation（末）。**不是机制解释论文**。

**headline 指标（4 个）**：effective rank、anisotropy、LoRA update overlap、future-task score。（PR/CKA/logit熵-ECE/domain separability 为二级。）

**ER 定义**：原始（**仅中心化，不白化**）残差流 hidden state $H_\ell$ → 协方差 $\Sigma=(n{-}1)^{-1}\bar H^\top\bar H$ → 特征值 $\lambda_i$ → $\widetilde{\mathrm{erank}}=d^{-1}\exp(-\sum p_i\log p_i)$，$p_i=\lambda_i/\sum\lambda_j$。在**固定 12k 多域探针**上、teacher-forced、逐层。

**训练**：**基本全程 LoRA**（默认 r=16/α=32/dropout=0.05，attn+MLP）；full-param 只在 1B 上做一次消融（查"塌缩是不是 LoRA 造成的"）。⚠️ **他们也是 LoRA → LoRA 不是我们的差异点**。

**怎么预测**：混合效应回归 `future_score ~ ER + anisotropy + CKA_drift + logit熵 + LoRA_overlap + (控制: 当前目标分/验证损失/总token) + 随机截距(模型族, 阶段序列)`。**in-sample 拟合、无 CV、无 R²/系数报告**（只有 Fig5 散点）→ 是**总体关联筛查**，不是强预测器。**明确不声称因果**（回归只是 screening）。

**核心结果**：ER 单调降(0.74→0.58)同时目标分升；塌缩集中中后层、目标依赖；ER@t 预测未来任务分@t+1；mitigation（replay/diversity）降塌缩且保 ~80% 目标增益。

**怎么做因果（干预）**：加/不加 anti-collapse 机制对比、**控住目标表现**、看未来任务是否变好。四机制：mixed-domain replay(5/10/20%、总token固定)、periodic feature refresh(ER 0.58→0.64、未来分 60.7→65.2)、表示多样性正则 $\mathcal{L}_{\cos}$、LoRA 更新去相关 $\mathcal{L}_{\text{decorr}}$。三向 Pareto 判定（降塌缩+保目标+提未来）。

**方向**：**压缩越多 → OOD 越差**（与我们 D08/C08 的"OPD 压缩却 OOD 保持更好"相反）。

---

## 2. ER 构念对照：我们 ≠ 他们（不止差白化）

| 维度 | 2605.30524 | 本项目 GetSlice（源自 SVD-LLM） | 含义 |
|---|---|---|---|
| **白化** | ❌ 只中心化，raw 表示秩 | ✅ Cholesky-jitter 协方差白化后 SVD | **最大差异**：SVD-LLM 白化使奇异值与该层**输出误差截断最优**→ 我们测"**输出相关谱**"，他们测"原始表示铺开度" |
| 对象粒度 | 残差流整层 hidden state | 逐**模块**(q/k/v/o/gate/up/down) | 我们更细、贴权重侧 |
| 谱幂次 | $p_i\propto\lambda_i=\sigma_i^2$ | $p_i\propto\sigma_i$（一次） | 熵分布不同 |
| 归一化 | $\times d^{-1}$，[0,1] | 无 $d^{-1}$ | 数值不可直接比 |
| 探针/参考 | 固定 12k 多域、teacher-forced | S(自训练数据)/X(冻结) + drift-from-θ₀ + X-S gap | 我们是条件化+相对参考 |

**结论**：因构念不同，**"方向冲突"未必是正面矛盾**（他们=原始表示秩，我们=SVD-LLM 输出相关白化谱）——**卸掉了"我们只是错了/被反驳"的最大风险**。代价：我们的 ER 更 bespoke，reviewer 会问"和 2605.30524 谁对"。
→ **建议 Experiment Design 同时计算两种 ER**（我们的白化 GetSlice ER + 他们的标准 raw 表示 ER）在同一批 OPD/SFT checkpoint 上，三种结局见 §4。SVD-LLM（2403.07378）应作为我们 ER 的方法学引用。

---

## 3. 精修护城河地图

| 护城河组件 | 状态 | 说明 |
|---|---|---|
| OPD weight 方向(off-principal) | ❌ **已抢发** | 2606.07082/13657。framing (a) 死 |
| "引入 activation ER 作观测量" | ❌ **非新** | 2605.30524 已 headline effective rank |
| "activation 压缩↔OOD" | ⚠️ **部分被占+方向冲突** | 2605.30524 已发"压缩→OOD 差"(SFT族,LoRA)。我们只剩"OPD特定+方向相反" |
| **OPD**（vs 只做 SFT/DPO/safety 的他们） | ✅ 开 | 三篇都不做 OPD |
| **OPD-vs-SFT 对比** | ✅ 开 | 无人做 |
| **单run内 dip-recover 轨迹** | ✅ 开 | Rethink SFT 只做 SFT dip；2605.30524 的"恢复"是跨阶段刻意 refresh，非训练自发 |
| **机制/过程 framing** | ✅ 开 | 2605.30524=静态预测；我们=训练过程 OPD-vs-SFT 动态机制。贡献类型不同 |
| **SVD-LLM 输出相关白化谱** | ✅ 开 | 他们用 raw 协方差 |

**真正无人占的护城河 = 交集**：
> "**OPD vs SFT，在 SVD-LLM 白化的输出相关激活谱上，OPD 的压缩伴随更早更锐的 dip-recover transient（step_5 vs step_20），且（与 2605.30524 的 SFT 方向相反地）保持 OOD**"——单看任一组件都被碰过，**交集无人做**。

> ⚠️ **对 ER 主线的技术威胁（2509.12235，数据筛选补搜时发现，但影响整条主线）**：该文（RL 修复 SFT 的 OOD 遗忘）用 SVD 分析发现 **OOD 跟 singular-vector 旋转、而非 singular-value 变化**。这直接质疑我们**基于 effective rank（奇异值分布）**的核心 observable——若 OOD 由方向旋转决定，rank/值分布可能不是对的量。**Theory 必须回应**：(i) 我们的 SVD-LLM 白化 + principal_angle（已在 geometry 指标表，虽 UV 未存）能否捕捉旋转？(ii) 是否需把 observable 从纯 ER 扩到"旋转 + 值分布"？(iii) 或论证 OPD-vs-SFT 差异在值分布上确实可见。**这是与 2605.30524 并列的第二个必答威胁。**

---

## 4. 三个 framing 的可辩护性（Related Work 判断，Theory 拍板）

- **(a) LoRA supplement**：❌ **建议放弃**。weight 方向被 2606.* 占；LoRA 表示塌缩被 2605.30524 占（他们也 LoRA）。
- **(b) new observation space (activation-ER)**：⚠️ 原样不新颖；**必须锐化**成"OPD-vs-SFT + SVD-LLM 输出相关谱 + 轨迹 + 方向对比"才成立。
- **(c) geometry→process→result 链**：✅ **最稳**——天然落在"无人占的交集"。代价：最贵、最需实验。

**双 ER 实验的三种结局**（决定 (b)/(c) 强度）：
1. 标准 ER 下 OPD 也压更多却 OOD 更好 → **对 2605.30524 的反例/精修（强新颖，直接对话最强威胁）**。
2. 只白化 ER 有信号 → 信号特定于 SVD-LLM 谱（较窄，措辞谨慎）。
3. 标准 ER 复现"压缩→OOD 差" → 白化 ER 捕捉了"好压缩 vs 坏压缩"之别 → 仍新颖，需讲清两种 ER 分工。

---

## 5. 新增创新点评估：ER-based 数据筛选让 SFT 逼近 OPD

**用户想法**：既然 ER 有预测/机制价值，用 **ER 准则筛选训练数据**使 SFT 轨迹避开 rank bump、逼近 OPD 的 OOD 保持——作为新创新点。呼应项目早期设计 `source_notes/ood_id_positioning_with_ppl.md`（几何/PPL 数据筛选缓解 OOD 退化，已有完整实验设计草案：静态筛选主 + SVD 机制分析 + 动态 iterative-SFT + PPL baseline + 3B/7B）。

**为什么强**：
- 把**观测/相关升级为干预/因果**（正是 2605.30524 用 mitigation 做因果的同款逻辑）→ 直接补上 claims_allowed 的"n=2 相关、不可因果"短板。
- **可操作 payoff**（"用数据筛选让便宜 SFT 逼近贵 OPD"）→ AAAI 强卖点（actionable，非仅诊断）。
- **与全部 scoop 论文差异化**：无人做"用 ER/几何筛数据复现 OPD 效果"。

**关键科学风险/看点**：静态数据筛选能否复现**on-policy**效果？
- 若能 → "OPD 收益（部分）可由数据几何复现，机制关于数据而非纯 on-policy"（惊人、高影响）。
- 若不能 → "OPD 收益不可约为数据选择，on-policy 本质"（也是机制结论）。
- **线索**：2606.07082 的 control（off-policy rollout 保持 weight-rank 动态）暗示 on-policy 未必是驱动 → **偏向"可复现"**，值得一试。**两种结局都可发表。**

**scoop 状态**：**已补搜（2026-07-07）。判决：未被直接占，但邻域拥挤 + 两个新威胁，novelty 窄且有条件。**

- **目标"按原则筛数据降遗忘"已拥挤**：GrADS（2511.08620，gradient-aware，5% 超全量+降遗忘）、FisherSFT（Fisher 信息）、gradient-orthogonality（2602.06359）、layer-wise gradient（2504.10766）——**criterion 都是梯度/信息，非 ER/表示几何**。→ 必打的 baseline。
- **⚠️ 新威胁 A — 2507.12856 "Curated-data SFT is RL (and can be improved)"**：已证 **SFT on 筛选数据 = 最大化 RL 目标下界**（iw-SFT 收紧）。→ "数据筛选让 SFT 逼近 RL/on-policy" 前提**已发表**，不能当新颖；只能把"用 **ER 几何 criterion** 复现 OPD 的几何/OOD 签名"当差异点。
- **⚠️ 新威胁 B（同 2605.30524 级）— 2509.12235 "RL Fine-Tuning Heals OOD Forgetting in SFT"**：RL/on-policy 修复 SFT 的 OOD 遗忘 + SVD 分析；**关键结论：OOD 跟 singular-vector 旋转、非 singular-value 变化**。→ **直接质疑基于 effective rank（奇异值）的 criterion**（若 OOD 由方向旋转决定，按 ER 筛可能选错维度）。**必须回应**：我们的 SVD-LLM 白化 + principal angle 或可捕捉旋转，否则需重构 criterion。这也部分威胁主线的 ER 叙事（不仅是数据筛选点）。
- **仍属于我们的窄缝**：① ER/SVD-LLM 白化表示几何作 selection criterion（vs 梯度/Fisher/reward）；② 专筛数据复现 OPD 的几何+OOD 签名（压缩/dip-recovery/OOD 保持）。**但需 ①打赢 gradient/Fisher/reward/PPL baseline ②区分 2507.12856 + 2509.12235 ③先解决"旋转 vs 秩"。**
- **对定位影响**：更坚定"**Cycle 10 核心 + 谨慎定位**"，非 Cycle 09 救命稻草。项目早期 ood_id 笔记的 PPL baseline 设计仍有用（回应"是否只是 likelihood/difficulty heuristic"）。

**节奏警告**：这是**新实验轴**。Cycle 09 intent 明令"P1 控制/robustness 推迟 Cycle 10""不启动新轴"。→ **数据筛选创新点建议定位为 Cycle 10 核心实验**（或若早期 ER-切分模块已有结果，Cycle 09 只做小规模 preliminary）。不要让它冲散 Cycle 09 的"完成 AAAI 论文"焦点。

---

## 6. AAAI 竞争力判断（Related Work 视角，最终归 Theory/PI）

- **当前护城河（纯观测交集）**：真实、可辩护，但**偏薄**——n=2 臂、相关非因果、且 2605.30524/2606.* 三篇近邻 scoop 抬高了门槛（reviewer 必问"beyond these 你加了什么"）。作为 AAAI 主轨**偏冒险**。
- **强化路径（择一或组合）**：① 双 ER + 方向对比直接 engage 2605.30524（结局1=反例，强）；② dip-recover transient 的 CI + ER 共位（B09）；③ **ER 数据筛选干预**（升级为因果+actionable，最能补短板，但成本/节奏在 Cycle 10）。
- **建议**：Cycle 09 以 **(c)/锐化(b) + 双 ER + dip-recover CI** 完成一篇**机制论文**；把**数据筛选干预**作为**最强的下一步/Cycle 10 核心**（若能挤进 Cycle 09 preliminary 更好）。单靠当前观测交集冲 AAAI 主轨，建议至少加双 ER 的直接对话 + dip-recover 统计化。

---

## 7. 给 Theory / Experiment Design 的具体交办

**Theory（Tier-0 framing 决策）**：
1. 在 (a)死/(b)锐化/(c)最稳 中选 framing（建议 c 或锐化 b）。
2. 表态如何 engage 2605.30524 的方向冲突（走"反例/精修"还是"不同构念"）。
3. 决定数据筛选创新点是 Cycle 09 preliminary 还是 Cycle 10 核心。
4. 更新 claims_allowed：activation-ER moat 是 SVD-LLM 白化输出相关谱（≠ 2605.30524 raw 表示秩）；dip-recover OPD-vs-SFT 为无人占交集；不可声称 compression→OOD 因果（除非做数据筛选干预）。

**Experiment Design**：① 双 ER（我们白化 GetSlice + 他们标准 raw 表示 ER）同批 checkpoint；② dip-recover transient 的 paired bootstrap CI（B09）；③（若采纳）ER 数据筛选 arm 复用 ood_id 笔记的静态筛选+PPL baseline+SVD 机制设计。

---

## 8. 需登记的新文献（已加入 related_work.md / source_matrix.md）

- 2606.07082、2606.13657（OPD weight geometry，scoop，weight-space）
- 2605.30524（representation collapse，头号威胁 #1，representation-space 预测诊断，LoRA）
- **2509.12235 "RL Heals OOD Forgetting in SFT"（头号威胁 #2，RL-vs-SFT + SVD；"OOD 跟 singular-vector 旋转非 singular-value" 直接质疑 ER criterion）**
- **2507.12856 "Curated-data SFT is RL (iw-SFT)"（"数据筛选让 SFT 逼近 RL"前提已发表，数据筛选创新点须区分）**
- SVD-LLM 2403.07378（本项目白化 ER 方法学来源）
- **数据筛选 baseline 簇**：GrADS 2511.08620（gradient）、FisherSFT（Fisher）、gradient-orthogonality 2602.06359、layer-wise gradient 2504.10766——数据筛选创新点的必打 baseline（criterion 均非 ER/几何）
- 相关机制近邻：Subspace Geometry Governs Forgetting in LoRA 2603.02224、Low-rank Example Associations 2406.14026（geometry↔forgetting，待精读）
- **补搜结论**：ER/几何 data-selection criterion 未被直接占，但邻域拥挤 + "旋转非秩"技术雷 → 数据筛选点定位 Cycle 10 核心、谨慎。
