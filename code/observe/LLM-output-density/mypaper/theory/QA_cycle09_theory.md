# QA: Cycle 09 — Theory Phase (Tier-0 Contribution Positioning)

```yaml
phase: theory
cycle: cycle_09_aaai_competitiveness_completion
status: discussion_complete
created: 2026-07-08
discussion_closed: 2026-07-08
blocking: true   # intent 规定：framing 未定，任何新实验不得启动
internalize_to:
  - theory/current_theory_update.md
  - theory/current_theory_update-ch.md
  - theory/geometry_metric_definitions.md
  - theory/geometry_metric_definitions-ch.md
read_scope_confirmed:
  - readme.md
  - next_cycle/current_cycle09_intent.md
  - result/cycle09_seed_handoff.md
  - result/current_picture.md
  - result/claims_allowed.md
  - result/cycle08_rho_metric_spec.md
  - related_work/related_work.md (Cycle 09 scoop 节)
  - related_work/source_notes/cycle09_scoop_audit_and_gap_map.md
  - local_experiment_results/cycle_08_h_opd_vs_sft_comparison/run_01/RESULTS_08.md
  - local_experiment_results/cycle_08_h_opd_vs_sft_comparison/run_01/geometry/opd_vs_sft_geometry.md
  - local_experiment_results/cycle_08_h_opd_vs_sft_comparison/run_01/geometry/rho_summary.md
  - theory/current_theory_update.md (至 Cycle 08 节)
  - theory/geometry_metric_definitions.md (至 Cycle 07 节)
write_scope: theory_only
cross_conversation_flags:
  - "⚠️ 待同步 result/：claims_allowed.md 的 framing 边界更新归 Result 对话（Tier-0 完成后）"
  - "⚠️ 待同步 result/：current_picture 'Both dips co-locate with the ER uptick' 表述与 opd_vs_sft_geometry.md mean 表矛盾，待 L18 层级重推导后由 Result 修正（见 Q4）"
```

本 QA 是 Cycle 09 Theory 阶段（Tier-0 贡献定位）的审查清单。按工作区规则只写 `theory/`。2026-07-08 对话中已完成相关工作内化分析并确认三项方向性决策（见下），本文件将其固化并提出剩余决策点。

---

## 角色与工程边界

当前对话角色：**Theory**。允许写入：`mypaper/theory/`。
Theory 定贡献框架、机制假设、观测量定义、解释树和 claim 边界；不写数字阈值、不定实现细节（归 Experiment Design）、不改 result/（标记待同步）。

---

## 本轮定位：Tier-0 —— 被抢发后的贡献重定位

### 已死的贡献点（scoop 审计结论，不再争论）

1. "OPD 权重方向 off-principal" —— 2606.07082/13657 全参发表。
2. "引入激活 ER 作观测量"（独立卖点） —— 2605.30524 已 headline。
3. "激活压缩↔OOD"（独立卖点） —— 被部分占据且方向相反。

### 活着的交集护城河（精确表述，2026-07-08 内化分析确认）

> 在 OPD-vs-SFT 匹配轨迹对照下（同 student/数据/checkpoint 网格/LoRA），以**输出相关白化激活谱**（SVD-LLM lens）为观测量：两种监督通道产生**相反的谱-暂态签名**——SFT 出现中层（L18）暂态秩扩张 + 权重方向早期 off-principal 重定向（ρ: step_5–10 off → step_20 翻 on），伴随 OOD 侵蚀；OPD 方向全程稳定、谱单调收缩，伴随 OOD 保持与全程能力优势（MATH500 每步 +0.10~0.24）。与塌缩文献"压缩→坏"方向相反：**决定 OOD 结局的不是压缩量，而是暂态破坏的存在与幅度**。

三个独立空间信号时序咬合（ρ 权重方向 step_5–10 → ER bump step_20 → 能力/OOD 行为），是所有 competitor 单篇都不具备的证据形态。

### 领域发展线路（论文 related-work 章骨架）

阶段1 权重空间静态（"更新落在哪"，2606.×2 完成）→ 阶段2 表示空间静态/预后（"训完表示长什么样"，2605.30524）→ 阶段3 训练动态（"过程中发生什么"，Rethink SFT / 2509.12235）→ **汇合点=本项目**：过程视角 × 输出相关谱观测 × 监督通道平行对照 × OOD 结局绑定。

---

## 📌 已确认决策（2026-07-08 对话，用户拍板）

| # | 决策 | 影响 |
|---|---|---|
| D1 | **off-policy KD control 进 Cycle 09**，不推迟 Cycle 10（"这一轮希望比较完善"） | 机制归因实验入场；(c) framing 的天花板打开；Q3 定解释树 |
| D2 | **principal angle 纳入 Cycle 09**（UV 保存开启，Cycle 07 遗留决定到期） | 正面回应威胁#2（2509.12235 旋转 vs 值）；Q5 定判定实验 |
| D3 | **L18 中层信号单独拉出作主观测目标**（mean-over-layers 降为汇总） | 几何观测层级化；Q4 定预注册规则 + 共位矛盾处理 |
| D4 | **LoRA 补充点保留为条件性贡献**（2606.×2 未做 LoRA，我们的 LoRA 读数与全参相反可能本身是发现） | Q2 定理论解释与决策树；一切以 A09 fp32 为条件 |

---

## Q1：论文贡献结构拍板（framing 主从关系）

**背景：** intent 给出 (a) LoRA supplement / (b) 观测空间 / (c) 几何→过程→结局链 三选。scoop 审计判 (a) 原样已死、(b) 需锐化、(c) 最稳但最贵。D1 使 (c) 的归因实验（off-policy control）进入本轮，D4 使 (a) 以条件形式复活为 (a')。据此贡献结构候选：

- a.（建议）**分层结构**：
  - **主贡献 = 锐化 (b)**：OPD-vs-SFT × 白化输出相关谱 × 轨迹 × 方向反转——现象+观测量论文内核，不依赖任何未完成实验即可成立；
  - **机制层 = (c) 作显式假设框架**，off-policy control 与双 ER 落地且干净 → (c) 升级为共同主贡献；不及时 → (c) 保持 hypothesis 语言；
  - **条件副贡献 = (a') LoRA-regime 差异**（系于 A09 fp32 结局，两种结局都可写，见 Q2）；
  - **方法贡献 = 测量学批判**（OverlapLift 撤回 + bf16/LoRA 使权重方向指标失效）。
- b. 直接以 (c) 机制链为主贡献（赌 control + 双 ER + fp32 全部按时且干净——风险高，AAAI 时窗 ~4 周）。
- c. 只写锐化 (b)，control/A09 结果全部留给 Cycle 10 论文（保守；浪费 D1/D2 决策带来的上升空间）。

**Theory 判断与推荐：a。** 分层结构让论文在"Tier B 训练实验任一延误"的情况下仍可提交（内核是分析型证据），同时保留 control 及时落地时升格 (c) 的通道。审稿人视角：主 claim 永远只依赖已完成证据，条件贡献显式标注条件——这是 rebuttal 期最能扛的结构。

**你的回答：**
你说的太好了，就按照你说的a。

---

## Q2：LoRA-vs-全参相反结果的理论定位（回答你的问题 1）

**背景：** 你问"和全参量相反的结果能不能说明什么"。先立事实边界：Cycle 08 的 LoRA ρ 读数（OPD on-principal）计算在 **bf16 部署件**上，bf16 量化偏 on-principal，所以**当前读数方向未定**——A09 fp32 retrain 是判真伪的前提。两种结局的理论含义：

**结局 A（fp32 后仍 on-principal，与全参文献相反）→ "更新几何是参数化依赖的，不是方法内禀属性"：**
同一 OPD 目标在 LoRA 低秩流形与全参空间走出**不同的几何路径**。这是对 2606.×2 结论普适性的边界修正——他们的 off-principal 是全参结论，**不能外推到 LoRA 部署**，而 LoRA 恰是实践主流。真 LoRA-regime 补充贡献成立。
注意两点必须写清：(i) 与 LoRA-illusion（intruder dimensions，off-base 新方向）的表面矛盾可以调和——我们 ρ²≈0.02（k=32）说明更新**大部分弥散**，on-principal 只是相对 spectrum-null 的统计偏置，少量 intruder 方向与轻微 on 偏置可共存；(ii) LoRA 差异点**只对 2606.×2 这对权重几何论文成立**——2605.30524 自己就是 LoRA（r16），塌缩比较上 LoRA 不是我们的差异。

**结局 B（fp32 翻转为 off-principal，与全参文献一致）→ 测量学贡献强化：**
"bf16 量化足以翻转方向性判定"——权重方向指标在 bf16 部署件上不可信，这是社区需要的警告；LoRA 补充点消亡，但诚实报告。

**两种结局都可发表；A09 之前禁止声称任何一侧。**

**选项：**
- a.（建议）按上述双结局决策树写入 theory：(a') 作为**条件性副贡献**注册，解释树预注册（防止 fp32 出来后临时编方向），claim 边界写明"A09 前无任何 on/off-principal 声明"。
- b. 更保守：LoRA 对比只写进 limitations/discussion，不作为贡献点。
- c. 更激进：把 LoRA-vs-全参差异直接写成主贡献之一（风险：A09 结局 B 时主贡献塌一根柱子）。

**Theory 判断与推荐：a。**

**你的回答：**
a，同时我想知道为什么一定要使用fp32，TPNT在文中就明确表示要使用bf16截断，这样对应真实的模型，现在实验中出现的达不到阈值，是lora本身的结构导致的吧。


---

## Q3：off-policy KD control 的理论设计与解释树（D1 落地）

**背景：** OPD 与 SFT 之间差着**三个混淆变量**：(1) 输入分布（on-policy rollout vs 静态语料）、(2) 监督信号（稠密 teacher logits vs one-hot 文本）、(3) teacher 身份（8B logits vs 32B 生成文本）。单加一条什么样的臂，才能最大化归因力？

**推荐设计（变体 i，"OPD 的 off-policy 版"）：** 同 8B teacher、同 forward_kl_topk loss、同 LoRA r=32、同 checkpoint 网格；唯一改变——**输入从 student rollout 换成 teacher 离线生成的静态回复**（经典 off-policy 序列级 KD）。则：
- off-KD vs OPD：**干净隔离 on-policy 采样**（其余全同）；
- off-KD vs SFT：隔离监督密度+teacher 身份（混淆对，Cycle 10 再拆）。

工程注：off-policy KD 无 rollout 环路，teacher top-k 可离线预计算——比 OPD 臂**更便宜**（cycle08 verl 环境直接复用）。

**预注册解释树：**

| off-KD 臂的签名 | 结论 |
|---|---|
| SFT-like（L18 bump + 早期 off-principal + OOD 侵蚀） | **on-policy 采样是 OPD 签名的驱动**（H-onpolicy 支持） |
| OPD-like（无 bump + OOD 保持） | **稠密 logit 监督已足够，on-policy 非驱动**——与 2606.07082 权重侧 control 结论汇合（他们：off-policy 保持 rank 动态），且直接强化 Cycle 10 数据筛选猜想 |
| 混合（如无 bump 但 OOD 侵蚀） | bump 与 OOD 保持**驱动因素解耦**——信息量最大的结局，写作需最谨慎 |

三种结局都有明确论文写法；这是"两头赢"设计。

**选项：**
- a.（建议）按变体 i + 上述解释树预注册写入 theory；具体语料/规模/步数由 Experiment Design 定。
- b. 变体 ii（固定 SFT 语料 + 8B logits——对 SFT 的对照更近，但对 OPD 的隔离不干净）。
- c. 双变体都跑（最完备，预算 ×2，AAAI 时窗风险）。

**Theory 判断与推荐：a。** 本轮核心问题是"OPD 签名的驱动是什么"，变体 i 对 OPD 的单变量隔离最干净。

**你的回答：**
a


---

## Q4：L18 层级化 + OPD-dip 共位矛盾的处理规则（D3 落地）

**背景（记录矛盾，2026-07-08 内化时发现）：** `current_picture.md`/intent 写 "Both dips co-locate with the ER uptick"，但 `opd_vs_sft_geometry.md` 的 **mean-over-layers** 表显示 OPD ER 从 step_0 单调下降（argmax=step_0，无 uptick）；OPD 侧与 dip 邻近的是 **xs_log_spectrum_gap argmin（step_10）**。"两臂 dip 都共位 ER 上凸"与"OPD 无 bump"（D08 结论）在 mean 层面不能同时为真。**若 OPD 的 dip 在 L18 层级也没有 ER 签名，紧版 compression-break 故事必须弱化**——但"两种监督通道的 dip 是几何上不同类的暂态（SFT: ER-bump 型；OPD: xs-gap 型）"可能是更有趣的结论。

**层级化预注册规则（建议）：**
1. **L18（或效应最大的中层）为 headline 层**，选层标准预注册为"OPD-vs-SFT 判别效应最大的层"且必须报告**全层剖面**作稳健性（防 cherry-pick 质疑）；外部锚：2605.30524 也发现塌缩集中于中后层——中层信号有跨论文一致性。
2. **在 L18 层级重推导两臂 dip 的共位关系**（ER、xs_gap、drift、θ_r 四量齐上），结果决定机制故事的措辞分支：
   - L18 上 OPD dip 有 ER uptick → "共位"表述保留（限定 L18）；
   - L18 上仍无 → 改写为"双通道 dip 几何异质"分支，`current_picture` 表述标 ⚠️ 待同步由 Result 修正。
3. mean-over-layers 降为附录汇总。

**选项：**
- a.（建议）按上述三条写入 theory + geometry 文件；矛盾处理分支预注册。
- b. 只做 L18 headline，不预注册矛盾分支（结果出来再说——违反本项目"先注册后解读"的一贯护栏，不建议）。
- c. 保持 mean-over-layers 为主（放弃 D3——与你已确认的方向矛盾，仅列出备完整性）。

**Theory 判断与推荐：a。**

**你的回答：**
a，我其实还有问题，现在的X和S是不是不太够，还需要更丰富的来源？请你复述一下现在的X和S同时指出你认为还能添加什么来源？

---

## Q5：principal angle 判定实验——正面回应"旋转 vs 值"威胁（D2 落地）

**背景：** 2509.12235 发现（权重侧 SVD）"OOD 行为跟奇异向量**旋转**、非奇异值变化"，直接质疑我们以 ER（值分布统计量）为核心观测。我们的辩护起点：白化算子由输入协方差决定，**权重方向旋转会在白化激活谱上表现为谱形变化**——我们的 ER 非纯"值"统计量，间接编码方向-协方差对齐。但这个论证**必须实证**，恰好 D2 已批准开 UV。

**判定实验（预注册）：** 在现有 checkpoint 网格上（三臂：OPD/SFT/off-KD）重跑 GetSlice 开 `save_uv` —— **纯探针成本，无需重训**。计算：
1. **θ_r 轨迹**：相邻 checkpoint 间 + 各 checkpoint vs base 的激活 top-r 子空间主夹角（r 按 Cycle 07 规则：energy-ratio@~90% 或 round(ER)，逐层逐模块）；γ_r（eigengap）继续作容量代理。
2. **预测**：SFT L18 在 step_10–20 出现 θ_r 旋转尖峰（与 bump 共位）；OPD 平滑。
3. **裁决量**：ER 与 θ_r 谁更好地 (i) 分离两臂 (ii) 跟踪 MMLU-Pro Δ。

**裁决树：**
- θ_r 是更好的跟踪量 → 采纳"旋转+值分布"双轴观测框架，引 2509.12235 为收敛证据（注意标明他们是权重侧、我们是激活侧——跨空间收敛反而是强化）；
- ER 仍是激活侧的判别量 → 获得对威胁#2 的实证回应："在白化激活空间，判别 OPD-vs-SFT 的是值分布暂态而非子空间旋转（旋转结论或为权重空间特有）"；
- 两者互补（如 θ_r 管 OOD、ER 管 dip）→ 观测量分工框架，同样可写。

**选项：**
- a.（建议）按上述判定实验+裁决树写入 theory 与 geometry 文件；实现细节（r 的每格自适应、UV 压缩格式）归 Experiment Design。
- b. 只在 discussion 里语言回应 2509.12235，不做 θ_r 实验（威胁#2 悬置——不建议，D2 已批 UV）。

**Theory 判断与推荐：a。**

**你的回答：**
a，同时为了有稳定性r是不是可以多采样一点？

---

## Q6：双 ER 实验——engage 2605.30524 的正面对话

**背景：** 与头号威胁#1 的方向冲突（他们"压缩→坏"，我们"OPD 压缩→好"）可能只是**构念不同**（raw 表示秩 vs 白化输出相关谱）。策略 = "不同构念"打底 + "潜在反例"进攻：在同一批 checkpoint（三臂）上**同时计算两种 ER**：
- 我们的：GetSlice 白化谱 ER（p ∝ σ，SVD-LLM lens）；
- 他们的：raw 中心化残差流协方差 ER（λ=σ²，d-归一化，固定多域探针——**探针须含 OOD 域输入**，与 intent 的 "ER on OOD-domain inputs" 合并）。

**三结局解释树（预注册）：**
1. raw ER 下 OPD 也压更多且 OOD 更好 → **对 2605.30524 headline 的直接反例/精修**（strong-accept 级材料）："压缩量不是 OOD 损伤的度量，暂态破坏才是"。
2. 只有白化 ER 有判别信号 → 信号特定于输出相关谱——较窄，措辞降为"观测量选择敏感"。
3. raw ER 复现"压缩→坏"、白化 ER 反转 → **两种压缩构念分工**（"表示铺开度"照他们的方向走，"输出相关谱"照我们的方向走）——仍新颖，需讲清分工机制。

**选项：**
- a.（建议）双 ER + 三结局树 + OOD 域探针合并，写入 theory 与 geometry 文件；SVD-LLM 注册为我们 ER 的正式方法学引用。
- b. 不做双 ER，只写构念差异防御（放弃结局 1 的进攻机会——不建议）。

**Theory 判断与推荐：a。** 这是纯分析实验（现有 checkpoint 重探针），是本轮性价比最高的一项。

**你的回答：**
a

---

## Q7：Cycle 09 claim 边界与禁止项

1. **A09 fp32 之前**：不声称任何 on/off-principal 权重方向结论（含 LoRA 侧）；不引用 Cycle 08 OverlapLift（已撤回）。
2. **off-KD control 结果之前**：不把 OPD 签名归因于 on-policy（"on-policy 是驱动"是待检验假设）；机制语言保持非因果（"co-locate / consistent with"，不用 "causes / drives"）。
3. **双 ER 之前**：不声称与 2605.30524 的方向反转成立——当前只能说"在白化构念下观察到相反方向，构念可比性待双 ER 判定"。
4. **θ_r 实验之前**：不声称 ER 是"正确的"观测量——2509.12235 的旋转质疑未回应前，ER 的地位是"当前唯一干净分离两臂的已测观测量"。
5. **压缩措辞**：任何 "compression" 声明必须限定构念（whitened output-relevant vs raw representational），不使用裸的"压缩好/坏"。
6. **L18**：选层标准（判别效应最大）必须预注册并报告全层剖面；不允许只报 L18。
7. **n 的诚实**：单模型/单数据集/单 seed/臂数≤3——所有主张限定在"this setting"；CI（B09）落地前，轨迹相关性（n=10 Spearman）只作 diagnostic。
8. **能力优势的输出控制维度**：OPD 响应极长（step_624 均值 ~4k tokens，16384 cap 下 trunc 0.92）——能力 margin 报告必须伴随 C09 输出控制分解与推理成本说明。
9. 论文正文不得出现 "Cycle N" 叙事（intent 规定）。
10. 不比较 1.7B 旧系列与 4B 系列的数字（不同 checkpoint/数据/格式，长期规则）。

**选项：**
- a. 全部正确，写入 claim 边界（并标记 ⚠️ 待同步至 result/claims_allowed.md，由 Result 对话执行）。
- b. 需修改某几条（请说明）。

**Theory 判断与推荐：a。**

**你的回答：**
a

---

## Q8：优先级阶梯与 AAAI 时窗（scope 现实检查）

**背景：** D1/D2 使 Cycle 09 含两条新训练臂（A09 fp32 retrain、off-KD control）+ 大量分析。AAAI 全文截稿通常在 8 月初（**确切日期请你核对**），今天 7 月 8 日，窗口 ~4 周。关键事实：**Tier A 全部是分析（无需重训）**——UV/θ_r、双 ER 都只是在现有 checkpoint 上重探针。

**建议的优先级阶梯（预注册，防中途乱序）：**

| Tier | 内容 | 性质 | 论文角色 |
|---|---|---|---|
| **A（第 1–2 周，阻塞主线）** | B09 CI（paired bootstrap）；双 ER；L18 层级重推导 + dip 共位矛盾裁决；UV 重探针 + θ_r；C09 输出控制分解 | 纯分析 | **主贡献内核（锐化 b）全部依赖 Tier A** |
| **B（并行启动，能赶则赶）** | A09 fp32 retrain；off-KD control 臂 | 训练 ×2 | (a') 条件贡献 + (c) 升级通道；**赶不上 → 论文按 hypothesis 语言写，不 block 提交** |
| **C（Tier A 落地后）** | theory 定稿 + paper rewrite（framing 按 Q1 结构） | 写作 | — |

**应急预案（预注册）：** Tier B 任一实验未在写作截止前完成 → 对应贡献自动降级（(a') 移入 future work；(c) 保持 hypothesis），主线不受影响；**不允许**为等 Tier B 结果推迟 Tier A 分析或论文提交决策。

**选项：**
- a.（建议）按上述阶梯+应急预案写入 theory，作为本轮执行纪律。
- b. 调整优先级（请说明）。

**Theory 判断与推荐：a。**

**你的回答：**
a。同时在QA的最后我想表明，为了提高速度，现在可以不用按照完整的轮次，我希望不需要训练的结论、可以快速得到结果的结论直接theory->coder->theory，视为mini-cycle09，重新更新理论之后，需要训练等开销大的再走完整路径theory->coder->result...，视其为完整的一轮。

---

## 通过 QA 后的预期写入

假设按推荐通过（Q1=a … Q8=a）：

1. **`theory/current_theory_update.md` + `-ch.md`** 新增 Cycle 09 大节：
   - 贡献分层结构（锐化 b 主 / c 假设层可升级 / a' 条件副 / 测量学方法贡献）+ 领域三阶段发展线路定位；
   - 交集护城河精确表述 + 三空间时序咬合证据链；
   - LoRA-vs-全参双结局决策树（Q2）；
   - off-KD control 解释树（Q3）；
   - 双 ER 三结局树（Q6）与 2605.30524 engage 策略；
   - θ_r 裁决树与 2509.12235 回应路线（Q5）;
   - Cycle 09 claim 边界 10 条（Q7）+ 优先级阶梯与应急预案（Q8）。
2. **`theory/geometry_metric_definitions.md` + `-ch.md`** 新增 Cycle 09 节：
   - 层级化规则（L18 headline + 全层剖面强制 + 选层标准预注册）（Q4）；
   - principal angle θ_r 正式激活（UV 开启，r 自适应规则沿用 Cycle 07，γ_r 转为容量代理配角）（Q5）；
   - 双 ER 构念规范（两种 ER 的公式/探针/归一化并列定义，SVD-LLM 正式引用）（Q6）；
   - OPD-dip 共位矛盾的层级裁决程序（Q4）。
3. **跨对话标记**（不由 Theory 写入）：
   - → Result：claims_allowed.md 边界更新 + current_picture "both dips co-locate" 表述修正（待 L18 裁决）；
   - → Experiment Design：off-KD 臂实现（变体 i）、fp32 retrain、UV 重探针、双 ER 探针集（含 OOD 域）、CI 协议、L18 全层剖面输出为 required artifacts。

---

## 📌 讨论结论（2026-07-08，两轮讨论后闭环）

**Q1–Q8 全部按推荐选项（a）通过。** 两轮讨论新增以下决议（E1–E5）：

### E1：A09 双轨制 + "过程 vs 终态"框架（Q2 讨论闭环）

用户提出的理解——**fp32 vs bf16 = 过程 vs 终态**——被采纳为正式框架：
- **bf16 轨 = 终态**（部署件真实携带的更新，TPNT-conform，生态效度）；
- **fp32 轨 = 过程**（训练过程实际做出的几何更新，归因干净）。
- 归因问题（LoRA-vs-全参矛盾）住在**过程侧**；与论文的过程/轨迹视角自洽。双轨都报告，两轨之差 = bf16 对方向指标的扭曲量（测量学贡献的数据）。

### E2：测量修正 = 直接算 BA，A09 第一步免重训（Q2 讨论闭环）

- 用户确认解法：把测量对象从 `bf16(W0+BA) − bf16(W0)` 改为 **直接计算 ΔW = B@A**（fp32 优先；bf16 存储的 adapter 也可接受——其舍入误差相对 adapter 元素量级 ~1e-3，远小于 merge−subtract 相对 W0 量级的误差）。
- **数值秩证据**记录在案：测得 ΔW numerical rank ≈ 2121，而 LoRA r=32 真实更新秩 ≤ 32 → 现有读数的秩结构被满秩量化噪声主导；ρ²≈0.02 的"弥散"是 **bf16 噪声地板特征而非 LoRA 结构特征**（纯 BA 应高度集中于 32 方向）——用户"弥散是 LoRA 导致"的初始猜想经数据讨论后反转。
- **A09 执行序**：第一步查 Cycle 08 LoRA adapter 文件（两臂都是 LoRA r=32，应有）→ 在则 fp32(B@A) 零训练成本出结论；仅当 adapter 不可得或需排除训练级 bf16 效应时才 retrain。

### E3：探针电池 v2.1（Q4 讨论闭环）

在 v2 基础上用户新增 **X_math_hard（AIME 式难题探针）**——额外收益：激活探针是 generation-free 的前向传播，**绕过 AIME24 行为评测的非终止失效**，让难数学 regime 有了可观测通道。最终电池：

| 探针 | 域 | 状态 |
|---|---|---|
| X_math（MATH500 式） | ID | 沿用 |
| X_math_hard（AIME 式，建议用历史 AIME 题避免与 eval 集耦合） | ID-难 | **新增** |
| X_bos | 无条件 | 沿用 |
| X_ood_knowledge（MMLU-Pro 式） | OOD 知识 | **新增，最高优先** |
| X_teacher（8B teacher 固定回复集） | 监督源 | **新增** |
| X_general（开放域文本） | 通用 | **新增** |
| S（各臂沿用；off-KD 臂 S = 其 teacher 静态训练回复） | 训练目标 | 沿用 |

预算紧时优先级：X_ood_knowledge > X_teacher > X_general ≈ X_math_hard。

### E4：θ_r 稳定化两维（Q5 讨论闭环）

- **r-网格**：r ∈ {8,16,32,64,128} ∪ {energy@80/90/95%}，主 r = energy@90%，其余附录稳健性曲线。
- **探针数敏感性**：用户先前消融显示探针级采样影响不大；按讨论执行一次 **采样数扫描复核**（如 n_samples ∈ {8,16,32,64}）确证不敏感后固定 n——作为附录稳健性证据，堵审稿质疑。

### E5：mini-cycle09 执行协议（Q8 补充闭环）

- **mini-cycle09**（Tier A 分析）：theory → coder → theory 快速环；coder 产物落 `local_experiment_results/cycle_09.../mini/` 保 provenance；theory 可持 provisional reading 快速迭代。
- **cycle09-full**（Tier B 训练）：完整 theory → exp → coder → result 链。
- **护栏（用户确认）**：mini-cycle 读数进入 claims_allowed/current_picture 前仍须过 Result 整合（可 Tier A 收尾一次性过）——加速理论迭代，不豁免 claim 门禁。

### 四点确认记录

用户对 Theory 提出的 4 个待确认点（A09 双轨+adapter 检查 / 电池全加 / r-网格+bootstrap / mini-cycle 护栏）**全部赞成**。

### 待同步（跨对话）

- → **Result**：claims_allowed 边界 10 条 + current_picture "both dips co-locate" 修正（待 L18 裁决）。
- → **Experiment Design / Code**：A09 adapter-first 执行序（E2）、off-KD 臂变体 i、探针电池 v2.1（E3）、θ_r r-网格与采样扫描（E4）、双 ER 探针集、CI 协议、L18 全层剖面、mini/ 目录约定（E5）。
