# Paper Thesis Framework（论文论点框架）

```yaml
view_type: paper_thesis_framework
cycle: cycle_09_aaai_competitiveness_completion
status: active — root sentence FROZEN (user-approved 2026-07-13); claim tree tracks evidence tiers
created: 2026-07-13
source_discussion: thesis 讨论（2026-07-13，Theory 对话）
maintenance_rule: 主句冻结不轻改；claim 树的证据等级随 mini-round/Tier-B 回程更新
depends_on:
  - theory/current_theory_update.md (Cycle 09 节全部裁决与 claim 边界 1–20)
  - theory/geometry_metric_definitions.md (Round-4 Spec: 窗口 v2 / M1–M3 / S-E-X-H)
downstream: paper_drafts/（写作阶段以本文件为唯一论点源头）
```

本文件是论文的"宪法"：§0 的两句话是全文唯一的论点源头，正文每个 section、每个图表都必须能追溯到 §1 claim 树的某个节点。写作阶段不得引入不在树上的主张。

---

## §0 论点（冻结版，2026-07-13）

**主句（论点）：**

> 本文将激活感知压缩（SVD-LLM）的白化谱几何重铸为微调过程的功能坐标系，在匹配的 OPD–SFT 轨迹上刻画输出相关方向的重组动力学，并发现一条与"压缩即损伤"塌缩叙事相反的规律：**ID 能力暂跌与中层输出相关谱的暂态破坏共位，OOD 能力的去留追踪其幅度，而两者都与重组或压缩的总量脱钩**——重组最剧烈、可压缩性提升最大的 on-policy 蒸馏臂以更早更短的暂态同时取得更高 ID 增益与完好 OOD 保持；移动最小却破坏最重的 SFT 臂增益更低且侵蚀 OOD。

**护盾句（紧随主句）：**

> 这一联合模式同时排除两类平凡解释：OOD 的保持既非"少动"的红利（OPD 的旋转与谱收缩皆为最大），也非"少学"的代价（其 ID 增益反而更高）。

**紧凑版**（摘要首句）：

> 我们把激活感知压缩几何重铸为微调过程的功能坐标系，在匹配 OPD–SFT 轨迹上揭示：ID 暂跌与 OOD 去留都系于中层输出相关谱的暂态破坏，而与重组总量脱钩——重组最多的臂在两个轴上同时更优。

**English:**

> We recast activation-aware compression geometry (the SVD-LLM whitened spectrum) as a functional coordinate system for fine-tuning dynamics and show, on matched OPD–SFT trajectories, that ID dips co-locate with — and OOD retention tracks the magnitude of — transient disruptions of mid-layer output-relevant spectra, while neither tracks the total amount of reorganization or compression: the most-reorganizing arm (on-policy distillation) achieves both higher ID gains and intact OOD, whereas SFT, which moves least but breaks hardest, gains less and erodes OOD.

**措辞守界（写作时不可违反）**：共位=时序主张、追踪幅度只落在 OOD（ID dip *深度*臂间差不显著）；全程用"追踪/共位/脱钩"，禁用"决定/导致"（因果边界，off-KD 未跑）；"臂"是描述性命名，不归因于 on-policy 机制本身。

---

## §1 Claim 树（从句 → 等级 → 证据 → 边界 → 归属 → 依赖）

等级：★已确立（CI/预注册标准达成）｜◐先导层（内部有效，待 v2 转正）｜○待实验。

### C1 镜头主张："白化谱几何 = 微调过程的功能坐标系"

| 子节点 | 等级 | 证据 | 边界 | 依赖 |
|---|---|---|---|---|
| C1a 判别信号是白化构念特有（raw 表示秩静默） | ★ | T8'：whitened OPD−SFT −15.8 vs raw −0.11（同批探针） | 只写"构念特定"，不写"优于"（#15） | — |
| C1b 权重空间不判别（零假设腿） | ★ | 干净 BA 轨 ρ 两臂全程弱 on（z+5~9）；θ_w ≤2–5° 含 1.2° 地板 vs 激活 17–23° | 量级对照=描述性；OPD 侧 top-32 近似 caveat | A09（OPD 干净 BA 终值） |
| C1c 观测空间优越性（统计版） | ○ | 待配对判别力检验 | #15 | R4-3.2 |
| C1d 测量学贡献（merge−subtract 翻转方向判定；OverlapLift 撤回） | ★ | r1-T4 数值秩 2121 vs 32；bf16-BA≈fp32-BA | 双轨制表述（bf16=终态/fp32=过程） | — |

**归属**：Methods（lens 定义 + M1–M3）+ 一节测量学批判。

### C2 设定主张：匹配 OPD–SFT 轨迹

| 等级 | 证据 | 边界 |
|---|---|---|
| ★ | 同 student/数据/checkpoint 网格/LoRA r32；Cycle 07 SFT 基线 + Cycle 08 OPD 臂 | teacher 通道三重混淆（on-policy/密度/身份）如实声明，归因留待 off-KD |

**归属**：Setup。

### C3 共位律：ID 暂跌 × 中层暂态破坏（跨通道，幅度不对称）

| 子节点 | 等级 | 证据 | 边界 | 依赖 |
|---|---|---|---|---|
| C3a 两臂 dip 与 L18 uptick 共位且显著 | ◐ | 样本级 bootstrap：OPD +0.83[0.65,1.08]@step_5、SFT +4.76[3.70,5.70]@step_20；14/14 模块格 | 层限定（#13）；先导层口径="题面+解答开头窗"（#19） | **R4-3.1 v2 重推导（转正的唯一通道）** |
| C3b 幅度不对称 ~5.7×，MLP+o_proj 承载 | ◐ | 同上逐模块表 | 同上 | 同上 |
| C3c 暂态时序：OPD 更早更短（step_5→20）vs SFT（step_20→160） | ★ | 轨迹网格事实 + 行为恢复 | dip **深度**差 CI 含 0——禁写"更深"（#11） | — |

**归属**：Results §1（机制 headline）。

### C4 追踪律：OOD 去留 ↔ 暂态破坏幅度

| 子节点 | 等级 | 证据 | 边界 | 依赖 |
|---|---|---|---|---|
| C4a OOD 双轴同向：OPD 保持/略升，SFT 侵蚀 | ★ | MMLU-Pro（C08：+0.016 vs −0.029）+ IFEval（R3-6：+4.4 vs −5.7 pts）；TruthfulQA/GPQA 平=preservation check | IFEval 保持 preservation 预注册身份，不升 gate（#9 旧例） | — |
| C4b SFT 的 OOD 条件谱自带小峰（同域证据） | ◐ | E_ood 上 SFT step_20/40 小峰 | 探针内比较承重；跨域幅度禁比（#20） | R4 v2 |
| C4c 幅度→去留的"追踪"关系 | ◐ | n=2 臂同向（大破坏→侵蚀，小破坏→保持） | 非因果措辞；"in this setting" | off-KD（第 3 臂）加密 |

**归属**：Results §2。

### C5 脱钩律：ID/OOD 结局 ⊥ 重组与压缩总量（方向反转）

| 子节点 | 等级 | 证据 | 边界 | 依赖 |
|---|---|---|---|---|
| C5a OPD 动得最多：θ_r 17–23°、drift 更大、全域谱收缩最深 | ★ | T7'、r1 drift、X 条件化 ER（各域 OPD ≫ SFT 收缩） | 跨域排序已撤回，逐探针内表述 | — |
| C5b 同域 eviction 证伪 | ◐ | E_ood 探针内：OPD −21.3 保持 vs SFT −4.7 侵蚀 | ER=组织度非容量（解释裁定）；方向级封口未做（#16） | **M3 e_keep_U/V（终审）** |
| C5c "有效压缩"定性（收缩=整合非删除） | ○ | 待 EC 判据 + e_keep | — | M1 EC + M3 |

**归属**：Results §3（与 2605.30524 正面对话的位置）。

### C6 例证主张：臂级结局

| 子节点 | 等级 | 证据 | 边界 |
|---|---|---|---|
| C6a OPD ID 增益更高 | ★ | MATH500 final +0.096[.058,.134]/peak/AUC 全显著；numina@12288 全步领先；AIME24 avg@10 同向（secondary） | ID 数字必须伴随输出控制分解（#8）；AIME 截断 caveat 内联（#18）；推理成本如实报告 |
| C6b SFT 增益更低且侵蚀 OOD（**非"两头皆输"**——SFT ID 有绝对增益） | ★ | 0.752>base 0.636 + C4a | 忠实措辞 |

**归属**：Results §0（能力表，最先出场）。

### C7 护盾：排除"少动/少学"

| 等级 | 推导 | 归属 |
|---|---|---|
| ★（由 C5a+C6a 直接合成） | 少动排除 ← C5a；少学排除 ← C6a | 紧随主论点段；Discussion 再展开 |

### C8 措辞人质："可压缩性提升最大"

| 等级 | 依赖 | 降级预案 |
|---|---|---|
| ○ | M1 的 r_ε/EC 读数 | EC 不成立 → 主句该从句退回"谱收缩最深"（C5a 已确立的弱表述），主句骨架不变 |

---

## §2 论点级相关工作对位（每从句打谁）

| 主句从句 | 对手/邻居 | 关系 |
|---|---|---|
| "重铸为功能坐标系" | SVD-LLM（方法源）；2605.30524（raw 构念） | 移植 + 构念划界 |
| "匹配 OPD–SFT 轨迹" | 2606.07082/13657（权重、无轨迹）；Rethink SFT（行为、无 OPD） | 无人占的交集 |
| "共位 + 追踪幅度" | Rethink SFT（只有行为 dip）；2605.30524（静态预后） | 加机制层 / 加过程维 |
| "与总量脱钩"（反转） | **2605.30524 headline 的直接修正**；2509.12235（其因果层已被我们分析瓦解，观测层与我们双空间数据互证） | 正面对话主战场 |
| "on-policy 蒸馏臂双优" | on-policy forgetting 簇（行为层） | 补几何观测量 |
| C1d 测量学 | TPNT / LoRA-illusion / 2603.02224 | 方法贡献 |

## §3 未决依赖账本（谁不回来，哪句要改）

| 未决实验 | 人质从句/节点 | 回不来的降级 |
|---|---|---|
| **R4-3.1 L18 v2 重推导** | C3a/C3b（机制 headline） | 不显著 → 共位降 suggestive，主句"共位"改"伴随"，论文重心移向 C4/C5/C6 |
| **M1 EC 判据** | C8 | 退"谱收缩最深" |
| **M3 e_keep_U/V** | C5b/C5c 封口 | 不做方向级定性，保留"组织度"解释层 |
| R4-3.2 判别力检验 | C1c | 保持描述性（现状） |
| R4-3.3 共位矩阵 + R4 v2 S/X/H | P-R6 跨空间链、S/X/H 仪器主张 | 不进正文，Discussion 一句或删 |
| off-KD（Tier B） | 不在主句内（主句刻意非因果） | 及时→机制归因升级进 Discussion/共同主贡献；不及→维持现状 |
| A09（Tier B） | 不在主句内（(a′) 是副贡献） | 及时→副贡献节；不及→future work |

## §4 论文骨架 ↔ claim 树映射

```text
Abstract        §0 紧凑版 + 护盾句
Intro           §0 主句展开；三阶段发展线路（权重静态→表示预后→过程机制）收束到交集
Related Work    §2 对位表成文；2509.12235 三层评估；2605.30524 构念划界
Setup           C2（匹配设计 + 混淆声明）
Lens & Metrics  C1（白化坐标系 + M1–M3 + S/E/X/H）+ C1d 测量学批判
Results 0       C6 能力表（ID 三轴 + OOD 双轴，全 caveat）
Results 1       C3 共位 headline（L18 逐模块 + 时序）
Results 2       C4 追踪律（OOD 双轴 × 破坏幅度）
Results 3       C5 脱钩律（θ/收缩/同域 eviction 证伪 → 对话塌缩文献）
Discussion      C7 护盾展开；break-not-movement 机制假设（标注 hypothesis）；
                function-level reorganization 解释；limitations（n=2 臂/单模型/单数据/
                单 seed/LoRA regime/相关性）；off-KD 与 A09 作 ongoing
```

## §5 下一步

1. R4 回程 → 按 §3 账本更新各节点等级（◐→★ 或降级），主句从句随之定稿；
2. 等级稳定后进入写作：**只写 ★ 节点，◐ 节点带口径声明，○ 节点不进正文**；
3. claims_allowed 的同步（边界 1–20 + 本框架）在 Tier A 收尾一并过 Result 整合。

---

## §6 Round-5 后的 claim 树修订（2026-07-14，provisional；θ/CI 未到）

论点句（2026-07-13 冻结，收束版）不变：

> 在白化激活空间——其谱尾直接量化"压缩到 r 维所损失的输出精度"——比较匹配的 OPD 与 SFT 轨迹，发现二者以相当的函数移动量换来截然不同的功能可压缩性：OPD 持续大幅提升，SFT 几乎不变；且这一可压缩性的起伏与 ID 能力的 dip–recover 同步。

### 升级

| 节点 | 变动 | 依据 |
|---|---|---|
| **C-压缩程度** | ◐ → **★ 且泛化范围扩大**：五个探针（训练域 / E_ood / E_general / E_math_hard / **S_bos 自由生成**）**零例外**，OPD 提升 2.1–5.3 倍于 SFT | P5-1。主句"功能可压缩性"从"外部语料上"升级为"**所有已探测域上**" |
| **C-共位（新子节点 C3d）** | 新增 ★（点估计级）：**域特异共位 + 向外扩散** —— SFT 的几何暂态按域依次出现（训练域@20 → OOD 知识@40 → 通用@160），每个都与**同域** benchmark 的 dip 共位 | P5-2。这是比"L18 有个峰"强得多的证据形态 |

### 降级 / 撤回

| 节点 | 变动 | 依据 |
|---|---|---|
| γ_{r_ε}（压缩质量） | **撤回全部读数**（"dip 时塌陷、之后 +91%"作废）——r_ε 处谱太平坦（γ/σ≈0.3%），信噪比不足 | P5-6。可用形式改为**谱顶部**（k=64：OPD +137% vs SFT +42%） |
| "OPD 全程无损" | **禁止**——MMLU-Pro/GPQA 的 step-40 dip **两臂共有**，OPD 甚至更深（−0.084 vs −0.036）；差异在**恢复**不在**是否 dip** | P5-3 |
| H-mismatch（B 线机制） | 预注册预测失败 → **exposure-bias 解释被证伪**（决定性反证：两臂都 dip，而 OPD 是 on-policy 本不该有 exposure bias） | P5-4 |

### 待检验（新假设，用户提出）

**H-transition（压缩方案切换）**：dip = 旧压缩方案已破、新方案未立的过渡窗口；两臂**都有**过渡，差别在**速度**（OPD dip@5→恢复@20；SFT dip@20→恢复@160）与**落点质量**（r_ε 终值 −25 vs −4.9）。
- 满足：从压缩角度、与全文一致、统一解释两臂 dip、不借 exposure bias。
- **主检验 = θ_{r_ε}**（运行中）：预测 dip 时刻主子空间转动最快。
- 次检验 = 稳健化的 γ（窗口平均 / log 斜率 + CI）。

### 术语固定

"mismatch" **不再作为伞状词**：Mismatch（维度**宽度**差）与 xs_gap（谱**能量**差）是两个量，禁止混用（P5-5）。其中 **SFT 的 xs_gap 在其 dip 步取峰（.0795）** 是臂内形状事实、有效；跨臂 xs_gap 幅度不可比（两臂的 X 是不同性质文本）。

### 方法论规程（本轮教训）

**任何机制主张，必须先在全部探针 × 全部 benchmark 上铺开，再挑 headline。** 本轮此前数轮只盯训练域探针，因而漏掉 P5-2（域特异扩散）这一更强模式，也差点把两臂共有的 MMLU-Pro dip 误作 SFT 独有。
