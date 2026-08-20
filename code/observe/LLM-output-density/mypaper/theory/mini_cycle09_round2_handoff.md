# Theory → Coder 转交：mini-cycle09 Round 2（统一探针战役，32G / 零训练）

```yaml
view_type: handoff_note
cycle: cycle_09_aaai_competitiveness_completion (mini, E5)
round: 2
from: Theory
to: Code Execution (mini-cycle09 coder)
created: 2026-07-09
constraints: 单卡 32G；零训练（任何训练属 Tier B，走完整路径，不在本 round）
copyback_target: local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/
handin_target: code/code_evolution.md（追加 round-2 handin 子节）
qa_convention: 开工前在 code/QA_cycle09.md 追加 round-2 节提问，用户答后执行
guard: 读数原样记录、不解释；θ_r/双ER/层剖面结果交回 Theory 按预注册裁决树裁定；
       不做 on/off-principal 结论；不写 result/、不碰 claims 文件
```

## 前置阅读（按序）

1. 本文件
2. `theory/current_theory_update.md` — Cycle 09 节，重点 **Mini-Round 1 Adjudication**（R1–R4 裁决 + R2b 层规则 + claim 边界 11–14 + round-2 优先级）
3. `theory/geometry_metric_definitions.md` — Cycle 09 节（θ_r 规格、双 ER 构念表、探针电池 v2.1、Mini-Round 1 测量裁定）
4. 自己 round-1 的产物（`mini/` 下 T1–T4 + findings）
5. `code/current_code_structure.md` + `../Readme-experiments.md`（GetSlice 探针机械）

**背景一句话**：round-1 裁决后，headline 层已由用户拍板为 **L18**（判据="dip 暂态最清晰"，预注册的是**判据**非层号）；"SFT 早期 off-principal"已退役为 merge−subtract 伪迹；dip"更深"无 CI 支持退役。Round 2 的使命是给幸存故事上保险：L18 小峰是否超噪声、旋转是否比值分布更判别、raw 构念下方向反转是否成立。

---

## 核心设计：一次统一探针战役，五个产出

**不要拆成五次跑。** 一次 GetSlice 战役覆盖：

```text
两臂（OPD/SFT）× 10 checkpoint × 全 36 层 × 电池探针 × 加大采样数，UV 保存开启
```

从同一批前向输出派生 T5–T9：

### T5 — 全 36 层 ER/xs_gap/drift 剖面
- 两臂 × 10 ckpt × 36 层（round-1 只有 {9,18,27}）。
- 用途（Theory 裁决）：(i) 反 cherry-pick 展品；(ii) 检查是否有比 L18 更清晰的 dip 暂态层（若有→并报，判据不变）；(iii) L27 终态分歧是否真是全层最大。
- 输出：`T5_full_layer_profile.csv` + 每层 dip-uptick 有/无 判定表。

### T6 — ER 探针 bootstrap 误差带 + 采样数扫描（R2 的生死项）
- 对探针样本 bootstrap，给每个 (层, ckpt) 的 ER 附置信带；采样数扫描 n_samples ∈ {8,16,32,64} 确证不敏感后固定 n。
- **裁决问题**：L18 的 OPD +1.3 / SFT +3.6 uptick 是否超出误差带？不过关 → R2"共位"降级为 suggestive。
- 输出：`T6_er_error_bands.csv` + 敏感性表。

### T7 — UV 重探针 + θ_r（实测子空间旋转）
- UV 开启；θ_r：(i) 相邻 checkpoint 间 (ii) 各 ckpt vs base，逐层×逐模块。
- r-网格：{8,16,32,64,128} ∪ {energy@80/90/95%}，主 r = energy@90%；θ_r 同样附探针 bootstrap 置信带。
- 预注册预测：SFT 在 L18 step_10–20 出现 θ_r 旋转尖峰（与 ER bump 共位）；OPD 平滑。
- **裁决量**（交 Theory 按裁决树）：ER vs θ_r 谁更好地 (i) 分离两臂 (ii) 跟踪 MMLU-Pro Δ。
- 输出：`T7_theta_r.csv` + r-稳健性曲线数据。

### T8 — 双 ER（engage 2605.30524）
- 同批 checkpoint 同时算两构念（规格表见 geometry 文件 Cycle 09 节）：
  - 白化 GetSlice ER（p∝σ，逐模块）——现有机械；
  - **raw ER**（仅中心化残差流整层 hidden state，λ=σ²，×1/d 归一化，teacher-forced，多域探针）——需新实现。
- **裁决问题**（三结局树在 theory 文件）：raw 构念下 OPD 是否也"压更多且 OOD 更好"。
- 输出：`T8_dual_er.csv`（两构念并列，逐层逐 ckpt 逐探针域）。

### T9 — 探针电池 v2.1 构建（T5–T8 的输入，先行）
- 纯文本准备（零 GPU）：`X_ood_knowledge`（MMLU-Pro 式 prompt）、`X_general`（开放域文本）、`X_math_hard`（**历史 AIME**，避免与 eval 集耦合）。
- 需 8B 生成（32G 小 batch 可跑）：`X_teacher`（8B teacher 在探针 prompt 上的固定回复集）。
- 全部冻结、版本化；优先级（预算紧时）：X_ood_knowledge > X_teacher > X_general ≈ X_math_hard。
- S 探针各臂沿用不改。

### T10（可选备料，GPU 空闲时）— off-KD teacher logits 预计算
- 8B 推理批作业：对 off-KD 语料离线算 teacher top-k logits。**这不是训练**——为 Tier B off-KD 臂铺路，等大卡时直接开训。

---

## QA_cycle09.md round-2 节必问事项

1. **UV 存储预算**：全量（36 层 × 7 模块 × 10 ckpt × 2 臂 × 多探针 × top-128 双侧）估计几十 GB。建议裁剪方案：UV 全存仅 {L9, L18, L27} × dip 邻近 ckpt {0,5,10,20,40}，其余层/步只存奇异谱——θ_r 主战场本来就在 L18 dip 步。请用户确认或调整。
2. 电池实例化：MMLU-Pro 取哪些 prompt/多少条；X_general 语料源；历史 AIME 年份范围；X_teacher 生成参数。
3. raw-ER 探针集与 2605.30524 对齐程度（12k 多域是否缩减）。
4. 探针采样数上限（时间预算）。

## 完成后交回 Theory 的裁决清单

| 来源 | 裁决问题 | 预注册裁决树位置 |
|---|---|---|
| T6 | L18 uptick 是否超误差带 → R2 确立 / 降级 suggestive | current_theory_update R2 |
| T5 | 36 层中 dip 暂态最清晰层是否 L18 → headline 确认 / 并报 | R2b 层规则 |
| T7 | 旋转 vs 值三分支 | Q5 裁决树 |
| T8 | 双 ER 三结局 | Q6 三结局树 |

护栏照旧：产物落 `mini/`，handin 追加 `code/code_evolution.md`，读数不解释，claim 门禁不豁免。
