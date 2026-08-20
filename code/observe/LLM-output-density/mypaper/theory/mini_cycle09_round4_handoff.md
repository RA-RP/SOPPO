# Theory → Coder 转交：mini-cycle09 Round 4（窗口 v2 战役 + 指标套件定稿版，96G×1，零训练）

```yaml
view_type: handoff_note
cycle: cycle_09_aaai_competitiveness_completion (mini, E5)
round: 4
from: Theory
to: Code Execution (mini-cycle09 coder)
created: 2026-07-12
gpu: 96G × 1
constraints: 零训练
copyback_target: local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/
handin_target: code/code_evolution.md（追加 round-4 handin 子节）
qa_convention: 开工前在 code/QA_cycle09.md 追加 round-4 节提问
guard: 读数原样记录、不解释；裁决交回 Theory 预注册树/判据；不写 result/、不碰 claims 文件
supersedes: round-3 handoff 的"追加任务"节（R3-8/9/10 不在 v1 数据上跑——机械保留，
            数据换 v2；R3-11 被 v2 多种子生成设计吸收）
```

## 前置阅读（按序，前两项必须精读）

1. 本文件
2. `theory/geometry_metric_definitions.md` — **Round-4 Spec 节**（S/E/X/H 四型命名、
   窗口 v2 六条规则、证据分层规则、M1/M2/M3 指标定稿）——这是本轮的执行规格本体
3. `theory/current_theory_update.md` — Cycle 09 节的 Round-4 amendments（claim 边界 19/20）
4. 自己 round-1/2/3 的产物与 manifest
5. `code/current_code_structure.md`

## 任务清单（按执行序）

### R4-0（P0，CPU，立即）— v1 窗口构成审计
- 对 round-2/3 的每个（探针, 域, 臂）格，计算首窗内 prompt-token 占比的分布
  （min/mean/max + "全 prompt 窗"计数）。
- 用途：量化 v1 前缀偏置的实际严重度，供 Theory 写先导层 caveat 的措辞强度。
- 输出：`R4_window_audit.csv`。

### R4-1（P0，96G 主战役）— v2 统一探针战役
严格按 geometry 文件 Round-4 Spec 执行，要点复述：
- **四型探针**：S_d（base 生成，含新增 **S_bos**）/ E_d（外部语料：E_ood、E_general、
  E_math_hard）/ X_arm（X_OPD 逐 ckpt rollout；X_SFT 数据集 CoT，题面 mask）/
  H_d（**仅非训练域**：H_ood、H_general、**H_bos**；两臂对称，无 math）。
- **窗口 v2**：512 窗全落生成区（prompt 只进前向不进统计）；每样本固定 k=3 均匀随机
  偏移窗（不足则重叠/减 k 并记录）；窗种子与生成种子分离；每窗记相对位置（三分箱）。
- **生成**：S / X_OPD / H 各 **3 个生成种子批**（temp 0.6 / top_p 0.9 / max_new 1024，
  沿用 v1 参数便于对照）；批间 mean±SD。
- **层级归一化**：窗级 token 均值 gram → 样本级窗均值 → 语料级样本等权均值；
  白化矩阵由该层级化协方差构造。
- **bootstrap 单元 = 样本**（窗嵌套）。
- 固定文本探针（旧 S=数据集 CoT、全部 E 型）**同批换随机窗重探**（供 R4-3 重推导）。
- 层：{9, 18, 27} 因子 + 全 36 层谱（沿用 r2/r3 分辨率）；7-ckpt 网格；两臂。
- 输出：`R4_v2_spectra_*.csv` + `R4_v2_manifest.json`（生成/窗种子、k 实录、mask 规则）。

### R4-2（P0，CPU/轻 GPU）— 指标套件 M1/M2/M3 在 v2 上计算
- **M1**：tail_energy(r) + r_ε（ε∈{0.05, 0.01}）+ **EC 判据**（r_ε 变化 × drift_core），
  白化双轨（逐 ckpt 主 / 冻结 base 副）；核对预注册预测（OPD 终点 EC 成立；
  SFT dip 步核心稳定性暂态违反）。
- **M2**：‖ΔW·X‖_F/‖W₀·X‖_F（X₀ 主 / X_t 副；`tr(ΔWᵀΔW·Σ_X)` 从 v2 因子 + BA 算；
  OPD top-32 近似 caveat）+ **M2b** 表示漂移 ‖h_t−h₀‖/‖h₀‖。
- **M3**：e_keep 与 φ 的 **U/V 双侧**（公式见规格表；r 取 M1 的 r_ε 衔接；
  全族用冻结 base 基底）；核对 P-tail-2 定稿版（OPD e_keep(E_ood)≈1 = 良性整合）。
- 输出：`R4_m1_tail_ec.csv`、`R4_m2_output_drift.csv`、`R4_m3_keep_aim.csv`。

### R4-3（P0/P1，随 R4-1/2 产出）— v2 重推导与既定检验
1. **L18 uptick v2 重推导**（P0——先导层→论文级的关键 robustness）：随机窗 +
   样本级 bootstrap 下两臂 dip-uptick 是否仍显著；逐模块表照旧。
2. **配对判别力检验**（原 R3-9 规格，数据换 v2）：d_whitened − d_raw 配对 bootstrap；
   两侧 θ 误差带。
3. **暂态共位矩阵**（原 R3-10 规格）：几何列用 v2；行为列（IFEval/MMLU-Pro）做
   prompt 级配对 bootstrap。
- 输出：`R4_l18_rederivation.csv`、`R4_discriminability.csv`、`R4_transient_colocation.csv`。

## QA_cycle09.md round-4 节必问

1. v2 生成与因子的存储预算（3 种子批 × 四型 × 7 ckpt × 两臂；因子只存 {9,18,27} 确认）。
2. k=3 的边缘处理确认（生成 <512 token 的样本：取单窗还是记缺格）。
3. E_math（教材式外部数学文本）本轮缓做——确认跳过。
4. 位置三分箱默认输出——确认。

## 完成后交回 Theory 的裁决清单

| 项 | 裁决问题 | 依据 |
|---|---|---|
| R4-0 | v1 前缀偏置实际严重度 → 先导层 caveat 措辞定稿 | 证据分层规则 |
| R4-3.1 | L18 uptick 在 v2 下存活与否 → 论文 headline 的最终地位 | P-R1 修订 |
| R4-2 M1 | EC 判据两臂读数 → "有效压缩 vs 破坏性暂态"故事定稿 | EC 预注册 |
| R4-2 M3 | e_keep_U/V(E_ood) → eviction/整合的方向级终审 | P-tail-2 定稿版 |
| R4-3.2 | 判别力 CI → "观测空间优越性"能否升级为统计主张 | Q4 标准 / claim 边界 15 |
| R4-3.3 | 共位矩阵 + 行为 bootstrap → P-R6 定稿或撤回 | claim 边界 17 |
| v2 S/X/H | X_OPD 平滑收缩、H 暂态（含新 H_bos）在 v2 口径下的形态 | S/X/H 预注册规则 |

护栏照旧：产物落 `mini/`，handin 追加 `code/code_evolution.md`，claim 门禁不豁免。
