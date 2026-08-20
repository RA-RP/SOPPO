# Theory → Coder 转交：mini-cycle09 Round 3（96G×1，全部零训练）

```yaml
view_type: handoff_note
cycle: cycle_09_aaai_competitiveness_completion (mini, E5)
round: 3
from: Theory
to: Code Execution (mini-cycle09 coder)
created: 2026-07-10
gpu: 96G × 1（用户已切换；vLLM 批量吞吐远优于 32G）
constraints: 零训练（A09 retrain / off-KD / protected-SFT 式前向干预均属 Tier B，不在本轮）
copyback_target: local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/
handin_target: code/code_evolution.md（追加 round-3 handin 子节）
qa_convention: 开工前在 code/QA_cycle09.md 追加 round-3 节提问，用户答后执行
guard: 读数原样记录、不解释；裁决交回 Theory 预注册树；不写 result/、不碰 claims 文件
scope_note: 因子化干预（原 B1 交换实验）已由用户裁定推迟至理论主体完成后——本轮不做。
```

## 前置阅读（按序）

1. 本文件
2. `theory/geometry_metric_definitions.md` Cycle 09 节 —— 重点两个新增小节：
   **Mini-Round 1 Measurement Rulings**（层限定规则、样本级误差带缺口）与
   **S/X/H Probe Semantics**（R3-5 的语义与预注册规则，刚正式化）
3. `theory/current_theory_update.md` Cycle 09 节 —— 重点 **Addendum 2026-07-10**
   （2509.12235 精读结论、θ_r^W 预期近零的理论依据）
4. 自己 round-1/2 的产物（`mini/` 全部）
5. `code/current_code_structure.md` + `../Readme-experiments.md`

---

## 任务清单（按执行序）

### R3-1（P0，CPU，立即）— L18 逐模块拆解
- 从现有 `T5_full_layer_profile.csv` / `T7_theta_r.csv` 按 7 模块拆开 L18（及 L9/L27 对照）的
  ER uptick、θ_r：uptick 是否集中在特定模块（如 down_proj）？
- 输出：`R3_module_breakdown.csv` + 简表。

### R3-2（P0，CPU，立即）— X 条件化白化 ER（预览版）
- 从 round-2 已存的各探针域 `xMat_X.json` 谱直接算 ER：{9,18,27} × 5 域 × 7 步 × 两臂。
- 这是缺口 #5（"OOD 域输入下的输出相关谱"）的**预览**；正式版（含样本级误差带）由 R3-4 战役产出。
- 输出：`R3_xcond_whitened_er_preview.csv`。

### R3-3（P1，轻 GPU）— 权重侧 θ_r^W
- 对每步干净 BA（SFT = r1 的 fp32 BA；OPD = top-32 近似，标 caveat）计算合并权重
  W₀+BA 的 top-r 奇异子空间相对 W₀ 的主夹角，r-网格同激活侧 {8..128}∪energy 档。
- **预注册预期：≈0°**（LoRA 更新范数小，Davis–Kahan 上界钳制）——近零结果本身入表：
  与激活侧 θ_r（17–20°）构成双空间对照，并对标 2509.12235 Table 7（其全参 top-8 秩也 ≈0°）。
- 输出：`R3_theta_w.csv`。

### R3-4（P0，96G 探针战役）— GetSlice v3：样本级误差带 + X 侧 ER + anisotropy
一次战役，三个产出（沿用 r2 的 7-ckpt 网格与电池）：
1. **per-probe grams 落盘** → 真探针样本级 bootstrap 误差带，替换 T6 的"跨域×模块"伪重复 CI
   （有效 n=7 的问题）——**L18 headline 的统计地基，本轮最重要单项**；
2. X 侧白化谱正式产出（覆盖 R3-2 预览）；
3. raw 构念加算 **anisotropy**（2605.30524 的另一 headline 指标，同批 hidden states 顺带）。
- 输出：`R3_er_sample_bands.csv`、`R3_xcond_whitened_er.csv`、`R3_anisotropy.csv`。

### R3-5（P1，96G 生成+探针）— S/X/H 新探针轴
- **严格按 `geometry_metric_definitions.md` 新增的 S/X/H 语义与 4 条预注册规则执行**：
  - S = base 各域生成（跨臂共享，一次生成）；
  - X = 真实训练信号：OPD 侧**逐 checkpoint 用该 ckpt 自身 rollout 生成**（math 域）；SFT 侧 = 数据集 CoT（现成）；
  - H = 各 ckpt 自生成非训练域文本（两臂；SFT 另含 math 自生成）。
- 生成参数冻结、版本化、入 manifest；两套分类法 ER **分表**，禁止与旧 S/X 混用。
- 输出：`R3_sxh_er.csv` + 生成语料 manifest。

### R3-6（P1，轻评测）— OOD 面扩展
- 两臂 × 7 步：**IFEval**（规则判分，短生成）+ **TruthfulQA-MC1**（loglikelihood）。
- 预注册定位：**preservation 轴**；MMLU-Pro 仍是唯一判别性 OOD 主轴，禁止事后升 gate（Cycle 08 禁项 #9）。
- 输出：`R3_ood_expansion.csv`。

### R3-7（P2，重评测，GPU 空闲时）— ID 轴补全
- numina：step ≥40 两臂 @统一 cap 16384 重测（早期步 cap-robust 不动）；
- aime24：final + 各臂峰值步，avg@10 @16384，N=30 caveat 照标，secondary。
- 输出：`R3_id_completion.csv`。

---

## QA_cycle09.md round-3 节必问

1. R3-4 grams 存储预算（per-probe grams 体积估算与裁剪：可只在 {9,18,27} + 全 36 层谱）。
2. R3-5 生成参数：各域 n、长度上限、采样温度；H 的非训练域取哪几个（建议 MMLU-Pro 式 + 通用文本两域起步）。
3. R3-7 numina 是否维持 N=1024。
4. 执行序确认（建议：R3-1/2 当天出 → R3-4 → R3-3/6 穿插 → R3-5 → R3-7）。

## 完成后交回 Theory 的裁决清单

| 项 | 裁决问题 |
|---|---|
| R3-4 样本级误差带 | L18 uptick（OPD +0.32 / SFT +3.43）在真样本级 CI 下是否仍显著 → R2 最终定稿 |
| R3-2/4 X 条件化 ER | OOD 域条件下白化谱是否复现压缩/暂态签名 → "压缩挤占 OOD 方向"首次同域读数 |
| R3-1 模块拆解 | uptick 的模块定位 → 机制叙事细化 + 未来干预的作用域 |
| R3-3 θ_r^W | 近零预期是否成立 → 双空间对照表定稿（vs 2509.12235 Table 7） |
| R3-5 S/X/H | L18 暂态出现在 S/X/H 哪种条件化下 → 监督通道观测轴首批读数 |
| R3-6 | IFEval/TruthfulQA 是否出现 preservation 差异（仅记录，不升 gate） |

护栏照旧：产物落 `mini/`，handin 追加 `code/code_evolution.md`，claim 门禁不豁免。

---

## 追加任务（2026-07-11，Q1–Q5 讨论后新增；全部零 GPU / CPU 级，从现有产物计算）

规格与预注册预测见 `theory/geometry_metric_definitions.md` 的 **"Output-Precision Metrics + Discriminability Standard (Round-3 additions)"** 节——执行前必读。

### R3-8（P0）— 输出精度指标三件套
1. **域条件化 tail_energy(r) / 功能秩 r_ε**：对现有各域白化谱计算（ε 建议档 {0.05, 0.01}，r 网格沿用）；重点核对预测 P-tail-1（SFT X_ood 在 step_20 的尖峰 vs OPD 平滑下降）。
2. **模块输出漂移** `‖ΔW·X‖_F / ‖W₀·X‖_F`：用 `tr(ΔWᵀΔW·Σ_X)` 恒等式从 R3-4 已存 per-probe 输入因子 + BA 计算（SFT fp32；OPD top-32 近似标 caveat），逐（模块×层×域×步）。
3. **输出空间分解**：`‖ΔW·X‖²` 在 base top-r 输出方向内/外的能量比（核对 P-tail-2 eviction 检验）。
- 输出：`R3_output_precision.csv`（三指标合表）。

### R3-9（P0）— 配对判别力检验（Q4 标准执行）
- 在同批探针样本上分别计算 whitened 与 raw 的无量纲判别力 d_space = |mean(OPD)−mean(SFT)| / 合并 bootstrap SD，对 **d_whitened − d_raw** 做配对 bootstrap（95% CI）；
- 同时给激活侧 θ_r 与权重侧 θ_w 各补误差带（θ_r 用探针 bootstrap；θ_w 用模块间分布 + identity 地板扣除说明）。
- 输出：`R3_discriminability.csv`。CI 排除 0 与否**只记录**，"优越性"结论交 Theory。

### R3-10（P1）— 暂态共位矩阵 + 行为侧统计
- 汇总对齐表：{几何 uptick 步、ID dip 步、H/ood ER 暂态步、IFEval/MMLU-Pro/TruthfulQA 各自的局部低点步} × 两臂；
- IFEval 与 MMLU-Pro 的 dip 步 delta（vs step_0 与 vs 相邻步）做 **prompt 级配对 bootstrap**；
- 输出：`R3_transient_colocation.csv` + 简表。定位 exploratory，不升 gate。

### R3-11（P2）— S/X/H 的 H 行加固
- H 条件化 ER 补不确定性：窗口 bootstrap（若单批生成可切窗）或补 1–2 批不同种子生成（96G 生成快）；重点核对 SFT H/ood 的 step_20 暂态（−45.7）是否稳。
- 输出：`R3_sxh_bands.csv`。
