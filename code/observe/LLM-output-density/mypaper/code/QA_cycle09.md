# QA:Cycle 09 —— mini-cycle Coder 阶段(Tier-A 纯分析 / 零 GPU / 不训练)

```yaml
phase: code_execution (mini-cycle09, E5 协议)
cycle: cycle_09_aaai_competitiveness_completion
status: answered_and_executed (mini round-1 完成 2026-07-08;见 mini/mini_round1_theory_handoff.md)
created: 2026-07-08
tasks: T1(L18 层级几何 + dip 共位裁决)、T2(B09 paired bootstrap CI)、
       T3(C09 输出控制分解)、T4(A09 adapter 优先 fp32/bf16 B@A 的 ρ)
out_dir: local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/
handin: code/code_evolution.md ；写边界:仅 local_experiment_results/ 与 code/
护栏: 不训练；不做 on/off-principal 结论(A09 双轨数据交回 Theory 裁);
      与 theory 预期不符的读数只记录、不解释。
```

按 E5 与本轮指令,我已读 theory 规格(QA_cycle09_theory 的 E1–E5、current_cycle09_intent 的 B09/C09 清单、
rho_metric_spec)并探测了文件系统。下面分两部分:(§A)我已自查确认、你不用答的事实;(§B)真正需要你拍板的实现
决策。每条给了推荐,你可以像 theory QA 那样直接回 "a"/"b"。

---

## §A —— 已确认事实(不用答,列出供你核对)

- **几何只在 {9, 18, 27} 三层**上算过(两臂、逐模块),列为 `effective_rank, spectral_gap, drift_from_base,
  xs_log_spectrum_gap`。全 36 层剖面需重跑 GPU GetSlice(Tier B),本轮范围外。所以本轮"全层剖面"= 这三个已测层。
- **per-sample 评测文件都在**,字段 `{gold, pred, ok, resp_len, finish, level, subject, id, gen}`:
  - `ok`=对错(bootstrap 用)、`finish`="stop"(终止)/"length"(截断)(非终止用)、`resp_len`、`gen`=全文(boxed 审计用)。
  - as-run(cap 4096,`cycle08 eval/step_*/`)+ 统一重测(cap 16384,`cap_unified_retest/{opd,sft}/step_*/math500/`)。
  - **两臂**都有统一 16384 的 MATH500 samples,各 n=500 → paired bootstrap 可行。
  - `id`="None",所以配对**按行号 index**(两臂以同一 seed 顺序遍历 MATH500);我会用 `gold[i]` 校验对齐,不齐就中止。
- **cycle07 `nontermination_trajectory.csv`** 列:`{task}_nonterm_pct, {task}_finish_n, {task}_trunc_but_correct,
  {task}_mean_resp_len` —— T3(C09)将严格平行,并补 as-run/finisher acc。
- **T4 adapter:OPD 的 LoRA adapter 已没了**(cycle08 convert 时 prune 了 raw,只剩 bf16 merged)。**SFT adapter 在**
  `cycle07.../checkpoints/step_*/adapter_model.safetensors`,dtype **F32**。

---

## §B —— 需要你拍板的实现决策

### QA1 —— T4 范围(因为 OPD adapter 没了)
干净 `ΔW = B@A`(E2)需要 LoRA 的 A/B 矩阵。**只有 SFT 有(fp32),OPD 的被 prune 了。**
- a.(**推荐**)本轮:算 **SFT 的 clean-BA ρ,双轨(fp32 的 B@A 和 bf16 取整的 B@A)**,对照 `rho_summary.md` 里已有的
  bf16-merged SFT ρ —— 给出我们**能干净测的那一臂**上 bf16/merge 的扭曲量。**OPD 的 clean-BA 标记为 blocked → 需
  A09 fp32 retrain(Tier B),本轮不做**,只记录。
- b. 额外用 (bf16-merged − base) 的 **top-32 SVD 去噪**近似 OPD 的干净 ΔW 并报告,明确标注为近似(非真 B@A)。
- c. T4 整个跳过,等 A09 retrain。

**你的回答:**
b，虽然收到精度影响，但是这个成本不高，不妨先试试看。

### QA2 —— T1 headline 层选取 + dip 裁决
"headline 层 = OPD-vs-SFT 判别效应最大的层(期望 L18)",全层剖面强制报。
- 选层判据(在 {9,18,27} 里):a.(**推荐**)逐层取"轨迹上 |ER_OPD(step) − ER_SFT(step)| 的最大值(跨模块均值)"
  —— 两臂 ER 轨迹分离最大的层;无论选哪层,三层曲线 + argmax/argmin(ER、xs_gap、drift)表都全报。
  b. 换判据(请说明,比如以 xs_gap 或 bump/no-bump 对比为准)。
- dip 裁决(两臂、逐层):报 OPD step_5 在**层级上有没有 ER uptick**(ER 的 argmax 是否落在 step_5 附近)对比 SFT
  step_20 的 bump;θ_r 一列留 "待第二轮 UV"。可以吗?

**你的回答:**
a，可以

### QA3 —— T2 paired bootstrap 参数
- draws / seed / CI 方法:a.(**推荐**)10000 次重采样、seed 42、**percentile** 95% CI。b. 其他(请说明;E4 提了
  bootstrap 但没定 B09 的 draws 数)。
- 5 个 CI 都基于 MATH500。除非你反对,我按以下定义:
  - **final** = step_624(统一 16384);**peak** = 轨迹上 acc 最大值(每臂各自 argmax 步);
  - **dip depth** = base(step_0) − 谷底 acc,OPD 谷@step_5、SFT 谷@step_20(as-run 4096,cap-robust);
  - **AUC** = acc 轨迹在 grid 上的梯形积分(混合 cap:早期 4096、后期 16384,与 `trajectory_scores_unified.csv` 一致),
    两臂同 x-grid;
  - **non-term peak** = 各步 MATH500 非终止率的最大值(取自 `finish`)。
  - 每项报 OPD−SFT 的 paired 差值 + 95% CI。a.(**推荐)接受这些定义**;b. 调整(请说明)。

**你的回答:**
b，为什么要这么多，我认为不需要这么多，你多测试一些比如256、1024就会知道实际上数值很稳定。
a，这些定义我同意。

### QA4 —— T3 输出控制表:用哪个 cap、哪些列
- a.(**推荐**)平行 cycle07 的 `nontermination_trajectory.csv` 列(nonterm_pct / finish_n / trunc_but_correct /
  mean_resp_len)+ 补 `as_run_acc` 和 `finisher_acc`(在 `finish=="stop"` 里的正确率),逐步。MATH500 在**两个 cap**
  都报(早期 0–20 @4096;40–624 @16384),直接对齐修正轨迹;numina/aime24 只有 as-run(没重测)。b. 单 cap / 换列。
- "boxed-before-truncation 审计" = 在 `finish=="length"` 的样本里,`gen` 中在被切之前**已出现 `\boxed{}`** 的比例
  (即虽截断仍可判分),和 cycle07 的 `trunc_but_correct` 同一思路。可以吗?

**你的回答:**
a，可以的

### QA5 —— 产物位置 + provenance
产物落 `local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/`(新目录),含一个
`mini_round1_findings.md` + 各 CSV(T1 层级表、T2 CI 表、T3 C09 表、T4 SFT 双轨 ρ)。handin 追加到
`code_evolution.md`。与理论预期不符的读数**原样记录、不解释**。a.(**推荐**)OK;b. 换路径。

**你的回答:**
a

---

**你答完后,我执行 T1–T4(零 GPU)然后停;结果交回 Theory 做 provisional reading 与裁决。**

---

# Round 2 —— 统一探针战役(单卡 32G / 零训练 / T5–T9)

```yaml
round: 2
status: waiting_user_response
spec: theory/mini_cycle09_round2_handoff.md
tasks: T5(全36层剖面)、T6(ER 误差带+采样扫描,生死项)、T7(UV+θ_r)、T8(双ER)、T9(电池v2.1)
```

已读:round2 handoff、`current_theory_update` Cycle09+Mini-Round1 裁决(R1–R4/R2b)、`geometry_metric_definitions`
Cycle09(θ_r 规格/双ER 构念表/电池 v2.1)、round-1 产物、GetSlice 机械。**headline 层已定 L18**(判据=dip 暂态最清晰,R2b)。

## §A(round2)已确认事实
- **8B teacher 在** `/root/autodl-tmp/model/Qwen/Qwen3-8B`(5 分片)→ X_teacher 生成可行。**GetSlice batch=1 是硬约束**
  (FINDING_06:>1 会崩)→ per-probe 前向,采样数直接乘时间。现有探针 X_base/S 各 **32 条**。SFT 各步需现场 merge adapter。
- **UV 存储在 round-1 没开**,`slice.py` 现在只出谱、**需我加"存 top-r U/V"的代码**(θ_r 前置)。**raw-ER 是新实现**(整层残差流、λ=σ²、×1/d)。
- MMLU-Pro/AIME 本地无现成探针文件,需从数据源取(MMLU-Pro 在 HF 缓存里,AIME 历史题需另找源)。

## §B(round2)必问事项 —— 每条给了推荐

### QA-R2-1 —— UV 存储预算(handoff 必问#1)
θ_r 需要 top-r 的活化 U/V。全 36 层全存 = 几十 GB。
- a.(**推荐,微调 Theory 方案**)UV 只存**候选 headline 层 {9,18,27} × 全 10 ckpt × 7 模块 × 2 臂**(top-128,≈4–6GB);
  **全 36 层各处只存谱**(ER/xs_gap/drift);其余 33 层不存 UV。**理由:θ_r 裁决树要 θ_r 跟踪 MMLU-Pro Δ(在后期步
  40–624 才动),Theory 原案把 UV 限在 dip 邻近 {0,5,10,20,40} 会导致后期步算不出 θ_r → 跟不了 MMLU。** 若 36 层剖面
  发现比 L18 更清晰的暂态层,再对那一层补存 UV。
- b. 就按 Theory 原案({9,18,27} × {0,5,10,20,40} full UV,其余存谱)——省空间但 θ_r 只有 dip 段。
- c. 全 36 层全存(几十 GB,32G 数据盘要看空间)。

**你的回答:**
a

### QA-R2-2 —— 电池 v2.1 实例化(handoff 必问#2)
- **X_ood_knowledge(MMLU-Pro 式,最高优先)**:a.(**推荐**)从 lm_eval 用过的 MMLU-Pro 14 子科各抽固定条数(seed42),
  总数 = 采样数 n(见 R2-4);prompt 取"题干+选项"文本、teacher-forced(不生成)。b. 换取法/条数。
- **X_math_hard(历史 AIME)**:a.(**推荐**)用 **AIME 1983–2023**(排除 2024=eval 集)题干为探针。**但本地没有历史 AIME
  文件** → 需你指路:(i) 允许我从 HF 拉某个历史 AIME 数据集(指定名),或(ii) 你提供文件路径,或(iii) 本轮先跳过
  X_math_hard(优先级最低)。
- **X_general(开放域文本)**:a.(**推荐**)从 HF 拉 wikitext/C4 切片固定 n 条(需联网),或(b) 你给本地语料路径,
  或(c) 复用某现有文本。请选。
- **X_teacher(8B 固定回复)**:a.(**推荐**)8B 在 X_math + X_ood_knowledge 的 prompt 上生成,temp0.6/top_p0.9/
  max_tokens 2048,冻结存盘。b. 换参数/prompt 集。

**你的回答:**
1. a
2. i
3. a
4. 为什么使用8B，你生成的目标是什么？

### QA-R2-3 —— raw-ER 探针集与 2605.30524 对齐程度(handoff 必问#3)
他们用 12k 多域 teacher-forced。
- a.(**推荐**)**不复刻 12k**;raw-ER 用我们电池的多域探针(X_general + X_ood_knowledge + X_math)各 n 条,teacher-forced,
  整层残差流构念——足够做"构念可比性"对话,不追求逐条复刻。b. 需要更严格对齐(请说明域/条数)。

**你的回答:**
a，但是我的问题是为什么使用teacher？它是训练集，但是我如果要是训练集的话SFT就是具体数据，OPD是学生生成轨迹。

### QA-R2-4 —— 采样数上限 / 时间预算(handoff 必问#4)
`batch=1` 下,采样数 n × 电池域数 × 20(臂×ckpt)× 36 层捕获 = 主要开销;n_samples 扫描 {8,16,32,64} 会再 ×4。
- a.(**推荐**)**扫描只在 L18 做一次**(base + 一个 dip ckpt)确认不敏感后**固定 n**,全战役用固定 n;主 **n=32**
  (预算松可 64)。这样避免全战役 ×4。请给**时间预算上限**(比如"≤6h")和可接受的主 n(32/64)。
- b. 全战役都做多档 n(贵)。

**你的回答:**
a

### QA-R2-5 —— 一次性统一跑 + 产物
按 handoff:不拆成五次,一次 GetSlice 战役(2 臂 × 10 ckpt × 36 层 × 电池 × 加大采样 × UV 开)派生 T5–T9;
产物落 `mini/`(T5_full_layer_profile / T6_er_error_bands / T7_theta_r / T8_dual_er + 电池版本化);
读数原样记录不解释,交 Theory 裁决树(T5→L18 确认、T6→R2 生死、T7→旋转vs值、T8→双ER三结局)。a.(**推荐**)OK。

**你的回答:**
a

---
**你答完后,我实现(加 UV 存储 + raw-ER + 电池构建 + θ_r/双ER)并一次性跑完 T5–T9,然后停,交回 Theory 裁决。**

## Round-2 定稿(2026-07-09 讨论后)

QA-R2-2#4 与 QA-R2-3 触发了 S/X 语义讨论。**结论(用户拍板):本轮按 handoff 原白化探针约定跑,不做 S/X/H 重设计。**
- **X_teacher defer**(用途=off-KD control,Tier B;本轮 T5–T8 用不到)。
- **S/X/H 重定义(S=base、X=checkpoint、H=half)= 好设计但 defer 至 Theory 正式化 + 独立(Tier B)一轮**;
  理由:换探针会破坏 R1 的 L18 confirm、破坏预注册裁决树、且需逐 ckpt 生成(生成密集)。详见
  `mini/probe_semantics_SXH_note_for_theory.md`。
- **本轮探针 = 原约定**:X = 共享多域参考电池(X_math 现有、X_ood_knowledge=MMLU-Pro、X_general、X_math_hard=AIME、
  X_bos 现有;**X_teacher 不含**);S = 每臂训练目标(复用 cycle07,OPD 侧 off-target 记 caveat)。
- 无 "CoT_human" 独立探针(= 就是 SFT 数据集)。
- 执行:UV 存储({9,18,27}×全 ckpt)+ raw-ER(T8)+ θ_r(T7)+ 全 36 层(T5)+ ER 误差带/采样扫描(T6,生死)。

---

# Round 3 —— 样本级误差带 / S-X-H 轴 / 补全评测（96G×1，零训练）

```yaml
round: 3
status: answered_code_in_progress
spec: theory/mini_cycle09_round3_handoff.md
tasks: R3-1(L18逐模块拆解,CPU)、R3-2(X条件化白化ER预览,CPU)、
       R3-3(权重侧theta_r^W)、R3-4(GetSlice v3样本级误差带/X侧ER/anisotropy)、
       R3-5(S/X/H新探针轴)、R3-6(IFEval+TruthfulQA-MC1)、
       R3-7(Numina+AIME24 ID轴补全)
out_dir: local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/
handin: code/code_evolution.md（仅追加 round-3 handin）
guard: 零训练；读数原样记录、不解释；不写 result/，不碰 claims 文件；裁决交回 Theory。
```

已读 round-3 handoff、Cycle09 的 Mini-Round1 Measurement Rulings 与 S/X/H Probe Semantics、
`current_theory_update` 的 Addendum 2026-07-10、round-1/2 产物和现有 GetSlice 机械。

## §A(round3)已确认事实（不用答，供核对）

- **R3 尚未开始**：目前没有 `R3_*` 产物；本节是执行前的正式 QA 停点。
- **R3-4 是 Round 2 T6 的统计修正**：T6 的跨域×模块 CI 不是独立探针样本 bootstrap，且白化 ER 的
  S 侧谱使五个 X 域重复。R3-4 必须保留可按探针重采样的充分统计量，才能回答 L18 uptick 的真样本级 CI。
- **S/X/H 与 Round 2 旧 S/X 分类分表**：新分类严格按几何文件的定义执行，不能把两套 taxonomy 的 ER 混合比较。
- **GPU 预检**：转交文件指定 96G×1；上一 Coder 会话曾观测到 32G RTX 4080 SUPER。实际运行 GPU 任务前我会记录
  `nvidia-smi` 的型号、总显存和可用显存；若仍是 32G，则在 R3-4/R3-5/R3-7 前停下请你确认，R3-1/2 不受影响。

## §B(round3)必问事项

### QA-R3-1 —— R3-4 per-probe grams 的落盘范围与预算

现有 4B 模型中，若把每个样本的七模块 `d×d` float64 Gram 直接落盘，单个三层样本约 3.0 GiB，
完整电池会到数 TiB，不能采用。可改存能精确重建 `G_i = H_i^T H_i` 的逐样本 activation-factor：
每层存四份不重复的投影输入（q/k/v 共享、o、gate/up 共享、down）；重采样时以原生
fp16 activation 转 fp32 后再合成 Gram。

- a.（**推荐**）**S + 五个既有 X 域**、两臂×7 ckpt、每域 n=32，在 **{9,18,27}** 存 native-fp16
  factor 和样本 provenance（bootstrap 中转 fp32）；全 36 层只存聚合谱。按全部 512 token 的
  实测维度后的上界约 **146 GiB**，预留 180 GiB（当前 `/root/autodl-tmp` 可用约 499 GiB），足以同时给
  headline 和 L9/L27 对照做真样本 bootstrap。
- b. 只在 **L18** 存同样 factor（保守上界约 **78 GiB**），全 36 层仍存聚合谱；能完成 headline 裁决，
  但 L9/L27 只保留点估计、无样本级带。
- c. 只存 **S 侧** 的 {9,18,27} factor（约 **39 GiB**）；可修正 T6 的 S 侧 uptick CI，X 条件化 ER 仍只报
  点估计，不能为其提供样本级带。

**你的回答：**
a

**实现更正（2026-07-10）：** 原题把 `gate/up` 误并入 `q/k/v` 输入；Qwen block 实际有四条输入流。smoke 实测维度为 q/k/v=2560、o=4096、gate/up=2560、down=9728。选项 a 的实验范围不变；native-fp16 保存的是模型的原始激活值，重建 Gram 时仍沿用 GetSlice 的 fp32 累积路径。


### QA-R3-2 —— R3-5 的冻结生成参数与 H 域

S/X/H 必须各自版本化并入 manifest。建议使 n 与现有 Round 2 电池对齐，同时把“生成长度”与 GetSlice 的
512-token profiling 截断分开记录。

- a.（**推荐**）每个域/ckpt **n=32**；生成 `max_new_tokens=1024`，`temperature=0.6`，`top_p=0.9`，
  固定 prompt 抽样 seed=3；进入 GetSlice 时统一截断为 512 token。H 的非训练域取
  **MMLU-Pro 式知识题干 + 通用文本** 两域。S=base step-0 在各域生成一次且两臂共享；X=OPD 逐 ckpt math
  rollout / SFT 固定 dataset CoT；H=两臂逐 ckpt 自生成，SFT 另含 math 自生成。
- b. 调整 n、生成长度、采样参数或 H 域（请直接写参数）。

**你的回答：**
a

### QA-R3-3 —— R3-7 Numina 统一 cap 的样本数

- a.（**推荐**）维持 **N=1024**：两臂、step≥40、统一 cap=16384；AIME24 仍为 final+各臂峰值步、avg@10，
  并保留 N=30 的 secondary caveat。
- b. 改为其他 N（请写数值）；我会把与 N=1024 的可比性限制写入 provenance。

**你的回答：**
b，正式 Numina 改为 N=200。先用固定 64 题、两臂最终 checkpoint 做 cap pilot，
比较 12288 / 16384 / 24576 的 paired accuracy、truncation 与 boxed-before-truncation；
正式任务采用满足“相对下一档准确率差不超过 2pp”的最短 cap。AIME24 固定 24576，
final + 各臂 peak、avg@10，仍保留 N=30 secondary caveat。

### QA-R3-4 —— 执行序确认

- a.（**推荐**）`R3-1/2` 当天先出（CPU）→ `R3-4` → `R3-3/6` 穿插 → `R3-5` → `R3-7`。
- b. 调整顺序（请说明依赖或优先级）。

**你的回答：**
a

---

**你答完以上四项后，我才会实现并执行 Round 3；每项产物落 mini/，完成后按转交文件的裁决清单交回 Theory。**

---
# Round 4 —— 窗口 v2 战役 / M1-M2-M3 指标套件（96G×1，零训练）

```yaml
round: 4
status: answered_storage_audit_before_code
spec: theory/geometry_metric_definitions.md#round-4-spec-2026-07-12-probe-nomenclature-sexh-windowing-v2-metric-suite-final-form
handoff: theory/mini_cycle09_round4_handoff.md
supersedes: round-3 handoff 的追加任务节；R3-8/9/10 机械保留但正式版只用 v2 数据
layer_scope: spectra_and_factors=[9,18,27]
layer_override: 用户 2026-07-12 裁定；覆盖 round-4 handoff 第 50 行的“全 36 层谱”
order: R4-0(CPU) -> R4-1(v2 campaign) -> R4-2(M1/M2/M3) -> R4-3(rederivations)
out_dir: local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/
handin: code/code_evolution.md（仅追加 round-4 handin）
guard: 零训练；读数原样记录、不解释；不写 result/，不碰 claims 文件；裁决交回 Theory
```

已核对 `coder01` 断点：该会话只读取了 round-4 handoff，随后额度中止；尚未追加本节、
实现代码或启动 R4。`code/current_code_brief.md` 仍为 Cycle 08 旧内容，本轮不作为规格来源。

**层范围更正：** Round 2 已完成全 36 层筛查并据此固定 L18 headline、L9 confirmatory、
L27 终态支持层。Round 4 的目标是窗口 v2 robustness 与指标重推导，不重复做层发现；
因此所有四型探针的谱和逐样本因子均只计算 `{9,18,27}`。

## §A(round4)开工前成本估计（2026-07-12）

Round 3 实测：84 个“模型×探针”格、`n=32`、单个 512-token 窗、全 36 层谱及
`{9,18,27}` 因子耗时 3h25m，因子占 146 GiB；Round 3 的三层 S/X/H 战役 91 个
“模型×探针”格实测约 31 分钟。按 Round-4 四型探针、3 个生成种子、固定/外部文本重探、
长生成上下文、三层谱/因子及 M1/M2/M3 双轨折算：

| 阶段 | 96G×1 墙钟估计 | 说明 |
|---|---:|---|
| 实现、静态检查、smoke | 2.5–4 h | 含 mask、随机窗、三层归一化、双 seed manifest 与断点续跑 |
| R4-0 CPU 审计 | 0.2–0.5 h | v1 每个（探针, 域, 臂）格的 prompt-token 占比 |
| R4-1 生成 | 1.5–3 h | 共享 base/step-0 去重；正式生成仍保留 3 seed |
| R4-1 v2 三层前向、谱与因子 | 6–10 h | 仅 L9/L18/L27；样本等权层级 gram，长上下文仍是主耗时 |
| R4-2 M1/M2/M3 | 3–6 h | 仅三层，尽量随 R4-1 流水派生以避免重复前向 |
| R4-3 三项重推导 | 4–7 h | 样本 bootstrap、配对判别力、暂态共位矩阵 |
| 校验、manifest、handin | 0.5–1 h | 完整性矩阵与 Theory 裁决清单 |
| **合计** | **约 18–32 h（中心约 25 h）** | smoke 后用实测速率再收紧 ETA |

若永久保存所有三层、三窗、三生成种子的 native-fp16 逐样本因子，粗估需约
2.0–2.7 TiB；当前 `/root/autodl-tmp` 只余约 305 GiB，因此当前磁盘无法采用全量永久归档。

## §B(round4)必问事项

### QA-R4-1 —— v2 生成与三层因子的存储预算

所有方案的**谱与逐样本因子均只做 `{9,18,27}`**，不重跑全 36 层；已有
`/root/autodl-tmp/cycle09_r3`（约 194 GiB）不动。三层限制大幅减少计算，
但逐窗因子仍是存储主项。

- a.（**推荐**）采用**滚动 scratch**：永久保留生成文本、窗口索引、聚合谱、bootstrap draws、
  M1/M2/M3 表和 manifest；native-fp16 因子按同探针/配对格临时保存，完成派生、数量核验和
  可复算抽检后只删除 **R4 临时因子**。scratch 峰值硬限制 **120 GiB**，预计最终新增低于 20 GiB。
- b. 永久保留全部 `{9,18,27}` 因子；需先提供约 **2.0–2.7 TiB** 的额外磁盘路径，当前盘不能跑。
- c. 指定其他 scratch 上限或选择性永久保留范围（请写预算与探针/层）。

**你的回答：**
a


### QA-R4-2 —— k=3 的短生成边缘处理

生成区 `>=512` token 时严格取 k=3 个均匀随机 512-token 窗；生成区在
`[512,1536)` 时允许重叠并仍记实际 `k=3`。必问边缘是生成区 `<512` token：

- a.（**推荐**）保留样本，取覆盖全部生成 token 的**单个变长窗**，记 `actual_k=1`、实际 token 数；
  prompt 仍不进统计。若生成区为 0 token，才记缺样本并写入 manifest。
- b. `<512` token 一律记缺样本/缺格，不进入 gram 与 bootstrap，并逐格报告缺失率。

**你的回答：**
a


### QA-R4-3 —— E_math 本轮缓做确认

- a.（**推荐**）确认本轮跳过可选的教材式 `E_math`；**保留并正式执行 `E_math_hard`**
  （AIME statements）。manifest 明记 `E_math=deferred_by_user`。
- b. 本轮也执行 `E_math`（需先指定固定语料版本/路径和样本数）。

**你的回答：**
a


### QA-R4-4 —— 位置 early/mid/late 三分箱确认

- a.（**推荐**）确认默认输出：每窗保存相对起止位置与中心位置，并按生成区内归一化中心位置
  分为 early/mid/late；逐格输出三分箱读数、样本数和实际窗数。总体主结果仍按窗→样本→语料
  三层归一化，三分箱只作诊断，不改变样本权重。
- b. 不输出三分箱（仍保留逐窗相对位置）。

**你的回答：**
a


---

**请先回答 QA-R4-1～4。收到四项确认后才实现与执行；正式顺序固定为 R4-0 → R4-1 →
R4-2 → R4-3，R3-8/9/10 不会在 v1 窗口数据上跑正式版。**

---

## Round-5 QA（coder → 用户，2026-07-14）

> 回答方式：直接在每题的 **【答】** 行后面写。写完告诉我，我读文件后开工。

### Q5-1 — B2 全 7×7 矩阵的存储/时长预算（转交文件必问 1）

**实测依据**：R4-1 战役 14 次权重加载 × 28 探针任务 = 2h25m，即 **~23 s / 32-样本任务**、**~10 min / 权重加载**。

**B2 预算**：7 权重 × 7 文本源 × 2 臂 = 98 格；每格 3 生成种子 → 每次权重加载要跑 21 个任务 ≈ 8 min + 加载 1.5 min ≈ **10 min/权重** × 14 = **2.5–3 h**。**不需要裁剪**（默认全跑）。

**存储**：谱 json 约 1–2 MB/格 × 98 ≈ 200 MB（可忽略）。**但 gram 是大头**：若每格存 gram（3 层 × 7 模块）≈ 420 MB/格 × 98 ≈ **41 GB**。数据盘现剩 198 GB。

- 【建议】**只对「对角线 + 第一列」28 格存 gram（≈12 GB）**——它们是 B3 Mismatch 和 A2 逐-ckpt 轨唯一用得到的；行内其余格只需谱（B2 的 ER/r_ε/tail/γ 都能从谱算）。
- 【答】按照你的建议

### Q5-2 — B1 的 SFT 自生成 prompt 池（转交文件必问 2）

转交文件建议：**与 `legacy_S_math` 同一批题面**（保证与静态参考同源）。

- 【建议】采纳。即：取 `legacy_S_math` 的 32 道题面（NuminaMath CoT 的 question），两臂都用同一批；SFT 侧生成 = "SFT 自己怎么解这些题"，静态参考 = 数据集对同题的 CoT。这样 B3 的 Mismatch 是**同题面**下的差，题面效应被消掉。OPD 侧沿用 R4 已有的 X_OPD rollout（其 prompt 池是 numina math，**与 legacy 是否同一批需我核对**——若不同，我会补一版同题面的 OPD rollout 以保证两臂可比，代价 +20 min GPU）。
- 【答】按照你的建议

### Q5-3 — r_ε 查表实现（转交文件必问 3）

A1/A2 要求「逐格用各自的 r_ε，不得用全局均值、不得 k=1」。r_ε 已在 `R4_m1_tail_ec.csv` 的 `r_epsilon_current` 列（31752 行，键 = arm × step × task_id × track × layer × module × epsilon）。

- 【建议】建一个 `R5_reps_lookup.parquet` 查表（同键），A1/A2/B2 全部从它取 r_ε；ε=0.05 为主、0.01 为副；**并行报 k=64 固定对照**（转交文件要求）。缺格（若有）记为 NaN 并在 manifest 里列出，不回退到均值。
- 【答】按照你的建议

### Q5-4 — A6 raw 修复口径（转交文件必问 4）

转交文件给的公式：`erank~ = d^{-1}·exp(−Σ pᵢ log(pᵢ+ε))`，`pᵢ = λᵢ/Σλⱼ`，报**归一化值**以便与 2605.30524 Fig 2 直接对照。

**我这边的现状**：`scratch/references/*.pt` 里存了 raw 二阶矩（`residual_second` / `residual_mean`），**base 侧**的 raw 协方差特征值可以直接算（零 GPU）。但**逐 ckpt 的 raw 协方差没存** → 逐 ckpt 的归一化 raw ER 需要 GPU 重跑（可搭 B2 顺带存，+0 额外加载）。

- 【建议】(a) ε 取 1e-12（数值保护，不影响量级）；(b) 用**中心化**协方差（减 `residual_mean`，与 anisotropy 复现口径一致）；(c) base 侧先出（CPU），逐 ckpt 的**搭 B2 便车**一起存；(d) 同时报前 5 个特征值能量占比（验证 outlier 维度主导假设）。**是否需要我去查原文其他细节**（如它是否中心化、是否只取残差流而非模块输入）——若你有原文，请指出；否则我按上述口径执行并在产物里标注"口径自定，未核原文"。
- 【答】是的，按照你的建议

---

## off-KD 阶段 1 · 第 0 步阻塞检查结论（coder，2026-07-14）

**问题**：verl 的 `forward_kl_topk` 消费的 teacher logprob 是 raw 还是 processed？

**结论：RAW（temperature=1.0 的 log_softmax），不是 processed。**

证据（verl 源码）：
1. 学生侧 `verl/trainer/distillation/fsdp/losses.py:compute_forward_kl_topk` →
   `student_log_probs = F.log_softmax(student_logits, dim=-1)`，**无 temperature/top-p 处理**。
2. teacher 侧 `verl/experimental/teacher_loop/teacher_manager.py:40-55` → 用 vLLM 的
   **`prompt_logprobs`** 且**强制 `temperature: 1.0`**。源码注释：
   *"Temperature has no effect on prompt_logprobs: the teacher performs a forward pass over
   existing tokens (no sampling). Always use temperature=1.0 regardless of the config value."*
   并警告 distillation.yaml 默认用 Hydra 插值抄 student 的 rollout temperature 会导致 spurious crash。
3. `verl/workers/config/rollout.py:196` 的 `logprobs_mode='processed_logprobs'` **只作用于 student
   rollout**（PPO ratio 需要 processed），**与 teacher 无关** —— 这正是转交文件担心的混淆点。

**⇒ 对 rollout 的执行含义（两遍）**：
1. 采样生成：`temperature=0.6 / top_p=0.9 / top_k=-1 / n=1 / seed=42 / max_tokens=10240`；
2. **另做一遍前向**：对 `prompt + generation` 用 `prompt_logprobs=32, temperature=1.0` 取 **RAW** top-32。
   **不可**在采样时用 `logprobs=32` 顺手取（那是 processed 口径，KL 会算错 ⇒ 实验作废）。

---

## off-KD 阶段 2 · 实现、smoke 与正式挂载（coder，2026-07-15）

**规格来源说明**：仓库中只有 `theory/offkd_rollout_handoff.md`（Stage 1），其正文明确说
Stage-2 另发，但未找到独立 Stage-2 转交文件。本实现因此只采用该文件末尾摘要，并逐项回查
Cycle08 正式 Hydra 配置 `outputs/2026-07-02/10-40-01/.hydra/config.yaml`：
Qwen3-4B-Base、LoRA r32/alpha64/all-linear、AdamW lr5e-5、batch16、2 epochs、
shuffle=false、token-mean、forward_kl_topk=32、clamp(-10/0/10)、624 步、
checkpoint {0,5,10,20,40,160,624}。

**Stage-1 正式 pass1 已完成**：5000/5000；34,486,621 response tokens（均值 6897.3）；
length 截断 1072/5000=21.44%；boxed 84.44%；实测 352.0 min。pass2-only 续跑新增
`--pass1-minutes`，防止 manifest 把已完成的 pass1 时间错误写成 0。

**pass2 smoke**：8/8、50,173 token；exact pass1 token sequence 不变；top-32
`(50173,32)`，offset 全对齐，missing id=0，NaN/Inf=0，正 logprob=0，排序错误=0。

**训练 smoke（最长样本）**：10240 response tokens；完成真实 forward/backward/clip/AdamW；
loss=0.3174503，grad norm=0.578125，2.97 s，峰值显存 18.26 GiB。checkpoint 0→1：
504 个 adapter tensor 中 252 个发生变化（LoRA B 先更新，符合零初始化），全部有限，
max |delta|=5.054e-5；自动恢复从 step1 正确判定流程已完成、不重复更新。

**正式挂载**：`screen: cycle09_offkd`，脚本 `watchdog_cycle09_offkd.sh`；严格串行
`formal_pass2 → validate → smoke(skip if complete) → formal train`。任一步失败立即停止、
不自动重试；状态 `/root/autodl-tmp/cycle09_offkd/watchdog_status.json`；不自动关机。

### pass2 主存事故与流式修复（2026-07-15）

第一版正式 pass2 把 5000 条请求一次性交给 `llm.generate`，使已完成请求的
`prompt_logprobs`（约 3449 万 token × top-32 的 Python dict/Logprob 对象）持续常驻
主存。日志停在约 1947/5000，随后 GPU 利用率归零、进程无输出约 22 min，实例整体重启；
重启后 dead screen、无 pass2 正式产物、pass1 文件仍完整。内核 OOM 日志在容器内无权限
读取，但重启时间、对象增长路径和停顿形态共同指向 host-RAM OOM，而非 GPU OOM/网络中断。

修复：pass2 改为每批 4 条，返回后立即抽取到磁盘 `.npy` memmap，flush 后原子记录
`next_record/cursor/source_sha256`，释放 Python 对象；每批检查 `MemAvailable`，
低于 128 GiB 主动失败并保留断点。流式 smoke 每批 2 条时可用主存稳定在
874.0–874.3 GiB；重跑前后 rollout JSON、top-32 NPZ、pass1 JSON 的 SHA-256 三者
完全一致；完成态重入直接从 8/8 结束，不重做 teacher forward。
