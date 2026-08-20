# Theory → Coder 转交：Stage 1（确定，立即执行）+ 夜间块（暂定）

```yaml
view_type: handoff_note
cycle: cycle_09 → 判别电池阶段
created: 2026-07-17
status:
  stage_1: CONFIRMED —— 立即执行，当前 96G×1
  night_block: TENTATIVE —— 今晚 2×96G，需 Theory 在切换前发 GO（内容可能随 Stage 1 结果修订）
  stage_3_4: PLACEHOLDER —— 仅供预告，勿按此执行
guard: 全部任务只回报原样读数，不解释、不裁决；预注册见各任务内嵌条款
context: T1{0,624}/T2/T3 已完成并裁决（offkd_geometry_handoff.md §7）——G1 升正式层、
  H-opd-no-transient 严格口径成立。本文件是其后续。
```

---

# Stage 1（确定）—— 审计 + 地基，当前单卡 + CPU 并行

执行顺序建议：CPU 件（S1-1/2/6/7）先发后台，GPU 件按 S1-3 → S1-4 → S1-5。

## S1-1 — MMLU-Pro 提取失败审计【CPU，~1h】

lm-eval 的逐题输出已在盘（`{eval_root}/lm_eval/step_*/mmlu_pro`，off-KD 在
/root/autodl-tmp/cycle09_offkd/ 下；OPD/SFT 用其既有正式评测的对应目录）。
对**三臂 × 十点**统计：
- 提取失败率（正则链没匹配到任何 A–J 字母的样本占比）；
- 失败样本的形态分布（空回答 / 无字母 / 有字母但格式不合 / 截断）；
- 回答长度分布（mean/median）随 step 的变化。

产物：`mini/S1_mmlupro_extract_audit.csv`（arm × step × 上述字段）。

## S1-2 — MMLU-Pro flexible 提取链离线重评【CPU，~1h】

对已存生成文本套宽松提取链（按序回退）：
1. `answer is (X)`（现行）→ 2. `Answer: X` / `答案…X` → 3. `\boxed{X}` → 4. 文末最后一个独立大写字母 A–J。
重算 exact_match，输出三臂 × 十点的 `mmlu_pro_flexible` 与原 `exact_match` 并列表。

**预注册（Theory 冻结）**：损伤分解 = 知识成分（flexible 口径下仍在的跌幅）+ 格式成分（原口径与
flexible 口径的差）。判定影响 claim 31 的行为标签，Theory 裁决。

产物：`mini/S1_mmlupro_flexible.csv`。

## S1-3 — 暂态窗样本级 CI【GPU 2.5–6h，本阶段主件】

T1 的暂态窗补全：**E_ood 单探针 × step {5,10,20,40,80} × 三臂**，keep_factors=True，
draws=256，配对 bootstrap，L18、7 模块均值。输出各步 ER 相对 base 偏移的两两臂差 CI
（追加 R5_bootstrap_ci.csv 同 schema）。

**预注册用途（冻结）**：claim 31 幅度排序的 CI 判定——
off-KD 峰值(+2.42@20) 与 SFT 峰值(+1.82@40) 的差是否排零；OPD 各步偏移 ≤0 是否在 CI 意义下成立。

## S1-4 — wikitext PPL 轨迹【GPU ~1h】

三臂 × 十点 × wikitext 固定切片（与 E_general 探针同语料族；固定 tokenization 与切片 seed，
记录 provenance）。产物：`mini/S1_wikitext_ppl.csv`（arm × step × ppl）。

**预注册（Theory 冻结）**：off-KD 的 general 域相对损伤（PPL 相对 base 的恶化比例）**小于**其
MMLU-Pro 相对损伤——与"E_general 无暂态"的域选择性一致。若 off-KD 的 PPL 恶化 ≥ 其
MMLU-Pro 相对损伤 → 域选择性的行为面证伪，Theory 重审。

## S1-5 — 三种训练文本的 base-PPL【GPU ~0.5h】

base 模型对三臂训练语料各取同规模样本（≥500 条、同截断规则）算 PPL：
(a) X_OPD（自身 rollout 文本，用 step-0/最早可得的 rollout 存档）；(b) X_SFT（数据集 CoT）；
(c) X_offKD（teacher rollout 文本）。产物：`mini/S1_train_corpus_base_ppl.csv`。
用途：陌生度排序（H2/H5 输入），只出读数。

## S1-6 — R6-2 方向分析【CPU，数小时，可后台】

从已存 UV/谱 artifact（fp64 θ 管线的中间产物）计算，三臂 × 关键步 {5,20,40,160,624} × L18：
1. **主角谱全量**：每格输出全部 r_ε 个主角 θ₁…θ_r（现只有 max/mean）；稀疏度 = 转动 >1°/>5° 的方向数占比；
2. **大转动方向的 base-σ 排名**：top-10 转动方向在 base 谱里的排名分布；
3. **臂间重叠**：两两臂的"大转动方向集合"（>5°）重叠率。

**预注册（沿 round6_queue.md R6-2 原文，冻结）**：H-where——OPD 重定向高 σ 核心方向、
SFT 重定向低 σ 边缘方向；三臂版补充预测：off-KD 与 OPD 的大转动方向集合**低重叠**（同量不同向）。
裁决树按 round6_queue.md R6-2 三分支执行，Theory 裁决。

产物：`mini/S1_direction_analysis.csv` + 排名分布表。

## S1-7 — H 生成文本退化度审计【CPU，~1h】

对已存 H/B1 生成文本（R4/R5 的 H_bos/H_general/H_ood + B1，OPD/SFT 七步）计算：
长度、4-gram 重复率、distinct-2、截断率，按（probe × arm × step）出表。
用途：闭环 r_ε 骤降（OPD H_bos@5 −27 / SFT H_bos@20 −155）的文本侧刻画——
回归"文本统计能解释多少 r_ε 变化"，只出读数。
产物：`mini/S1_h_text_stats.csv`。

## Stage 1 回报格式

每任务一张原样读数表 + 一行 provenance；不解释。handin 追加 code_evolution.md。
完成即回交，**不要等夜间块指令**——夜间块以 Theory 的 GO 为准。

---

# 夜间块（暂定 —— 今晚 2×96G，切换前需 Theory GO）

⚠️ **本块内容可能随 Stage 1 结果与 Theory 侧讨论修订，以切换前的最终版为准。**
seqKD 的行为/几何预注册树由 Theory 在 GO 之前冻结并追加至本文件（占位见下）。

## 切机同步清单（预留 ~1h）

- teacher rollout jsonl + top-32 logprob memmap（~10GB，seqKD 训练 + X_offKD 探针都要）
- 四组权重：base 模型、OPD merged（十点）、SFT adapters（十点）、off-KD adapters+merged（十点）
- 探针语料全套 + `scratch/references/` base 白化缓存（避免重算 base 前向）
- 评测数据集（MATH500/GPQA/MMLU-Pro/IFEval/TruthfulQA/Numina）+ lm-eval 环境
- 仓库 + conda env（对齐 off-KD 训练用的 HF+PEFT 环境）

## 夜间排程（两卡分工）

```
T+0h    卡A: N-1 seqKD 训练 (3.5h)
        卡B: N-2a X_offKD 探针注册+十点 (~1.5h) → N-2b off-KD H_bos/H_ood 七步含生成 (~3h)
             → N-2c T5 off-KD raw ER (~1.5h)；N-2d T4 off-KD 权重空间 ρ/θ_w (CPU 随行)
T+6h    两卡分单双 ckpt: N-3 seqKD 评测电池 十点×五任务 (~3.8h)
T+10h   两卡分单双 ckpt: N-4 seqKD 几何五探针 十点 (~2.5h)
T+12.5h 完成；held-out eval 不进夜块（切回单卡后执行，Stage 3 暂定）
```

## N-1 seqKD 训练规格（暂定）

- 数据：**与 off-KD 完全相同的 teacher rollout 文本**（同 4,999 样本、同顺序、同截断保留）；
- loss：**标准 CE**（response-token only、token-mean）——与 off-KD 唯一差别是 loss（CE vs KL），
  与 SFT 唯一差别是数据（teacher 文本 vs 数据集 CoT）；question 掩码规则对齐 SFT（cycle07）惯例；
- 其他全同 off-KD：Qwen3-4B-Base、LoRA r32/α64/all-linear、AdamW lr 5e-5、batch 16、
  2 epoch、shuffle=false、drop-last；
- checkpoint：**原生保存十点全网格 {0,5,10,20,40,80,160,320,480,624}**（离线训练无需回放）；
- 记录：loss 轨迹、grad norm（与 off-KD 训练表同 schema）。

## N-2 ~ N-4 规格要点（暂定）

- **X_offKD**：teacher rollout 文本注册为固定探针语料（与其它探针同 v2 窗口协议；取样规则
  与 X_SFT 一致：response 区、question 掩码）；一次 base profile + 三臂无关（它只属 off-KD 的 X 列，
  但作为固定语料照常测四臂——命名 X_offKD，arm 列如实记录被测臂）；
- **off-KD H_bos/H_ood**：七步 {0,5,10,20,40,160,624}，生成参数对齐旧臂 H 协议（同 prompt 池、
  同采样参数、3 种子若旧协议如此）；测量走 v2 全套；
- **T5 raw ER**：严格按 R5-A6 的 2605.30524 复刻实现，off-KD 十点；
- **T4 权重空间**：ρ 双轨 + θ_w canonical 角，off-KD fp32 adapter 轨，schema 对齐 T4_rho/R3_theta_w；
- **N-3 评测**：与三臂正式评测同协议（MATH500 cap 规则、MMLU-Pro --limit 100/class、
  IFEval、GPQA、TruthfulQA-MC1）；
- **N-4 几何**：五静态探针 × 十点，管线与 off-KD 那轮完全一致（adapter BA fp32、fp64 θ、缓存复用）。

## seqKD 预注册占位

（Theory 将在 GO 前冻结并追加：H4 行为裁决树 + seqKD 的 G 树几何归属判据。冻结前 coder 不需要、也不应读它。）

---

# Stage 3 / Stage 4（占位，勿执行）

- Stage 3（暂定）：ρ_core 跨条件能量分解 Phase A（判 H1/H2/H3）、overlap(X核心,E_ood核心)@step0、
  held-out Math-CoT eval（四臂×十点，N=500）；
- Stage 4（暂定）：P4 基底移植（=2509.12235 换回协议）、2606.x 权重协议严格复刻、
  MMLU-Pro loglikelihood 重评（视 S1-1/2 结果）、ρ_core(H_t) Phase B（视夜间块 H 结果）、
  X̂（训练域自生成）待用户裁定名分。

---

# 紧急追加（2026-07-17 晚，CONFIRMED —— 立即执行，优先于夜间块其余项）

背景：S1-1/S1-2 审计显示 MMLU-Pro strict exact_match 被提取伪迹重度污染
（OPD=纯截断型，off-KD=持续格式型），flexible 重评反转了"OOD 塌陷"叙事，
但 flexible 的 tier-4 末位字母回捞对截断样本不可靠。需要格式免疫的金标准终审。

## S1-8 — MMLU-Pro loglikelihood 重评【金标准，GPU ~3–5h，最高优先】

- **口径**：每题对全部 10 个选项计算条件 loglikelihood，argmax 为预测；
  **两种归一化都报**：raw LL 与 长度归一化（acc_norm 风格）。
- **题目集**：与既有评测完全相同的 1400 题子集（100/class，同 seed）——保证与
  strict/flexible 三口径逐题可比。
- **prompt**：标准 MC 格式（非 CoT 生成式）；三臂 + base 同一 prompt 模板，记录模板全文。
- **范围**：三臂 × 十点（30 格）＋ base（=step 0 共享，测一次）。
- **产物**：`mini/S1_mmlupro_loglik.csv`（arm × step × acc_ll × acc_ll_norm），
  与 strict/flexible 并成四口径对照表。

**预注册（Theory 冻结于数据未出）**：
> 知识通道判定量 = 各臂 loglikelihood acc 相对自身 step-0 的轨迹（模式内自比，
> 不与生成式口径比绝对值——两种模式测的不是同一问题）。
> H6 双通道假设的预测：三臂全部平坦或上升；具体判据：**off-KD 终点 ≥ 其 step-0 − 2 pts**。
> 若 off-KD 的 LL 轨迹也塌（终点 < step-0 − 2 pts）→ 知识损伤为真，
> H6"标签改写"裁决回滚，双通道框架重审。
> 附属：OPD 在 40–80 的 LL 轨迹若无 dip → OPD 的 MMLU dip 确认为纯伪迹。

## S1-9 — IFEval 逐指令类型审计【CPU 优先，若样本缺失才用 GPU】

- 从已存 lm-eval 逐样本输出提取（三臂 × 十点）：**按指令类型**（length_constraints /
  format 类 / content 类等 IFEval 原生分类）分解 prompt-strict 通过率轨迹；
  另附各类型样本的回复长度统计。
- 目的：判定各臂的 IFEval 下降由哪类指令驱动——长度类（=冗长度病灶的直接投影）
  vs 格式类 vs 内容类。这决定 IFEval 在"双通道 OOD"框架中的归类粒度。
- 若逐样本输出未存盘：先回报缺失清单，**不要自行重跑生成**（重跑规格另批）。
- **产物**：`mini/S1_ifeval_breakdown.csv`（arm × step × instruction_category × pass_rate × resp_len）。

回报：只给原样读数表，不解释。handin 追加 code_evolution.md。
夜间块其余项（seqKD 等）等 Theory 的 GO——本追加件不需要等。

---

# Execution Return（2026-07-17/18，Coder 写回；原样产物，不解释、不裁决）

## S1-8 / S1-9

- 状态：`COMPLETE`；S1-8 唯一模型格 `28/28`，三臂十点输出 `30` 行；S1-9 三臂十点 × 9 个 IFEval 原生类别，输出 `270` 行。
- S1-8 raw：`mini/S1_mmlupro_loglik.csv`。
- S1-9 raw：`mini/S1_ifeval_breakdown.csv`。
- 完整 raw 表 + provenance：`mini/mini_stage1_emergency_s1_89_handoff.md`。
- 校验清单：`mini/S1_89_handoff_manifest.json`。

## Pulled-forward N-2

- 状态：`COMPLETE`；完成时间 `2026-07-17 18:49:09 UTC`。
- N2-1 X_offKD：spectra `3360` / M1 `6720` / M2 `3600` / theta `4536`。
- N2-2 OPD H@80：spectra `252` / M1 `504` / M2 `270` / theta `378`。
- N2-3 off-KD raw ER：`150`。
- N2-4 off-KD H 七步：spectra `1764` / M1 `3528` / M2 `1890` / theta `2268`。
- N2-5 off-KD weight geometry：rho `756` / theta_w `945`。
- 严格完成 manifest：`mini/n2_completion_manifest.json`。
- raw 产物索引 + provenance：`mini/mini_n2_theory_handoff.md`。
- shutdown：`disabled`；未执行未获 GO 的 seqKD/night-block 任务。

---

# 第二执行块（2026-07-18，2×96G，CONFIRMED —— 本节取代前文"夜间块（暂定）"成为现行指令）

```yaml
status: CONFIRMED
gpu: 2×96G 已就位
milestones: T+7~8h 首批可分析（seqKD 关键步行为 + Llama 预检/rollout）；T+16h 大批结果
guard: 只回报原样读数；seqKD 预注册树已冻结于本节末，coder 不需要读、不应被其影响
```

## 排程（T=0 = 收到本单）

```
      卡 A                                  卡 B                          CPU 后台（立即发）
T0    G1 seqKD 训练 (3.5h)                  G4 Llama 预检 (0.7h)          C1 方向分析扩全探针
                                            G5 Llama rollout (~4.5h)      C2 剂量-响应表
T3.5  G2 seqKD 评测·关键步优先序
T5.2                                        G7 off-KD H_bos/H_ood (3h)
T8    G2 补全余格 (~2h)                     G6 Llama 三离线训练串行 (~7.5h)
T10   G3 seqKD 几何十点（单卡 ~5h）
T15   G8 adapter 分层消融（起跑，跑到哪算哪）
```

## G1 — seqKD 训练

规格 = 前文 N-1 原样（数据/掩码/超参/十点原生保存全同 off-KD，仅 loss 换标准 CE）。

## G2 — seqKD 评测（关键步优先序）

- **执行顺序**：先 {0,5,20,40,624} 后 {10,80,160,320,480}；每步内任务序 MATH500 → MMLU-Pro → IFEval → GPQA → TruthfulQA。
- 协议与三臂正式评测完全一致；**生成落盘后立即做 CPU 后处理**：MMLU-Pro strict 提取失败分解 + flexible 重评（S1-1/S1-2 同实现）、IFEval 九类别分解（S1-9 同实现）——这些是裁决口径，随批产出，不要等全网格。
- 产物追加 three_arm_full_trajectory 同 schema（arm='seqkd'）+ S1 系列审计 CSV 追加 seqkd 行。

## G3 — seqKD 几何

五静态探针 × 十点，管线与 off-KD 那轮完全一致（adapter BA fp32、fp64 θ、base 缓存复用、
谱全量单独文件 R4_v2_spectra_seqkd.csv）。

## G4 — Llama 预检（model-2 发车判据）

Llama-3.2-3B-Base 的 MATH500 N=100：zero-shot 与 4-shot（boxed 风格 prompt）各一遍。
回报两个 acc + 生成样例 5 条。判据（Theory 冻结）：任一协议 acc ≥ 0.10 且答案可提取 → GO；
双双 <0.10 → 停，回报后等 Theory 改选型（备选 Qwen2.5-3B←7B）。

## G5 — Llama teacher rollout

- teacher = Llama-3.1-8B-Instruct；prompts = 同一 5k（opd_prompts_5k 的 query，Llama chat template）；
- 采样对齐 model-1 惯例：temp 0.6 / top_p 0.9 / top_k −1 / n 1 / seed 42 / max_tokens 10240；
- **两遍法**：pass1 生成；pass2 对 prompt+generation 以 temperature=1.0 取 RAW top-32 logprob
  （与 off-KD 数据管线同实现）；记录 finish_reason / n_tokens / has_boxed / 截断率。
- 产物 {EXP_ROOT}/model2_llama/rollout/，schema 同 offkd rollout。

## G6 — Llama 三离线训练（G4 GO + G5 完成后，卡 B 串行）

三臂：SFT-llama（数据集 CoT）、off-KD-llama（teacher 文本 + forward_kl_topk32 RAW）、
seqKD-llama（teacher 文本 + CE）。全部：Llama-3.2-3B-Base、LoRA r32/α64/all-linear、
AdamW lr 5e-5、batch 16、2 epoch、shuffle=false、十点原生保存。评测与几何**不在本块**（排下一块）。

## G7 — off-KD H_bos/H_ood

规格 = 前文 N-2b 原样（七步、生成协议对齐旧臂、v2 测量全套）。

## G8 — Adapter 分层消融（off-KD@624，纯推理）

- 配置：六个层组 {0–5}{6–11}{12–17}{18–23}{24–29}{30–35} 逐组**关闭**该组全部模块的 adapter
  （其余层保留），加 全开 / 全关 两个参照 = 8 配置；
- 每配置评测（缩减集）：MATH500 N=200 @cap8192 → acc + 截断率；MMLU-Pro N=500 子集
  （固定 seed 抽取，8 配置同一子集）→ strict 失败率 + flexible acc；
- 产物：`mini/G8_adapter_ablation.csv`（config × math_acc × trunc × strict_fail × flex_acc）。
- 目的（coder 无需理会）：格式压制分量与 math 增益分量的层定位与可分离性。

## C1 — S1-6 扩展全探针【CPU 后台】

对 legacy_S_math / E_general / E_math_hard / S_bos 补齐方向分析三件套
（主角谱全量 / base-σ 排名 / 臂间重叠），与 E_ood 已有输出同 schema 追加。

## C2 — 剂量-响应表【CPU 后台】

汇一张表：每（arm × 域）行 —— x 列 = E_域探针 ER 暂态峰值与峰步（S1-3 有 CI 的用 CI 均值，
无 CI 的用点估计并标注）；y 列 = 该域格式通道损伤（MMLU-Pro strict 失败率终点增量、
IFEval detectable_format/startend/length 类终点变化、@40 崩塌深度）。只出表不回归，Theory 自己拟合。

## 回报节律

- **T+7~8h 中间回报一次**：G2 关键步行为表（含 CPU 后处理审计列）、G4/G5 状态、G7 进度、C1/C2 表；
- T+16h 总回报：全部完成件原样读数。
- handin 追加 code_evolution.md；一切只报读数不解释。

---

## seqKD 预注册裁决树（Theory 冻结 2026-07-18，数据未出；填充前文占位）

**设计角色**：seqKD 与 off-KD 同数据异 loss（CE vs KL）；与 SFT 同 loss 异数据（teacher 文本 vs 数据集 CoT）。
全部判定量为**审计后口径**。参照值取自现有三臂十点数据。

### 树 1 — H4 格式通道归因（主判定量：MMLU-Pro strict 提取失败率终点增量 Δfail = fail(624)−fail(0)）

参照：Δfail_offKD = +.352（且不恢复）；Δfail_SFT = +.165；（OPD +.182 但恢复形）。
归属规则：|Δ_seqKD − Δ_offKD| vs |Δ_seqKD − Δ_SFT| 最近参照；辅助判定量 = IFEval
detectable_format @40 是否出现崩塌（off-KD .374→.161 有；SFT 无）。
- **D1（数据驱动）**：seqKD 归 off-KD 侧且 @40 崩塌出现 → 格式重组由 teacher 文本驱动，KL 非必需；
- **D2（交乘驱动）**：seqKD 归 SFT 侧且无 @40 崩塌 → KL×off-policy 交乘才是格式损伤动力（J4 支持）；
- **D3（分裂）**：两判定量方向不一致 → 按通道拆分报告，禁单因归因。

### 树 2 — J6 "logits 是廉价补丁"（判定量：MATH500 终点 acc）

参照：off-KD .794、SFT .752。
- seqKD ≤ off-KD − 2pts → J6 成立（同数据下 logits 带来实增益）；
- seqKD ∈ off-KD ± 2pts → J6 证伪（off-KD 的 math 增益主要来自 teacher 文本，logits 边际小）；
- seqKD > off-KD + 2pts → 异常，标旗待查。

### 树 3 — F-10 终止归因（判定量：MATH500 截断率终点）

参照：off-KD .048、SFT .352。数据-EOS 假说预测 seqKD ≤ .10。
- 成立 → 终止能力归数据 EOS 定案；
- seqKD > .25 → F-10 修订（loss 类型参与终止行为）。

### 树 4 — F-1 压缩指纹的判决性检验（判定量：d(r_ε) 五探针，G 树同口径同归属规则）

参照：d_KL臂 ∈ [−24.3, −9.0]；d_SFT ∈ [−6.4, −3.3]。
**F-1 的预测（冻结）**：seqKD（CE）尽管与 off-KD 数据完全相同，压缩应归 **SFT 侧**——
压缩跟信号不跟数据。
- 归 SFT 侧（多数探针）→ **F-1 升最终形态**："同数据异信号分离 + 异数据同信号汇聚"双重控制齐全；
- 归 KL 臂侧 → **F-1 证伪**，压缩由数据驱动，G1 裁决连锁重审（重大修订，标最高旗）；
- 分裂 → 按探针域拆分。

### 树 5 — 暂态的力度×支撑集检验（判定量：E_ood ER 偏移轨迹的峰值与峰步）

参照：SFT +1.77@40；off-KD +2.23@20；OPD 无。
预测（F-4/J4，冻结）：seqKD **有**暂态（外部支撑集）、**峰步靠后**（弱信号 → @40 型）、
**幅度近 SFT**（力度主导幅度）。
- 三项全中 → J4 升级；幅度反而近 off-KD → 幅度由数据驱动，J4 修订；无暂态 → F-4 支撑集条款重审。

**违规即弃**：改判据须在本文件追加修订记录说明原判据失效原因，不允许静默换口径。

---

# 裁决记录（Theory，2026-07-19，中间回报后；树 1–3 按冻结口径机械执行，树 4–5 待 G3）

## 树 1（H4 格式归因）→ **D1：格式模板损伤由 teacher 文本（数据）驱动**
Δfail_seqKD = .541−.126 = **+.415**；|Δ−.352(offKD)|=.063 < |Δ−.165(SFT)|=.250 → 归 off-KD 侧。
辅助判定量：detectable_format@40 下跌存在（.303→.219，−8.4pts；弱于 off-KD 的崩塌但方向同）。
⇒ 格式重组跟数据不跟 loss：两个 teacher-文本臂同病，KL 非必需。

## 树 2（J6）→ **成立**：seqKD .724 ≤ off-KD .794 − 2pts
同数据下 logits 带来 +7pts math。附带（未注册）：seqKD .724 **< SFT .752**——
无 logits 时 teacher 文本反而不如数据集 CoT。

## 树 3（F-10 终止）→ **修订触发**：seqKD trunc@624 = .730 ≫ .25
数据-EOS 假说死亡（同一批 78.6%-EOS 文本，CE 训出 .73 runaway）。四臂全表：
OPD(KL×自文本) .918 / off-KD(KL×teacher文本) .048 / SFT(CE×数据集) .352 / seqKD(CE×teacher文本) .730
⇒ **F-10 修订：干净终止 = KL 稠密信号 × 可终止文本 的交乘**（仅 off-KD 兼备；
teacher 文本内 KL vs CE 翻转 .048/.730，KL 臂内自文本 vs teacher 文本翻转 .918/.048）。

## 未注册但记录（待正式裁决）
1. **F-12 受挑战**：seqKD（CE）dip@5（.522）——dip 时机可能跟**数据源**
   （teacher/自文本臂 @5-10，数据集臂 @20）而非信号类型；改判需全网格补齐后执行；
2. F-19 第三臂闭环列**混合**：off-KD H_ood 微 dip@5（−24，与其 ID dip 共时）✓，
   但 H_bos 大骤降在 @160（−124）无行为事件对应，@20–40 反而扩张——分型学在第三臂上不干净；
3. **F-17 终局**：C1 全探针 σ-rank 三臂全同（差 ≤12/500–750），方向身份线全域关闭；
4. seqKD MMLU flexible 终点 .581 ≥ base——知识通道四臂无损，双通道框架四臂成立。

---

# 第三执行块（2026-07-20，CONFIRMED SPEC —— 待 Coder 发车）

```yaml
status: CONFIRMED_SPEC__PENDING_EXECUTION
objective:
  - 补齐 Llama-3.2-3B 的 OPD 臂与第二模型主复现
  - 以 Qwen alpha=0.5 support-mixture 检验 self-support fraction 的剂量响应
  - 修复 probe–eval lead–lag 配对与 probe 样本量稳健性证据
budget: 新训练+评测+几何总计不超过 3 GPU-days，逐项记录实际 GPU-hours
seed_policy: 不增加独立训练 seed；训练继续 seed=42；bootstrap/subsample seed 不算训练 seed
scope_guard:
  - 用户自行处理 abstract/title/OpenReview/投稿；Coder 不写、不修改这些内容
  - 不重训已完成的 Llama SFT/off-KD/seqKD
  - 正文功能轨只用 per-checkpoint matched whitening W_t S_{D,t}
  - r_epsilon 只对白化功能算子 W_t S_{D,t} 定义；禁止生成 raw-weight/raw-activation r_epsilon 并继承输出误差解释
  - 只回报原始读数+协议+完成 manifest；理论裁决由 Theory 执行
```

## 优先级与停机门

1. **P0：Llama OPD 训练**；完成后立即回报实际 GPU-hours。
2. **P1：Llama 八点主评测 + 白化功能秩**；这是第二模型最小可发表闭环。
3. **P2：Qwen alpha=.5 先到 step160**；只在 P0/P1 后剩余预算足以完成并安全落盘时发车。
4. **P3：CPU 稳健性与 lead–lag 修复**；与 GPU 项并行。
5. **P4：Numina 欠账**；只在上述完成且仍有预算时执行。

Coder 必须维护 `gpu_budget_ledger.json`（task / gpu_count / wall_hours / GPU-hours / remaining）。
若剩余预算不足以完成下一个不可切分单元，停机并回报，不得超预算起跑。

---

## L1 —— Llama-3.2-3B OPD 臂【P0，新训练】

### 已有地基（禁止重训）

- `SFT-llama` / `offKD-llama` / `seqKD-llama` 已完成 step624，已有原生十点
  `{0,5,10,20,40,80,160,320,480,624}`。
- student = `/root/autodl-tmp/model/Meta/modelscope/Llama-3.2-3B`（28 layers，context 131072）。
- teacher rollout 与 RAW top-32 数据已完成；该固定 rollout 只供 off-KD/seqKD，**不得当作 OPD support**。

### OPD 训练规格

- teacher = `Llama-3.1-8B-Instruct`；student = `Llama-3.2-3B-Base`。
- prompts / prompt order = 与 Llama 三离线臂相同的 4,999 条有效 5k pool；`shuffle=false`。
- **support = 当前 student checkpoint 的 on-policy self rollout**，每个训练 step 重新访问当前策略分布。
- rollout: temperature=.6 / top_p=.9 / top_k=-1 / n=1 / seed=42 / max_response_tokens=10240。
- distillation: forward KL, teacher RAW top-32，response-token only，`token-mean`，不含 policy-gradient/reward 项。
- LoRA r32/alpha64/all-linear；AdamW lr=5e-5；effective batch=16；2 epochs；total steps=624。
- 原生保存十点 `{0,5,10,20,40,80,160,320,480,624}`；正式测量只用下文八点。
- 保存每步/batch 的 prompt id、rollout text、token length、finish reason、has_boxed、
  teacher/student mass、KL/loss、grad norm 与 wall time；所有 checkpoint 必须可恢复。

### L1 训练侧 support 统计

从已保存 rollout 离线计算，不额外生成：

- exact duplicate / normalized duplicate rate；
- near-duplicate cluster size distribution 与 cluster entropy，`effective_support_size = exp(H_cluster)`；
- distinct-2 / distinct-4 / 4-gram repetition；
- response length / cap-hit / EOS / has_boxed。

产物：`llama_opd_training_manifest.json`、`llama_opd_support_stats.csv`、十点 checkpoint inventory。

---

## L2 —— Llama 四臂行为复现【P1】

### 统一规则

- arms = `{opd,sft,offkd,seqkd}`。
- measured checkpoints = **`{0,5,20,40,80,160,320,624}`**。
- step0 全臂共享一次 base 运行，禁止伪造四个独立 base。
- 全 checkpoint/全 arm 使用同一 cap、sampling、prompt template、sample IDs 和 scorer；
  禁止复制 Qwen step20→40 的混合-cap 边界。

### 主任务

1. **MATH500**：N=500，`max_new_tokens=16384`，固定 seed=42；报
   `accuracy / SE / cap_hit / EOS / boxed_rate / trunc_but_correct / mean,median,p90 length`。
2. **MMLU-Pro generation**：与 Qwen 已有 100/class 子集协议对齐；同一批生成同时报
   `strict / flexible / extract-failure / response length`；不在本块新增 LL 全网格。
3. **IFEval**：全 541 prompts；报 prompt-strict / instruction-strict，并按现有 9 类审计链输出
   category pass rate + response length。

不跑 GPQA / TruthfulQA / AIME generation / Numina-llama；这些不属第二模型最小复现。

产物：`llama_behavior_8ckpt.csv`、逐样本文本，以及 task/arm/checkpoint 完成 manifest。

---

## L3 —— Llama 白化功能秩 + 相关工作 native-space 对照【P1】

### 固定探针电池（跨臂/跨 checkpoint 共用）

| probe | 数据/角色 | 实施约束 |
|---|---|---|
| `E_math` | MATH-style，直接对齐 MATH500 | 从非 MATH500 正式题面构建；与 5k 训练池指纹去重 |
| `E_math_hard_v2` | **AIME25** 题面 | 只用 prompt，不用答案；与 AIME24 eval 及训练池指纹去重 |
| `E_ood` | MMLU-Pro 题面 | 沿用已冻结 128-row corpus |
| `E_if` | IFEval prompt | 沿用已冻结全 541 prompts，不生成回答 |
| `E_general` | 通用文本 | 沿用已冻结 corpus |
| `S_math` | Llama base 生成的共同数学 support anchor | 只生成一次，32 rows，四臂共用，版本化 |

主复现不包含 `S_bos`、H/X 双动探针和 `E_numina`。

### 白化功能轨（本文主对象）

- arms/checkpoints 与 L2 相同。
- Llama 有 28 layers；**headline = L14**，八点全测。
- 层稳健性 = L7/L21 的 `{0,5,20,40,160,624}`；不在两个辅助层测 80/320。
- modules = q/k/v/o/gate/up/down；保留逐模块谱与预先固定的七模块等权均值。
- 只使用 `track=per_checkpoint`：对每个 cell 计算 `A_{D,t}=W_t S_{D,t}`，
  `S_{D,t} S_{D,t}^T = Sigma_{D,t}`。`frozen_base` 不进正文、不进主裁决。
- epsilon grid = `{.01,.025,.05,.10}`；headline = `.05`；完整阈值稳健性进附录。
- 保存全量 singular values 与 tail-energy curve；输出 `r_epsilon / delta_from_base`。
- 禁止对 raw residual / raw W 计算同名 `r_epsilon`。

### 相关工作 native-space 对照（landmarks 即可）

在 checkpoints `{0,5,20,40,160,624}` 和 raw residual streams 上计算：

- normalized entropy effective rank（raw centered covariance）；
- participation ratio；
- top-{1,8,32} explained-variance share；
- raw + centered anisotropy；
- CKA versus step0。

相关工作测量的完整分类必须在 manifest 标明 `native_object`：

1. raw representation：ER / PR / top-k share / anisotropy / CKA / domain separability；
2. weight/update：update norm/rank/stable-rank，spectral concentration/sparsity，near-zero-coordinate mass，
   principal/off-principal overlap，subspace locking；
3. singular-vector：principal angle / U,V rotation / intruder dimensions；
4. output/probability：logit entropy / ECE / PPL / KL / task score；
5. ours：白化 `W_t S_{D,t}` 的 r_epsilon/tail energy 及辅助谱量。

本块必做的 literature baseline 是上述 raw-representation suite；其余类别是完整定位和已有
Qwen 产物的对照索引，不得因清单存在就擅自扩成 Llama 全量网格。

产物：`llama_r_epsilon.csv`、`llama_tail_energy.csv`、`llama_raw_representation_suite.csv`、
full spectra inventory + probe provenance + strict manifest。

---

## Q1 —— Qwen alpha=.5 support-mixture arm【P2，新训练，先到160】

### 科学问题和端点

- alpha=0 = 现有 off-KD：frozen external teacher support。
- alpha=1 = 现有 OPD：current self-rollout support。
- alpha=.5 = 本新臂：每个 batch 同时使用 external/self support。
- 它检验的是 **self-support fraction dose response**，不单独证明“重复率/集中度导致过冲”。

### alpha=.5 严格实现

- student/teacher/prompts/optimizer/LoRA/total-step 规格与现有 Qwen OPD 一致。
- effective batch=16；每 batch 固定 8 条 current self rollout + 8 条 frozen external teacher rollout。
- 两个 epoch 交换每个 prompt 的 source assignment；同一 batch 仍保持8/8，避免 prompt 身份与 support source 永久混杂。
- self 部分：当前 student rollout + teacher RAW top-32 forward KL。
- external 部分：现有 off-KD teacher text + 其 RAW top-32 forward KL。
- **禁止将两源 token pool 后做一次 token mean**。正式 loss：

  `L = .5 * mean_token(L_self) + .5 * mean_token(L_external)`。

  这条保证长 self rollout 不会把名义 alpha=.5 改写成 token-share >.5。
- 记录 sample share、token share、两源独立 loss/KL/length，以及合并后 grad norm。

### 分阶段发车

**Stage A（必须先停在160）**：只运行至 step160，保存
`{0,5,10,20,40,80,160}`，并回报实际 GPU-hours。

- 几何：`{0,5,20,40,80,160}` × L18 ×
  `{S_math,E_math,E_math_hard_v2,E_ood,E_if,E_general}`，per-checkpoint r_epsilon，
  epsilon `{.01,.025,.05,.10}`。
- 训练文本统计：与 L1 同 schema，但 self/external 分源报告。
- 行为关键点：`{0,20,40,160}` 的 MATH500 / MMLU-Pro strict+flexible / IFEval，协议对齐现有 Qwen 正式运行。

**Stage B（需新 GO）**：只在 Stage A 完整交付、预算仍足且 Theory 明确 GO 后，
从 step160 恢复到320，补 step320 主测量；320 是 Q1 的最终终点。不得自动续跑。

### 预注册读数（Theory 裁决，Coder 只输出）

对每个 probe，将 step `{5,20,40,80,160}` 的 `delta r_epsilon@.05` 展平成轨迹向量，
并合并六 probe 得到 `z_offKD / z_half / z_OPD`。输出：

- segment projection
  `lambda_hat = <z_half-z_offKD, z_OPD-z_offKD> / ||z_OPD-z_offKD||^2`；
- distance from `z_half` to the line segment `[z_offKD,z_OPD]`；
- 逐 probe×step 的 `z_half` 是否位于两端之间。

主预测是 `0 < lambda_hat < 1` 且 `z_half` 比对侧 endpoint 更接近线段。
只有 `{0,.5,1}` 三个 alpha 时禁止回报“连续函数”；只回报有序中间点/轨迹插值是否成立。

产物：`qwen_alpha05_training_manifest.json`、`qwen_alpha05_support_stats.csv`、
`qwen_alpha05_r_epsilon.csv`、`qwen_alpha05_behavior_keypoints.csv`、`qwen_alpha05_segment_test.csv`。

---

## R1 —— fixed-probe 样本量敏感性重建【P3，CPU 优先】

本项必须分清三个不同轴：

1. epsilon sensitivity：已有 .01/.05，本块统一扩为 `.01/.025/.05/.10`；
2. bootstrap draws sensitivity：既有 `{256,1024,4096}` 稳定结果，只引用已有产物，不重跑；
3. **probe sample-count sensitivity**：当前缺少可审计成品，本项要补的是这一轴。

优先从 Stage-3 scratch 保留的 per-sample factors 重建；如任一必要 cell 缺 factors，先出
missing inventory，只允许对该格做 generation-free forward 补采，禁止重生成训练文本。

- 32-row probes：`n={8,16,24,32}`。
- E_ood/E_if 大探针：`n={16,32,64,128}`。
- 每个 n 做 200 次无放回子采样，seed=42；不用单一前缀样本代替重复子采样。
- 核心 cells：四臂 × steps `{0,20,40,624}` × L18 × 七模块，headline epsilon=.05。
- 输出 `r_epsilon median/IQR/95% interval`、相对全样本的绝对方向差、符号保留率、
  事件峰/谷步保留率，以及尾能阈值两侧的 margin。

产物：`probe_n_sensitivity.csv`、`probe_n_event_stability.csv`、`probe_factor_inventory.json`。

---

## R2 —— domain-matched probe–eval lead–lag 重建【P3，CPU】

现有 C7/C14 Cartesian-product descriptive table **禁止继续用于正文 lead–lag claim**。
新表只允许下列语义配对：

| behavior | matched probe |
|---|---|
| MATH500 accuracy / cap-hit / length | `E_math` |
| MMLU-Pro strict / flexible / extract-failure | `E_ood` |
| IFEval prompt/instruction/category pass | `E_if` |
| AIME24 eval（仅已有 Qwen 稀疏格） | `E_math_hard_v2` = AIME25，同域但独立题面 |
| Numina eval（仅完成 R3 后） | `E_numina` |

- `E_general` 只做跨域 control；`S_math` 只做 support anchor；两者都不强行匹配一个 behavior。
- 对每个 matched pair 和 arm 报几何事件步、行为事件步、
  `lag = t_behavior - t_geometry`；正值才表示几何领先。
- checkpoint 网格稀疏；主表使用 `same observed checkpoint / leads by one observed interval /
  lags by one observed interval`，精确 step 差只作审计列。
- Qwen MATH 的 0–20@4096 与 40–624@16384 是已知协议边界；跨 20→40 的精确 lag
  必须标 `cap_boundary_confounded=true`，禁止当作精确时差。
- Llama L2 使用全轨单一 cap，作为无该 confound 的主 lead–lag 复现。

为了修复 Qwen 对齐，纯探针补测（无训练）：

- `E_math`：四臂 × 十点 × L18；
- `E_math_hard_v2(AIME25)`：四臂 × 十点 × L18；
- 主量 = per-checkpoint `r_epsilon@.05`；全 epsilon grid 一并保存。

产物：`domain_matched_events.csv`、`domain_matched_lead_lag.csv`、
`qwen_emath_emathhardv2_r_epsilon.csv`。

---

## R3 —— Numina 稀疏欠账【P4，非 Llama 主复现】

只在 P0–P3 完成后执行：

1. Qwen seqKD Numina eval：steps `{40,160,624}`，N=200，cap=12288，协议与已有
   OPD/SFT/off-KD 三臂完全一致。
2. 建 `E_numina` fixed probe：Numina test 题面，排除 few-shot rows，与训练池/正式评测样本指纹去重。
3. `E_numina` 几何：Qwen 四臂 × `{40,160,624}` × L18，per-checkpoint r_epsilon 全 epsilon grid。

`legacy_S_math` 来自 NuminaMath CoT，但它不得代替 eval-matched `E_numina`。
不对 Llama 增加 Numina eval/probe，除非用户后续另行批准。

产物：`qwen_seqkd_numina_sparse.csv`、`E_numina_manifest.json`、`qwen_enumina_r_epsilon.csv`。

---

## 已知 cap 事实与禁止误读（全块通用）

1. SFT step80 的非配对 aggregate pilot：4096/N500 acc=.6200 vs 24576/N60 acc=.6167；
   这支持 SFT/as-run accuracy 近似 cap-robust，但不是逐题配对等价证明。
2. OPD step320 的配对 pilot：4096 acc=.680 vs 24576 acc=.910，+23pts；晚期 OPD 对短 cap 明显敏感。
3. Qwen 正式 MATH 使用 0–20@4096，40–624@16384；早期 OPD step5 dip 保留为真实事件，
   禁止写“4096 使整条轨迹无效”。
4. 本块不再跑 Qwen 0–20 的16k cap repair；Llama 用全轨统一 cap 规避边界。

---

## 回报节律与完成门槛

- L1 完成即交一次：训练 manifest + checkpoint inventory + GPU ledger + support 统计。
- L2/L3 按 checkpoint 可恢复；优先完成 `{0,5,20,40,624}`，再补 `{80,160,320}`，
  但最终主表必须是八点。
- Q1 Stage A 到160必须先交付；Stage B 需新 GO。
- R1/R2 与 GPU 块并行，不要等训练结束才开始清点 factors/建探针。
- 每个任务提交：原始 CSV/JSONL，provenance，row-count，SHA-256，strict completion manifest，
  以及一份人类可读 handoff。
- Coder 不做结论、不修改 human-read/正文/abstract；Theory 收到原始读数后再裁决。

---

# Theory → Coder 增量任务（2026-07-24）：domain-conditioned state–displacement–readout

```yaml
view_type: handoff_addendum
status: READY_FOR_ARTIFACT_AUDIT_AND_EXECUTION
scope: state_rank + fixed_support_displacement + direction + incremental_information + local_output/readout
non_scope: paper / abstract / human_read editing
execution_rule: 先审计已有产物；已完成则验证并登记，部分完成只补缺格，未完成的新增训练不阻塞现有 checkpoint 分析
```

## A0 —— 执行规则、状态码与命名冻结【P0，先执行】

开始任何新计算或训练前，先检查现有 handoff、manifest、CSV、checkpoint、per-sample
factor、激活二阶矩、奇异谱和 UV 产物。

状态码：

- `DONE_EXISTING`：此前已完成，本轮验证定义、范围和产物路径后登记；
- `DONE_NEW`：本轮新完成；
- `PARTIAL`：部分 cell 已完成，只补缺格；
- `NOT_FOUND`：未找到已有产物；
- `DEFERRED_NONBLOCKING`：未完成，但不阻塞现有 checkpoint 主分析；
- `BLOCKED`：存在具体技术阻塞，必须说明缺什么。

若 `frozenSelf0-KD`、独立 seed、更多 alpha 或相应评测已经存在，直接进入统一分析；若没有，
登记为 `DEFERRED_NONBLOCKING`，不要阻塞 A1–A7。禁止仅因本增量再次列出任务而重跑。

任何结论必须链接到 machine-readable 逐单元表、公式/实现版本、完整范围、聚合前读数、
provenance、row-count、SHA-256 和产物路径。不得只给七模块均值、相关系数或一句裁决。
Coder 不修改论文、abstract 或 `theory/human_read-ch.md`，也不替 Theory 升降 claim。

### A0.1 Probe / Eval 命名

| 名称 | 严格含义 |
|---|---|
| `E_general` | 通用文本 fixed probe |
| `E_math` | MATH500 语义配对数学 probe；不得指代 AIME |
| `E_ood` | MMLU-Pro 语义配对 probe |
| `E_if` | IFEval 语义配对 probe |
| `E_mathCoTtrain` | Math-CoT-20K 实际训练行 |
| `E_mathCoThold` | Math-CoT-20K 同分布未训练保留行 |
| `E_numina` | Numina eval-matched probe；仅在已有或 R3 完成时纳入 |
| `E_AIME24/E_AIME25` | 明确版本 AIME probe；不得改名为 `E_math` |

`X_*` 仅保留给 base/step0 rollout support；不得与外部 `E` 混名。行为评测显式写为
`Eval_MATH500 / Eval_MMLUPro / Eval_IFEval / Eval_mathCoThold / Eval_Numina`。

### A0.2 核心范围

- models：Qwen、Llama；
- core arms：`OPD / off-KD / seqKD / SFT`；
- 已存在则加入：`alpha=.5 / frozenSelf0-KD / 其他 alpha / 独立 seed`；
- checkpoints：使用真实 shared grid，优先 `{0,5,20,40,80,160,320,624}`，不得插值；
- headline layers：Qwen L18、Llama L14；
- sensitivity layers：Qwen L9/L18/L27、Llama L7/L14/L21；
- modules：`q/k/v/o/gate/up/down`。

先保存逐模块读数，再输出七模块等权均值、attention/MLP 分组和 output-energy weighting。

## A1 —— State 与 fixed-support displacement【P0，最高优先；不需要新训练】

### A1.1 二阶矩和数值协议

```math
\Sigma_{D,a,t}=N^{-1}\sum_i h_{i,a,t}h_{i,a,t}^{\top},
\qquad S_{D,a,t}S_{D,a,t}^{\top}=\Sigma_{D,a,t}.
```

未减均值时必须标 `uncentered_second_moment`，不得无条件写 covariance。记录样本/token 数、
manifest/hash、padding/position mask、square-root 方法、eig/ridge floor、rank-deficient 处理和精度。
正式矩阵与谱计算用 FP32/FP64，禁止直接使用 bf16 谱结果。

统一有效权重：

```text
W_t = base weight + effective LoRA update at checkpoint t
Delta W_t = W_t - W_0
```

若 step0 带 adapter，按真实有效权重计算并登记，不能默认零 adapter。

### A1.2 Current-state rank

```math
A^{state}_{D,a,t}=W_{a,t}S_{D,a,t},
```

```math
r^{state}_{\varepsilon}(D,a,t)=
\min\left\{r:
\frac{\sum_{i=1}^{r}\sigma_i^2(A^{state})}{\sum_i\sigma_i^2(A^{state})}
\ge 1-\varepsilon\right\}.
```

同时保存 `delta_from_base`。不得称 capability score；它只表示当前域条件局部模块输出能量的
谱维数。

### A1.3 Fixed-base displacement：共同 ruler

```math
A^{disp,0}_{D,a,t}=(W_{a,t}-W_0)S_{D,0}.
```

验证：

```math
\|A^{disp,0}\|_F^2=
\operatorname{tr}(\Delta W_{a,t}\Sigma_{D,0}\Delta W_{a,t}^{\top}).
```

同时计算：

```math
m^{disp,0}_{D,a,t}=
\frac{\|\Delta W_{a,t}S_{D,0}\|_F}{\|W_0S_{D,0}\|_F},
\qquad
r^{disp,0}_{\varepsilon}=r_\varepsilon(\Delta W_{a,t}S_{D,0}).
```

保存 norm numerator/denominator/ratio；`t=0` 时 `displacement_rank=NA`，不得当真实 0。
跨模型另报：

```math
\widetilde r^{disp,0}_{\varepsilon}=
\frac{r^{disp,0}_{\varepsilon}}
{\min(d_{out},\operatorname{rank}(\Sigma_{D,0}))}.
```

原始与 normalized rank 都保存。跨模型主结论优先比较模型内符号、排序和事件，不直接比较原始
rank 数值。

A1 最低覆盖：Qwen/Llama、四 core arms、全部 shared checkpoints、
`E_general/E_math/E_ood/E_if`、headline layer 七模块、epsilon=.05。
`E_mathCoTtrain/E_mathCoThold/E_numina` 和辅助层已有则纳入；没有出 missing inventory。

产物：

```text
state_rank_full_cells.csv
fixed_support_displacement_full_cells.csv
state_displacement_manifest.json
state_displacement_missing_inventory.json
```

## A2 —— Displacement magnitude + rank + direction 与 on-policy 对照【P0】

### A2.1 方向量

对 fixed-support displacement 计算：

```math
c_{ab}=\frac{\langle A_a,A_b\rangle_F}{\|A_a\|_F\|A_b\|_F}.
```

若 `A_a=U_a Sigma_a V_a^T`，计算：

```math
O_U(k)=k^{-1}\|U_{a,k}^{\top}U_{b,k}\|_F^2,
\qquad
O_V(k)=k^{-1}\|V_{a,k}^{\top}V_{b,k}\|_F^2,
```

并输出左右 principal-angle 全谱/摘要。主 k 可取双方 `r_epsilon` 较小值，但登记实际 k；另做
固定 rank-fraction sensitivity。

### A2.2 核心配对

1. `OPD vs off-KD`：逐 checkpoint/domain/module 比较 state rank、displacement
   magnitude/rank/direction。该配对识别 current-self generation pipeline 相对 external/frozen
   support 的总效应，包括 support、style、length、EOS、repetition、freshness；禁止单独归因 freshness。
2. `OPD vs alpha=.5 vs off-KD`：若 alpha=.5 已有，在 fixed-support matrix 上计算：

```math
\widehat\lambda=
\frac{\langle A_{.5}-A_{off},A_{OPD}-A_{off}\rangle_F}
{\|A_{OPD}-A_{off}\|_F^2},
```

```math
\rho_\perp=
\frac{\|A_{.5}-[(1-\widehat\lambda)A_{off}+\widehat\lambda A_{OPD}]\|_F}
{\|A_{OPD}-A_{off}\|_F}.
```

   输出所有 cell，不得只给均值或“多少格位于中间”。
3. `off-KD vs seqKD`：比较 state trajectory、displacement magnitude/rank/direction；rank 相似而
   direction/readout 不同只回报事实，不称 `r_epsilon` 为行为充分统计量。
4. `OPD vs SFT`：Qwen/Llama 分别比较。跨模型不强迫相同正峰/负过冲/回弹形状，先报告更弱但
   稳定的 OPD–offline 差异。

另做 general-adjusted difference-in-differences：

```math
R^{rel}_{D,a,t}=[R_{D,a,t}-R_{D,a,0}]
-[R_{general,a,t}-R_{general,a,0}],
```

R 分别取 state-rank、normalized displacement magnitude、normalized displacement rank。
绝对曲线仍是主结果；OPD–off-KD 必须逐 checkpoint，不只报终点。

产物：

```text
displacement_direction_full_cells.csv
opd_offkd_state_displacement_pairs.csv
alpha_segment_displacement_test.csv
offkd_seqkd_state_displacement_pairs.csv
general_adjusted_state_displacement.csv
```

## A3 —— Ruler、centered 与有限样本稳健性【P0】

主尺：

```math
A^{disp,0}=\Delta W S_{D,0}.
```

敏感性：

```math
A^{disp,t}=\Delta W S_{D,a,t},
```

```math
\Sigma^{sym}=\tfrac12(\Sigma_{D,0}+\Sigma_{D,a,t}),
\qquad A^{disp,sym}=\Delta W S^{sym}.
```

fixed-base 用于跨 checkpoint/arm 公平主比较；current 测更新在当前访问分布上的作用；symmetric
仅作 pairwise sensitivity，不得用 arm-specific symmetric ruler 取代共同主尺。已有 cross-arm
pooled ruler 则登记，没有不阻塞。

centered sensitivity：

```math
\mu=\mathbb E[h],
\qquad
\Sigma^{centered}=\mathbb E[(h-\mu)(h-\mu)^\top].
```

最低覆盖 OPD/off-KD、Qwen L18/Llama L14、early/deepest/recovery-or-terminal、
`E_general/E_math/E_if`、epsilon=.05。报告符号、arm 排序、rank 差、trajectory correlation 和
OPD 压缩/位移支配是否保持。

epsilon 统一为 `.01/.025/.05/.10`。probe sample-count 复用 R1 和已有 per-sample factors；已完成
则登记，不重跑。缺 factor 只对缺格 generation-free forward。另报七模块独立、attention/MLP、
等权/output-energy weighting 和辅助层 landmarks。

产物：

```text
support_ruler_ablation.csv
centered_uncentered_state_displacement.csv
state_displacement_epsilon_sensitivity.csv
state_displacement_module_layer_robustness.csv
```

## A4 —— 相较纯权重与纯激活空间的增量信息【P0，方法 gate】

### A4.1 同格 baseline

在与 A1 完全相同的 cell 计算或登记：

1. raw weight/update：`||Delta W||_F`、spectral norm、stable/effective rank、concentration/sparsity；
2. strict source-principal：

```math
p_k=\frac{\|U_{0,k}^{\top}\Delta W V_{0,k}\|_F^2}{\|\Delta W\|_F^2},
\qquad W_0=U_0\Sigma_0V_0^\top;
```

   left-only 不得冒充 strict joint `p_k`；
3. weight direction：PABS `theta_U/theta_V`、NSS、已有时加 mask overlap；
4. raw activation native-space：PR、top-share、anisotropy、CKA；legacy entropy-effective-rank 只能
   标 external raw-activation baseline，不得与本文 `r_epsilon` 混名；
5. ours：state rank、displacement magnitude/rank/direction。

先保留原始量，再做 fold 内标准化；不得比较不同尺度的“最大暂态均值”来宣称敏感性。

### A4.2 Track A：训练臂增量判别

主任务 OPD vs off-KD，比较：

```text
baseline only
baseline + state rank
baseline + displacement magnitude/rank/direction
baseline + state + displacement
```

至少按 checkpoint grouped cross-validation，防止同 checkpoint 的 module/domain 同时进 train/test。
标准化只用训练 fold。报告 held-out log-loss、AUC、balanced accuracy、Delta log-loss/Delta AUC、
grouped bootstrap/permutation uncertainty。训练集拟合优度不能代替样本外结果。

### A4.3 Track B：domain-matched 行为增量解释

只允许：

| behavior | geometry |
|---|---|
| `Eval_MATH500` | `E_math` |
| `Eval_MMLUPro` | `E_ood` |
| `Eval_IFEval` | `E_if` |
| `Eval_mathCoThold` | `E_mathCoThold` |
| `Eval_Numina` | `E_numina`，仅已有时 |

先拟合 `arm + progress + objective + model + weight/raw-activation baselines`，再加入 matched-domain
state rank、displacement magnitude/rank/direction summary。Qwen/Llama 先分开，合并只作补充。
使用 grouped held-out evaluation，报告 Delta R2、Delta MAE、contemporaneous；数据足够再做
one-observed-interval lead/lag。

本任务检验增量解释，不要求完美预测。负结果完整保存，不自动否定局部低秩构念。

产物：

```text
strict_weight_activation_baseline_full_cells.csv
incremental_arm_discrimination.csv
incremental_behavior_explanation.csv
incremental_information_cv_manifest.json
```

## A5 —— 实际局部输出差分与代数归因【P1，GPU forward；不需要新训练】

checkpoint 0 与 t 必须使用相同 prompt、token IDs、positions、attention mask、padding exclusion 和
sample order；保存 token manifest/hash。禁止对各 checkpoint 自由生成的不同文本直接相减。

```math
\Delta Y_{D,a,t}=W_{a,t}H_{D,a,t}-W_0H_{D,0}.
```

计算：

```math
m_Y=\frac{\|\Delta Y\|_F}{\|W_0H_0\|_F},
\qquad r_\varepsilon(\Delta Y).
```

最低范围：Qwen/Llama、OPD/off-KD；已有则加 SFT/seqKD/alpha=.5；Qwen L18/Llama L14；
early/deepest/recovery-or-terminal；`E_general/E_math/E_if`，已有或资源允许加 `E_ood`。

对称代数归因：

```math
\bar W=\tfrac12(W_t+W_0),\quad \bar H=\tfrac12(H_t+H_0),
```

```math
\Delta Y_W=\Delta W\bar H,
\qquad
\Delta Y_H=\bar W\Delta H.
```

先验证 `Delta Y = Delta Y_W + Delta Y_H`，再报三者 norm、relative fractions 和：

```math
c_{WH}=\frac{\langle\Delta Y_W,\Delta Y_H\rangle_F}
{\|\Delta Y_W\|_F\|\Delta Y_H\|_F}.
```

只称 weight-associated/activation-associated algebraic component 与 reinforcement/cancellation，
不得称 causal mediation。

产物：

```text
local_output_displacement_full_cells.csv
local_output_weight_activation_decomposition.csv
local_output_token_manifest.json
```

## A6 —— LoRA/module finite intervention【P1，GPU inference；已有则纳入】

对 checkpoint t 保留当前模型，只把目标层或模块 LoRA update 置零到 base/step0 有效权重。
优先：headline 中间层整体 zeroing → 七模块单独 zeroing → 已有时纳入相邻层。

固定上下文输出 full-vocabulary next-token KL、JS、answer-token margin、EOS probability、
formatting-token probability。少量 landmarks 行为生成输出 strict/flexible accuracy、truncation、
format failure、length、repetition。

登记 KL 方向、完整/截断词表、layer/module、merge 和 hybrid provenance。只解释“当前网络中撤销
该模块变化”的有限干预效应；禁止假设模块效应可加，不能把 hybrid 当自然训练轨迹。

产物：

```text
lora_zeroing_logit_effects.csv
lora_zeroing_behavior_landmarks.csv
lora_zeroing_manifest.json
```

## A7 —— 实现、top-32、行为和 cap 审计【P1；已有则登记】

逐 arm 登记 trainer、dataset builder、batch/packing、token normalization、gradient accumulation、
optimizer/scheduler/precision、LoRA settings、checkpoint save/merge、data order/seed。发现预期 treatment
bundle 外差异先登记，不自动重训。

已有 top-32 KL 时统计 retained mass 的 mean/median/p5/p25/p75/p95、omitted mass、随
checkpoint/support 变化、EOS 在 top-32 外比例。已有 top-64/128/exact-KL 则纳入；没有不阻塞。

行为登记 task version、template、cap、sampling、strict/flexible extractor、format/truncation、item
outputs，并报 item bootstrap CI。4096/16384 已有 audit 则登记，不因本任务重跑；fixed teacher-forced
geometry 不受 generation cap 影响，只有相反新证据才重启问题。

产物：

```text
trainer_arm_implementation_audit.md
teacher_top32_retained_mass.csv
behavior_protocol_and_ci_audit.csv
generation_cap_evidence_registry.md
```

## A8 —— 已有则纳入、未完成不阻塞【P2】

1. `frozenSelf0-KD`：若已存在，确认 rollout/support 永久来自 step0 student，target=teacher KL，
   trainer/batch/optimizer 尽量 matched；纳入 state/displacement/direction/behavior/support stats，比较
   current-self OPD。它估计 current-self/freshness bundle 总效应，不自动隔离单一文本中介。
2. 独立 seed：若已有，按 seed 分开纳入 OPD/off-KD shared window；不得把 Llama 当第二 seed。
3. 更多 alpha：若 `.25/.75` 已完成，复用 A2 的 lambda/rho/state/displacement/behavior 管线。

没有相应产物时登记 `DEFERRED_NONBLOCKING`，不阻塞 A1–A7。

## A9 —— 最终 schema、候选图与解释边界

主表至少包含：

```text
model, arm, seed, checkpoint, domain, probe_name, layer, module, epsilon,
support_ruler, centered, sample_count, token_count,
state_rank, state_rank_delta,
displacement_norm_raw, displacement_norm_denominator, displacement_norm_normalized,
displacement_rank, displacement_rank_normalized,
matrix_cosine, left_subspace_overlap, right_subspace_overlap,
principal_angle_summary, artifact_path
```

候选图：

1. state rank / displacement magnitude / displacement rank 三联图；
2. OPD–off-KD 逐 checkpoint 配对；
3. OPD–alpha=.5–off-KD interpolation 与 orthogonal residual；
4. Qwen/Llama 分模型轨迹；
5. fixed/current/symmetric/centered sensitivity；
6. strict p_k、PABS/NSS、raw activation suite 与本文量的增量信息；
7. Delta Y_W/Delta Y_H reinforcement/cancellation；
8. LoRA zeroing 的 logit/EOS/行为效果。

解释边界：

- state rank 不是 capability score；
- displacement rank 不得脱离 magnitude；
- 相同 rank 不代表相同方向；
- `Delta W S_0` 是 activation-conditioned weight-mediated local displacement，不是完整最终功能变化；
- 完整奇异谱对固定域局部低秩近似问题充分，单个 `r_epsilon` 不完备；
- OPD–off-KD 是 current-self support bundle 总效应，不单独识别 freshness；
- 行为增量解释为负不自动否定局部低秩构念；
- 多个 module/checkpoint cell 不是独立训练重复。

执行优先级：

```text
P0: A0 audit
 -> A1 fixed-support state/displacement
 -> A2 magnitude/rank/direction + on-policy pairs
 -> A3 ruler/centered/sample/module/layer robustness
 -> A4 strict baselines + incremental information

P1: A5 local output displacement
 -> A6 LoRA finite intervention
 -> A7 implementation/top-32/behavior audit

P2: A8 existing frozenSelf/seed/more-alpha results are integrated when present;
    absence does not block P0/P1.
```

## A10 —— 预计工时与 2×96G 并行建议（估算，不替代 A0 实际盘点）

以下为机器运行、产物核验和表格生成时间，不包含 Theory/Coder 对话以及论文、human-read 写作。
A0 完成后必须按 `DONE_EXISTING/PARTIAL/NOT_FOUND` 给出一次更新 ETA。

| 工作块 | 主要资源 | 已有 S/谱/factors 较完整 | 缺 centered/H factors 时 |
|---|---|---:|---:|
| A0 产物/定义盘点 | CPU + I/O | 1–2 h | 1–3 h |
| A1 fixed-support state/displacement 全格 | GPU 矩阵/SVD + CPU 汇总 | 4–8 h wall@2GPU | 8–16 h wall@2GPU |
| A2 方向与配对/alpha/DiD | CPU；大矩阵 overlap 可用 GPU | 1–3 h | 2–5 h |
| A3 ruler/centered/sample/module/layer | GPU forward/SVD + CPU 重采样 | 3–7 h | 6–14 h |
| A4 strict baselines + grouped CV | CPU，部分 SVD 可用 GPU | 2–5 h | 3–8 h |
| A5 local output Delta Y | GPU forward | 3–7 h wall@2GPU | 5–10 h wall@2GPU |
| A6 LoRA zeroing/readout | GPU inference | 3–8 h wall@2GPU | 5–12 h wall@2GPU |
| A7 实现/top32/行为审计 | CPU + I/O | 1–4 h | 2–6 h |

条件总工期：

- **P0 最小闭环 A0–A4，缓存较完整：约 8–16 h wall-clock**；
- **P0+P1 全闭环 A0–A7，缓存较完整：约 14–26 h wall-clock**；
- 若 centered、per-token H 或关键 factors 需要补 forward：约 **20–36 h wall-clock**；
- 若完全串行执行，同一工作量约 **35–60 h**。

A8 的训练不计入上述主线：已有则直接分析；未完成但之后决定补跑时，单独登记实际 GPU-hours，
不让它阻塞 A1–A7。离线 frozen-support/seed 训练通常比 current-self OPD 快；不得在盘点前用统一
时长替代真实 trainer 估算。

推荐极限并行：

```text
CPU lane:
  A0 inventory/provenance
  -> A2/A4 statistics and grouped CV as rows land
  -> A7 audits
  -> A3 sample-count/bootstrap resampling

GPU-0 lane:
  Qwen A1/A3 matrix + missing forward
  -> Qwen A5
  -> Qwen A6

GPU-1 lane:
  Llama A1/A3 matrix + missing forward
  -> Llama A5
  -> Llama A6
```

若一张 GPU 已在进行 A8 或其他已批准训练，另一张 GPU 可运行 A1/A3/A5/A6，CPU lane 照常并行；
不要在同一 GPU 上同时放训练与大矩阵 SVD/forward。两张卡同时读取/merge 大量 checkpoint 可能形成
I/O 瓶颈，应预缓存共同 `S_{D,0}`，按模型分盘/错开 merge，并让 CPU 统计消费已经落盘的分块结果。

完成顺序不要求等待整块：A1 每完成一个模型即交一次 partial manifest，A2/A4 随即消费；A5/A6
不必等待所有 sample-count sensitivity 完成。这样比“所有 P0 完成后才启动 P1”可再缩短约 25–40%
墙钟时间。

---

# Theory → Coder 增量任务（2026-07-25）：LoRA 双轨与 readout-conditioned output proxy

```yaml
view_type: handoff_addendum
status: READY_FOR_DUAL_TRACK_AUDIT_AND_PROXY_PILOT
scope: direct_BA_vs_merged_bf16 + matched_topk + lora_output_proxy + fractional_rollback_KL
non_scope: paper / abstract / human_read editing
relation_to_A_block: 本节不取消 A0--A10；细化 A1/A2/A4/A6 的 update、precision、top-k 和 intervention 口径
execution_gate: 先完成或登记 B0--B4；B5--B7 已有则纳入，未有则等 B4 回交后再决定
```

## B0 —— 冻结 LoRA 双估计对象【P0】

Track A，LoRA-action：

```math
\Delta W^{BA}_{t,m}=s_mB_{t,m}A_{t,m}.
```

- 使用真实 LoRA scaling；BF16 A/B 先转 FP32，再计算 `s*(B@A)`；
- 正式名 `direct_BA_from_bf16_factors_fp32_matmul`，未找到 FP32 master 时不得简称 FP32 checkpoint；
- 用于 adapter branch、rollback、LoRA rank 利用和再压缩。

Track B，serialized merged-state：

```math
\Delta W^{merged}_{t,m}=
\operatorname{FP32}(\widehat W^{BF16}_{t,m})-
\operatorname{FP32}(\widehat W^{BF16}_{0,m}).
```

- 先分别恢复落盘 BF16 权重到 FP32，再相减；
- 用于相关工作公平比较、落盘状态和 BF16 merge 审计；
- 不预设它等于或优于 direct BA。

每表登记：`weight_object/factor_storage_dtype/merged_storage_dtype/matmul_dtype/`
`subtraction_dtype/svd_dtype/lora_scaling/merge_version/checkpoint_hash`。

## B1 —— 冻结 matched top-k【P0】

正式公平比较逐 cell 固定相同 model/arm/checkpoint/layer/module/probe、weight object、精度、矩阵侧与：

```math
k\in\{4,8,16,32\}.
```

任一对象有效秩小于 k 时标 `NA_RANK_LIMIT`，不得零填充。相关工作 native k/rank-fraction 另表；
方法间结论只用共同 fixed-k。adaptive `r_epsilon` 仅作本文内部描述，k 确定后 epsilon 不再提供额外角度信息。

产物：`matched_topk_protocol.json`、`related_work_native_vs_matched_k_registry.csv`。

## B2 —— Landmark direct-BA / merged-BF16 审计【P0】

最低范围：Qwen/Llama，OPD/off-KD；Qwen `{20,160,624}`、Llama `{20,160,320}`；L18/L14；
`o_proj/down_proj`；功能审计用 `E_general/E_math/E_ood/E_if`；fixed k 四档。

输出：

```math
e_{rel}=\|\Delta W^{BA}-\Delta W^{merged}\|_F/\|\Delta W^{BA}\|_F,
\quad q_{norm}=\|\Delta W^{merged}\|_F/\|\Delta W^{BA}\|_F.
```

另报 algebraic/numerical/stable/effective rank、四 epsilon 的 `r_epsilon`、fixed-k cosine/PABS/左右
subspace overlap/principal angles、完整谱、merged rank-32 tail，以及同一 S/G 下的 `||Delta W S||`、
`||G Delta W S||`。

若双轨 landmark 一致，不全量双跑：direct BA 负责 proxy，merged 负责 related-work；若显著分歧，
按两个 estimand 分开报告。

产物：`lora_BA_merged_landmark_audit.csv`、`lora_BA_merged_spectra.parquet`、
`lora_BA_merged_topk_subspaces.csv`、manifest。

## B3 —— Readout-conditioned LoRA proxy【P0】

真实观测对象是固定 teacher-forced 上下文中的 fractional rollback 输出 KL：

```math
y^{(\lambda)}=W_0h+(1-\lambda)sBAh,
\quad
K(\lambda)=N^{-1}\sum_x KL[p_t(\cdot|x)\|p_t^{(\lambda,m)}(\cdot|x)].
```

保存逐 sample/token full-vocabulary KL；主方向固定为 current-reference 到 intervened。
从当前分布采样 `c`，令 `g=partial log p_t(c|x)/partial y`，使用二因子近似：

```math
\Sigma=E[hh^T]=SS^T,
\quad Q=E[gg^T],
\quad \mathcal E(E)\approx\|Q^{1/2}ES\|_F^2.
```

以 `G^T G≈Q` 的 gradient sketch 构造：

```math
P^{BA,cur}_{D,t,m}=s_mG_{D,t,m}B_{t,m}A_{t,m}S_{D,t,m},
```

```math
P^{BA,0}_{D,t,m}=s_mG_{D,0,m}B_{t,m}A_{t,m}S_{D,0,m}.
```

current 用于 rollback 校准，frozen-base 用于跨 arm/checkpoint。`r_epsilon(P)` 只指 LoRA 可用 rank 中
被域访问且能传播到 readout 的通道数，不是整模型功能维数。`AS`/`GB` 单独范数不 gauge-invariant；
headline 只用整体 `GBAS`。

保存 proxy norm、谱、`r_epsilon`、fixed-k tail、左右子空间和完整 estimator provenance。

## B4 —— 最小 rollback–KL pilot【P0，方法 gate】

冻结：Llama L14；`o_proj/down_proj`；OPD/off-KD；checkpoint `{0,20,160,320}`；
`E_general/E_math/E_ood/E_if`；lambda `{.1,.25,.5,1}`。step0 只建 G0/S0。
一次保存 nested `m=64` sketch，固定前缀复用 `m={16,32,64}`；同一次 backward 用 hooks 同时收集模块，
扩七模块时也不得重复七遍 backward。

代理预测：

```math
\widehat K(\lambda)=\lambda^2\|P^{BA,cur}\|_F^2/2.
```

逐 cell 输出 K 对 lambda^2、through-origin/带 intercept slope、绝对/相对误差、小扰动与全范围分表、
Pearson/Spearman、leave-one-checkpoint/domain-out、sketch-size 稳定性和 current/frozen 差异。
Coder 不裁决；B4 完成即回交，由 Theory 决定 B5/B6 GO。

产物：`lora_fractional_rollback_token_kl.parquet`、`lora_proxy_rollback_calibration.csv`、
`lora_proxy_sketch_sensitivity.csv`、manifest。

## B5 —— 全量 proxy 轨迹【P1；已有则纳入，否则等 B4 GO】

顺序：Llama 四臂 L14 全 checkpoint/七模块/四 probe → Qwen 四臂 L18 → 辅助层 landmarks。
alpha=.5/frozenSelf/seed 已有则纳入，未有不阻塞。保存 current/frozen、累计 update 和真实相邻 checkpoint
stepwise update。比较 OPD–offKD、OPD–alpha–offKD、offKD–seqKD、OPD–SFT、absolute 与 general-adjusted。
不得仅因轨迹分臂就称 output proxy；名分由真实 KL 校准决定。

## B6 —— Fixed-k 收缩【P1；B4 后】

主对象 direct BA，固定 `k={4,8,16,32}`：

```math
L_{tail}(k)=\frac12\sum_{i>k}\sigma_i^2(G(sBA)S).
```

若从截断 P 反解 adapter/update，记录 ridge/pseudoinverse、condition number、重建误差、最终 rank，
并在相同 probe 测真实 KL/PPL；行为 Eval 只在 KL 校准成功后运行。Merged 原始矩阵只做 related-work
fixed-k geometry；rank-32 merged 只能称 denoised sensitivity，不得冒充 LoRA action。

## B7 —— Conditional merged intervention【P2】

仅 B2 显著分歧时运行：

```math
\widehat W_t^{(\lambda)}=\widehat W_t-\lambda(\widehat W_t-\widehat W_0).
```

它只校准 `G Delta W_merged S`。不得用 adapter disabling KL 校准 merged proxy，反向也不允许。

## B8 —— 回报顺序

1. B0/B1 先回报定义与 inventory；
2. B2 weight-only 先交，不等 G/S；
3. B3 先做 `1 arm × 1 ckpt × 1 probe × 1 module × m=4` smoke；
4. smoke 后运行 B4，完成即停在 gate；
5. 已存在的 B5/B6/B7 不重跑，按 A0 状态码登记；
6. 每项需 raw/provenance/row-count/hash/strict manifest；
7. coder 不修改论文、abstract、human_read，也不自行将 proxy 升为贡献。

## B9 —— 工时与 2×96G 并行估算

包含实现/调试、机器运行、核验和表格，不包含沟通与论文写作；B3 smoke 后必须以单 cell profile 更新 ETA。

| 工作块 | 资源 | 1×96G | 2×96G |
|---|---|---:|---:|
| B0/B1 protocol/inventory | CPU | 0.5–1.5 h | 同左 |
| B2 landmark audit | CPU/GPU SVD | 1–4 h | 1–3 h |
| B3 estimator + smoke | CPU+GPU | 3–7 h | 3–6 h |
| B4 G/S pilot | GPU backward | 5–12 h | 3–7 h |
| B4 rollback KL | GPU forward | 2–5 h | 1–3 h |
| B4 calibration | CPU | 1–2 h | 1–2 h |
| B5 Llama 四臂全轨 | GPU | 12–28 h | 7–16 h |
| B5 Qwen 四臂全轨 | GPU | 14–32 h | 8–18 h |
| B6 landmark fixed-k + KL | GPU/CPU | 8–20 h | 5–12 h |
| B7 merged intervention | GPU | 4–12 h | 2–7 h |

条件总墙钟：

- B0–B4 最小决策闭环：**8–16 h@2×96G**，或 **12–24 h@1×96G**；
- 加 Llama 四臂全轨：**18–34 h@2×96G**；
- 双模型四臂全轨：**30–60 h@2×96G**；
- 再加 landmark fixed-k：**40–72 h@2×96G**；
- 若触发 B7/大量行为 Eval：**48–90 h@2×96G**；
- 完全单卡串行：约 **70–140 h**。

B0/B1/B2 与既有 CPU 任务并行；B3/B4 复用已有 S、token manifest、loader 和 LoRA zeroing；
七模块同次 backward；`m=16/32` 复用 `m=64`；base G0/S0 跨 arm 缓存。推荐：

```text
CPU: B0/B1 -> B2 -> matched top-k -> B4 calibration
GPU-0: B3 smoke -> Llama OPD B4
GPU-1: Llama off-KD B4；若正在训练，空闲后再接
```

B4 前不得启动 B6。若单卡空闲，B0–B4 仍约 12–24 h；若 B4 为负，停止 B5/B6 可避免约
1.5–3 GPU-days 的无效扩展。

---

# Theory → Coder 纠偏覆盖指令（2026-07-25）：真实更新—真实输出轨迹闭环

```yaml
view_type: superseding_handoff_addendum
status: READY_FOR_ACTUAL_CHECKPOINT_OUTPUT_TRAJECTORY
supersedes:
  - B3--B9 的 Fisher/readout-gradient proxy
  - B6 的 rank-k update compression 主实验
  - B7 的 merged 人工插值干预
  - B9 的旧工时与 GPU 排程
retains:
  - B0 的 direct-BA / merged-BF16 estimand 审计
  - B1 的 matched top-k 公平规则（用于 related-work 方向比较）
  - B2 的 weight-only 与 Delta-W-S 双轨 smoke；取消其中全部 G/readout-gradient 项
core_question: 真实微调权重轨迹在激活白化空间中的变化，能否比纯权重空间更好地预判真实 checkpoint 输出变化
non_scope: 复现或证明 SVD-LLM；Fisher/KFAC；大规模 rank-k 压缩；新增全轨行为 Eval
```

## O0 —— 立即停止的旧任务

除非相应程序在收到本指令前已经完成并落盘，否则不要启动：

1. `G^T G≈Q` gradient sketch；
2. `G(sBA)S` readout-conditioned proxy；
3. Fisher/Task-Tangent output ruler；
4. fractional module rollback 作为主闭环；
5. rank-k update reconstruction/compression；
6. spectral-band intervention；
7. 新的全 checkpoint 行为生成 Eval。

已有相应结果只登记，不删除、不重跑，也不自动进入正文。本节目标不是论证 SVD-LLM 压缩方法，
而是检验本文提出的动态激活白化几何是否含有真实输出后果信息。

## O1 —— 单一主权重对象与激活选择【P0】

### O1.1 理论对象与 LoRA 无关

正文与产物统一定义：

```math
\Delta W^{effective}_{a,t}=W^{forward}_{a,t}-W^{forward}_{0}.
```

它适用于 LoRA、其他 adapter 和全参微调。LoRA 的 `sBA` 只是本实验中产生 effective update 的实现。

先执行 B2 的小范围 direct-BA / merged smoke，再按以下预注册规则选择单一全量主轨：

1. 查明正式 Eval/geometry 实际 forward 使用 merged checkpoint 还是 unmerged PEFT；
2. 若正式 forward 为 merged，主轨使用 serialized merged effective difference；
3. 若正式 forward 为 unmerged，主轨使用 checkpoint 编码的 effective `sBA`；
4. 若两者 landmark 在能量、谱和 fixed-k 子空间上等价，论文为跨设定通用性优先称
   `effective Delta W`，实现表登记实际主轨；
5. 不得按哪个轨道与输出相关性更漂亮来选择；非主轨只留数值审计。

### O1.2 固定外部文本，current activation 为主

所有 checkpoint 使用相同 probe token IDs、sample order、positions、attention mask 和 token mask；
固定的是外部文本 `x`，不是隐藏表示。

主轨：

```math
D^{cum,cur}_{D,a,t}=\Delta W^{effective}_{a,t}S_{D,a,t},
```

```math
M^{cum,cur}_{D,a,t}=\|D^{cum,cur}_{D,a,t}\|_F^2
=\mathbb E_{x\sim D}\|\Delta W^{effective}_{a,t}h_{D,a,t}(x)\|_2^2.
```

它表示真实 update 在当前模型实际访问表示上的本层输出能量。

必要但次要的 fixed-base 审计：

```math
M^{cum,0}_{D,a,t}=\|\Delta W^{effective}_{a,t}S_{D,0}\|_F^2.
```

另报 activation exposure ratio：

```math
A^{act}_{D,a,t}=M^{cum,cur}_{D,a,t}/M^{cum,0}_{D,a,t}.
```

current 是输出预测主量；fixed-base 只用于隔离/审计激活重组，不作为第二套平行主方法。
不得把该 ratio 等同于既有 `Delta Y_H/Delta Y_W` 代数分解。

## O2 —— 真实累计与相邻 checkpoint 几何轨迹【P0】

### O2.1 累计轨迹

```math
\Delta W^{cum}_{a,t}=W_{a,t}-W_0,
```

逐 model/arm/checkpoint/domain/layer/module 保存：

```text
raw_weight_energy = ||Delta W_cum||_F^2
whitened_update_energy_current = ||Delta W_cum S_t||_F^2
whitened_update_energy_fixed = ||Delta W_cum S_0||_F^2
whitened_update_rank = r_epsilon(Delta W_cum S_t)
whitened_update_tail_share
state_rank = r_epsilon(W_t S_t)
```

epsilon 仍为 `{.01,.025,.05,.10}`；主文 epsilon=.05。保存七模块逐行，之后再形成等权、attention/MLP
和已注册 weighting summary，不得只保存均值。

### O2.2 相邻 checkpoint 轨迹

对真实相邻 checkpoint，不插值：

```math
\delta W_{a,t}=W_{a,t}-W_{a,t-1},
```

```math
D^{step}_{D,a,t}=\delta W_{a,t}S_{D,a,t-1},
\qquad
M^{step}_{D,a,t}=\|D^{step}_{D,a,t}\|_F^2.
```

另保存 raw `||delta W||_F^2`、step rank、matrix cosine 和方向重合。可计算：

```math
L^{white}_{D,a,t}=\sum_{\tau\le t}\|D^{step}_{D,a,\tau}\|_F,
\qquad
R^{white}_{D,a,t}=\|D^{cum,cur}_{D,a,t}\|_F,
```

但 path length 只作补充，不能代替累计净位移。

产物：

```text
actual_update_cumulative_geometry.csv
actual_update_stepwise_geometry.csv
actual_update_current_fixed_exposure.csv
actual_update_geometry_manifest.json
```

## O3 —— 真实 checkpoint 输出轨迹【P0；forward-only】

不生成回答。对与 O1/O2 完全相同的固定 probe token stream，运行 base 与真实 checkpoint，
在 batch 内以 full vocabulary 精确计算，保存逐 sample/token 聚合及 manifest；除非已有空间，不要求落盘
全部 logits tensor。

### O3.1 累计输出变化

主口径：

```math
Y^{cum}_{D,a,t}=\mathbb E_x KL[p_0(\cdot|x)\|p_{a,t}(\cdot|x)].
```

### O3.2 相邻输出变化

```math
Y^{step}_{D,a,t}=\mathbb E_x KL[p_{a,t-1}(\cdot|x)\|p_{a,t}(\cdot|x)].
```

### O3.3 Held-out token loss

同一次 forward 计算：

```math
\Delta NLL^{cum}_{D,a,t}=NLL_D(W_{a,t})-NLL_D(W_0),
```

```math
\Delta NLL^{step}_{D,a,t}=NLL_D(W_{a,t})-NLL_D(W_{a,t-1}).
```

主结果为 forward KL 与 NLL。reverse KL、symmetric KL/JS、entropy、EOS probability 可由同一 logits
低成本生成时作为 sensitivity，但不得增加模型 forward 或改变主口径。

最低 probes：`E_general/E_math/E_ood/E_if`。记录 sample/token 数、position mask、padding exclusion、
log-softmax precision、batch reduction、checkpoint/hash。禁止用不同 checkpoint 自由生成的文本比较 KL。

产物：

```text
actual_checkpoint_token_kl_cumulative.parquet
actual_checkpoint_token_kl_stepwise.parquet
actual_checkpoint_nll_trajectory.csv
actual_checkpoint_output_manifest.json
```

## O4 —— 最小 smoke 与执行 gate【P0】

先运行：

- model：Llama；
- arms：OPD、off-KD；
- checkpoints：`{0,20,160,320}`；
- headline layer：L14；
- modules：七模块逐行，聚合后分析；
- probes：`E_general/E_math/E_ood/E_if`；
- 权重：B2 smoke 后选定的单一 effective 主轨；
- output：累计/相邻 KL 与 NLL；无 generation，无行为 Eval。

必须形成匹配 cell：

```text
(model, arm, checkpoint/source_checkpoint, domain, layer, module)
-> raw update geometry
-> current/fixed whitened update geometry
-> actual cumulative/stepwise output KL and NLL
```

O4 完成即回交 raw+manifest；若相关脚本或结果已经存在，按 A0 状态码登记并只补缺格。

## O5 —— 公平“输出增量信息”分析【P0，核心裁决】

分别对累计和相邻目标，按相同 grouped held-out folds 比较：

```text
Model-W:     raw ||Delta W||^2 + raw weight baselines
Model-WS:    Model-W + ||Delta W S||^2 (current; fixed 作为 sensitivity)
Model-WSR:   Model-WS + r_epsilon(Delta W S) + tail/concentration summary
Model-State: 上述 + r_epsilon(W_t S_t)，只检验 state rank 是否再提供信息
```

输出目标：

```text
Y_cumulative_KL
Y_stepwise_KL
Delta_NLL_cumulative
Delta_NLL_stepwise
```

规则：

1. Qwen/Llama 先分开；合并只作补充；
2. OPD/off-KD/SFT/seqKD 使用相同 features 和 folds；
3. checkpoint grouped CV，防止同 checkpoint 的 module/domain 泄漏；
4. leave-one-arm-out，重点用离线臂训练后预测 OPD；
5. 标准化、特征选择和任何系数只在 train fold；
6. 报 held-out MAE/R2/log-error、rank correlation、Delta metric 与 grouped bootstrap/permutation；
7. 首先检验“小—小/大—大”的单调排序；固定比例只作更强的次级假设；
8. 负结果完整保存，不以 post-hoc 换目标或挑 checkpoint 修复。

这里的成功标准不是高绝对预测率，而是：在相同 folds 中，`Model-WS` 相对 `Model-W` 提供稳定的
held-out 增量；`Model-WSR` 再检验 rank/concentration 是否在控制能量后仍有贡献。

产物：

```text
actual_output_incremental_models.csv
actual_output_leave_arm_out_predictions.csv
actual_output_monotonicity_calibration.csv
actual_output_incremental_manifest.json
```

## O6 —— 全量扩展【P1；已有则纳入，否则 O4 回交后执行】

若 O4 有可计算信号或相应任务已经完成/在跑，则扩到：

1. Llama 四臂全部 shared checkpoints；
2. Qwen 四臂全部 shared checkpoints；
3. headline：Qwen L18、Llama L14；
4. sensitivity layers 只复用已有 S/geometry，缺失时不因本任务强制全层重算；
5. alpha=.5/frozenSelf/seed 已有则纳入，未有不阻塞。

O3 输出 KL/NLL 是 model×arm×checkpoint×domain 级，不按模块重复 forward；同一份输出目标与各模块
几何行配对。不得为了七模块重复七遍 checkpoint output forward。

## O7 —— 行为 Eval 规则【默认不新增】

最小闭环和全量 O1--O6 均不需要新增行为 Eval。

1. 现有 MATH500/MMLU-Pro/IFEval/其他正式结果只做离线链接，不重跑；
2. 不对新建 counterfactual/rank-k 模型做行为评测，因为本节不构造这些模型；
3. 只有 O5 显示稳定 KL/NLL 增量且现有行为 landmark 缺失时，Theory 可另发 GO；
4. 条件 smoke 每模型至多两个预注册 checkpoint：一个 geometry event、一个 terminal/recovery；
5. 同一模型所有 arms 使用相同 checkpoint，任务只选与 probe 严格配对的已有核心轴；
6. 无 GO 时登记 `NO_NEW_BEHAVIOR_EVAL_BY_DESIGN`。

无新增行为 Eval 时，论文只能主张 output-distribution change 与 held-out token loss 的增量解释，
不能将 O5 结果改写成对 downstream task accuracy 的预测。

## O8 —— 新工时与并行排程（覆盖 B9）

本节无 backward、无 Fisher、无生成行为 Eval。以下包含实现、forward、核验和表格，不含论文写作与沟通；
首个 model×checkpoint×probe profile 后更新 ETA。

| 工作块 | 资源 | 1×96G | 2×96G |
|---|---|---:|---:|
| O0/O1 inventory + 单轨选择 | CPU/I/O | 0.5–1.5 h | 同左 |
| O2 几何复用/补算 | CPU/GPU SVD | 1–4 h | 1–3 h |
| O3 Llama smoke logits/KL/NLL | GPU forward | 2–5 h | 1–3 h |
| O4 cell join + manifest | CPU | 0.5–1.5 h | 同左 |
| O5 grouped/leave-arm-out 分析 | CPU | 1–3 h | 同左 |
| O6 Llama 四臂全轨输出补全 | GPU forward | 3–8 h | 2–5 h |
| O6 Qwen 四臂全轨输出补全 | GPU forward | 4–10 h | 2–6 h |
| 条件行为 smoke | GPU generation | 2–6 h | 1–4 h | 默认不启动 |

条件总墙钟：

- O0--O5 最小 Llama 决策闭环：**3–7 h@2×96G**，或 **5–11 h@1×96G**；
- 双模型四臂全量真实输出轨迹：若 S/输出缓存较全，约 **6–14 h@2×96G**；
- 若关键 S 与 logits 都需补 forward，约 **10–22 h@2×96G**；
- 行为 Eval 不计入默认主线，只有单独 GO 才增加 1–4 h@2×96G。

推荐并行：

```text
CPU: O0/O1 -> O2 已有产物 join -> O5 随输出分块滚动分析
GPU-0: Llama OPD checkpoints/probes -> Llama 其他 arms
GPU-1: Llama off-KD checkpoints/probes -> Qwen 分块
```

base logits、token manifest 和 `S_D0` 每 model/domain 只计算一次并跨 arm 复用；checkpoint logits 每个
model×arm×checkpoint×domain 只 forward 一次，同时得到累计/相邻 KL、NLL 及低成本 sensitivity。

---

# Theory → Coder 覆盖指令（2026-07-26）：全量基线归一化功能秩压缩 \(c_{\varepsilon,D,t}\)

状态：`READY_FOR_FULL_RELATIVE_FUNCTIONAL_CONTRACTION`  
优先级：**P0**  
核心量仍为 \(r_\varepsilon(W_tS_{D,t})\)；\(c_\varepsilon\) 是其基线归一化读数，不是第二个核心指标。  
主输出链接：真实 checkpoint 相对 base 的 full-vocabulary 累计 KL。  
次输出链接：signed/absolute cumulative delta-NLL；stepwise KL/NLL 仅作诊断。  
不新增行为 Eval、Fisher、backward、人工 rank-k 模型或新训练。

本覆盖指令修正并取代旧 O5 中两项 state-rank output-link 实现：

1. 不再把七个 module 行作为七个独立回归观测；
2. 不再跨 probe 直接使用绝对 state rank。

O1--O4 的固定 token、真实 checkpoint、full-vocabulary FP32 输出协议继续有效；O6 双模型四臂
全量 forward 扩展继续执行。

## C0 —— 精确定义与边界

对每个 module：

\[
c_{\varepsilon,D,t,m}
=
\frac{
r_{\varepsilon,D,0,m}-r_{\varepsilon,D,t,m}
}{
r_{\varepsilon,D,0,m}
}.
\]

主聚合：

\[
c_{\varepsilon,D,t}
=
\frac{1}{|\mathcal M|}
\sum_{m\in\mathcal M}c_{\varepsilon,D,t,m},
\qquad
\mathcal M=\{q,k,v,o,gate,up,down\}.
\]

必须先逐模块用其自己的 base rank 归一化，再七模块等权平均。同时输出“先平均 rank、再计算比值”
作为 sensitivity，但不得混称为同一量。

- \(c>0\)：相对 base 压缩；
- \(c=0\)：不变；
- \(c<0\)：相对 base 膨胀。

产物字段命名为 `relative_functional_contraction` 或 `c_epsilon`。禁止命名为 `NCD`；现有 NCD 是沿
log-step 积分的负压缩剂量。

主假设：

> baseline-normalized functional-rank contraction 是否与真实 checkpoint 的累计输出分布漂移呈稳定
> 的“小—小/大—大”关系，并提供纯权重空间缺少的 domain-conditioned 增量信息。

不得声称固定比例、因果关系、任务能力分数或输出改善/恶化由 \(c_\varepsilon\) 单独决定。

## C1 —— 全量覆盖矩阵【P0】

先从 registry/manifest 构造机器可读覆盖表，不凭目录名猜测。纳入：

1. models：Qwen、Llama；
2. core arms：OPD、SFT、off-KD、seqKD；
3. checkpoints：各 model/arm 全部正式保存点；跨臂主比较使用 shared checkpoints；
4. probes：每个模型完整正式 probe panel；跨模型合并只使用语义和 token protocol 严格匹配的交集；
5. headline layers：Qwen L18、Llama L14；
6. 已有 sensitivity layers：Qwen L9/L27、Llama L7/L21；
7. epsilon：\(\{.01,.025,.05,.10\}\)，主文冻结 .05；
8. alpha=.5、frozenSelf、独立 seed：只要已有正式 checkpoint 与同协议 probe，就纳入独立 panel；
   若上游尚未完成，登记 `PENDING_UPSTREAM`，不阻塞核心四臂。

最低核心 output-link 覆盖：

```text
model x core-arm x all-shared-checkpoint x
{E_general,E_math,E_ood,E_if} x headline-layer x 7 modules x 4 epsilons
```

Numina、hard-math 等其他正式 probes 在具有完全相同 fixed-token manifest 的 base/current forward 时
纳入扩展表。不得按近似名称强行匹配不同数据集、tokenization 或 sample manifest。

输出：

```text
relative_functional_contraction_coverage.csv
relative_functional_contraction_missing_registry.csv
```

逐格状态限定为 `AVAILABLE_COMPLETE`、`PENDING_UPSTREAM`、`MISSING_STATE_RANK`、`MISSING_BASE_RANK`、
`MISSING_MATCHED_OUTPUT`、`PROTOCOL_MISMATCH`。

## C2 —— 从现有 \(r_\varepsilon\) 表生成全部 \(c_\varepsilon\)【P0；CPU 优先】

对每个 model×arm×checkpoint×probe×layer×module×epsilon，严格匹配同
model×probe×layer×module×epsilon 的 base rank，保存：

```text
state_rank_base
state_rank_current
state_rank_delta = current - base
absolute_contraction = base - current
relative_functional_contraction_module
relative_functional_contraction_equal7
relative_functional_contraction_ratio_of_means_sensitivity
attention_group_relative_contraction
mlp_group_relative_contraction
```

硬规则：

1. denominator 必须大于零，不得添加任意小常数；
2. 不跨 module 或 probe 使用 denominator；
3. module 行只用于审计；模型状态统计必须先聚合；
4. 统计观测单位是 model×arm×checkpoint×probe×layer×epsilon，不是 module；
5. current 主定义为 \(r_\varepsilon(W_{t,m}S_{D,t,m})\)；
6. fixed/centered state-rank 已有则生成同公式 sensitivity，缺失不阻塞 current；
7. merged-BF16 update rank/tail 不得混入本量；这里测实际状态 \(W_tS_t\)，不是
   \(r_\varepsilon(\Delta W_{\rm merged})\)。

输出：

```text
relative_functional_contraction_all_cells.csv
relative_functional_contraction_module_audit.csv
relative_functional_contraction_aggregation_sensitivity.csv
```

## C3 —— 补齐全部匹配的真实输出轨迹【P0；O6 forward】

复用 O3，同一 fixed-token stream 计算：

\[
Y^{cum}_{D,a,t}
=
\mathbb E_x KL[p_0(\cdot|x)\|p_{a,t}(\cdot|x)],
\]

\[
\Delta NLL^{cum}_{D,a,t}
=
NLL_D(W_{a,t})-NLL_D(W_0).
\]

同一次 forward 保存 cumulative KL、signed delta-NLL、absolute delta-NLL。已有 parquet/logits 只聚合，
不得重跑；不得为七模块重复 forward；不得生成自由回答或新增行为 Eval。

输出：

```text
relative_contraction_matched_cumulative_outputs.csv
relative_contraction_matched_stepwise_outputs.csv
```

## C4 —— 全量相关与公平增量分析【P0；纠正旧 O5】

### C4.1 必报描述性统计

对 cumulative KL、signed delta-NLL、absolute delta-NLL 分别报告：

1. 每个 model×arm 跨 checkpoint×domain 的 Spearman/Kendall；
2. 每个 model×checkpoint 的域内相关；
3. 每个 model×arm×domain 的时间相关；
4. checkpoint 内 rank/demean 后的 pooled association；
5. Qwen/Llama 分开，合并只作补充；
6. epsilon=.05 主结果与四 epsilon sensitivity；
7. headline layer 主结果与已有上下层 sensitivity。

并列比较 absolute state rank（只作同域审计）、\(\Delta r_\varepsilon\)、absolute contraction、
\(c_\varepsilon\)、raw weight energy、whitened update energy。

### C4.2 模型状态级 held-out 分析

每行必须是聚合后的 model-state cell。至少比较：

```text
Model-W:    raw update magnitude baselines
Model-C:    c_epsilon alone
Model-WC:   Model-W + c_epsilon
Model-WS:   Model-W + whitened update energy
Model-WSC:  Model-W + whitened update energy + c_epsilon
```

merged-BF16 update spectrum 不得进入 headline model。主 folds：

1. leave-one-checkpoint-out / grouped-checkpoint CV；
2. leave-one-arm-out，重点离线臂预测 OPD；
3. leave-one-domain-out sensitivity；
4. base0 不进入主相关/误差，只作零点 sensitivity；
5. scaling、transform、regularization 只在 train fold；
6. 线性和 log 尺度都报告，不得择优隐藏；
7. uncertainty 以 checkpoint/trajectory block 为单位；
8. module/probe cell 不得冒充独立 seed。

当前 24-cell Llama smoke 只作为待复现线索：

```text
Spearman(c_epsilon, cumulative_KL):
  OPD = 0.888
  off-KD = 0.874
  pooled = 0.943

mean within-checkpoint domain-order Spearman:
  approximately 0.80 across six arm x checkpoint groups
```

正式产物必须说明这些数值在全点、双模型、四臂、四 epsilon 下是否保持。

输出：

```text
relative_contraction_output_correlations.csv
relative_contraction_within_checkpoint_correlations.csv
relative_contraction_grouped_models.csv
relative_contraction_leave_arm_domain_out.csv
relative_contraction_raw_predictions.csv
```

## C5 —— 累计与相邻量分工【P1；不增加 forward】

\(c_{\varepsilon,D,t}\) 的首要匹配对象是 cumulative KL/NLL。相邻诊断定义：

\[
\delta c_{\varepsilon,D,t}
=c_{\varepsilon,D,t}-c_{\varepsilon,D,t^-}.
\]

完整保存 signed delta-c vs signed delta-NLL、absolute delta-c vs stepwise KL、absolute delta-c vs
absolute stepwise delta-NLL。不得因 stepwise 弱而删除。

## C6 —— 预注册裁决树

1. 跨模型、主要 arms、within-checkpoint 均稳定为正，且 WC/WSC held-out 优于 W：
   支持 \(c_\varepsilon\) 提供 domain-conditioned output-departure 增量信息。
2. cumulative KL 稳定而 signed NLL 不稳定：
   支持压缩比例关联 departure magnitude，但不决定 readout valence。
3. 仅 OPD 稳定：限定为 OPD/on-policy 条件经验关系。
4. pooled 强而 within-checkpoint/held-out 弱：
   判定主要由训练进度或 arm separation 驱动，不作输出代理结论。
5. 跨模型/epsilon 符号不稳：
   \(c_\varepsilon\) 只保留为归一化描述量，主线回到 \(r_\varepsilon\) 轨迹分离。

任何分支都不得声称 \(c_\varepsilon\) 是能力分数、压缩必然导致行为下降、单层空间对最终 logits
全局完备，或相关识别了单一 on-policy 因果机制。

## C7 —— 执行顺序、并行与回交

执行：C1 coverage → C2 CPU 生成全量 \(c_\varepsilon\) → C4 对已有 outputs 滚动分析；同时 C3 在 GPU
只补缺失 fixed-token forward；随后重算 C4、完成 C5，并按 C6 原样回交。

并行：

```text
CPU:   coverage、c_epsilon、聚合修正、grouped analysis
GPU-0: Llama 剩余 arms/checkpoints/probes fixed-token outputs
GPU-1: Qwen 四臂 shared checkpoints/probes fixed-token outputs
```

预计：

- state-rank 与 outputs 大部已有：CPU **1–3 h**；
- 需补四臂 output forward：约 **6–14 h@2×96G**；
- 两模型多个 probe 均缺 logits：约 **10–22 h@2×96G**；
- 不增加行为 Eval、backward、Fisher 或新训练。

最终回交：

```text
full_relative_functional_contraction_theory_handoff.md
full_relative_functional_contraction_manifest.json
```

handoff 只报告覆盖、原始数值、缺失格和各裁决分支所需事实，不自行宣布理论成立。

---

# Theory → Coder 补齐与纠偏指令（2026-07-26）：完成全量 \(c_\varepsilon\) 输出闭环

状态：`RETURN_FOR_COMPLETION_AFTER_PARTIAL_POSITIVE_RESULT`

优先级：**P0**

上一轮 `full_relative_functional_contraction_theory_handoff.md` 已确认一个重要正信号，但其状态
`COMPLETE_WITH_DECLARED_COVERAGE` 只表示已排入队列的 140 个 output cells 完成，不表示上一条
Theory 指令要求的双模型×四臂×全 shared checkpoint×四核心 probe 矩阵完成。本节要求补齐缺格、
纠正分析单位和交付状态；不得用“缺失已登记”再次替代目标矩阵完成。

## D0 —— 上一轮已确认的正结果：必须原样复现，不得重新选点

headline \(\varepsilon=.05\)、七模块先逐模块归一化再等权平均，当前正式相关为：

| model | arm | matched cells | Spearman \(c_\varepsilon\) vs cumulative KL | vs signed cumulative NLL | vs absolute cumulative NLL |
|---|---|---:|---:|---:|---:|
| Llama | OPD | 24 | .943478 | .846087 | .847826 |
| Llama | SFT | 24 | .954783 | .164348 | .413043 |
| Llama | off-KD | 24 | .895652 | −.056522 | .180000 |
| Llama | seqKD | 24 | .808696 | .172174 | .183478 |
| Qwen | OPD | 18 | .948400 | .822497 | .760578 |

Llama 固定 checkpoint、跨 4 arms×4 domains 的相关依次为：

```text
step5=.650000
step20=.870588
step40=.826471
step80=.747059
step160=.705882
step320=.711765
```

Llama checkpoint-demeaned pooled correlation 为 `.748223`；Qwen 当前只有 OPD×两域，demeaned
correlation 为 `-.223942`，不得隐藏。以上均为需复现的审计锚点，不是允许 coder 选择的结论。

## D1 —— 上次执行的问题与原因登记

### D1.1 “140/140 complete”不是全任务完成

registry 共 3168 格：

```text
AVAILABLE_COMPLETE = 1052
MISSING_STATE_RANK = 628
PENDING_UPSTREAM = 1488
```

即正式可用约 33.2%，缺 state rank 约 19.8%，pending 约 47.0%。最终状态应解释为
`PARTIAL_CORE_COMPLETE`，不能继续称“full completed”。只有 D2--D7 的 core matrix 满足 done criteria 后
才可使用 `COMPLETE_CORE_MATRIX`。

### D1.2 Qwen 没有完成四臂四域闭环

当前 Qwen output table 只有 base+OPD；与 \(c_\varepsilon\) 严格匹配的正式格进一步只有：

```text
arm = OPD
probes = E_general, E_ood
layers = 9,18,27
epsilon = .01,.05
```

Qwen SFT/off-KD/seqKD、\(E_{math}\)、\(E_{if}\)、\(.025/.10\) 尚不能支撑“双模型四臂”主张。
`relative_functional_contraction_all_cells.csv` 中存在的 legacy Qwen 行不能绕过 coverage/protocol 状态
直接进入正式结论。

### D1.3 Model-C 被错误限制到 24 个共同特征格

当前脚本先执行：

```text
model_data = joined.dropna(
  raw_update_energy,
  whitened_update_energy,
  c_epsilon
)
```

然后才分别拟合 Model-W/Model-C/Model-WS。因此即使 Model-C 在 Llama 四臂 96 个状态上可用，也被
raw-W/WS 的缺失压缩到 OPD/off-KD×{20,160,320}×四域，共 24 行、3 个 checkpoint groups。

必须拆成两张结果：

1. `full-availability`：每个 feature set 使用自身全部合法 cells；
2. `common-grid fair comparison`：W/C/WS 使用完全相同的共同 cells/folds。

两张都要报告；不得用 common-grid 删除 Model-C 的全量复现，也不得用各自 coverage 表冒充公平比较。

### D1.4 四 epsilon、上下层、逐模块只有 raw cells，没有正式统计表

上一条指令要求 epsilon/layer/module sensitivity 的正式相关与 coverage；当前 correlation CSV 只冻结
headline layer、\(\varepsilon=.05\)。原始表虽可复算，但仍属于未交付分析。

### D1.5 within-checkpoint 的名称和估计对象不清楚

当前 `model_checkpoint_within_domain` 实际是在固定 model×checkpoint 后，把 4 arms×4 domains 共
16 行放在一起，并不是“同 arm 内跨 domain”。必须重命名并同时输出：

1. fixed model×checkpoint，across arm×domain；
2. fixed model×arm×checkpoint，across domains；
3. fixed model×domain×checkpoint，across arms；
4. checkpoint-demeaned pooled。

### D1.6 公平权重空间对照仍只覆盖最小 Llama 子集

raw update energy 与 whitened update energy 只来自上一轮 Llama OPD/off-KD 三 checkpoint O5 缓存。
因此当前结果能说明 `Model-C` 在该小交集优于 W/WS，不能说明它已经在双模型四臂全点公平优于
权重空间。

### D1.7 Qwen 的高 pooled correlation 主要仍是时间轨迹证据

Qwen OPD 两域各自沿 checkpoint 的 \(\rho=.983333\)，但 checkpoint-demeaned 为负。当前证据支持
“Qwen OPD 中 \(c_\varepsilon\) 与累计 KL 共同随训练发展”，不支持 Qwen 稳定跨域排序。补齐四域和
离线臂前不得升级语言。

## D2 —— 先解决 inventory/path/protocol，不得立即重算【P0；CPU/I/O】

对所有 `MISSING_STATE_RANK` 和 `PENDING_UPSTREAM` 做真实路径审计：

1. 搜索已有 merged checkpoints、state spectra、probe manifests、token outputs；
2. 记录“实际不存在”与“registry 路径写错/未发现”两种状态；
3. 现有正式 artifact 只做 schema/protocol adapter，不重复 GPU forward；
4. legacy Qwen state 行必须逐格验证 model hash、checkpoint、probe sample IDs、layer、module、epsilon、
   current whitening protocol；验证后才能升级为 AVAILABLE_COMPLETE；
5. 对找不到的模型目录写出 searched paths 与上游 producer，不得只写一个预期路径。

输出：

```text
relative_contraction_gap_path_audit.csv
relative_contraction_recovered_artifacts.csv
relative_contraction_unrecoverable_registry.csv
```

## D3 —— 正式补交 Model-C 全可用点 held-out【P0；CPU】

不等待 W/WS 补齐，先在所有合法 \(c_\varepsilon\)+output cells 上正式运行：

```text
Model-C = relative_functional_contraction_equal7 alone
```

主 folds：leave-one-checkpoint-out；另报 leave-one-arm-out、leave-one-domain-out。base0 不进入主误差。

最低应复现的诊断范围：

```text
Llama four arms: 96 rows, 6 checkpoint groups
Qwen current OPD scope: 18 rows, 9 checkpoint groups
```

特别注意：上一轮 `relative_contraction_leave_arm_domain_out.csv` 中的
OPD↔off-KD leave-one-arm-out 只覆盖

```text
arms = OPD,off-KD
checkpoints = 20,160,320
probes = E_general,E_math,E_ood,E_if
```

即 24-row 受限共同网格。它之所以只有三个 checkpoint，是 D1.3 的错误交集造成的，不是确认性
landmark 设计，也不是 Model-C 的完整轨迹。该表可以保留为 W/C/WS 的旧 common-grid 审计，但以下
数值不得再称为“Llama 全轨迹跨臂迁移”：

```text
held-out off-KD: actual mean KL=.041494, Model-C prediction=.043832
held-out OPD:    actual mean KL=.180404, Model-C prediction=.102365
```

正式 Model-C leave-one-arm-out 必须使用四臂全部 96 rows，即
`{5,20,40,80,160,320} × 4 probes × 4 arms`；step0 是
\(c_\varepsilon=0,\ KL=0,\ \Delta NLL=0\) 的定义锚点，不进入主预测误差。若完整结果仍显示
held-out OPD 被低估，再登记为跨臂 calibration gap；在此之前只能称旧受限网格现象。

Theory 用交付表的只读复算锚点为：

| scope | rows/groups | MAE | R2 | held-out Spearman |
|---|---:|---:|---:|---:|
| Llama four arms | 96/6 | .025813 | .772978 | .912520 |
| Qwen OPD current scope | 18/9 | .019372 | .747355 | .915377 |

coder 必须用正式脚本复现并输出 raw predictions；数值不一致时先查 fold、Ridge scaling、cell 去重，
不得以 Theory 复算替换正式产物。

输出：

```text
relative_contraction_model_c_full_grouped.csv
relative_contraction_model_c_full_predictions.csv
```

## D4 —— 补齐 Qwen 核心四臂四域矩阵【P0；GPU forward 仅补缺】

Qwen 本轮正式改用与 Llama 同类的 **full merged-state track**。这是状态指标
\(r_\varepsilon(W_tS_{D,t})\) 的主轨，不是 adapter update-space 轨。目标矩阵分为两套：

```text
model = Qwen
arms = OPD,SFT,off-KD,seqKD
full-availability checkpoints = each arm's all recoverable formal checkpoints
common-grid checkpoints = four arms' formal checkpoint intersection
probes = E_general,E_math,E_ood,E_if
layer = 18
modules = 7
epsilon = .01,.025,.05,.10
```

全可用轨用于每条 arm 的轨迹、相关和 Model-C；共同网格用于四臂 cellwise 比较以及 W/C/WS 公平
比较。不得为了共同网格删除 full-availability 结果，也不得将不同 checkpoint coverage 的 pooled
相关直接当成跨臂比较。

### D4.1 merged-state 数值协议

每个 Qwen checkpoint 必须按下列单一状态轨计算：

1. 从该 checkpoint 的完整 merged model 得到当前权重 \(W_t\)；
2. 用同一个 merged model、同一组冻结 probe sample IDs 前向得到当前激活并构造
   \(S_{D,t}\)；
3. 计算 \(W_tS_{D,t}\) 的奇异谱、七模块 \(r_\varepsilon\) 和
   \(c_{\varepsilon,D,t}\)；
4. cumulative KL、signed/absolute NLL 必须由同一个 merged checkpoint、同一 probe token
   mask 与 sample IDs 得到；
5. step0 必须经过与训练 checkpoint 相同的保存/加载/materialization 路径，避免把 base 与
   checkpoint 的 dtype 路径差异写进 \(c_\varepsilon\)。

manifest 对每个 artifact 必须逐项记录：

```text
checkpoint_storage_dtype
merge_compute_dtype
model_load_dtype
activation_dtype
gram_and_whitening_dtype
WS_matmul_dtype
svd_input_dtype
singular_value_accumulation_dtype
logit_forward_dtype
logit_storage_dtype
KL_NLL_compute_dtype
```

禁止笼统写“BF16 计算”或“FP32 计算”。如果实际路径是
`merged BF16 on disk -> FP16 materialization -> FP32 W/S/matmul/SVD statistics`，必须原样写出。

Adapter \(sBA\) FP32 只进入独立的 update-space/displacement 数值审计；不得用 \(BA\) 代替
本节的完整 \(W_t\)，也不得通过 `merged_t - merged_0` 恢复本节状态量。merged-subtract 的低精度
尾部问题针对 \(\Delta W\) 谱，不应误用于否定完整状态 \(W_tS_{D,t}\)。

在不妨碍全量补算的前提下，抽取 `base + OPD@早/中/晚 + 至少一条离线臂@中期` 做数值 parity：
比较落盘 merged-state 与 on-the-fly \(W_0+sBA\) 物化后的模块级 \(r_\varepsilon\)、七模块均值及
四个 epsilon 排序。该 parity 只审计量化稳健性，不新建第二条论文主轨。

执行顺序：

1. 优先恢复现有 state spectra/checkpoints；
2. 已有 legacy Qwen 行只有通过 D4.1 的 checkpoint/probe/dtype/protocol 验证后才能并入；
3. 缺 \(c_\varepsilon\) 时只补 merged-state rank 所需 forward/SVD；
4. 缺 output 时按 O3 fixed-token full-vocabulary KL/NLL 协议 forward；
5. base logits 每 probe 只计算一次；模型 checkpoint output 不按 module 重复；
6. 仍禁止自由生成回答和新增行为 Eval。

若某条 arm 的 checkpoint 实际不可恢复，最终状态必须是 `BLOCKED_MISSING_UPSTREAM_MODEL`，并将正文
主张固定为“Llama 四臂 + Qwen OPD cross-model replication”，不得继续称双模型四臂。

## D5 —— 全量 raw-W/WS 公平共同网格【P0/P1】

围绕论文“相较纯权重空间多表达什么”的问题，补：

```text
raw_update_energy_equal7
whitened_update_energy_equal7
c_epsilon
```

优先顺序：

1. Llama 四臂×6 checkpoints×四域；
2. Qwen 四臂 shared checkpoints×四域；
3. current-S 主轨；fixed-S 仅 sensitivity；
4. update rank/tail 不进入 headline，因为 merged-BF16 差分谱已确认受量化尾部污染。

同一个 common grid、同一 folds 比较：

```text
Model-W
Model-C
Model-WC
Model-WS
Model-WSC
```

必须同时保留 D3 的 full-availability Model-C。共同网格若因 W/WS 缺失缩小，要在表头写出精确 rows、
arms、checkpoints、probes，不能只写 `complete`。

## D6 —— 将现有 robustness 正式落表【P0；CPU】

从现有 all-cells/module audit 生成正式表，不需要新 forward：

1. 每 model×arm×layer×epsilon 的 \(c_\varepsilon\)-KL Spearman/Kendall/rows；
2. module-first normalization 与 ratio-of-means sensitivity；
3. 七模块逐模块相关；
4. attention/MLP group；
5. 每个 fixed arm×checkpoint 的跨域相关；
6. 每个 fixed domain×checkpoint 的跨臂相关；
7. checkpoint-demeaned pooled；
8. cumulative KL、signed NLL、absolute NLL 分开。

Theory 复算的检查范围：

```text
Llama OPD across 3 layers x 4 eps: .785--.971
Llama off-KD: .762--.943
Llama SFT: .674--.961
Llama seqKD: .544--.853
Qwen OPD current available 3 layers x {.01,.05}: .876--.948
```

模块审计检查：Llama OPD 与 Qwen OPD 七模块应全部为正；off-KD/seqKD 的 o_proj 可能为例外，不能隐藏。

输出：

```text
relative_contraction_epsilon_layer_correlations.csv
relative_contraction_module_correlations.csv
relative_contraction_within_arm_checkpoint_domain.csv
relative_contraction_within_domain_checkpoint_arm.csv
relative_contraction_demeaned_correlations.csv
```

## D7 —— 累计、NLL 与 stepwise 的理论分支必须机械输出【P0；CPU】

最终表必须明确区分：

1. cumulative KL：output-distribution departure magnitude；
2. signed NLL：变化效价/likelihood 改善或恶化；
3. absolute NLL：token-loss 变化强度；
4. stepwise KL/NLL：相邻变化诊断。

当前结果显示四条 Llama arms 的 cumulative KL 都强相关，但 signed NLL 主要在 OPD 强；不得合成一个
“输出表现相关”数字。stepwise 当前约为：

```text
abs(delta-c) vs step-KL:
Llama OPD=.598
Llama off-KD=.587
Llama SFT=.428
Llama seqKD=.310
Qwen OPD=.404
```

### D7.1 OPD-specific contraction--readout coupling【确认性分析】

当前待确认的关键结构是：

```text
                           c_epsilon vs cumulative KL   c_epsilon vs signed NLL
Llama OPD                           .943478                       .846087
Llama off-KD                        .895652                      -.056522
Llama seqKD                         .808696                       .172174
Llama SFT                           .954783                       .164348
Qwen OPD current scope              .948400                       .822497
```

其中

\[
\Delta\mathrm{NLL}_{D,t}
=\mathrm{NLL}_{D,t}-\mathrm{NLL}_{D,0},
\]

正值严格表示固定 reference token 的 likelihood 恶化。不得把 signed NLL 改成绝对值后继续使用
“恶化方向”语言，也不得把 token NLL 自动等同于行为准确率。

coder 必须机械输出以下检验，而不是只复报 pooled Spearman：

1. 每 model×arm 的全可用 checkpoint×domain Pearson、Spearman、Kendall、rows；
2. 去训练时钟：分别对 \(c_\varepsilon\) 与 signed NLL 回归/秩残差化
   `log1p(checkpoint)` 后的相关；
3. fixed arm×checkpoint 后跨 domain 的相关及 checkpoint-demeaned pooled；
4. fixed arm×domain 后沿 checkpoint 的相关，并报告各 domain，不只报告合并值；
5. 相邻 checkpoint 的 signed step-NLL 对
   \(\Delta c_\varepsilon\)、\(|\Delta c_\varepsilon|\)；
6. leave-one-checkpoint-out 与 leave-one-domain-out 预测；
7. checkpoint/domain grouped bootstrap CI 与 arm-label permutation；
8. 在统一 model/common-grid 上拟合
   `signed_NLL ~ c_epsilon + OPD + c_epsilon:OPD + log1p(step) + domain`，
   报交互系数与 grouped CI。不得用“OPD 显著、其他臂不显著”代替直接交互检验；
9. 按 D6 对 epsilon、层、模块、聚合方式做同方向稳健性；
10. Qwen merged-state 四臂补齐后，按相同协议完整复算；当前只有 Qwen OPD 的结果只能称
    cross-model OPD replication，不能称 Qwen 内四臂特异性。

分支判定：

```text
A. 两模型 OPD 均为正，去 step 后仍稳定，且 OPD×c 交互为正：
   登记为 OPD-specific contraction-readout coupling；
   可解释为压缩幅度在 OPD 中稳定投影到不利 fixed-token readout 方向。

B. pooled 为正，但去 step/first-difference 后消失：
   只能登记为 OPD contraction 与 signed NLL 共同随训练累积；
   禁止使用“OPD 特异方向耦合”。

C. Qwen 四臂补齐后离线臂也同样为正：
   改写为跨训练范式 contraction-readout coupling，不再称 OPD 特异。

D. 只在部分 epsilon/layer/module 成立：
   降为探索性、层/尺度条件结论。
```

若 A 成立，C6 总结应明确分成两层：累计状态压缩普遍追踪
`output-distribution departure magnitude`；OPD 进一步表现出
`adverse signed readout alignment`。若 B--D 成立则按分支降级。stepwise 映射较弱时必须如实并列。
coder 只交付检验、置信区间、coverage 与 branch code，不自行扩展因果语言。

新增/更新输出：

```text
relative_contraction_signed_nll_correlations.csv
relative_contraction_signed_nll_detrended.csv
relative_contraction_signed_nll_stepwise.csv
relative_contraction_signed_nll_grouped_models.csv
relative_contraction_signed_nll_interaction.csv
relative_contraction_signed_nll_predictions.csv
qwen_merged_state_numeric_protocol.json
qwen_merged_state_parity_audit.csv
```

## D8 —— 完成标准与最终回交

只有同时满足以下条件才可标记 `COMPLETE_CORE_MATRIX`：

1. D2 对全部 gap 完成路径/协议审计；
2. D3 Model-C 全可用点正式 held-out 产物完成；
3. Llama 四臂核心矩阵完整；
4. Qwen 四臂四域 merged-state 完整，且 numeric protocol/parity 审计落盘；或明确
   `BLOCKED_MISSING_UPSTREAM_MODEL` 并冻结降级主张；
5. D5 common-grid W/C/WS 公平表完成；
6. D6 四 epsilon、层、模块、聚合 sensitivity 正式落表；
7. D7 cumulative/signed/absolute/stepwise 分开，D7.1 去 step、交互和 grouped CI 已完成并返回
   A/B/C/D branch code；
8. manifest 逐文件记录 rows/hash/protocol，并给出 target-matrix completion fraction；
9. handoff 同时报告 full-availability 与 common-grid，不再仅报告 queue completion。

最终回交：

```text
full_relative_functional_contraction_completion_handoff.md
full_relative_functional_contraction_completion_manifest.json
```

建议状态码：

```text
COMPLETE_CORE_MATRIX
PARTIAL_CORE_COMPLETE_BLOCKED_UPSTREAM
FAILED_PROTOCOL_MISMATCH
```

禁止再次使用含糊的 `COMPLETE_WITH_DECLARED_COVERAGE` 作为全任务完成状态。

## D9 —— 资源与非范围

可并行：

```text
CPU: D2 inventory + D3 full Model-C + D6/D7 tables
GPU-0: Llama 缺失 W/WS 或 output cells
GPU-1: Qwen 四臂缺失 state/output cells
```

预计：

- D2/D3/D6/D7：CPU 约 1--3 h；
- Llama common-grid W/WS 补齐：依缓存约 2--6 h@1×96G；
- Qwen 四臂 fixed-token output/state 补齐：约 6--14 h@2×96G；
- 若 Qwen 上游 checkpoint 确实不存在，先完成 D2 后停止 GPU，不虚构重训权限。

本轮不新增：行为 Eval、Fisher/gradient proxy、rank-k intervention、alpha sweep、新 seed、新训练、
新指标或新理论分支。目标只是把已经出现的 \(c_\varepsilon\) 正结果按原预注册矩阵补齐并公平比较。

---

# D10 —— Llama state-rank matched-numeric parity【P0；CONFIRMED，立即执行】

```yaml
status: CONFIRMED
priority: P0
reason: >
  D8 已完成的 Qwen 与 Llama state-rank 测量的是同一数学对象，但来自两套不同数值实现。
  Llama legacy 为 BF16 checkpoint -> FP16 load/forward -> FP32 WS -> FP32 SVD；
  Qwen D4 为 BF16 checkpoint -> BF16 load/forward -> FP64 Gram eig ->
  FP32 WS -> FP64 SVD input。跨模型 signed-NLL alignment 在完成本 parity 前不得归因于模型架构。
scope: Llama headline L14 的完整 matched state/output 重算；Qwen D4 不重算
gpu: 1×96G 可执行；2×96G 推荐并行
expected_wall_time:
  one_gpu: 1--2 h GPU + 10--25 min CPU
  two_gpu: 40--75 min GPU + 10--25 min CPU，可流水并行
guard: 不训练、不自由生成、不新增行为 Eval、不覆盖 legacy artifact、不改 probe/sample IDs
```

## D10.1 为什么必须执行

本任务不是新增指标，而是统一论文核心状态量

\[
r_{\varepsilon,D,a,t}=r_\varepsilon(W_{a,t}S_{D,a,t})
\]

的跨模型数值协议。现有每个模型内部的四臂比较仍使用一致协议；待消除的是 Qwen–Llama 之间的
load/forward、白化和 SVD dtype 差异。

正式参考实现为：

```text
/root/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_qwen_d4_merged_state.py
```

Llama legacy 实现仅作为旧新 parity 输入：

```text
/root/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_r4_campaign.py
```

不得仅修改旧 CSV 的 dtype 标签；必须重新进行 Llama BF16 forward、当前激活 Gram、state spectrum
与 fixed-token output 计算。

## D10.2 冻结矩阵

```text
model       = Llama-3.2-3B
arms        = OPD,SFT,off-KD,seqKD
checkpoints = 5,20,40,80,160,320
base        = shared step0，不能按四臂重复计数
probes      = E_general,E_math,E_ood,E_if
layer       = L14
modules     = q/k/v/o_proj + gate/up/down_proj（七模块）
epsilon     = .01,.025,.05,.10
whitening   = current/per-checkpoint uncentered second moment
```

覆盖应为：

```text
nonzero matched state-output cells = 4 arms × 6 checkpoints × 4 probes = 96
shared base cells                  = 1 base × 4 probes = 4
total forward/state cells          = 100
unique loaded model states         = 24 nonzero states + 1 shared base = 25
```

同一模型状态应只加载一次，随后顺序处理四个 probes；禁止把 100 cells 实现为 100 次完整模型加载。
多个 epsilon 必须由同一条完整奇异谱一次派生，不重复 forward/SVD。

所有 probe 必须复用 D8 正式矩阵的：

```text
sample IDs
sample order
token mask / selected positions
token weights
reference token IDs
prompt formatting
```

每个 probe 的 `sample_ids_sha256` 必须与 D8 Llama 对应 artifact 精确一致；任何不一致均标记
`FAILED_SAMPLE_PARITY`，不得并入正式相关表。

## D10.3 冻结数值协议

新的 Llama 正式主轨统一为：

```text
checkpoint storage       = BF16 safetensors merged state
model load               = BF16
forward / hidden states  = BF16
activation cast          = FP32
Gram accumulation        = FP32
Gram symmetrization      = FP64
Gram factorization       = FP64 symmetric eigh
negative eigen handling  = clamp_min(0)，并额外审计 min eig / negative mass
S used in WS             = FP32
W used in WS             = FP32, read from the same loaded merged state
WS matmul                = FP32
SVD input                = FP64 cast of the FP32 WS result
SVD / energy accumulation= FP64
logit forward/storage    = BF16/BF16
KL/NLL computation       = FP32 full vocabulary
```

说明：

1. FP64 SVD 不会恢复 FP32 matmul 已损失的信息，其作用是避免分解和尾能量累计继续引入误差；
2. 对称平方根与 Cholesky 因子在精确 PSD 条件下只差右正交变换，不应改变 \(WS\) 奇异值；本任务
   审计的是有限精度、jitter、近零特征值与离散阈值造成的实际差异；
3. `min_eigenvalue`、`negative_eigenvalue_count`、`negative_eigenvalue_mass`、
   `lambda_max` 必须逐 cell×module 记录。主轨仍严格复刻 Qwen D4 的 `clamp_min(0)`，不得临时发明
   ridge/jitter 后混入；
4. step0 必须经过与非零 checkpoint 相同的 BF16 materialization/load/forward 路径；
5. 若 Llama 正式 checkpoint 只有 adapter，则可以从正式 base+adapter 物化 BF16 merged state，
   但必须记录 `merge_compute_dtype`、adapter 路径和 materialized model hash。优先使用 D2 已登记的
   正式 serialized merged checkpoint；
6. Adapter \(sBA\) FP32 不参与本任务的 state-rank；它继续只属于 update-space sensitivity。

每个 artifact 必须实际断言并记录，而不是手填：

```text
checkpoint_storage_dtype
merge_compute_dtype
model_load_dtype
activation_dtype
gram_accumulation_dtype
gram_factorization_dtype
gram_factorization_method
WS_matmul_dtype
svd_input_dtype
singular_value_accumulation_dtype
logit_forward_dtype
logit_storage_dtype
KL_NLL_compute_dtype
```

## D10.4 必须同时重算的 output

不得只重算 \(c_\varepsilon\) 后继续配对旧 FP16 Llama NLL。每个 BF16 state forward 必须同时产生：

```text
cumulative_kl_base_to_current
delta_nll_cumulative        # signed NLL = NLL_t - NLL_0
absolute_delta_nll_cumulative
```

协议继续使用：

```text
fixed probe/reference tokens
same selected positions/token weights
full vocabulary
shared BF16 step0 logits cache per probe
```

不执行 MATH500 自由生成、IFEval 生成、MMLU-Pro accuracy 或任何新行为 Eval。

## D10.5 GPU 后立即进行的 CPU 重建

新的 matched Llama 轨完成后，以其替换正式分析表中的 Llama legacy state/output 行；Qwen D4 行原样
保留。至少重建：

1. module-level 与 equal-seven \(r_\varepsilon,\Delta r_\varepsilon,c_\varepsilon\)；
2. Llama 四臂 \(c_\varepsilon\) 对 cumulative KL、signed NLL、absolute NLL 的
   Pearson/Spearman/Kendall；
3. checkpoint-demeaned、去 `log1p(step)`、within-domain、stepwise；
4. `signed_NLL ~ c + OPD + c:OPD + log1p(step) + domain` 及 grouped CI；
5. D7.1 mechanical branch code；
6. D5 Model-W/C/WC/WS/WSC common-grid 表：W/WS 原始特征不变，但必须以新的 matched
   \(c_\varepsilon\) 替换 Model-C 输入并重新拟合；
7. Llama Model-C full-availability 96-row held-out 表；
8. 双模型 36/36 early compression dominance、18/18 shared-axis dominance 和 \(T=320\) NCD；
9. `human_read` 所需的旧新协议差异摘要，但 coder 只交读数与 branch code，不做理论归因。

训练时钟残差、交互和相关分析不得把 legacy 与 matched Llama 行混合在同一模型中。

## D10.6 legacy → matched parity 审计

逐相同 cell 比较旧 Llama 与新 matched Llama：

```text
module_r_old
module_r_matched
module_rank_difference
equal7_delta_r_old / matched
equal7_c_old / matched
tail_margin_old / matched（若旧谱可恢复）
```

并汇总：

```text
exact-rank match fraction
|rank difference| <= 1 fraction
mean / median / max absolute rank difference
equal-seven MAE / Pearson / Spearman
arm ordering agreement
epsilon ordering agreement
36/36、NCD 与 D7 branch 是否改变
```

即使旧新结果高度一致，论文主轨也切换到 matched 版本；legacy 只作为 numerical sensitivity。若差异
显著，matched 版本优先，旧数字不得继续进入正文。

## D10.7 验收与状态码

完成条件：

1. `100/100` state-output cells；
2. 四 probe sample hash 与 D8 精确一致；
3. 25 个模型状态均有完整 dtype/materialization provenance；
4. 七模块完整 spectrum 和四 epsilon rank 均存在；
5. BF16 base/output cache 与当前 checkpoint output 同协议；
6. D10.5 九类 CPU 表重建完成；
7. legacy→matched parity 表完成；
8. 所有文件有 rows/bytes/SHA-256，最终 handoff 同时报 raw 与 aggregate coverage。

状态码：

```text
COMPLETE_MATCHED_NUMERIC_PARITY
PARTIAL_PARITY_BLOCKED_UPSTREAM
FAILED_SAMPLE_PARITY
FAILED_NUMERIC_PROTOCOL
```

建议输出目录：

```text
/root/autodl-tmp/cycle09_relative_functional_contraction/d10_llama_numeric_parity/formal/
```

必须回交：

```text
llama_matched_state_numeric_protocol.json
llama_matched_state_module_ranks.csv
llama_matched_state_equal7.csv
llama_matched_fixed_token_outputs.csv
llama_legacy_matched_numeric_parity.csv
llama_legacy_matched_numeric_parity_summary.csv
relative_contraction_*_matched_numeric.csv
d10_llama_numeric_parity_handoff.md
d10_llama_numeric_parity_manifest.json
```

handoff/manifest 镜像到：

```text
/root/LLM-output-density/mypaper/local_experiment_results/
cycle_09_aaai_competitiveness_completion/run_01/mini/
```

## D10.8 并行排程

2×96G 推荐：

```text
T0:
  GPU-0 生成共享 BF16 base 四 probe cache；完成后开始 OPD + SFT
  GPU-1 等待 base cache COMPLETE marker；随后 off-KD + seqKD
  CPU   准备 legacy 索引、sample-hash audit 与输出 schema

T+20 min:
  CPU   对已完成 cell 流水执行 parity；不得提前拟合不完整相关表

T+40--75 min:
  GPU   预期完成 100 cells
  CPU   重建 D10.5 正式表、manifest 与 handoff（10--25 min）
```

共享 base cache 必须采用 lock + atomic write；GPU-1 不得各自生成另一份语义相同但 hash 未核对的 base。
若只有 1×96G，则按 `base → OPD → SFT → off-KD → seqKD` 串行，预计 1--2 h。

本任务可与 strict \(p_k\)、论文写作和其他 CPU 分析并行；不得与另一项占满相同 GPU 的 forward 任务
叠跑。L7/L21、Qwen 重算、行为生成、独立 seed、frozen-self 与新训练均不属于 D10。

---

# A10-SEED —— Llama 独立 seed matched pair 到 step160【Stage A；已纳入 A0--A10 AUTO-GO】

> 本节是当前 reviewer-response 排程中的 A10，与文件前部历史 addendum 的“A10 工时估算”不同。
> machine-readable task ID 固定为 `SEED_A10_STAGE160`，避免脚本命名冲突。

```yaml
status: AUTO_GO_AFTER_PROTOCOL_AND_GPU_PREFLIGHT
priority: P2_HIGH_VALUE
task_id: SEED_A10_STAGE160
model: Llama-3.2-3B
new_master_seed: 43
arms: OPD,off-KD
terminal_step: 160
purpose: >
  以一个额外独立训练 seed 复现核心 matched on/off-policy early contrast，
  并为 A11 的 160->320 续训建立可恢复 checkpoint、数值缓存和早期分析。
estimated_wall_time: 9--14 h@2×96G，包括训练、probe/output 和 CPU 分析
guard: >
  A10 与 A11 是同一条 seed43 训练轨的两个阶段。A10 不得在 step160 只保存 adapter；
  必须保存可无缝续训所需的完整 trainer/RNG/data-position 状态。
authorization: >
  用户于 2026-07-27 授权 A0--A10 无人监管自动执行。A10 不再等待科学结果判断或单独 GO；
  当 A1 数值协议冻结、A10 preflight 通过且 GPU lane 空闲时自动启动。该授权不包含 A11。
```

## A10-SEED.1 冻结训练与 checkpoint

训练条件与 seed0 正式 Llama 完全一致，仅将 master seed 冻结为43：

| arm | support | objective |
|---|---|---|
| OPD | seed43 current-student self-rollout | dense teacher forward-KL |
| off-KD | 与 seed0 完全相同的冻结 teacher rollout | dense teacher forward-KL |

checkpoint 原生保存：

```text
{0,5,10,20,40,80,160}
```

正式分析网格：

```text
{0,5,20,40,80,160}
```

OPD 必须重新生成 seed43 current-self rollout；off-KD 继续使用冻结 teacher support。不得改变原正式
shuffle、mask、token normalization、optimizer、LoRA、teacher、batch、scheduler、sequence length
或 top-32 KL 协议。

`master seed=43` 不能只改 behavior/eval seed，也不能只改 vLLM rollout seed。A10 preflight 必须
展开最终 resolved config，并机器校验至少以下字段：

```text
PYTHONHASHSEED=43
actor_rollout_ref.rollout.seed=43
actor_rollout_ref.actor.data_loader_seed=43
actor_rollout_ref.actor.fsdp_config.seed=43
off-KD Python/NumPy/PyTorch CPU/CUDA/trainer seed=43
```

若某字段在实际 trainer 版本中改名，controller 必须从 resolved config 和首个 checkpoint RNG
manifest 中找到对应项；不得默认沿用42。现有正式 Llama 训练 manifest 与 OPD runner 已核实使用42，
此前目录中出现的 seed43 行为生成文件不是训练 seed，因此不构成冲突。最终训练 manifest 必须同时
登记原正式轨42与新轨43的全部 seed 字段。

## A10-SEED.2 step160 必须是可恢复训练状态

除 adapter/merged model 外，必须原子保存并校验：

```text
optimizer state
lr-scheduler state
global step / epoch / micro-step
AMP/gradient-scaler state（若使用）
Python/NumPy/PyTorch CPU/CUDA RNG states
dataloader/sampler state and next sample index
OPD prompt/batch cursor
rollout sampler RNG and decoding state
teacher-cache/data hashes
trainer config and code commit
```

必须执行一次 dry-resume smoke：

1. 从 step160 checkpoint 启动到 step161；
2. 验证 next batch/prompt IDs、lr、global step 和 RNG provenance；
3. smoke 产物写入隔离目录，不能污染正式 step160；
4. 只有 dry-resume 通过，A10 才能标记为 A11-ready。

状态字段：

```text
resume_ready=true/false
resume_smoke_next_prompt_hash
resume_smoke_lr
resume_smoke_rng_hash
resume_smoke_model_hash_before_step
```

## A10-SEED.3 可被 A11 直接复用的测量

数值协议与 D10 完全一致。几何 panel 使用六个原正式 Llama probes：

```text
S_math,E_math,E_math_hard_v2,E_ood,E_if,E_general
```

output-link panel 使用：

```text
E_general,E_math,E_ood,E_if
```

范围：

```text
arms        = OPD,off-KD
steps       = 0,5,20,40,80,160
layer       = L14
modules     = seven
epsilon     = .01,.025,.05,.10
outputs     = cumulative/stepwise KL, signed/absolute NLL
```

A10 必须保存完整 spectrum、current Gram、激活一阶矩、sample/token manifest 和 BF16 base logits cache，
使 A11 能直接复用：

```text
seed43 shared BF16 base state/output cache
OPD/off-KD step<=160 state ranks and spectra
OPD/off-KD step<=160 matched outputs
probe sample hashes and token positions
current Gram / mean sufficient statistics
```

不得只保存聚合相关系数。

## A10-SEED.4 Stage-A 分析

只分析 matched pair，不冒充四臂复制：

\[
M^{\mathrm{pair}}_{s,D,t}
=c_{\varepsilon,s,D,t}^{\mathrm{OPD}}
-c_{\varepsilon,s,D,t}^{\mathrm{offKD}}.
\]

必须输出：

```text
seed0/seed1 OPD-offKD 逐 checkpoint/probe margin
early {20,40,80} sign agreement、Pearson/Spearman/MAE
c_epsilon vs cumulative/signed/absolute output
seed0 -> seed1 与 seed1 -> seed0 的两臂 cross-seed transfer
OPD support length/EOS/repetition/finish-reason seed difference
```

状态码：

```text
COMPLETE_SEED_STAGE160_REUSABLE
COMPLETE_STAGE160_NOT_RESUMABLE
PARTIAL_STAGE160
FAILED_TRAINING_OR_PROTOCOL
```

只有 `COMPLETE_SEED_STAGE160_REUSABLE` 才允许 A11 从 step160 接续。

必须交付：

```text
a10_seed43_stage160_training_manifest.json
a10_seed43_stage160_resume_audit.json
a10_seed43_stage160_state_outputs.csv
a10_two_seed_pair_replication.csv
a10_seed43_stage160_handoff.md
a10_seed43_stage160_manifest.json
```

raw 根目录：

```text
/root/autodl-tmp/cycle09_a10_llama_seed43_stage160/
```

---

# A11 —— 接续 A10 的 Llama 四臂独立 seed 完整 step320 复现【Stage B；等待 A10 与显式 GO】

```yaml
status: DESIGNED_BLOCKED_ON_A10_AND_EXPLICIT_GO
priority: P2_HIGH_VALUE
relation_to_A10: >
  A11 必须消费 A10 的 COMPLETE_SEED_STAGE160_REUSABLE 产物。
  OPD/off-KD 从同一 seed43 step160 trainer state 接续到320，不从0重跑；
  A10 已完成的 base、step<=160 geometry/output 和早期分析全部复用。
purpose: >
  在 A10 matched-pair early replication 之上，续训到320并补入 seed43 SFT/seqKD，
  将证据升级为至少一个模型上的四臂完整独立训练复制；
  同时检验早期 OPD compression dominance、T=320 NCD、c_epsilon-output relation
  以及 signed-readout branch 是否能跨 seed 保持。
model: Llama-3.2-3B
new_master_seed: 43（执行前验证未被现有正式轨使用；冲突则停止并回报，不得看结果后选 seed）
arms: OPD,SFT,off-KD,seqKD
terminal_step: 320
gpu: 2×96G
estimated_incremental_wall_time_after_A10: 11--18 h
estimated_total_wall_time_A10_plus_A11: 20--32 h
guard: >
  不改变数据、teacher、loss、优化器、LoRA、batch、scheduler 或原始 shuffle 规则；
  不新增自由生成行为 Eval；不把 probe/checkpoint/module 当作独立 seed。
```

## A11.1 分阶段设计与新增范围

A10 已经完成 seed43 matched pair 的 step0--160：

\[
c_{\varepsilon,\mathrm{OPD}}-c_{\varepsilon,\mathrm{offKD}}.
\]

A11 新增且只新增：

1. 从 A10 step160 精确接续 OPD 到 step320；
2. 从 A10 step160 精确接续 off-KD 到 step320；
3. 训练 seed43 SFT 与 seqKD 到 step320，补足四臂；
4. 对 OPD/off-KD 只新增 step320 的 matched measurements；
5. 对 SFT/seqKD 补齐正式 shared grid；
6. 合并 A10 与 A11，执行四臂 early、NCD320 与 output-transfer 分析。

论文的 36/36 压缩支配比较 OPD 与 SFT、off-KD、seqKD 三条离线臂。A10 单独只能写“matched pair
在一个额外 seed 上复现”；A11 补齐 SFT/seqKD 后才能写：

> Llama 四臂在一个额外独立训练 seed 上完整复现，Qwen 继续提供第二模型家族的一条正式轨迹。

总 seed 数仍只有两个，因此只能称 `replicated under one additional independent seed`，不能声称已
精确估计训练随机性的总体方差。

## A11.2 训练条件冻结

四臂定义保持原正式轨；其中 OPD/off-KD 的 0--160 部分来自 A10：

| arm | sequence support | objective |
|---|---|---|
| OPD | A10 seed43 current-student self-rollout；从160继续动态刷新 | dense teacher forward-KL |
| off-KD | A10 冻结 teacher rollout；从160继续 | dense teacher forward-KL |
| seqKD | 与 off-KD 完全相同的冻结 teacher rollout、行序和 mask；A11 从0训练 | hard-label CE |
| SFT | 与 seed0 完全相同的外部/reference CoT 数据；A11 从0训练 | hard-label CE |

只改变 master seed，并将其传播到原实现允许的随机源：

```text
LoRA initialization
dropout / CUDA random stream
dataloader RNG（仅当原协议启用 shuffle；原协议 shuffle=false 时仍保持 false）
OPD rollout sampling/decoding seed
worker RNG
```

不得改变：

```text
base/teacher model
prompt/data split
teacher rollout（off-KD/seqKD）
dataset order policy
response mask/token normalization
optimizer/lr/scheduler
batch/gradient accumulation
LoRA rank/alpha/target modules
top-32 KL 规范
maximum sequence length
checkpoint save/merge protocol
```

OPD step161--320 必须由 A10 step160 current student 继续生成新的 seed43 rollout。不得重新从 base
开始，也不得复用 seed0 rollout。off-KD 必须恢复 A10 的 optimizer/scheduler/data cursor；不能把
step160 adapter 作为新初始化后重新建 optimizer。

checkpoint 原生保存：

```text
{0,5,10,20,40,80,160,320}
```

正式 seed 比较网格使用与现有 Llama 主轨严格共有的：

```text
{0,5,20,40,80,160,320}
```

step10 只作可恢复性 artifact；除非 seed0 也按 D10 协议补出 step10，否则不得进入跨 seed 统计。

## A11.3 训练过程必须保存的审计量

每个 arm 保存：

```text
training loss / KL or CE
gradient norm
learning rate
wall time and GPU-hours
checkpoint hash and adapter/merged provenance
all effective random seeds
data order/hash
resume/restart events
```

OPD 逐 step 额外保存：

```text
rollout sample/prompt IDs
response token length
EOS / finish reason / cap-hit
4-gram repetition / distinct-2
teacher retained top-32 mass
rollout and teacher-inference latency
```

离线三臂登记相同 support 的静态长度/EOS/repetition 分布，但不得伪造为逐 checkpoint 动态统计。

A11 启动前必须读取并验证 A10 的 `resume_ready=true`、checkpoint/hash、step161 dry-resume 与全部
sample/probe cache。任何一项不匹配时状态为 `BLOCKED_A10_NOT_RESUMABLE`，不得用仅 adapter 权重
近似续训。

## A11.4 matched geometry 与 output 测量

全部使用 D10 冻结后的正式数值协议：

```text
BF16 merged checkpoint/load/forward
FP32 activation and Gram accumulation
FP64 symmetric eig with negative-eigen audit
FP32 WS matmul
FP64 SVD input and tail-energy accumulation
BF16 logits; FP32 full-vocabulary KL/NLL
```

### 几何 panel

为了复现原 Llama 四臂 18-cell 早期排序，几何使用原正式六 probe 的完全相同 sample manifest：

```text
S_math
E_math
E_math_hard_v2
E_ood
E_if
E_general
```

论文统一命名由 Theory 后处理；coder 不自行重命名 artifact。范围：

```text
layer   = L14
modules = q/k/v/o/gate/up/down
epsilon = .01,.025,.05,.10
steps   = 0,5,20,40,80,160,320
```

复用规则：

```text
OPD/off-KD:
  reuse A10 = {0,5,20,40,80,160}
  compute A11 only = {320}

SFT/seqKD:
  compute A11 = {5,20,40,80,160,320}

shared base:
  reuse A10 BF16 base state/probe cache；不重新 forward
```

由此 A11 新增 geometry model-state×probe cells：

```text
OPD/off-KD @320:       2 arms × 1 step × 6 probes = 12
SFT/seqKD full grid:   2 arms × 6 steps × 6 probes = 72
total new geometry cells                         = 84
```

### output-link panel

与 D8/D10 严格共同的四个核心 probes：

```text
E_general,E_math,E_ood,E_if
```

同一次 forward 产生：

```text
cumulative KL
signed cumulative delta-NLL
absolute cumulative delta-NLL
stepwise KL/NLL（由相邻真实 checkpoint）
```

output 也必须复用 A10：

```text
OPD/off-KD @<=160:     reuse A10
OPD/off-KD @320:       2 × 1 × 4 = 8 new output cells
SFT/seqKD full grid:   2 × 6 × 4 = 48 new output cells
total new output cells              = 56
```

step160→320 的 stepwise output 由 A10 step160 cache 与 A11 step320 output 配对，不重算 step160。

不新增 MATH500、MMLU-Pro 或 IFEval 自由生成行为评测。若之后需要 seed-level 行为复现，必须由 Theory
另发 GO，并单独预注册任务/landmark；不得在 A11 中顺手启动。

## A11.5 一级分析：四臂压缩制度的 seed replication

每个 seed、probe、checkpoint 定义 OPD 对最近离线臂的压缩 margin：

\[
M_{s,D,t}
=c_{\varepsilon,s,D,t}^{\mathrm{OPD}}
-\max_{a\in\{\mathrm{SFT,offKD,seqKD}\}}
c_{\varepsilon,s,D,t}^{a}.
\]

\(M>0\) 表示 OPD 比三条离线臂都更压缩。必须报告：

1. seed0、seed1 分开的逐 cell 四臂原始值与 \(M_{s,D,t}\)；
2. 早期窗口 \(t\in\{20,40,80\}\)：
   - 六 probe 严格 win count（每 seed 18 cells）；
   - 四 core probes win count（每 seed 12 cells）；
   - 每 checkpoint 的 probe-mean margin；
3. step160、step320 的终态/后期 margin，不强迫保持早期 18/18；
4. 每条 arm 的 seed0–seed1 trajectory Pearson/Spearman/MAE；
5. 四臂排序 agreement 与 OPD margin curve 的跨 seed correlation/MAE；
6. 不把 18 cells 当作 18 个随机重复；逐 seed 曲线和 effect size 是主报告。

严格复现不要求 seed1 机械达到 18/18。分级：

```text
FULL_STRICT_REPLICATION:
  seed1 early six-probe 18/18 且三个 checkpoint 的 mean margin 均 > 0

EFFECT_REPLICATION:
  三个 early checkpoint 的 mean margin 均 > 0，
  至少 75% early cells 同方向，且 seed1 early mean margin
  >= 0.25 × seed0 early mean margin

PARTIAL_REPLICATION:
  aggregate mean 为正，但某 early checkpoint 反号、cell agreement <75%，
  或 seed1 margin < 0.25 × seed0 margin

FAILED_REPLICATION:
  至少两个 early checkpoint 的 mean margin <=0，
  或总体 OPD-offline margin 反号
```

Coder 只机械输出状态码与读数；Theory 决定论文措辞。

## A11.6 二级分析：完整 \(T=320\) 轨迹

使用与现有主结果完全相同的 log-time 积分协议，逐 seed 计算四臂 NCD：

\[
\mathrm{NCD}_{s,a}(320)
=
\frac1{|\mathcal D|}
\sum_D
\int_0^{\log(321)}
[-\Delta r_{\varepsilon,s,a,D}(\tau)]_+
\,d\tau.
\]

必须报告：

```text
seed0/seed1 四臂 NCD 原始值
OPD - second-largest NCD margin
四臂 NCD ordering agreement
各 probe 对 NCD 的贡献
early 0--80 / middle 80--160 / late 160--320 分段面积
```

判定：

```text
NCD_REPLICATED:
  两个 seed 均为 OPD 最大、OPD margin 同号，且
  seed1 OPD NCD margin >= 0.25 × seed0 OPD NCD margin

NCD_SEED_SENSITIVE:
  seed1 中 OPD 不再最大、margin 反号，或
  seed1 margin < 0.25 × seed0 margin
```

该分析是 A11 相对 A10 到 step160 的主要新增价值。

## A11.7 三级分析：\(c_\varepsilon\) 与真实 output 的跨 seed 迁移

每个 seed×arm 分别计算：

```text
c_epsilon vs cumulative KL
c_epsilon vs signed NLL
c_epsilon vs absolute NLL
checkpoint-demeaned
log1p(step)-detrended
within-domain temporal
stepwise delta-c vs step-KL/NLL
```

之后进行双向 cross-seed transfer：

```text
train seed0 -> predict seed1
train seed1 -> predict seed0
```

模型至少包括：

```text
Model-C:  output ~ c_epsilon + progress + arm + domain
Model-WC: output ~ weight baselines + c_epsilon + progress + arm + domain
```

报告：

```text
held-out MAE/R2/Spearman
calibration slope/intercept
OPD/offline 分解
seed fixed-effect pooled sensitivity
```

只能称 cross-seed transfer，不能把两个 seed 的大量 cell 当作 seed-level 大样本。只有两个 seed 时不报
正态近似 seed CI；可以报告两 seed 均值、范围和逐 seed block bootstrap，但必须标为
`conditional measurement uncertainty`，不是训练随机性总体 CI。

## A11.8 signed-readout branch 的 seed 检验

只有 D10 matched-numeric parity 后仍保留 signed-NLL 结构，A11 才继续检验：

```text
signed_NLL ~ c + OPD + c:OPD + log1p(step) + domain + seed
signed_NLL ~ c * OPD * seed + log1p(step) + domain
```

必须输出每个 seed 的 arm-wise相关与直接交互，不得用“一组显著、另一组不显著”替代交互。

分支：

```text
SEED_STABLE:
  OPD-vs-offline signed alignment 在两个 seed 中同方向，且 c:OPD 交互不反号

SEED_CONDITIONAL:
  seed0 强、seed1 弱，但方向不反号

NOT_REPLICATED:
  seed1 中方向反号或离线臂呈同等结构
```

该分支不影响一级 compression-regime replication；它只决定 output-valence 是否可写成稳定机制。

## A11.9 可选机制审计：support 统计与几何

以 seed 为单位比较 OPD rollout：

```text
length / EOS / repetition / finish reason
support statistics vs c_epsilon/NCD
seed0--seed1 support difference
seed0--seed1 geometry-effect difference
```

只作探索性机制分析：

\[
\text{seed}\rightarrow
\text{realized current-self support}
\rightarrow
\text{functional trajectory}.
\]

不得因两个 seed 的相关就声称 length/EOS/repetition 是唯一中介。

## A11.10 完成标准、交付与状态码

A11 只有同时完成以下内容才算结束：

1. 验证并消费 A10 `COMPLETE_SEED_STAGE160_REUSABLE`，不得复制为新轨；
2. seed43 OPD/off-KD 从 A10 step160 精确续训至 step320；
3. seed43 SFT/seqKD 从 step0 训练至 step320；
4. seed43 四臂 step320 durable checkpoint；
5. checkpoint `{0,5,10,20,40,80,160,320}` inventory 和 hash；
6. A10+A11 合并后的六 probe geometry、四 core probe output 正式共同网格；
7. D10 matched numeric protocol 与 sample-hash parity；
8. A11.5 early four-arm seed replication；
9. A11.6 \(T=320\) NCD replication；
10. A11.7 cross-seed output transfer；
11. 条件执行 A11.8 signed branch；
12. support/training provenance、GPU ledger、rows/bytes/SHA-256；
13. handoff 只交事实、effect sizes、coverage 与机械状态码。

总状态：

```text
COMPLETE_FULL320_SEED_REPLICATION
BLOCKED_A10_NOT_COMPLETE
BLOCKED_A10_NOT_RESUMABLE
PARTIAL_A11_OFFLINE_COMPLETE
PARTIAL_A11_OPD_OFFKD_RESUME_RUNNING
BLOCKED_TRAINING_FAILURE
FAILED_PROTOCOL_OR_SAMPLE_PARITY
```

建议输出：

```text
llama_seed43_training_manifest.json
llama_seed43_support_stats.csv
llama_seed43_checkpoint_inventory.csv
llama_seed43_matched_state_ranks.csv
llama_seed43_matched_outputs.csv
llama_two_seed_early_regime_replication.csv
llama_two_seed_ncd320_replication.csv
llama_two_seed_output_transfer.csv
llama_two_seed_signed_branch.csv
a11_full320_seed_replication_handoff.md
a11_full320_seed_replication_manifest.json
```

正式 raw 输出位于独立目录，不覆盖 seed0：

```text
/root/autodl-tmp/cycle09_a11_llama_seed43/
```

handoff/manifest 镜像至：

```text
/root/LLM-output-density/mypaper/local_experiment_results/
cycle_09_aaai_competitiveness_completion/run_01/mini/
```

## A11.11 真实资源预算与并行排程

现有 Llama OPD 正式轨记录显示：step1--159 的正常 step 约 2.5--3.5 分钟，且 OPD 同时占用
GPU0 student rollout/train 与 GPU1 teacher inference。A10 已支付 0--160 的成本，因此 A11 只预算
OPD/off-KD 的 160--320 续训、SFT/seqKD 新臂和缺失测量：

| 阶段 | 资源 | 预计墙钟 |
|---|---|---:|
| A10 resume/hash/cache preflight | CPU | 0.25–0.75 h |
| seed43 SFT/seqKD 0→320 + off-KD 160→320 | 2×96G 并行排程 | 2–4 h |
| seed43 OPD 160→320 | 2×96G，student+teacher | 7–10 h |
| 84 geometry + 56 output 新 cells | 2×96G | 1–2.5 h |
| seed-level/NCD/output 分析与回交 | CPU | 1–2 h |
| **A11 增量墙钟** | 2×96G | **约11–18 h** |
| **A10+A11 总墙钟** | 2×96G | **约20–32 h** |

推荐：

```text
T0:
  CPU verify A10 resume-ready、hash、base/probe/output caches
  GPU-0 resume off-KD 160->320 -> SFT 0->320
  GPU-1 seqKD 0->320

T+2--4 h:
  两卡从同一 A10 step160 切换 OPD resume：
  GPU-0 student rollout/train，GPU-1 teacher
  CPU 继续 p_k、human_read、论文工作

OPD@320:
  两卡只补 A11 新增的 84 geometry / 56 output cells
  CPU join A10+A11，生成 four-arm seed replication、NCD 与 transfer
```

不得在 A11 OPD 运行时让 GPU1 同时承担其他大模型任务；teacher latency 是 OPD 主循环的一部分。
A11 不得重算 A10 的 base、OPD/off-KD step≤160 state/output；发现缺失时先回报
`A10_REUSE_GAP`，由 Theory 决定补缺。A11 可与 CPU 的 strict \(p_k\)、D10 后处理、human_read 和
论文写作并行，但 OPD 续训阶段会占用全部两张 GPU。

---

# 当前 reviewer-response 实验总表【机器任务唯一总索引，2026-07-27】

```yaml
view_type: machine_experiment_master_index
status: CURRENT
scope: GPU/CPU experiment, numerical analysis, protocol audit
non_scope: human_read, paper writing, abstract, Theory/Coder discussion
supersession_rule: >
  本表负责当前优先级、依赖、资源和执行状态；各任务的公式、矩阵、协议和验收条件仍以
  D10、A10-SEED、A11 及相应历史正式小节为准。若摘要表与详细协议冲突，以时间较新的详细协议为准。
```

| ID | 优先级 | 实验/分析 | 核心问题与冻结范围 | 资源与预计墙钟 | 依赖 | 当前状态 |
|---:|---|---|---|---:|---|---|
| A0 | 已完成基础 | D8 双模型全矩阵 | Qwen 144 + Llama 96，共240个 state-output cells；D5 W/C/WS、D6、D7 | 已完成 | — | `COMPLETE_CORE_MATRIX`；不重跑 |
| A1 | P0 | Llama matched-numeric parity | L14、四臂×6 ckpt×4 probes；BF16 forward、FP64 eig/SVD；同时重算 KL/NLL | 2×96G 40–75 min；1卡1–2 h | 复用现有 checkpoint/probes | 待执行；详细协议=`D10` |
| A2 | P0 | A1 后处理与 legacy parity | 重建 \(c_\varepsilon\)、D7 branch、detrend、interaction、36/36、NCD、Model-C/WC/WSC；比较旧新 rank | CPU 10–30 min | A1 | 待执行 |
| A3 | P0 | strict \(p_k\)、PABS/NSS 权重基线 | 两模型统一 serialized BF16 merged effective difference；joint \(p_k\)；fixed \(k=\{4,8,16,32\}\) | 1×96G 0.5–2 h + CPU 0.5–1 h | A1 时可准备 source SVD | 待执行 |
| A4 | P0 | \(c_\varepsilon\) beyond \(p_k\) 增量信息 | 同 cells/folds 比较 W、\(p_k\)、PABS/NSS、C、WC；KL/NLL、OPD/offKD 判别和已有行为 coverage | CPU 0.5–1.5 h | A2+A3 | 待执行 |
| A5 | P0 | current/fixed/centered 构念消融 | 检查 per-checkpoint activation metric 与激活均值是否驱动主结果；headline 层、核心 probes、landmarks | 复用A1为0.5–1.5 h；缺均值缓存则1–3 h GPU | 最好与A1合并 Gram+\(\mu\) | current已有；fixed/centered待闭环 |
| A6 | P0 | 有限样本、tail-margin、epsilon 稳健性 | sample-count、阈值安全边际、rank跳变；\(\varepsilon=.01/.025/.05/.10\) | CPU 0.5–1.5 h；缺 factors 加1–2 h GPU | 可与A1并行 | pilot已有，正式总结待做 |
| A7 | P1 | 模块、层与聚合稳健性 | 七模块、attention/MLP、等权/output-energy weighting、Qwen L9/18/27、Llama L7/14/21 | CPU 20–60 min | 数据大部来自D6 | 主要是正式汇总 |
| A8 | P1 | 训练实现与 top-32 审计 | trainer、batch、token normalization、precision、scheduler、data order；teacher retained/omitted mass | CPU/I/O 0.5–2 h | 无 | 部分已有，待正式表 |
| A9 | P1 | 行为协议与不确定性审计 | strict/flexible/extract/format定义；item-bootstrap CI；4096/16384 cap证据登记 | CPU 1–2 h | 使用已有逐题输出 | 不新增行为Eval |
| A10 | P2高收益 | 新 seed Stage A：OPD/offKD 到160 | Llama seed43；六probe几何、四probe KL/NLL；early matched contrast；保存完整可续训状态 | 2×96G 9–14 h | A1协议冻结；GPU preflight | `AUTO_GO_AFTER_PROTOCOL_AND_GPU_PREFLIGHT` |
| A11 | P2高收益 | 接续A10到320并补成四臂 | OPD/offKD 从A10 step160续训；SFT/seqKD到320；复用A10 early结果；四臂seed replication、NCD320、cross-seed transfer | A10后增量11–18 h@2×96G | A10=`COMPLETE_SEED_STAGE160_REUSABLE`且显式GO | 已设计；被A10阻塞 |

## 总表执行依赖

必要闭环：

```text
GPU lane:
  A1
  -> A3 / A5 missing cells

CPU lane:
  A3 source/protocol preparation + A6/A7/A8/A9
  -> A2 when A1 lands
  -> A4 when A2+A3 land
```

预计 A1–A9 在2×96G与CPU极限并行下为约 **3–6 h wall-clock**。

独立 seed 是严格串联的 Stage A/Stage B：

```text
A1 numeric protocol frozen
  -> A10: seed43 OPD/offKD 0->160
       -> dry-resume smoke
       -> Stage-A matched-pair analysis
       -> COMPLETE_SEED_STAGE160_REUSABLE
  -> explicit A11 GO
       -> OPD/offKD 160->320
       -> SFT/seqKD 0->320
       -> reuse A10 base + step<=160 state/output
       -> four-arm seed replication + NCD320 + output transfer
```

## A10 → A11 复用契约摘要

| A10 已完成内容 | A11 处理 |
|---|---|
| seed43 OPD/off-KD step0–160 | 从step160恢复 optimizer/scheduler/RNG/data cursor 后续训 |
| step0 BF16 base cache | 直接复用，不重算 |
| OPD/off-KD step≤160 六probe几何 | 直接复用 |
| OPD/off-KD step≤160 四probe KL/NLL | 直接复用 |
| sample/token manifests | hash校验后复用 |
| current Gram、激活均值、完整奇异谱 | 直接用于构念消融和后续分析 |
| Stage-A matched-pair分析 | A11只追加四臂与 \(T=320\)，不得覆盖 |

A11 新增测量严格为：

```text
geometry:
  OPD/off-KD @320      = 2 × 1 × 6 = 12 cells
  SFT/seqKD full grid  = 2 × 6 × 6 = 72 cells
  total                = 84 new cells

output:
  OPD/off-KD @320      = 2 × 1 × 4 = 8 cells
  SFT/seqKD full grid  = 2 × 6 × 4 = 48 cells
  total                = 56 new cells
```

## 当前总墙钟估算

| 执行范围 | 2×96G 极限并行墙钟 |
|---|---:|
| 必要闭环 A1–A9 | 约3–6 h |
| A1–A9 + A10，得到额外 seed early matched-pair 复制 | 从启动起约12–20 h |
| A1–A9 + A10 + A11，得到完整四臂 step320 seed replication | 从启动起约21–34 h |

硬边界：

1. A0 已完成，不得因总表重列而重新执行；
2. A1–A9 不新增训练、自由生成行为 Eval、Fisher 或 gradient proxy；
3. A0--A10 已获无人监管 AUTO-GO；A11 仍未获 GO，不得自动启动；
4. A11 不得从 step0 重跑 OPD/off-KD，不得重算 A10 已完成的 state/output；
5. probe/checkpoint/module cells 不是独立 seed；A10+A11 完成后总 seed 数仍为2，只能称一个额外
   独立 seed replication。

---

# A0--A10 无人监管自动执行协议【2026-07-27，覆盖旧的逐项 GO 要求】

```yaml
authorization: AUTO_GO_A0_THROUGH_A10
authorized_utc_date: 2026-07-27
included: [A0,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10]
excluded: [A11]
scientific_result_gate: none
human_approval_between_tasks: false
protocol_mutation_on_failure: forbidden
result_dependent_seed_or_metric_selection: forbidden
```

这里的“无人监管”具有严格含义：

1. A0 已完成，controller 只校验完成标记，不重跑；
2. A1--A10 的正结果、负结果或混合结果均按冻结协议保存，**任何科学结果都不阻断后续任务**；
3. 不需要 User/Theory 在 A1、A4、A5 或 A10 前后判断“结果是否好看”；
4. A10 在 A1 数值协议冻结、必要 GPU 前置计算结束后自动启动；
5. A10 完成或失败后，controller 生成最终 handoff 并停止；**不得自动进入 A11**；
6. 无人监管不授权脚本在失败后改变 seed、样本、probe、checkpoint、精度、epsilon、top-k、
   batch size、loss、数据顺序或统计口径。

## 自动 DAG

```text
A0 completion/hash audit
  |
  +--> CPU lane:
  |      A3 source/protocol preparation
  |      + A6/A7/A8/A9 可立即完成的缓存分析与审计
  |      A1 complete -> A2
  |      A2 + A3 complete -> A4
  |
  `--> GPU lane:
         A1
         -> A3/A5/A6 仅在缺少必要 forward/factor 时补算
         -> A10 preflight
         -> seed43 OPD/off-KD 0--160
         -> step160 dry-resume 到隔离的 step161
         -> A10 geometry/output
         -> A10 Stage-A analysis
         -> final A0--A10 handoff
         -> STOP（A11 HOLD）
```

CPU lane 与 GPU lane 可并行。A10 OPD 期间两张 GPU 分别承担 student rollout/train 与 teacher
inference，不与其他 GPU forward 叠跑。A2/A4/A7/A8/A9 不应为等待 A10 而闲置；反之，A10
也不等待 A4 的科学结论，只等待冻结协议和所需机器资源。

## 机器可判定 gate

只有以下技术条件可以阻断一个任务或其依赖分支：

```text
missing/corrupt upstream checkpoint or cache
manifest/hash/sample/token mismatch
dtype or numeric-protocol mismatch
NaN/Inf or failed spectrum invariant
GPU unavailable
OOM or transient runtime failure after fixed retry budget
nonzero exit after fixed retry budget
A10 seed43 collision with an existing training master seed
A10 full optimizer/scheduler/RNG/data-cursor checkpoint incomplete
A10 isolated step160->161 dry-resume mismatch
```

不得把 Pearson/Spearman、36/36、NCD、AUC、KL/NLL、行为显著性或论文主张是否成立设置为
execution gate。一个分析得到负结果时，任务状态仍可为 `COMPLETE_NEGATIVE_RESULT`，随后任务继续。

## 重试、恢复与幂等

外层 controller 必须使用：

```text
states = PENDING,RUNNING,COMPLETE,COMPLETE_NEGATIVE_RESULT,
         BLOCKED_DEPENDENCY,FAILED_RETRY_EXHAUSTED
max_attempts_per_command = 2
atomic output + lock + COMPLETE marker
resume from last validated durable checkpoint/cache
never overwrite a completed artifact with a partial artifact
```

第一次出现进程退出、GPU 服务暂态错误或主机中断时，清理孤儿进程并按原协议恢复一次。OOM
只能按完全相同的配置重试一次；不得为“救活任务”自动减小 batch、缩短 sequence 或更换精度。
第二次仍失败则写 `FAILED_RETRY_EXHAUSTED`，继续执行不依赖该任务的分支，并在最终 handoff 中列出
阻塞链。这样全程不需要人做科学判断，同时也不会静默改变实验。

A10 的现有 VERL 正式 checkpoint 已知可包含 model、optimizer、LR scheduler、CPU/CUDA/NumPy/Python
RNG 与 dataloader snapshot；但新 seed43 controller 仍必须逐项验收，并以隔离的 step161 dry-resume
证明恢复一致。仅保存 adapter 或 merged weights 时只能标为
`COMPLETE_STAGE160_NOT_RESUMABLE`，不能伪造 `COMPLETE_SEED_STAGE160_REUSABLE`。

## 最终交付

除各 A 任务原定产物外，controller 必须生成：

```text
a0_a10_autorun_manifest.json
a0_a10_task_status.csv
a0_a10_retry_and_failure_audit.json
a0_a10_final_handoff.md
```

其中登记每个任务的起止时间、输入/输出 hash、命令、commit、GPU、退出码、重试次数和最终状态。
预计在 2×96G 与 CPU lane 极限并行下，A0--A10 从启动到最终 handoff 约需 **12--20 小时**；
暂态故障重试或 A5/A6 缺少缓存时可能延长。该时间不包含 A11。

# D11-PK-TPNT —— Llama merged-\(p_k\) 修正与 TPNT 最小完整对照【2026-07-27 新冻结协议】

```yaml
task_id: D11_PK_TPNT
status: READY_TO_RUN
purpose:
  - 修正当前 Llama strict joint p_k 使用 adapter BA、而 Qwen 使用 deployed merged delta 的协议不一致
  - 在统一 deployed-update 口径上重建 p_k 的增量信息分析
  - 将目前仅覆盖 Qwen OPD/SFT 的 TPNT-style 审计补成边界清晰的相关工作对照
not_in_scope:
  - 不重算 Qwen strict p_k
  - 不重新训练任何模型
  - 不新增自由生成行为 Eval
  - 不重算已有 matched r_epsilon/c_epsilon state track
  - 不把 TPNT 的全部论文实验或因果干预机械复现一遍
```

本节是当前最新协议；若与前文 A3 中“两模型均需重算 \(p_k\)”的旧摘要冲突，以本节为准。
当前决定是：

1. **Qwen strict \(p_k\) 保留现有结果。** OPD 使用保存的 BF16 merged checkpoint；
   SFT/off-KD/seqKD 使用同一个 BF16 base 上构造的 deployed BF16 effective weight。用户已确认不要求
   为这三臂额外执行一次落盘—重载。
2. **只重算 Llama 四臂 strict \(p_k\)。** OPD、SFT、off-KD、seqKD 均改用 BF16 deployed
   merged-minus-base，不再用 FP32 adapter \(sBA\) 作为正式 related-work track。
3. **TPNT 补全是另一项任务。** “Qwen \(p_k\) 不重算”不等于“Qwen TPNT 不补”；
   Qwen off-KD/seqKD 的 principal-mask/angle/NSS 仍属于缺失的 TPNT 对照。

## D11.1 全部权重空间指标的冻结数值协议

正式 deployed update 定义为

\[
\Delta W_{t,m}
=
\operatorname{FP32}\!\left(W_{t,m}^{\mathrm{BF16}}\right)
-
\operatorname{FP32}\!\left(W_{0,m}^{\mathrm{BF16}}\right).
\]

不得用 FP32 \(sBA\) 代替上式进入 Llama 正式 \(p_k\)/TPNT 主轨。若某个 Llama 臂只有 adapter，
必须先在冻结的 BF16 base 上确定性 merge，**保存 BF16 merged artifact，重新加载后**再按上式
构造 \(\Delta W\)，并登记 base/merged hash。adapter \(sBA\) 可保留为数值审计支线，但不得与正式
merged track 混表。

| 项目 | 冻结协议 |
|---|---|
| base | serialized BF16 base |
| trained state | serialized BF16 deployed merged checkpoint |
| delta | `FP32(Wt_bf16) - FP32(W0_bf16)` |
| source SVD | FP32 source tensor；分解可用 FP64，必须登记 dtype |
| projection/norm | 至少 FP32；聚合前不得回写 BF16 |
| Qwen \(p_k\) | 复用现有 deployed-effective 结果，不重算 |
| Llama \(p_k\) | 四臂全部重算为 deployed merged-minus-base |
| \(r_\varepsilon,c_\varepsilon\) | 复用 D10 matched state-rank，不重算 |
| output | 复用已有 cumulative KL、signed NLL、absolute NLL 和已有行为结果 |

所有产物必须保留 `delta_construction`、`native_source`、`checkpoint_materialization`、
base/checkpoint hash、source/projection/SVD dtype。不得再把 `adapter_ba` 与
`bf16_merged_minus_base` 的行放入同一个正式跨模型回归或 AUC 表。

## D11.2 实验总表

| ID | 优先级 | 实验 | 模型与训练臂 | checkpoint | 层/模块 | 主要输出 | 新增开销 |
|---|---:|---|---|---|---|---|---:|
| E0 | P0 | Llama merged-\(p_k\) 重算 | Llama：OPD、SFT、off-KD、seqKD | 5/20/40/80/160/320 | L14；七模块 | joint \(p_k\)，\(k=\{4,8,16,32\}\) | 1×96G，约0.5–1.5 h |
| E1 | P0 | 重建 \(p_k\) 增量信息分析 | Llama 新结果；Qwen 原结果 | 与 E0/D10 共同网格 | 同 cell 聚合 | KL/NLL held-out \(R^2\)、OPD 判别 AUC、\(C+p_k\) 增量 | CPU 0.3–1 h |
| E2 | P1 | TPNT principal-mask 主比较 | 两模型、四训练臂 | 5/20/40/80/160/320 | Qwen L18；Llama L14；七模块 | coverage、OverlapLift、random-null 差值 | 1×96G，约1–2 h |
| E3 | P1 | TPNT principal-angle 与 NSS | 两模型、四训练臂 | 同 E2 | headline 层；七模块 | 左/右 principal angles、PABS、NSS | 与 E2 共载入；增量约1–2 h |
| E4 | P1 | 权重基线与 \(c_\varepsilon\) 公平比较 | 两模型、四训练臂 | 严格共同 cells/folds | 聚合后比较 | arm discrimination、KL/NLL 增量解释、跨模型一致性 | CPU 0.5–1.5 h |
| E5 | P2 | 层稳健性 | 两模型、四训练臂 | 20/160/320 | Qwen L9/18/27；Llama L7/14/21 | 各层方向、效应量和排序 | 1×96G，约1–3 h |
| E6 | P2 | TPNT 参数敏感性 | 两模型、四训练臂 | 20/160/320 | headline 层 | source-rank 与 mask-density 敏感性 | 复用 E2 cache，约0.5–1 h |
| E7 | P2 | 严格随机更新零假设 | 两模型、四训练臂 | 20/160/320 | headline 层 | real update 对 spectrum-matched random update 的 \(z\)-score | CPU/GPU约0.5–1.5 h |

执行原则：

- E0--E1 是**必须完成的协议修正**。
- E2--E4 是 TPNT 的**最小完整 related-work comparison**；若截稿资源允许则完成。
- E5--E7 是稳健性增强；做了则报告，未做不阻断主论文和 E0--E4。
- 正结果、负结果或 mixed result 均算实验完成，禁止按结果好坏改变网格。

## D11.3 E0：Llama strict joint \(p_k\) 重算

定义

\[
p_k(W_0,\Delta W_t)
=
\frac{\left\|U_k^\top \Delta W_t V_k\right\|_F^2}
{\left\|\Delta W_t\right\|_F^2},
\qquad
W_0=U\Sigma V^\top .
\]

正式网格：

```text
model       = Llama-3.2-3B
arms        = OPD,SFT,off-KD,seqKD
checkpoints = 5,20,40,80,160,320
layer       = 14
modules     = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
fixed_k     = 4,8,16,32
delta       = BF16 deployed merged-minus-base
```

预期正式行数为

\[
4\ \mathrm{arms}\times6\ \mathrm{checkpoints}
\times7\ \mathrm{modules}\times4\ k
=672.
\]

必须逐 cell 校验 \(0\le p_k\le1\)、无 NaN/Inf、相同 source basis hash，并输出：

```text
llama_joint_pk_merged_rows.csv
llama_joint_pk_merged_manifest.json
llama_joint_pk_adapter_vs_merged_audit.csv
```

第三个文件只负责说明旧 adapter-\(BA\) 结果与新 deployed 结果差多少；论文正式表只消费第二条
merged track。

## D11.4 E1：重建同 cell/fold 的增量信息分析

E0 完成后，用新的 Llama \(p_k\) 替换旧 adapter-\(BA\) 行；Qwen 行保持不变，重新运行 A4。
不得只替换图、不重建统计表。至少比较：

\[
W,\quad p_k,\quad C,\quad W+C,\quad p_k+C,
\]

其中 \(C=c_{\varepsilon}\)，并在 TPNT E2--E3 完成后追加
principal-mask、PABS/NSS 及其与 \(C\) 的组合。

必须在完全相同的 model-arm-checkpoint-probe cell 与 checkpoint-grouped fold 上报告：

| 问题 | 因变量 | 基线 | 完整模型 | 主输出 |
|---|---|---|---|---|
| 输出漂移 | cumulative KL | \(W\) 或 \(p_k\) | baseline+\(C\) | held-out \(R^2\)、\(\Delta R^2\)、MAE |
| 输出幅度 | absolute NLL | 同上 | 同上 | held-out \(R^2\)、\(\Delta R^2\)、MAE |
| 输出效价 | signed NLL | 同上 | 同上 | held-out \(R^2\)、\(\Delta R^2\)、MAE |
| 训练范式 | OPD vs non-OPD | 单一权重指标 | weight+\(C\) | GroupKFold AUC、balanced accuracy、log-loss |
| 跨模型迁移 | 同上 | train-one-model | test-other-model | Spearman、排序、迁移误差 |

同一 checkpoint 下重复到不同 probe 的 domain-independent 权重量不得被当作独立样本；
fold 必须按 checkpoint 分组，并同时提供 model-wise 表，不能只给 pooled cell 数。

## D11.5 E2：TPNT principal-coordinate mask 最小补全

TPNT principal coordinate 指标不是 joint \(p_k\)。其正式定义为：

\[
W_0^{(k)}=U_k\Sigma_kV_k^\top,
\qquad
M_{\mathrm{princ}}
=\operatorname{Top}_{\alpha}\!\left(\left|W_0^{(k)}\right|\right),
\]

\[
M_\Delta=\mathbf 1[\Delta W\ne0],
\qquad
\mathrm{Coverage}_{k,\alpha}
=\frac{|M_{\mathrm{princ}}\cap M_\Delta|}{|M_\Delta|},
\qquad
\mathrm{OverlapLift}_{k,\alpha}
=\frac{\mathrm{Coverage}_{k,\alpha}}{\alpha}.
\]

主网格：

| 维度 | 设置 |
|---|---|
| 模型 | Qwen3-4B、Llama-3.2-3B |
| 训练臂 | OPD、SFT、off-KD、seqKD |
| checkpoint | 5、20、40、80、160、320 |
| headline 层 | Qwen L18；Llama L14 |
| 模块 | 七个 LoRA target modules |
| source reconstruction rank | \(k_{\rm src}=\{16,32,50\}\) |
| mask density | \(\alpha=\{1\%,10\%\}\) |
| deployed delta | D11.1 的统一口径 |

现有 Qwen OPD/SFT、\(k=50,\alpha=1\%\) 的结果直接复用；只补缺失 cells。Qwen
off-KD/seqKD 与 Llama 四臂必须进入同一正式表。

主结果不得只报告“OPD 比 SFT 高/低”，还要报告真实更新相对低秩随机零假设的差距。现有
rank/norm-matched random LoRA 可作为主零假设；E7 有余量时再升级为 spectrum-matched random
subspace。

## D11.6 E3：principal angles、PABS 与 NSS

在 E2 相同的 deployed weights、checkpoints、layers 和 modules 上共同计算，避免重复加载。

principal angles 使用：

\[
\cos\theta_i^{(U)}
=\sigma_i(U_{0,k}^{\top}U_{t,k}),
\qquad
\cos\theta_i^{(V)}
=\sigma_i(V_{0,k}^{\top}V_{t,k}),
\]

固定 \(k=\{4,8,16,32\}\)，输出全部角、mean/max angle 和冻结定义的 PABS 聚合。NSS 必须输出
source/checkpoint 的归一化奇异谱、逐 cell scalar drift，并在 manifest 中写出使用的精确公式；
不得只写“NSS”而没有 estimand。

本项用于回答：

1. TPNT-style 主子空间旋转能否稳定区分 OPD 与离线训练；
2. 该差异是否超过低秩随机更新零假设；
3. 它相较于 \(c_\varepsilon\) 是否提供跨域、跨模型的额外信息。

不得把 Qwen LoRA 上“random LoRA 接近 OPD”的既有结果写成对 TPNT 全参量结论的普遍否定。
正式边界表述应为：

> 在当前 Qwen LoRA 设置中，TPNT-style principal-mask/rotation 的部分变化可由低秩、尺度匹配
> 的随机更新复现；这提示其训练特异性弱于激活条件功能轨迹，但不否定 TPNT 在全参量 RLVR
> 设置中的发现。

## D11.7 E4：最终公平比较与论文可用判据

最终比较必须回答“激活条件空间相较于纯权重空间多表达了什么”，而不是只排列单指标的训练集
相关系数。

| 科学问题 | 主要比较 | 可写入正文的判据 |
|---|---|---|
| OPD 是否具有不同轨迹 | TPNT/\(p_k\)/PABS/NSS vs \(C\) | \(C\) 在 checkpoint-grouped held-out 上提供稳定增量判别 |
| 是否解释输出漂移 | weight-only vs weight+\(C\) | \(\Delta R^2>0\) 且不是单个 checkpoint 驱动 |
| 是否具有训练特异性 | real update vs random low-rank null | 报 null-adjusted effect；不得只报 raw overlap |
| 是否跨模型成立 | Qwen 与 Llama 分别拟合并互相迁移 | 至少方向/排序稳定；量级可模型依赖 |
| 指标边界是什么 | \(C\) 对 KL、abs/signed NLL 的差异 | 不要求 \(C\) 成为 signed behavior 的充分统计量 |

若结果仍显示纯权重指标在某一模型上判别很强，必须如实报告。核心主张只要求：

> \(c_\varepsilon\) 描述的是权重与领域激活共同定义的功能状态；它是否在纯权重几何之外提供
> 可复现的增量信息，应由同 cell、同 fold、同 deployed-update 协议的实验决定。

## D11.8 E5--E7 可选增强

### E5：层稳健性

只在 landmark \(20/160/320\) 补：

```text
Qwen  = L9,L18,L27
Llama = L7,L14,L21
```

输出逐层和 headline-layer 一致性，禁止把 layer×module 当作独立 seed。

### E6：TPNT 参数敏感性

主网格已冻结为 \(k_{\rm src}=\{16,32,50\}\)、\(\alpha=\{1\%,10\%\}\)。若资源允许，在
landmark cells 追加 \(\alpha=\{5\%,20\%\}\)，并总结结论是否依赖单一 \(k/\alpha\)。

### E7：spectrum-matched random-subspace null

保留真实 \(\Delta W\) 的非零奇异值，随机旋转其左右子空间，构造：

\[
\Delta W_{\rm null}=Q_U\Sigma_{\Delta W}Q_V^\top .
\]

每个 landmark cell 固定 10 个随机 seed，报告：

\[
Z_{\mathrm{TPNT}}
=
\frac{
\mathrm{OverlapLift}_{\rm real}
-\mathbb E[\mathrm{OverlapLift}_{\rm null}]
}{
\operatorname{Std}[\mathrm{OverlapLift}_{\rm null}]
}.
\]

该 null 同时控制 rank、Frobenius norm 和奇异值谱，比仅控制 rank/norm 的随机 LoRA 更严格；
它是可选增强，不得阻断 E0--E4。

## D11.9 执行 DAG、资源和完成状态

```text
E0 Llama merged-p_k
  -> E1 replace old Llama p_k and rebuild A4

E2 TPNT mask
  + E3 angles/PABS/NSS       # 同一次 checkpoint load/SVD 尽量共算
  -> E4 final fair comparison

E5/E6/E7 may run only after E0--E4 inputs are immutable
```

E0--E1 与 E2--E3 可在资源不冲突时并行；E1/E4 的 CPU 后处理可以与 GPU 权重分解并行。不得在
A10/A11 的 OPD teacher/student 双卡训练期间抢占其 GPU。

| 范围 | GPU/CPU 预算 | 极限并行墙钟 |
|---|---:|---:|
| E0--E1 必做修正 | GPU 0.5–1.5 h + CPU 0.3–1 h | 约1–2.5 h |
| E2--E4 TPNT 最小补全 | GPU 2–4 h + CPU 0.5–1.5 h | 约2.5–5 h |
| E0--E4 合计 | GPU约2.5–5.5 h + CPU约1–2 h | 约3–6 h |
| E5--E7 可选增强 | 增量约2–5 h | 不计入必做闭环 |

E0--E4 完成时必须交付：

```text
d11_pk_tpnt_task_status.csv
d11_llama_merged_pk.csv
d11_llama_pk_numeric_audit.csv
d11_tpnt_principal_mask.csv
d11_tpnt_angles_pabs_nss.csv
d11_same_cell_incremental_comparison.csv
d11_modelwise_and_crossmodel_summary.md
d11_pk_tpnt_manifest.json
```

允许的最终状态：

```text
COMPLETE_D11_CORE
COMPLETE_D11_CORE_NEGATIVE_OR_MIXED_RESULT
PARTIAL_D11_E0_E1_ONLY
PARTIAL_D11_TPNT_MISSING_CELLS
BLOCKED_MISSING_MERGED_CHECKPOINT
FAILED_NUMERIC_PROTOCOL_MISMATCH
FAILED_RETRY_EXHAUSTED
```

只有 checkpoint/cache 缺失、hash/dtype 协议不匹配、NaN/Inf、OOM 或命令失败可以阻断任务；
结果是否“支持论文”不得作为执行 gate。

# FAT-OUTLINK —— Prompt / CoT / Format / Answer / Termination 概率分解【2026-07-28】

```yaml
task_family: FAT_OUTLINK
status:
  first_round: CONFIRMED_READY_TO_RUN
  followup_rounds: HOLD_FOR_THEORY_DESIGN
authorized_first_round: [FAT-R1-S0, FAT-R1-M1, FAT-R1-M2]
explicitly_removed: [MATH_DIRECT_NO_COT]
gpu_target: 1x RTX_5090_32GB
expected_first_round_wall_clock: 40-75 min; reserve 1-1.5 GPU hours
guard:
  - zero training
  - zero new rollout
  - zero new free-generation behavior evaluation
  - teacher-forcing forward only
  - raw readings and provenance only; coder does not adjudicate theory
```

## 0. 非破坏性追加与编号边界

本节只追加新的 output-link 概率分解任务，**不得删除、重命名、截断或覆盖**本文件此前的
L1--L3、Q1、R1--R3、A0--A11、B/O/C/D 系列和 D11/E0--E7。此前任务、产物与完成状态继续
保留。本节中的 `M1/M2/S0` 在文件名和 manifest 中必须写成
`FAT-R1-M1/FAT-R1-M2/FAT-R1-S0`，避免与旧谱矩阵 `M1/M2` 字段混淆。

### 0.1 已完成上游：EQUAL5_NON_QK【禁止重跑】

```yaml
task: EQUAL5_NON_QK
status: COMPLETE_EXISTING_DO_NOT_RERUN
canonical_created_utc: 2026-07-28T06:18:15.495260+00:00
canonical_manifest: mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/equal5_non_qk/EQUAL5_manifest.json
canonical_handoff: mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/equal5_non_qk/EQUAL5_theory_handoff.md
script: experiments/opd_sft_h1/scripts/cycle09_equal5_non_qk.py
execution: reuse-only CPU aggregation
```

正式版本是 `06:18:15` manifest；`code_evolution.md` 和 reviewer handoff 中仍保留的
`06:17:19` 条目是中间版，其中 `EQUAL5_spectrum_robustness.csv=469` 行。不得使用该中间
row-count 覆盖最终版的 `517` 行。

冻结完成性：

| check | canonical observed |
|---|---:|
| Llama uncentered equal-5 early | 48/48 |
| Llama centered equal-5 early | 48/48 |
| Qwen epsilon=.05 early | 12/12 |
| Qwen all-epsilon early | 47/48 |
| pooled epsilon=.05 | 24/24 |

该任务只是在测量侧将七模块等权聚合改为
`v_proj/o_proj/gate_proj/up_proj/down_proj` 五模块等权，排除 `q_proj/k_proj`；它不是 LoRA
模块 zeroing、adapter ablation、新 forward 或新 SVD。正式边界：

1. epsilon=.05 下，去除 q/k 后两模型 pooled early cells 为 24/24 OPD strict deepest；
2. 全 epsilon 为 47/48，唯一例外是 Qwen epsilon=.10 的一个 offline-strictly-deeper cell；
3. 该结果说明核心 OPD 压缩支配不由 q/k 异质性单独制造，但不能写成“q/k 参数无效”；
4. Qwen centered/full spectrum、Qwen per-module \(p_k\) 仍为
   `BLOCKED_NO_MODULE_SOURCE`；禁止从 equal-7 aggregate 反推；
5. RR2D displacement 已标 `SUPERSEDED_SOURCE_FORBIDDEN`，不得重新纳入正式 state-rank
   robustness。

EQUAL5 与 FAT-R1 没有执行依赖，不提供 F/A/T logits，也不替代 FAT-R1-M1/M2。Coder 只需在
FAT-R1 inventory 中登记其 canonical manifest/hash，不得重跑或把 equal-5 当作新的模型状态。

本节按“第一轮 / 后续轮次”管理：

1. **第一轮已确认，可执行：**只运行 `FAT-R1-S0 → FAT-R1-M1 + FAT-R1-M2 → S0 postprocess`。
2. **后续轮次仅保留设计槽位：**General、IFEval、matched `_P`、MATH 精确 `_C`；Theory
   冻结协议前不得自行发车。
3. **MATH direct/no-CoT 轨永久删除：**不得构造
   `[problem][\boxed{answer}]` 来替代完整 gold solution。该截断条件改变了数学任务，
   不属于可解释的正式对照，任务编号不得复用。

## 1. 统一序列、区域和指标命名

具有规范参考回答的统一序列为

\[
[P]\,[C]\,[F_{\mathrm{pre}}]\,[A]\,[F_{\mathrm{post}}]\,[T].
\]

若原始 gold response 在最终答案 span 后仍有关闭数学环境、标点或说明文本，使用不截断版本

\[
[P]\,[C_{\mathrm{pre}}]\,[F_{\mathrm{pre}}]\,[A]\,
[F_{\mathrm{post}}]\,[C_{\mathrm{post}}]\,[T],
\qquad C=C_{\mathrm{pre}}\cup C_{\mathrm{post}}.
\]

| 下标 | 冻结含义 | 第一轮实例 |
|---|---|---|
| \(P\) | prompt / input | MMLU question+options；MATH problem |
| \(C\) | gold CoT / solution 正文，严格排除最终答案格式；可以由答案前后的两个不连续 span 组成 | MMLU test 无 \(C\)；MATH 的 \(C_{\rm pre}\) 为 final boxed span 之前部分，\(C_{\rm post}\) 为 matching brace 后至原始 solution 末尾的原文 |
| \(F_{\rm pre}\) | 最终答案之前的固定严格格式 token | MMLU `The answer is (`；MATH `\boxed{` |
| \(A\) | 正确答案内容；可为一个或多个 tokenizer tokens | MMLU A--J；MATH boxed 内数学表达式 |
| \(F_{\rm post}\) | 答案之后的固定格式闭合 token | MMLU `).`；MATH 与 `\boxed{` 配平的 `}` |
| \(F\) | \(F_{\rm pre}\cup F_{\rm post}\)，不包含 \(A\) | 两侧格式 span 的 sample-level token mean |
| \(T\) | 当前模型协议的 EOS/EOT | 必须记录模型、token id 和 terminator 类型 |

原始量统一命名：

\[
\mathrm{NLL}_{Z,D,t},\qquad
\mathrm{KL}^{0\rightarrow t}_{Z,D,t},
\qquad Z\in\{P,C,F,A,T\},
\]

\[
\mathrm{KL}^{0\rightarrow t}_{Z,D,t}
=
\frac{1}{|I_Z|}
\sum_{i\in I_Z}
D_{\mathrm{KL}}\!\left(
p_0(\cdot\mid z_{<i})
\Vert
p_t(\cdot\mid z_{<i})
\right).
\]

CSV 字段使用小写：

```text
nll_p,nll_c,nll_f_pre,nll_f_post,nll_f,nll_a,nll_t
kl_p,kl_c,kl_f_pre,kl_f_post,kl_f,kl_a,kl_t
delta_nll_z = nll_z(checkpoint) - nll_z(step0)
abs_delta_nll_z = abs(delta_nll_z)
```

禁止使用含糊的 `absolute_nll` 表示 `abs(delta NLL)`。KL 方向固定为
`base_to_checkpoint = D_KL(p0 || pt)`；manifest 必须登记方向，不允许交换方向后沿用同名列。
没有实际测量的区域写 `NA`，不得用 0 填补。例如 MMLU test 没有 gold CoT，所以全部 `_c`
字段为 `NA`。

## 2. 第一轮总表【已确认】

| ID | 数据与固定样本 | 固定 teacher-forced 序列 | 第一轮正式输出 | 对齐的既有行为 | GPU 范围 | 预计 5090 墙钟 |
|---|---|---|---|---|---|---:|
| FAT-R1-S0 | 协议/样本/token mask/数值 preflight | 不做科学裁决；先校验 M1/M2 的完整输入和 span | sample manifest、mask audit、checkpoint inventory、smoke、统一聚合与 paired bootstrap | 只登记行为文件及 sample-ID 对齐 | CPU + 极小 GPU smoke | 5--15 min |
| FAT-R1-M1 | MMLU-Pro 固定 1,400 题；14 类×100；严格复用现有 strict/flexible 行为评测的 question IDs | `[P][Fpre][A][Fpost][T]` | `NLL_F/A/T`、`KL_F/A/T`、`delta_nll_F/A/T`、`abs_delta_nll_F/A/T`；`_C=NA` | strict、flexible、strict extract-fail；\(F\) 对格式差，\(A\) 对 flexible，\(T\) 对长度/截断辅助量 | Qwen 37 + Llama 25 个唯一正式状态 | 25--45 min |
| FAT-R1-M2 | MATH500 全部 500 题；原始 `problem/solution/answer` | `[P][Cpre][Fpre][A][Fpost][Cpost][T]`，其中 \(C=C_{\rm pre}\cup C_{\rm post}\)；不得删除或截断 \(C\) | 主输出 `NLL_F/A/T`、`KL_F/A/T`；相同 forward 顺手保存 `NLL_C`；第一轮不算全位置 `KL_C` | MATH accuracy、boxed-format（若既有日志可 CPU 抽取）、cap-hit、mean/median length | 同 M1 全正式状态 | 15--30 min |
| FAT-R1-TOTAL | M1+M2+S0 | 无 rollout、无自由生成 | 第一轮完整交付 | 同题/同 checkpoint join | 单张 RTX 5090 32GB 串行即可 | 40--75 min；保守预留1--1.5 h |

### 2.1 第一轮模型、训练臂和 checkpoint

```text
Qwen3-4B-Base:
  arms  = OPD,SFT,off-KD,seqKD
  steps = 0,5,10,20,40,80,160,320,480,624
  unique states = 1 shared base + 4*9 = 37

Llama-3.2-3B:
  arms  = OPD,SFT,off-KD,seqKD
  steps = 0,5,20,40,80,160,320
  unique states = 1 shared base + 4*6 = 25
```

step0 物理前向每个模型只计算一次。为便于轨迹 join 可以生成 arm aliases，但必须标
`is_shared_base=true`，不得把四个重复别名当作四个独立观测。第一轮不自动加入 alpha=.5、
frozenSelf、A10 seed43 或任何新臂；后续如需扩展必须另行列入 round-2 scope。

### 2.2 统一模型和数值协议

1. 使用当前正式 deployed checkpoint/state；Qwen 与 Llama 都按同一加载协议，不混用
   legacy FP16 state-rank 轨或 adapter-only 数值轨。
2. 模型前向可使用 BF16；进入 `log_softmax`、NLL、KL 和最终累计前至少转 FP32。
3. KL 必须使用**精确 full-vocabulary distribution**，不是 top-32 训练缓存近似。
4. 只保留被请求 span 的 logits/统计量；禁止把完整
   `[sample,sequence,vocabulary]` tensor 全量写盘。
5. base distribution 在固定语料和 target position 上每模型只计算一次并缓存；不得为每个
   checkpoint 重跑相同 base，除非缓存 hash 校验失败。
6. 每个 checkpoint 加载一次后连续处理 M1 和 M2，再卸载；禁止按任务重复加载所有 checkpoint。
7. 按长度分桶或动态 batch；不得逐题单独 forward。OOM 时只能按预注册的等价 dynamic-batch
   规则降低 batch，不得截断序列、样本或更换精度。

### 2.3 FAT-R1-S0：preflight、mask 与 parity gate

S0 必须先完成以下技术检查：

1. 从正式行为日志恢复 MMLU-Pro 的 canonical 1,400 `question_id`；验证所有 arm/checkpoint
   的行为结果使用同一集合。若不一致，写 `BLOCKED_MMLU_SAMPLE_PARITY`，不得偷偷重新抽题。
2. MATH500 必须恰为 500 个唯一 `unique_id`，且 500/500 `solution` 含最终 `\boxed{...}`。
3. 冻结两个模型各自 tokenizer、special-token ids、EOS/EOT 与 chat/base template 状态。
4. 对每模型至少审计 MMLU 14 题（每类1题）和 MATH 20 题的字符 span→token span。
5. token 与区域边界重叠时不得强行归类。先使用 offset mapping；输出
   `boundary_crossing=true`。若 \(A\) 与格式字符被 tokenizer 合并，必须停在 S0 回报，
   不得更换答案模板后静默继续。
6. 技术 smoke：每模型的 base、OPD terminal、seqKD terminal；MMLU 每类1题，MATH 20题。
   验收无 NaN/Inf、所有 KL≥数值容差内的0、各 span 非空、样本数/类别数/hash 正确后自动进入
   M1/M2。科学结果正负不是 gate。

S0 聚合协议：

1. 每个 sample 内先对对应 span token 等权平均；
2. 然后对 sample 等权平均，禁止长 CoT/长答案支配总体；
3. MMLU 同时报告 sample-macro 和 14-category macro；
4. MATH 报 overall，并可按 subject/level 作附录审计；
5. CI 使用相同 item indices 的 paired item bootstrap，seed=42、draws=1,024；
6. 所有 arm/checkpoint/model 比较使用相同 bootstrap resample indices。

### 2.4 FAT-R1-M1：MMLU-Pro 格式—答案—终止

固定单题模板：

```text
Question:
{question}
Options:
A. {option_A}
...
{last_letter}. {last_option}
Answer:
The answer is ({gold_letter}).
<MODEL_TERMINATOR>
```

区域：

```text
P      = 从 Question 开始至 Answer: 后换行结束
Fpre   = "The answer is ("
A      = gold_letter
Fpost  = ")."
T      = model-specific EOS/EOT
```

`Fpre/A/Fpost` 必须按字符 offset 映射，不能假设两个 tokenizer 有相同 token ids；但模板字符
内容必须一致。M1 的正式 cell table 至少包含：

```text
model,arm,checkpoint,domain,question_id,category,
n_tokens_p,n_tokens_f_pre,n_tokens_a,n_tokens_f_post,n_tokens_t,
nll_f_pre,nll_f_post,nll_f,nll_a,nll_t,
kl_f_pre,kl_f_post,kl_f,kl_a,kl_t,
strict_exact,flexible_exact,strict_extract_fail,
response_length,truncated,
sample_manifest_hash,template_hash,tokenizer_hash,base_cache_hash
```

主要机械对齐表必须逐 checkpoint 报：

\[
\mathrm{flexible}-\mathrm{strict},\quad
\mathrm{strict\ extract\ failure},\quad
\Delta\mathrm{NLL}_{F/A/T},\quad
\mathrm{KL}_{F/A/T}.
\]

Coder 只输出相关、回归或 paired 差值读数，不得写“知识保留”“格式损伤成立”等理论裁决。

### 2.5 FAT-R1-M2：MATH gold-CoT 条件下的 boxed/answer/termination

MATH 必须使用原始 `solution`，并用 balanced-brace parser 定位**最后一个**完整
`\boxed{...}`：

```text
P      = problem + 冻结的 Solution cue
Cpre   = gold solution 中 final "\\boxed{" 之前的全部原文
Fpre   = "\\boxed{"
A      = final boxed span 内部的数学表达式
Fpost  = 与 final "\\boxed{" 配平的闭合 "}"
Cpost  = matching brace 后至原始 solution 结束的全部原文（可为空）
C      = Cpre union Cpost
T      = 完整原始 solution 结束后追加的 model-specific EOS/EOT
```

不得用简单 regex 截取嵌套 LaTeX 花括号。解析后用独立 `answer` 字段做规范化核验；原始
`solution` 与 `answer` 不一致、多个 box 无法裁决、空答案或 boundary-crossing 必须逐题登记，
不得静默改写 gold solution。`Cpost` 即使只包含关闭数学环境或标点也必须保留；不得把它并入
`Fpost`，否则 `_F` 会混入非统一正文。正式 F/A/T 结果条件于完整、未截断的 gold \(P,C\)。

第一轮：

1. 必算 `NLL_F/A/T` 与 exact `KL_F/A/T`；
2. 同次 forward 可保存 `NLL_C`，因为目标 token gather 的增量很小；
3. **不算全 CoT 各位置的 exact `KL_C`**，也不因此缓存完整 sequence×vocab logits；
4. 不生成模型 CoT，不使用行为 rollout 作为 gold prefix；
5. 现有 MATH accuracy/cap-hit/length 只作 checkpoint-level 外部行为 join。

## 3. 第一轮必须交付的文件

建议根目录：

```text
mini/fat_outlink_round1/
```

必须交付：

```text
fat_r1_task_status.csv
fat_r1_sample_manifest.json
fat_r1_checkpoint_inventory.csv
fat_r1_template_and_tokenizer_manifest.json
fat_r1_mask_audit.csv
fat_r1_mmlu_samples.csv
fat_r1_mmlu_cells.csv
fat_r1_math_samples.csv
fat_r1_math_cells.csv
fat_r1_behavior_join.csv
fat_r1_bootstrap_ci.csv
fat_r1_base_cache_manifest.json
fat_r1_runtime_and_gpu_ledger.csv
fat_r1_handoff.md
fat_r1_manifest.json
```

其中 `*_samples.csv` 可以按 model/arm/checkpoint 分片保存，顶层 CSV 作为索引，避免单文件过大。
manifest 必须登记输入/输出 SHA-256、git commit、命令、环境、GPU、dtype、batching、
tokenizer/template、样本 IDs、KL 方向、base cache 与每项 row count。

允许状态：

```text
COMPLETE_FAT_R1
COMPLETE_FAT_R1_NEGATIVE_OR_MIXED_RESULT
PARTIAL_FAT_R1_M1_ONLY
PARTIAL_FAT_R1_M2_ONLY
BLOCKED_MMLU_SAMPLE_PARITY
BLOCKED_MATH_BOX_PARSE
FAILED_TOKEN_BOUNDARY_PROTOCOL
FAILED_NUMERIC_PROTOCOL_MISMATCH
FAILED_RETRY_EXHAUSTED
```

## 4. 后续轮次槽位【保留设计，不得执行】

以下内容必须保留在交接中，防止第一轮完成后遗忘，但当前状态均为
`HOLD_FOR_THEORY_DESIGN`。

| 后续 ID | 目标 | 当前冻结边界 | Theory 后续必须解决的问题 |
|---|---|---|---|
| FAT-R2-C | MATH `_C` 输出联系 | 可复用 R1 的完整 gold \(P,C,F,A,T\)；`NLL_C` 可复用 | 是否值得在全部 CoT 位置计算 exact `KL_C`；先 landmark 还是全轨；完整 full-vocab KL 的内存/时间协议 |
| FAT-R2-G | General 输出侧指标 | Wikipedia/general 没有 \(F/A\)，且普通 continuation 不得冒充 CoT `_C` | 为固定 reference continuation 冻结新下标（候选 `_R`，尚未批准）、prefix/continuation 切点、sample N、终止定义 |
| FAT-R2-I | IFEval 输出侧指标 | IFEval 无唯一 gold answer，所以不得构造统一 `_A` 或伪 gold response | required literal `_F`、forbidden-token mass、length-conditioned EOS/`_T` 如何分层；是否只复用已有冻结响应 |
| FAT-R2-P | 四域 same-corpus matched prompt control | 旧 `_P` 的样本规模不一致：MMLU 旧主轨约128、Llama Math 旧轨也非完整500 | 是否为 general/math/mmluPro/ifeval 建 `P_matched`；如何与 `P_legacy` 分名；是否计算全位置 exact `KL_P` |
| FAT-R2-X | 辅助臂扩展 | 第一轮只有双模型四核心臂 | 是否加入 alpha=.5、frozenSelf、seed43；必须复用 R1 模板、样本和 base cache，禁止结果驱动选臂 |

后续四域目标仍然是

\[
D\in
\{\mathrm{general},\mathrm{math},\mathrm{mmluPro},\mathrm{ifeval}\},
\]

但“对齐”不等于强迫四域都拥有相同 \(F/A/T\)。正式对齐至少要求：

1. 同一 model/arm/checkpoint 数值协议；
2. 每个 domain 内固定且可复现的 sample/item IDs；
3. 同一个 `base_to_checkpoint` KL 方向；
4. 明确区分 `P_legacy` 与未来 `P_matched`；
5. domain-specific output estimand 不被错误改名成另一个构念。

Theory 冻结 R2 协议前，Coder 完成第一轮 handoff 后停止；不得根据第一轮结果自行决定是否跑
General、IFEval、`KL_C`、matched `_P` 或辅助臂。

# FAT-R1-LINK —— 区域化 output 与 \(c_\varepsilon/p_k\) 的同格关联【2026-07-28】

```yaml
task: FAT-R1-LINK
status: READY_AUTORUN_CPU
priority: P0
dependencies:
  - FAT-R1-v2 complete
  - D10.5 matched c_epsilon complete
  - D11 deployed-weight p_k complete
authorization:
  - CPU-only join, bootstrap, correlation and grouped held-out analysis
  - no training
  - no model forward
  - no rollout or behavior eval
  - no new SVD
estimated_wall_clock: 15-45 min CPU
```

本任务不是 FAT-R2 forward。它只把已经完成的 FAT-R1-v2 区域化 NLL/KL 与现有
\(c_\varepsilon\)、strict joint \(p_k\) 按 `model,arm,checkpoint` 同格连接，并回答：

1. 功能压缩更接近无符号 output departure，还是有符号 gold-token readout；
2. 这种关系是否集中在 format/answer/termination 的特定区域；
3. \(c_\varepsilon\) 在纯权重 source-principal \(p_k\) 之外是否提供增量信息；
4. Qwen/Llama 的 signed readout 方向是否必须保留模型交互，而不能池化成统一符号。

## L0. 冻结输入与协议修正

FAT-R1-v2 正式输入：

```text
mini/fat_outlink_round1_v2/fat_r1_v2_mmlu_cells.csv
mini/fat_outlink_round1_v2/fat_r1_v2_math_cells.csv
mini/fat_outlink_round1_v2/fat_r1_v2_mmlu_samples.csv
mini/fat_outlink_round1_v2/fat_r1_v2_math_samples.csv
mini/fat_outlink_round1_v2/fat_r1_v2_region_contrasts.csv
mini/fat_outlink_round1_v2/fat_r1_v2_bootstrap_ci.csv
mini/fat_outlink_round1_v2/fat_r1_v2_manifest.json
```

几何输入：

```text
mini/d10_5_a4_feature_matrix.csv
mini/T_PK_qwen3_4b_fixedk.csv
mini/d11_llama_merged_pk.csv
```

冻结规则：

1. \(c_\varepsilon\) 主轨使用 per-checkpoint whitening、headline 中层
   （Qwen L18；Llama L14）、\(\varepsilon=.05\)、七模块 module-first equal-7；
2. MMLU-Pro output 配 `probe_name=E_ood`，正文显示名写作
   \(E_{\mathrm{mmluPro}}\)；这是 domain-matched probe，不是1400题逐题同语料；
3. MATH500 output 配 artifact `probe_name=E_math`，正文显示为当前
   external-math probe；它不是 MATH500 500题逐题同题 probe，禁止写成 item-matched；
4. \(p_k\) 必须使用 strict joint
   \[
   p_k=\frac{\|U_{0,k}^{\top}\Delta W_tV_{0,k}\|_F^2}
   {\|\Delta W_t\|_F^2},
   \qquad k\in\{4,8,16,32\},
   \]
   并在 q/k/v/o/gate/up/down 七模块分别计算后 equal-7；
5. Llama \(p_k\) 只读 `d11_llama_merged_pk.csv` 的
   `bf16_merged_minus_base` 正式轨；不得回退到 D10.5 中旧 adapter-BA \(p_k\) 列；
6. Qwen \(p_k\) 使用 `T_PK_qwen3_4b_fixedk.csv` 的 deployed merged 轨；
7. step0 全部量由定义等于0，所有相关和回归主分析必须排除 step0，防止共同原点抬高相关性；
8. MMLU 只使用 `aggregation=sample_macro` 作为正文主轨；category-macro 只作敏感性，
   不得与 sample-macro 重复计数；
9. \(p_k\) 不依赖 output domain。MMLU/Math 必须分别分析；池化时按
   `model,arm,checkpoint` 聚类/分组，不能把复制到两个 domain 的同一个 \(p_k\) 当成两次独立证据。

当前覆盖审计已经确定：

```text
c_epsilon-only:
  Qwen  = 4 arms x 9 checkpoints x 2 domains = 72 rows
  Llama = 4 arms x 6 checkpoints x 2 domains = 48 rows
  total = 120 rows

same-cell c_epsilon + p_k:
  Qwen p_k 缺四臂 step10，因此主公平交集排除 Qwen step10
  total = 112 rows
```

不得用插值、相邻 checkpoint 或 adapter-BA 倒填 Qwen step10 \(p_k\)。本任务不得因该缺口
阻断：完整 \(c_\varepsilon\)-only 表保留120行；涉及 \(p_k\) 的公平比较明确写
`QWEN_PK_STEP10_MISSING_MATCHED_EXCLUSION` 并使用112行。若未来另行授权真实 step10
\(p_k\) backfill，再追加 sensitivity，不覆盖本次 matched 结果。

## L1. 目标变量

所有量都使用相对同模型 step0 的 FAT-R1-v2 cell 结果。

MMLU-Pro：

```text
signed:
  delta_nll_p, delta_nll_f, delta_nll_a, delta_nll_t
unsigned:
  kl_f, kl_a, kl_t,
  abs_delta_nll_f, abs_delta_nll_a, abs_delta_nll_t
regional contrasts:
  delta_nll_f_minus_a,
  delta_nll_f_minus_p,
  kl_f_minus_a
```

MATH500：

```text
signed:
  delta_nll_p, delta_nll_c, delta_nll_b, delta_nll_t
unsigned:
  kl_b, kl_t,
  abs_delta_nll_c, abs_delta_nll_b, abs_delta_nll_t
regional contrasts:
  delta_nll_b_minus_c,
  delta_nll_b_minus_p
```

`kl_b_minus_c` 保持 NA，因为 FAT-R1 第一轮没有计算 `kl_c`；不得以0或 NLL 差替代。
`abs_delta_nll_z` 是样本内先取 \(|\Delta\mathrm{NLL}_{z,i}|\) 再 sample-macro，
不得误写成 \(|\operatorname{mean}_i\Delta\mathrm{NLL}_{z,i}|\)。

在进入相关性前，先从 sample rows 补正式 paired item-bootstrap contrast CI：

```text
MMLU:
  delta_nll_f_minus_a
  delta_nll_f_minus_p
  kl_f_minus_a

MATH:
  delta_nll_b_minus_c
  delta_nll_b_minus_p
```

seed=42、draws=1024；同一 contrast 两侧必须使用同一 item resample indices。现有
`fat_r1_v2_bootstrap_ci.csv` 只有 primitive-region CI，不能冒充 contrast CI。

## L2. 描述性关联：完整轨迹、去进度与同 checkpoint 排序

### L2.1 固定臂逐 checkpoint

对每个 `model x domain x arm x target` 分别计算：

\[
\rho_s(c_\varepsilon,Y),\quad
r_P(c_\varepsilon,Y),\quad
\rho_s(p_k,Y),\quad
r_P(p_k,Y),
\]

其中 Llama \(n=6\)，Qwen 的 \(c_\varepsilon\) 为 \(n=9\)，Qwen 的 \(p_k\) 为
\(n=8\)。必须登记 \(n\)，不得只报相关系数。四个 \(p_k\) 分别报告，不根据结果选择
最优 \(k\)。

### L2.2 checkpoint-demeaned

为去除“四臂都随训练时间累积”的共同趋势，在每个
`model x domain x checkpoint` 内对四臂去均值：

\[
\widetilde X_{m,D,a,t}
=X_{m,D,a,t}
-\frac14\sum_{a'}X_{m,D,a',t},
\qquad
\widetilde Y_{m,D,a,t}
=Y_{m,D,a,t}
-\frac14\sum_{a'}Y_{m,D,a',t}.
\]

分别报告
\(\rho_s(\widetilde c_\varepsilon,\widetilde Y)\) 和
\(\rho_s(\widetilde p_k,\widetilde Y)\)。这里“去 checkpoint 均值”必须按上式解释，
不是删除全局均值，也不是删除某个 checkpoint。

### L2.3 progress-residual

对每个 `model x domain x arm` 分别拟合：

\[
X=\alpha_X+\beta_X\log(1+t)+e_X,\qquad
Y=\alpha_Y+\beta_Y\log(1+t)+e_Y,
\]

报告 \(\rho_s(e_X,e_Y)\)。该表用于判断 raw trajectory correlation 是否只是共同训练进度。

### L2.4 四臂同 checkpoint rank

每个 `model x domain x checkpoint` 内以四臂为四个观测，报告
Spearman/Kendall 排序及跨 checkpoint 的中位数、符号一致率。单格 \(n=4\)，只能作为
cellwise 描述，不能把15个 checkpoint cell 当独立 seed 做显著性声明。

## L3. \(c_\varepsilon\) beyond \(p_k\) 的 grouped held-out 增量分析

Qwen、Llama 必须先分模型运行；signed targets 必须再给 pooled
`model x feature` interaction，禁止只报告抵消后的 pooled 系数。

外层按 checkpoint 分组留出：一个 fold 同时留出该模型该 checkpoint 的四臂。所有标准化只用
训练 fold。若使用 ridge，正则强度必须在训练 checkpoint 内层 grouped CV 选择，与现有
nested-regularization 协议一致。

连续 target 的模型块：

```text
M0    : arm one-hot + log1p(step)
MC    : M0 + c_epsilon
MP(k) : M0 + p_k                  # k=4,8,16,32 分别报告，公平单标量对照
MPall : M0 + p_4+p_8+p_16+p_32   # 强纯权重基线
MPC   : MPall + c_epsilon
```

报告：

```text
held-out R2
MAE
prediction Spearman
Delta R2(MC-M0)
Delta R2(MPC-MPall)
Delta MAE(MPC-MPall)
foldwise win count
```

`MC` 对 `MP(k)` 是一标量对一标量的公平敏感性；`MPC-MPall` 回答
\(c_\varepsilon\) 是否在完整 fixed-\(k\) 权重块之外提供信息。不得只挑对本文最有利的
\(k\)、target、模型或 domain。

预注册解释：

1. \(c_\varepsilon\) 对 KL/`abs_delta_nll` 强、对 signed NLL 弱，支持“功能压缩更接近
   unsigned output departure，而 signed readout 仍需额外方向信息”；
2. signed 区域关系在 Qwen/Llama 反号或模型交互显著，属于 model-dependent readout
   boundary，不得平均成统一损伤；
3. `MPC-MPall` 稳定为正才可声称 \(c_\varepsilon\) 在 \(p_k\) 之外有增量 output
   information；
4. raw correlation 强但 checkpoint-demeaned、progress-residual 和 grouped held-out
   均消失，只能写共同训练进度相关；
5. 负结果不阻断交付，也不得触发重新选择 epsilon、层、模块或 checkpoint。

## L4. 必须交付

目录：

```text
mini/fat_outlink_round1_v2_link/
```

文件：

```text
fat_r1_v2_link_task_status.csv
fat_r1_v2_link_feature_matrix.csv
fat_r1_v2_link_coverage.csv
fat_r1_v2_link_region_contrast_bootstrap_ci.csv
fat_r1_v2_link_within_arm_correlations.csv
fat_r1_v2_link_checkpoint_demeaned_correlations.csv
fat_r1_v2_link_progress_residual_correlations.csv
fat_r1_v2_link_within_checkpoint_rank.csv
fat_r1_v2_link_incremental_models.csv
fat_r1_v2_link_incremental_predictions.csv
fat_r1_v2_link_model_interactions.csv
fat_r1_v2_link_handoff.md
fat_r1_v2_link_manifest.json
```

handoff 必须机械报告完整 coverage、缺失 cell、相关系数、held-out 指标和分支代码；Coder 不得
写“压缩导致格式损伤”“\(c_\varepsilon\) 优于所有权重指标”等理论裁决。

允许状态：

```text
COMPLETE_FAT_R1_LINK
COMPLETE_FAT_R1_LINK_MIXED_OR_NEGATIVE
PARTIAL_FAT_R1_LINK_PK_STEP10_EXCLUDED
FAILED_FAT_R1_LINK_JOIN_PARITY
FAILED_FAT_R1_LINK_NUMERIC_OR_SCHEMA
```
