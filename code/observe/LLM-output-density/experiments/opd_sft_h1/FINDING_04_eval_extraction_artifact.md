# 重大发现 04 / MAJOR FINDING（Cycle 04）

```yaml
artifact_type: major_finding
cycle: cycle_04_opd_stability_gain
date: 2026-06-15
status: confirmed_with_evidence
severity: HIGH  # 影响所有数学评测的解读，后续分析必须知晓
discovered_during: cycle04 medium-scale run + NuminaMath-test floor probe
```

> **⚠️ 给后续分析（result interpretation / Cycle 05 seed）的强提示**：
> 本轮发现了一个会**系统性误导结论**的评测假象，并据此找到一个**更干净的指标结构**。
> 凡是引用 Cycle 04 数学评测数字的人，**先读完本文件再下结论**。

---

## 一句话结论

**Cycle 04 里 `MATH500 = 0.0`（所有模型）不是"模型不会做"，而是 lm_eval 严格答案抽取在我们的输出格式上系统性失败造成的"抽取地板"（scoring artifact）。** 用宽松抽取 + 同分布留出集（NuminaMath-test）重测，同一批模型拿到 **open-answer ≈ 0.38**，证明能力非零。由此 NuminaMath-1.5 的留出集可作为**域内（ID）主指标**，让 GSM8K 退为**泛化（OOD）指标**——这是比 GSM8K 身兼二职更干净的设计。

---

## 发现 1：竞赛数学评测被"抽取地板"打成 0（scoring artifact）

任何评测分两步：**(a) 模型生成解题文本 → (b) 打分器从文本里抽取最终答案并判等价**。若 (b) 期望的格式与模型实际输出不一致，就会抽不到答案 → 全判错 → 总分 0.0。**模型可能答对了，但打分器"没读到"。**

### 铁证（同一次 cycle04 eval，同一批模型、同一批答案）

| 任务 / 抽取方式 | theta0 得分 | 说明 |
|---|---|---|
| GSM8K `exact_match, strict-match`（严格） | **0.000** | 严格抽取读出 0 |
| GSM8K `exact_match, flexible-extract`（宽松） | **0.413** | 宽松抽取读出 41% |
| `hendrycks_math500 exact_match, none`（仅严格路径） | **0.000** | 无宽松回退 → 卡在 0 |
| NuminaMath-test（本轮 floor probe，宽松 `\boxed`+sympy） | **0.388**(open) | 同难度，能力非零 |

> 同一个模型、同一批生成：GSM8K 严格=0%、宽松=41%。差别**只在"怎么读答案"**。
> MATH500 只配了严格路径，所以停在 0.0。这就是"抽取地板"。

### 对 Cycle 04 结论的影响（务必照做）

1. **`MATH500 = 0.0` 这一行不可解读为能力或 OOD 稳定性**，它**无信息量**。在 OOD-penalty / trajectory 分析中应**剔除或显式标注"extraction-floored, uninformative"**，不得纳入 worst-OOD-drop 的判断。
2. **Gate 结论本身不受影响**：Gate B/C/D 依赖的是 GSM8K（用的是 flexible-extract，0.33–0.43，有效）+ OOD-lite（MMLU / TruthfulQA / WinoGrande，多选 acc，不走 `\boxed` 抽取，有效）。MATH500 不是任何 Gate 的判据。
3. **项目级教训**：本仓库后续**任何接近 0 的数学评测**，先确认是 flexible / math_verify 抽取，再下"模型不会"的结论。

---

## 发现 2：NuminaMath-1.5 有一个可用的"域内（ID）留出集"

`/root/autodl-tmp/prepared/NuminaMath-1___5/test.jsonl`（1024 条），与训练 prompt **严格不相交**，可作 ID 下游主任务。

### 不相交的证据
- `split_meta.json`：`seed=42`，从 896,215 条里先切 1024 条 test、再从剩余 785,414 抽 40,000 进 train.parquet；**`train_row_ids ∩ test_row_ids = 0`**。
- 直接比对：cycle04 的 1024 条训练 prompt vs test.jsonl 1024 条，**重叠 = 0**。
- 训练池确为 NuminaMath train（同 prepared 目录的 train 切片）。

### 难度构成（test 1024 条来源分布）——不是纯奥赛，60% 易–中
| 来源 | 数量 | 难度 |
|---|---|---|
| cn_k12 | 378 (37%) | 易–中 |
| orca_math（GSM8K 级应用题） | 222 (22%) | 易 |
| synthetic_math | 213 (21%) | 中 |
| olympiads | 139 (14%) | 难 |
| cn_contest / aops / amc_aime / 其它 | ~58 (6%) | 很难（竞赛尾） |
| metamath | 14 | 易 |

题型：math-word-problem 828 + MCQ 196。

### floor probe 结果（N=96，非思考贪心，统一 boxed 指令）——未触地板，可用
| 模型 | overall | open-answer(N=85) | MCQ(N=11) | boxed 产出率 |
|---|---|---|---|---|
| teacher Qwen3-4B | 0.396 | 0.376 | 0.545 | 0.667 |
| theta0 1.7B 冷启 | 0.385 | 0.388 | 0.364 | 0.740 |
| opd_lmbda05 1.7B | 0.385 | 0.388 | 0.364 | 0.667 |

**含义**：open-answer ~0.38，动态范围充足 → NuminaMath-test **可作 ID 主指标**；GSM8K 可重定位为**泛化/迁移指标**。该 ~0.38 仍是**下界**（见发现 3 的 scorer 缺陷），真实更高。

---

## 发现 3：OPD 的可测优势在 OOD，不在 ID（重要 caveat，勿误读）

| 轴 | theta0 | opd_lmbda05 | 差 |
|---|---|---|---|
| **GSM8K（OOD / 泛化，N=1319，flexible）** | 0.413 | 0.431 | **+0.017** |
| **NuminaMath-test（ID，N=96 probe）** | 0.388 | 0.388 | **0.000** |

- theta0 与 opd_lmbda05 在 ID 上得分相同（boxed 率不同→确为两个不同模型，分数为巧合相等）。
- **OPD 的信号体现在 OOD（GSM8K）保持/略升，而非 ID 增益。** 因此把 NuminaMath-test 设为"增益主指标"，OPD 多半同样**过不了 Gate D**——OPD 的故事仍是**稳定性 / OOD 保持**，不是 **ID 增益**。这是对叙事的**重构**而非对 Gate D 的**拯救**。
- N=96 的 ID 持平可能是巧合，**需全 1024 + 硬化 scorer 复核**（见待办）。

---

## scorer 当前缺陷（为何 0.38 是下界 / 为何要"硬化"）

floor probe 用的是手写 scorer，存在**假阴性**（把对的判成错）：
1. **MCQ 漏判**：`\text{C}`、`(C)`、多选 `(B),(C),(D)` 未被识别为选择题 → 当成 open-answer 后判错（样例：gold=`\text{C}`、pred=`\text{C: }...` 被判 False）。
2. **区间 / 元组 / 集合**：`x∈(0;4]`、`(x;y)=(0;0)` 归一化不准。
3. **boxed 产出率仅 0.67–0.74**：1/4–1/3 输出没有可抽取的 `\boxed{}`（teacher 尤甚，故 teacher 被多扣分、看起来不比 1.7B 强）。

**"硬化 scorer"** = 加固判分鲁棒性、堵假阴性：装 `math_verify`（HF 出品的数学答案判等库，LaTeX/符号/区间/集合等价，竞赛数学判分事实标准）+ 修 MCQ/区间识别 + 更强 boxed 指令（把产出率提到 ~0.95）。目标：**分数反映"题做没做出来"，而非"格式有没有撞上判分器口味"**。

---

## 待办（geometry 收尾后执行，已与用户确认）

1. 安装 `math_verify`，硬化 `scorer.py`（MCQ `\text{}`/多选、区间/元组/集合、math_verify 等价）。
2. 在**全部 7 个 merged 模型 × NuminaMath-test 全 1024 条**上重评（**无需重训**，模型已在盘）。
3. 产出 **ID（NuminaMath）× OOD（GSM8K）双轴表**，看 OPD/theta0/4×SFT 在 ID 上全量 N 下是否分离。
4. 据此决定：纳入 **Cycle 04 附录** 还是作 **Cycle 05 主指标设计**。

## 证据出处（artifacts）

- cycle04 eval 汇总：`/root/autodl-tmp/cycle04_opd_stability_gain/eval/csv_results/target_metrics_results.csv`
- GSM8K strict vs flexible：`eval/origin/_combined/theta0_*.json`（`results.gsm8k`）
- NuminaMath split 元数据：`/root/autodl-tmp/prepared/NuminaMath-1___5/split_meta.json`
- floor probe 代码 + 结果：`/root/autodl-tmp/floor_probe/{scorer.py, probe.py, run_all.sh, *.json, floor_probe.log}`
