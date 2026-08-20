# QA: Cycle 07 — Experiment Design Phase

```yaml
phase: experiment_design
cycle: cycle_07_base_model_sft_feasibility
status: concluded_2026-06-24
created: 2026-06-24
write_scope: exp_only (含 code/current_code_brief.md)
internalize_to:
  - exp/current_experiment_design.md
  - exp/current_acceptance_criteria.md
  - code/current_code_brief.md
  - exp/experimental_methodology/  # 仅在 methodology 实际变化时
```

本 QA 文件是 Cycle 07 experiment design 阶段开始前的审查清单。
用户回答并讨论确认后，写正式设计文件。

---

## 背景摘要（已确认，来自上游阶段）

**来自 Theory 对话（QA_cycle07_theory.md，active）：**
- 核心框架升级为 **OPD vs SFT_dip vs SFT_recover** 三方对照。
- OOD-drop 被重定义为优化轨迹上的 checkpoint 属性，非方法属性（Rethink SFT dip-and-recovery）。
- H-OPD（OPD ≈ 直达 recover、跳过 dip 暂态）是 Cycle 08 的预注册假设，Cycle 07 不训练 OPD arm。
- **几何 + response_length 必须沿 SFT 轨迹多 checkpoint 采样**——这是 Experiment Design 的硬要求。

**来自 Related Work 对话（source_matrix.md，更新至 2026-06-24）：**
- 核心新来源：**Rethink SFT（Ren et al. arXiv 2604.06628）**。
- 模型方向：Qwen3-4B-Base（pretraining checkpoint），1.7B-Base 能力地板已由 Rethink SFT Table 6 独立复现（MATH500 净负）。
- 数据方向：**放弃 NuminaMath-1.5 短解**，换 verified long-CoT 配方（NuminaMath 在 Rethink SFT Table 2 中被点名为把 OOD reasoning 打到地板的最差数据）。
- 格式方向：**翻转为 think-format**（`<think>...</think>` long-CoT 为训练 target，对应 Rethink SFT 实验格式）。
- 评测轴方向：MATH500 + AIME24（ID）；GPQA-D + MMLU-Pro（OOD reasoning）；LCB v2（代码 OOD）。

**外部数值锚（Rethink SFT Table 9，Qwen3-4B-Base + Math-CoT-20k，step 640）：**

| 指标 | 未训练 base | 训练后 (step 640) | Δ |
|---|---|---|---|
| MATH500 | 70.8 | 86.2 | +15.4 |
| AIME24 | 11.0 | 30.7 | +19.7 |
| LCB v2 | 20.6 | 24.4 | +3.8 |
| GPQA-D | 14.6 | 37.9 | +23.3 |
| MMLU-Pro | 35.5 | 64.8 | +29.3 |

dip 可见（MATH500 step 20 = 62.9 < base 70.8，后回升至 86.2）。判 fail 前须看完整 trajectory。

**Cycle 06 遗留（须在 Cycle 07 代码阶段解决）：**
- pre-experiment Task 1：base 系列 stop-fix（stop=["\n\nProblem:"]），解决 A06b UNDETERMINED
- pre-experiment Task 2：Cycle 06 geometry copyback（geometry_metrics.csv + principal_evidence.csv）

---

## 核心问题

### Q1：Qwen3-4B-Base 本地路径确认

**背景：** `/root/autodl-tmp/model/Qwen/` 目录下已有：
- `Qwen3-1.7B`（instruct，已用于 Cycle 04–06）
- `Qwen3-4B`（instruct，目前只用作 teacher）

Cycle 07 的主模型是 **Qwen3-4B-Base**（pretraining-only checkpoint，无 instruction tuning）。

**问题：**
- a. `/root/autodl-tmp/model/Qwen/Qwen3-4B-Base` 是否已存在本地？
- b. 如果不存在，需要下载——是否有网络/存储可以在代码阶段完成？或者需要提前手动下载？
b 我将下载到同个文件夹下

---

### Q2：Long-CoT 训练数据来源（关键瓶颈）

**背景：** Rethink SFT 使用的 Math-CoT-20k = OpenR1-Math-220k 的 query + Qwen3-32B（thinking）生成的 long-CoT + math-verify 过滤。本地没有这份数据，NuminaMath-1.5 短解已放弃。

**可选方案：**

| 方案 | 描述 | 可行性 |
|---|---|---|
| **A. 用现成预生成数据集** | 例如 HuggingFace 上的 OpenR1-Math 或类似含 long-CoT 的数学数据集（如 `open-thoughts/OpenThoughts-114k`、`Bespoke-Stratos` 等） | 需确认可访问、格式适配 |
| **B. 用 Qwen3-4B-Instruct 自生成** | 用本地 teacher（`/root/autodl-tmp/model/Qwen/Qwen3-4B`）对 NuminaMath 或其他 math query 库开思考模式生成 long-CoT，math-verify 过滤 | 轻量可控，但生成质量低于 32B；需时间 |
| **C. 其他已有数据** | 是否有其他 long-CoT 数学数据集已在本地或 autodl-tmp？ |  |

**问题：** 以上哪种方案可行？目标数量级是多少（5k / 10k / 20k 样本）？
使用现成数据集，我将下载到autodl-tmp/prepared 文件夹下
但是在此之前请你先访问https://www.modelscope.cn/collections/nebularaid/Rethink_SFT_generalization这个网址，判断我们下载哪个数据集比较好，之后我会自己下载

---

### Q3：Think-format 训练与评测协议

**背景：** Rethink SFT 的训练 target 是 `<think>...</think>` long-CoT，训练时用 Qwen chat template 包装（user/assistant turns）。评测时同样用 chat template，模型生成含 think 的完整回答，提取 `</think>` 后的内容打分，max 32768 tokens。

这是对当前项目训练/评测协议的**两处重大变更**：

1. **训练格式翻转**：从 no-think 非思考 target → think-format `<think>...</think>` target
2. **评测格式翻转**：从 `enable_thinking=False` → think 模式（max_gen_toks 需扩至 32768，提取器改为 `</think>` 后内容）

**问题：**
- a. 确认 Cycle 07 采用 think-format（与 Rethink SFT 对齐）？
- b. 评测时 max_gen_toks 上限：32768（Rethink SFT 默认）还是更保守的值（如 8192）？较长的生成预算对训练中间 checkpoint 的 response_length 诊断是必要的（dip 期 response_length 会暴涨至 28–30k）。

这两个我都确认，都需要和rethink sft对齐

---

### Q4：评测轴可行性确认

**背景：** Rethink SFT 使用 MATH500 + AIME24（ID reasoning）+ LCB v2 + GPQA-D + MMLU-Pro（OOD reasoning）。

| 轴 | N | 挑战 | 本项目现状 |
|---|---|---|---|
| MATH500 | 500 | 无 | 有 runner，已验证 |
| AIME24 | 30 | N 极小，avg@10 = 300 次生成；pass@1 高方差 | 无 runner |
| LCB v2 | ? | 需要 code execution 环境 | 无 |
| GPQA-D | 198 | 无 | lm-eval 可运行 |
| MMLU-Pro | ? | 无 | lm-eval 可运行 |

**问题：** 
- a. AIME24：是否纳入？（推荐：pass@1，N=30，可作为"有没有很难题的 gain"的粗略信号，不作为主要 gate）
- b. LCB v2：是否纳入？（代码执行环境复杂；4B 上 gain 也最小 +3.8。若不纳入，理由是工程成本 > 信息增量）
- c. 其他：NuminaMath-test 是否还需要？（format 不同，不可与之前直接比较，但可以作为"是否还保持领域内能力"的诊断）

LCB先暂时不使用，AIME24纳入，Numina还需要它退化为一般的ID。现在的MMLU-pro我希望能够简化一些，我在cycle06的代码中可能对mmlu进行过修改，测试不需要那么多可以少一点，这样每个测试都比较均衡，请你判断要如何减少一定量的测试。

---

### Q5：训练规模与 Checkpoint 采样网格

**背景：** Theory 硬要求：**必须沿 SFT 轨迹多 checkpoint 采样**几何指标 + response_length，否则 dip 不可见、Cycle 08 三方对照缺基线。

Rethink SFT 的配方：20k 样本 × 8 epoch × bsz 256 = 640 steps；checkpoint 在 steps {10, 20, 40, 80, 160, 320, 480, 640} 采样。

本项目约束：单卡 48GB，LoRA（全量 FT 的内存不支持）。

**问题：**
- a. 数据量：用可用的 long-CoT 数据全量还是限额（如 5k/10k/20k）？
- b. Batch size + gradient accumulation：有效 batch size 多少？（影响 steps per epoch，进而影响 checkpoint 网格密度）
- c. Checkpoint 网格：为了确保 dip（step 20–40 附近）可见，需要至少在前 50 步密集保存一个 checkpoint（如 step 5/10/20/40/80/160/320/640）。请确认是否接受这个保存频率。

a.全额的话时间大致是多少？
b.原文中的batch size我们是不是用不了，太大了？如果是的话减小到我们适合的大小，同时这也会改变steps吗？还是其实关系不大，steps还是只需要和之前保持一致？我认为这个问题的核心是每个batch的梯度是不是会归一化，这样batch只起到稳定的作用，steps才是决定移动多少？
c.是的确实需要这样，而且我还想知道现在的LoRA是不是会快速收敛一些，相较于原文的全参数？同时这样我们是不是应该删除一些之前的cycle的结果，释放一些磁盘空间，为了现在和未来的checkpoints能够存储的下？

---

### Q6（快速确认）：LoRA 配置 + Pre-experiment 任务安排

**6a. LoRA：** 沿用 Cycle 04–06 的 LoRA 配置（r=16 / alpha=32 / dropout=0.05，target_modules=[q/k/v/o/gate/up/down_proj]）？还是为 4B 模型调高 rank（如 r=32/64）？

> Rethink SFT 用全量 FT，LoRA 的 dip-and-recovery 幅度可能不同——是否接受这是一个 project-specific variant，和 Rethink SFT 的数字不直接可比？

**6b. Pre-experiment 任务安排：** 
stop-fix（A06b）和 geometry copyback（C06 PARTIAL）是否作为代码 brief 的 Phase 0 独立运行（先于主训练），还是并行？这些任务不依赖 Qwen3-4B-Base 数据，可以立即开始。

6a.我认为需要调高一些，你能不能联网搜索一下类似的训练使用的大致是怎么样的LoRA参数？
6b.我认为是先解决，并行时显存可能不够用。同时stop-fix真的需要吗？我们训练的不是think格式的吗？

---

## 讨论结论

```yaml
status: concluded_2026-06-24
```

经 Q1–Q6 讨论确认的所有设计决策，已全部写入正式文件（见 `internalize_to` 列表）。

### Q1 结论
- 模型路径：`/root/autodl-tmp/model/Qwen/Qwen3-4B-Base`（已下载并移动至该位置，2026-06-24 确认）

### Q2 结论
- 使用 `nebularaid/Math-CoT-20k`（Rethink SFT 论文官方数据集，ModelScope 镜像）
- 已下载至 `/root/autodl-tmp/dataset/Math-CoT-20k/Math-CoT-20k.parquet`（159MB，parquet 格式）
- 训练取 5000 样本，seed=42

### Q3 结论
- 训练格式：think-format（`<think>...</think>` long-CoT 为训练 target），与 Rethink SFT 对齐
- 评测：max_gen_toks=32768，think 模式，与 Rethink SFT Table 9 对齐
- 提取器：last `\boxed{}` in full output（`</think>` 前后都扫描）

### Q4 结论
- LCB v2：暂不纳入（工程成本 > 信息增量）
- AIME24：纳入，pass@1，N=30（OOD hard math signal）
- NuminaMath-test：纳入（作为 ID 退化检查，不与 Cycle 06 直接比较——格式已改变）
- MMLU-Pro：纳入，使用 `--limit 500 --seed 42` 缩减至约 500 题（与 MATH500 规模对齐）
- 最终评测集：MATH500 + NuminaMath-test + AIME24 + GPQA-D + MMLU-Pro（共 5 个任务）

### Q5 结论
- 数据量：5k 样本（≈ 5000/64×8 ≈ 624 步，近似 Rethink SFT 640 步）
- Batch size：per_device=1，gradient_accumulation=64 → effective_bsz=64（原文 256 显存不够，但步数决定优化轨迹，batch 主要起稳定作用）
- Checkpoint 网格：{5, 10, 20, 40, 80, 160, 320, 480, 624}（步数 + step_000 共 10 个）
- LoRA 相比全参数可能收敛更快，dip 可能较浅或提前——需实测观察
- 旧 checkpoint 清理：根据磁盘情况在 Cycle 07 结束后处理（非阻塞本次训练）

### Q6 结论
- LoRA 参数：升级为 r=32 / alpha=64（适合 4B 模型 + 复杂推理任务；联网搜索确认 r=32 是 4B 规模的主流选择）
- Pre-experiment 安排：
  - **stop-fix（A06b）取消**——Cycle 07 用 think 格式，base 系列 plain-format stop-fix 已无意义；A06b 关闭为 deferred
  - **Geometry copyback（C06 PARTIAL）** 移至 Cycle 07 所有评测完成后执行（Phase 4 deferred task）
  - Phase 0 从 code brief 中删除，GPU 立即开始主训练
