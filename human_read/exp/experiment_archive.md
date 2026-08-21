# 所有实验设计归档

本文件按时间顺序追加所有定稿或实质修订的实验设计，包括最终证明无效或失败的设计。历史条目只追加勘误说明，不改写其原始判断标准。

## 归档模板

### <cycle ID> / <experiment ID> / 设计版本 <N> — <日期>

- 状态：计划中 / 已执行 / 取消 / 被替代
- 对应理论版本：
- 版本变化：
- 要回答的问题：
- 假设与预测：
- 变量、对照和数据：
- 指标与预先判断标准：
- 消融与执行顺序：
- `C_{\gamma}` 观测计划：
- 资源预算与停止条件：
- 实际实验目录或取消原因：
- 对应结果：

## 归档记录

### `cycle-20260818-01` / `exp-20260818-01-standard-mvp` / 设计版本 v0.1 — 2026-08-18

- 状态：设计中；已选择标准数据集路线，待实现代码并冻结数据 revision 与正式配置。
- 对应理论版本：`../theory/current_theory.md` v0.1。
- 版本变化：从目录占位稿形成首个实质实验设计；放弃用 toy experiment 代替正式 MVP，改用 UltraFeedback 标准偏好数据，同时保留 smoke/pilot 作为工程验收而非论文证据。
- 要回答的问题：群体级 label-encoding 结构监督能否在有限真实标签下利用方向未知的真实回复对，并优于 instance-level hard/soft pseudo-target。
- 假设与预测：H1 为 PE 优于 DPO-10% 且 Brier 不恶化；H2 为 PE 优于两个 pseudo 方法中的更强者；H3 为提升不能仅来自置信度变尖或预测坍缩。
- 变量、对照和数据：UltraFeedback；10% 有标签、80% 隐藏方向、10% 测试；主模型候选 Mistral-7B-Instruct-v0.2；方法为 DPO-10%、Pseudo-hard、Pseudo-soft、DPO + PE、DPO-100%；正式种子为 42/43/44。
- 指标与预先判断标准：Acc、Brier、paired bootstrap 差值区间、预测比例、两方向责任质量、熵、校准和数值稳定性；PE 需在三种子上方向一致且差值区间不跨 0，Brier 不恶化。
- 消融与执行顺序：先本地测试，再做预检、数据审计、五方法 smoke、单种子 pilot、pilot 决策门，最后执行 5 方法 × 3 种子的正式运行与独立聚合。
- `C_{\gamma}` 观测计划：待 `../../../observe/` 核验；若及时完成只作为预注册诊断指标，不阻塞核心实验。
- 资源预算与停止条件：A800 集群；smoke 1 GPU、pilot 5 个独立 1-GPU 任务、正式 15 个独立 1-GPU 任务；数据泄漏、未解决 NaN/Inf、输出不一致、预算不公平或配置未冻结时停止下游任务。
- 实际实验目录或取消原因：首次实际运行前创建 `../../exp/exp-20260818-01-standard-mvp/`；服务器操作见 `../../../machine/A800_standard_MVP_runbook.md`。
- 对应结果：尚无。

### `cycle-20260818-01` / `exp-20260819-01-mvp` / 设计版本 v0.5 30k MVP — 2026-08-21

- 状态：已批准；用户明确把数据规模扩大为 30k，并明确要求保持原始比例。当前处于 `CODE_IMPLEMENTATION`，服务器执行仍需代码交接后的单独授权。
- 对应理论版本：`../theory/current_theory.md` v0.2（已通过）。
- 数据合同：2,700 labeled-train（9%）+ 300 labeled-validation（1%）+ 24,000 unlabeled（80%）+ 3,000 test（10%）= 30,000；旧 10k 目录不覆盖，新目录为 `mvp-v0.5-30k`。
- Batch 合同：常规 labeled global batch=32（4×4×2）；DPO+PE 的 exact population global batch=128（8×8×2）；DPO-100 global batch=128（8×8×2）。
- smoke 合同：`gpu_test` 请求 2 张 GPU，wall time 50 分钟；不假定卡型，记录实际硬件；覆盖 Qwen3 离线加载、reference cache、SFT/DPO-10/Pseudo/DPO-100/DPO+PE、FSDP、前反向、精确 PE population batch 和 checkpoint round-trip。正式训练运行时硬检查 2×A800。
- 自动执行：`submit_all.sh` 先以 hold 提交完整 Slurm DAG，注册全部 job ID 后一次释放；所有边使用 `afterok`，任一上游失败会阻止下游。
- checkpoint 合同：不自动删除。预实验与 λ 搜索每 80 step 保存并保留最终点；主实验每 40 step 保存并保留最终点。保存内容为 HF 模型、tokenizer 与运行配置，不包含 optimizer state。
- 科学判断：H1/H2/H3、validation-only 选择、headroom ≥5%、单种子趋势边界和测试标签隔离保持不变。

### `cycle-20260818-01` / `exp-20260818-01-standard-mvp` / 设计版本 v0.2 — 2026-08-18

- 状态：设计中；替代 v0.1 的执行位置规则，研究问题、对照与判断标准不变。
- 对应理论版本：`../theory/current_theory.md` v0.1。
- 版本变化：确立“本地控制面、服务器执行面”的不可放宽边界。本地仅编辑源码/文档、SFTP 上传和接收聚合摘要；依赖、环境、测试、数据、模型、smoke、pilot、训练、评价、聚合、绘图及全部 GPU 工作均转为服务器任务。
- 要回答的问题：与 v0.1 相同，检验群体级 label-encoding 结构监督能否利用方向未知的真实回复对并优于 hard/soft pseudo-target。
- 假设与预测：H1/H2/H3 与 v0.1 相同，不因基础设施规则调整而改变。
- 变量、对照和数据：UltraFeedback 与五方法不变；数据/模型只在服务器共享存储下载、读取与保存，服务器计算 revision 校验值；本地不保存副本。
- 指标与预先判断标准：Acc、Brier、paired bootstrap、责任质量、熵、校准与坍缩诊断不变；所有评价和统计在服务器完成。
- 消融与执行顺序：本地纯文本设计/编码 → SFTP 上传 → 服务器环境与预检 → 服务器测试 → 服务器数据准备 → 服务器 smoke → 服务器 pilot → 服务器审计 → 15 个正式服务器任务 → 服务器评价聚合 → 本地只接收白名单摘要。
- `C_{\gamma}` 观测计划：不变；若实现，其计算也只能在服务器完成。
- 资源预算与停止条件：GPU 预算不变；新增硬停止条件为任何步骤要求本地安装包、读取数据/模型、运行测试/分析或执行 GPU 工作。
- 实际实验目录或取消原因：本地 `../../exp/exp-20260818-01-standard-mvp/` 只保存无样本级聚合摘要和远程索引；数据、环境、原始日志、逐样本输出、模型及 checkpoint 留在服务器。服务器操作见 `../../../machine/A800_standard_MVP_runbook.md` v0.2。
- 对应结果：尚无。

### 流程状态附注 — 2026-08-19（不构成新设计版本）

- v0.1 与 v0.2 均形成于理论逐段讨论和显式批准门建立之前，现统一视为未来 `EXP_DISCUSSION` 的预讨论草案。
- 当前理论 v0.1 尚待用户明确通过，因此实验阶段未进入；两个设计版本均未获用户批准，也没有服务器执行授权。
- v0.1 中“先本地测试”的历史表述已经作废，受全局本地执行硬约束禁止；不得据此在本地运行任何测试。
- 本附注只更正流程身份，不改写历史设计内容。进入实验讨论后，应以用户逐项审阅后的版本作为新的当前设计；在明确批准和授权前，不得实现、上传或运行。

### 流程状态更新 — 2026-08-19（不构成实验设计版本）

- 理论现已更新为 v0.2 并获用户明确通过；当前活动对象改为 `current_experiment.md` v0.1，处于 `EXP_DISCUSSION`。
- 正式流程新增 `CODE_IMPLEMENTATION`：实验设计通过后先在 `../../code/` 完成实现和 `CODE_OVERVIEW.md` 交接，再由用户确认代码版本是否可以提交服务器。
- 旧归档中任何“实验批准后直接执行”或“实现与服务器执行合并”的表述均已失效；历史条目保留仅用于追溯。

### `cycle-20260818-01` / `exp-20260819-01-mvp` / 设计版本 v0.4 MVP — 2026-08-21

- 状态：已批准（用户明确确认，2026-08-21）；当前处于 `CODE_IMPLEMENTATION`，服务器执行尚未授权。替代 v0.3 的服务器执行架构草案，研究问题、四组核心对照、单种子 MVP 与 H1/H2/H3 判断标准不变。
- 对应理论版本：`../theory/current_theory.md` v0.2（已通过）。
- 版本变化：模型固定为 ModelScope `Qwen/Qwen3-4B`；环境要求 Transformers ≥4.51.0 并锁定 ModelScope；模型在 `gn001` 下载校验。新增 `gpu_test` 单卡 10 分钟强制 smoke，并把剩余服务器工作改为一次提交的 Slurm job-array/`afterok` DAG。
- 要回答的问题：与当前 v0.4 完整设计 §1 相同，检验 population-level label-encoding 结构监督能否在有限偏好标签下优于 DPO-10% 与 instance-level pseudo target。
- 假设与预测：H1/H2/H3 及 Acc >2%、Brier 不恶化、C_ε 机制诊断不变；自动化不改变统计或科学判断标准。
- 变量、对照和数据：UltraFeedback 10k；DPO-10%、Pseudo-target、DPO+PE、DPO-100%；seed=42；Qwen3-4B；测试标签继续与训练入口隔离。
- 指标与预先判断标准：预实验 headroom ≥5%；ε/β/lr 与 λ 只依据 validation 和预注册 tie-break 规则选择；任一验收失败时自动链停止，不自行更换模型、数据或阈值。
- 消融与执行顺序：Qwen3 下载/manifest → 一次提交 DAG → CPU 测试 → 10 分钟 GPU smoke → 预实验数组/选择 → λ 数组/选择 → 主实验数组 → C_ε 数组 → 独立评价数组 → 聚合与白名单导出。
- `C_{\gamma}` 观测计划：沿用当前设计的 C_ε 方案；只有主实验 checkpoint 合同完整后才运行。
- 资源预算与停止条件：沿用 v0.3 约 53–103 GPU·小时估算，增加 smoke ≤10 GPU 分钟；smoke、headroom、NaN/Inf、标签隔离、依赖产物或状态合同失败均阻断下游。
- 实际实验目录或取消原因：尚未执行；代码实现已解锁，服务器授权仍锁定。
- 对应结果：尚无。

### `cycle-20260818-01` / `exp-20260819-01-mvp` / 设计版本 v0.6 SSPO-aligned 30k MVP — 2026-08-21

- 状态：已批准；用户确认设计无问题并明确要求开始编码。当前处于 `CODE_IMPLEMENTATION`，服务器执行仍未授权。
- 对应理论版本：`../theory/current_theory.md` v0.2（已通过）。
- 版本变化：替代 v0.5 的 SFT/Pseudo/DPO+PE 设计。删除 SFT、hard-static 和 linear scheduler；headroom 改为同一 mean-logp score 下 DPO-10 对训练前冻结 base 的提升，阈值仍为 .05；DPO-100 只作 oracle。训练统一 Qwen3-4B 标准 LoRA r8/alpha16/dropout0/all projections、max length 2048、global batch64。
- 数据合同：继续复用 30k 的 2,700 labeled-train + 300 validation + 24,000 unlabeled + 3,000 test；不覆盖已冻结的 `mvp-v0.5-30k` 数据目录。
- 八条最终轨迹：DPO-10、DPO-100、SSPO-hard-exp、SOPPO-PE-exp，以及 normalized fixed lambda `{0.1,0.3,0.5,1.0}` 的四条 PE。
- SSPO 对齐：labeled 为 SimPO mean-logp loss（beta10、margin2），hard unlabeled 为 single-response KDE Bayes threshold pseudo-risk，prior .5、EMA .95、200 grid；论文未给 bandwidth，冻结 Scott rule 作为复现决定。
- PE 对照：pair probability 不含 margin；PE responsibility/denominator 默认不断梯度；exp arm 与 hard 使用同一个 `gamma_t=max(gamma_min, exp(-0.01t))`，static arm 使用 `1/(1+lambda)` 与 `lambda/(1+lambda)`。
- 优化合同：DPO 1 epoch/lr1e-6；SSPO/PE 2 epochs/lr1e-5；AdamW、weight decay0、cosine、warmup.1、clip1、seed42。联合每步精确 8 labeled pairs + 56 unlabeled pairs。
- checkpoint：全部保留 PEFT adapters；DPO 每20 step、SSPO/PE 每40 step及 final；不保存 optimizer state。评价和 GetSlice 在内存中加载/合并 adapter。
- 执行顺序：CPU tests → strong smoke → reference → 两条 DPO/headroom → 四条 static PE/validation selection → hard-exp 与 PE-exp → C_epsilon → 8-arm independent test → aggregate。
- 2026-08-21 服务器路由勘误：当前账户仅关联 `gpu|normal`，且 `sbatch --hold`/typed `--gres` 被集群拒绝，`sbatch -G 1/2` 已实测成功；v0.6 实现改为辅助阶段1卡、smoke/正式训练2卡的直接 `afterok` 提交，并在中途提交失败时自动取消本次已提交 job。该变更只影响调度，不改变训练目标或超参。
- 2026-08-21 首次 DAG 启动证据：job `1500656` 在训练前因 Slurm spool 副本错误解析 `job_env.sh` 而失败，下游全部由 `afterok` 阻断，未发生训练。实现改为显式传递仓库 cluster 目录，并修正状态工具的 array task ID 字段。
- 2026-08-21 第二次 DAG 启动证据：tests 与两卡 strong smoke 成功；reference-cache job `1500671` 在完整30k编码时发现 Qwen3 tokenizer 跨 assistant 边界合并 token，使“整体编码必须以单独 prompt 编码为前缀”的旧假设失败。修正为 prompt 与 response+EOS 分别编码后拼接 IDs，使 response-only mask 按构造精确；正式训练仍未启动。
- 对应结果：尚无；当前只进行本地纯文本编码和静态复核。
