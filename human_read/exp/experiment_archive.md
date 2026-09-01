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

### Round1/Round2 current 文档转历史索引 — 2026-08-25

- 结构变化：`current_experiment.md`不再同时承载已批准Round1/Round2设计与Round3草案；当前文件只保留Round3 v0.9。
- 完整历史基线：Git commit `d338eb5bedef16d83a42790c3faa97f8f404315b` 的 `human_read/exp/current_experiment.md` 保存v0.6 SSPO-aligned 30k MVP完整设计和截至该commit的批准/失败记录。
- 归档摘要：本文件中的v0.1–v0.8、Round2 rollout addendum与v0.6条目继续提供设计演化索引，不因current清理而改写。
- 后续Round2执行事实：`../code/ROUND2_LIVE_HANDOFF.md`与`CODE_HANDOFF.md`继续保存依赖修复、TP兼容、smoke、正式长链、checkpoint保留和远程证据入口；这些属于Round2历史/执行面，不混入Round3 current设计。
- 科学内容变化：无。此条只改变文档导航；当前活动执行仍是Round2，Round3 v0.9仍未批准。

### `cycle-20260818-01` / Round3 resource-constrained unified validation / 设计版本 v1.0 — 2026-08-25

- 状态：锁定的预讨论草案；用户已明确批准把本版本新增配置内化，但未整体批准`r3-theory-v0.5`或`round3-exp-v1.0`，未解锁代码、上传或服务器执行；不覆盖当前Round2。
- 对应理论版本：Round3 `r3-theory-v0.5`。
- 相对v0.9的修订：六任务、数据、batch、epoch、250 steps、GitHub-loss SSPO、validation view和test均不变；冻结此前尚未明确的模型variant、优化、序列、PE lambda、rollout sampling、dynamic panel hash、eval tie/non-finite与SSPO eval-state验收。
- 模型与兼容性：ModelScope `Qwen/Qwen3-1.7B` post-trained Instruct/hybrid-thinking版本，命令`modelscope download --model Qwen/Qwen3-1.7B --local_dir ./dir`；不用`-Base`，不追加单文件参数。六方法共享resolved model/tokenizer/template manifest，使用native chat template且`enable_thinking=false`。Qwen3要求的兼容栈不能直接复用SSPO仓库固定的Transformers 4.46.1；实现只移植commit `2df9e9a`的loss/trainer语义。
- 统一优化：非量化LoRA r8/alpha16/dropout0，target为`q/k/v/o/gate/up/down_proj`且排除embedding/`lm_head`；AdamW torch、weight decay0、betas .9/.999、epsilon 1e-8、lr 1e-5、cosine、warmup .1、clip1；BF16 forward/autocast、FP32 LoRA trainables和optimizer states。六方法不做method-specific lr sweep。
- 序列合同：总长2048、prompt/completion各最多1024；prompt保留末端、fixed completion保留前端，response-only mask。动态rollout预先限制`max_new_tokens=min(1024,2048-prompt_tokens)`，EOS自然结束，cap-hit保留并显式标记；训练/validation汇总prompt/fixed-response截断、EOS、cap-hit和response-token统计。
- PE权重：static、SFT+rollout和rollout-only统一使用`(L_DPO+0.1*L_PE)/1.1`；不做lambda sweep。该0.1与SSPO decay .001无关。
- rollout sampling：两个dynamic方法及其train/checkpoint-validation均为Qwen3 non-thinking、sampling on、temperature .7、top-p .8、top-k20、min-p0、repetition penalty1、presence penalty0和Qwen resolved EOS；rollout-only两draw使用独立seed。train seed映射包含step/sample/draw，validation映射包含panel sample/draw且不含checkpoint；禁止跨方法/checkpoint复用文本。最终fixed-pair test不生成回复、没有temperature。
- dynamic panel：namespace `round3-dynamic-eval-panel-v1`、seed42；对role与sample ID的NUL分隔SHA-256排序，分别选4 labeled和28 unlabeled；两dynamic方法共享canonical ID manifest及SHA-256，但各checkpoint各自重新生成。
- checkpoint policy：只在finite eval losses中按原始`(eval_loss,step)`排序，完全并列选更早step。单checkpoint出现NaN/Inf即无效但继续；全无效则方法工程失败且不test；train loss/gradient非有限立即失败，禁止替换、静默跳过或用test补选。
- SSPO eval state：checkpoint显式保存running mean/var；eval从隔离副本开始并丢弃。代码交接前须通过pre/post state hash不变、重复eval、with/without-eval next-step分叉、checkpoint round-trip和missing-state fail-closed验收。
- 尚待解析：实际ModelScope revision/文件manifest、Round3 experiment ID、实现commit与精确dependency lock；这些不构成可事后调节的实验超参数。
- 实际实验目录或取消原因：尚未创建；理论与实验整体通过、代码实现/交接和执行授权完成后方可创建。
- 对应结果：尚无。

### `cycle-20260818-01` / Round3 resource-constrained unified validation / 设计版本 v0.9 — 2026-08-24

- 状态：锁定的预讨论草案，未获得`r3-theory-v0.4`整体理论通过、v0.9实验批准、代码交接或服务器执行授权；不覆盖当前Qwen3-4B Round2。
- 对应理论版本：Round3依据当前`../theory/current_theory.md`的未批准`r3-theory-v0.4`；已批准Round1/Round2 v0.2固定于Git commit `d338eb5bedef16d83a42790c3faa97f8f404315b`。
- 相对v0.8的修订：实际SSPO从论文v3 KDE/threshold-EMA profile改为作者GitHub commit `2df9e9a`的code-loss profile；保留本项目pair-derived singles并把随机合并sampler适配为固定1:7分层。六方法全部改为epoch=1，并以各自训练objective在validation上的同构loss选checkpoint；rollout方法采用固定单-batch轻量panel。
- 要回答的问题：有限标注DPO是否相对训练前模型增长；GitHub-loss SSPO、static PE和动态PE的相对行为；DPO-100 oracle gap；固定公开单回复锚点+rollout与rollout-only是否不同。
- 数据与隔离：同一冻结UltraFeedback-derived source split；1,000 labeled train、7,000 hidden-direction unlabeled train、200 validation source pairs、1,000 independent test pairs。DPO-100独占恢复7,000 hidden labels；其余方法禁止读取。SSPO以namespace `round3-sspo-single-v1`、seed42和sample ID从每个随机换位pair确定性选一条response，得到恰好7,000 singles。
- 六任务与顺序：`DPO-10 → SSPO-code-loss-stratified-pair-derived@2df9e9a → DPO+PE-static → DPO-100 → DPO+PE-SFT+rollout → DPO+PE-rollout-only`；前一任务的科学结果不作为后续强制gate。
- epoch、batch与step：所有方法epoch=1、250 optimizer steps。DPO-10为4 labeled pairs/step；SSPO为4 labeled pairs+28 unpaired singles；static PE和两个动态PE均为4 labeled pairs+28方法特定unlabeled pairs/prompts；DPO-100为32 labeled pairs。整数对齐为1000/4=7000/28=8000/32=250。`DPO-10`沿用旧ID，但本轮训练池实际标注率为12.5%，必须报告绝对数与真实比例。
- SSPO code-loss：length-normalized SimPO labeled risk使用`beta=10, margin=2`；按chosen/rejected/unpaired顺序更新同一running mean/variance，normalized values clamp到[-5,5]；threshold为当前normalized chosen最小值，无KDE和threshold EMA；unpaired使用threshold-centered、prior=.5的分支logistic risk；`gamma_t=max(.125, exp(-.001*t))`。固定4+28 sampler是稳定性适配，不冒充GitHub随机sampler。
- 动态训练：SFT+rollout使用已随机换位的公开`response_a`固定锚点，每prompt生成一条current-policy response，共7,000条；rollout-only每prompt独立生成两条，共14,000条。两方法不共享生成回复。
- checkpoint：step25/50/75/100/125/150/175/200/225/250保存，共60个adapter；step250与final为同一对象。每个checkpoint只用本方法的`eval_loss`选择，不横比不同objective的loss。
- validation：DPO-10在全部200 labeled pairs上按4计算；DPO-100在全部200上按32计算；SSPO/static PE使用固定25 labeled+175 unlabeled view，按4+28和最后1+7计算；两个动态方法共享从该pool一次选定的4 labeled+28 unlabeled source IDs，每checkpoint只评价一个logical batch，不重新抽样但由当前checkpoint重新生成回复。十次动态eval分别生成280与560条，为旧完整方案的1/6.25。
- eval完整性：SSPO evaluation从checkpoint running state建立隔离副本且结束后丢弃；dynamic evaluation冻结panel、采样配置与逐样本draw seeds。non-finite loss、state污染、重抽panel或跨checkpoint复用rollout均为实现失败。
- test与指标：每方法最低自身eval loss的一个checkpoint进入独立1,000-pair test；test直接评分冻结A/B pairs，不为动态方法再次生成rollout。跨方法只比较Accuracy、Brier、ECE、confidence分布等共同指标并记录score type。test不参与选择；单种子只作探索性趋势，不宣称显著性。
- 风险与解释：method-specific eval loss不可横比；动态32-source panel降低生成成本但增加checkpoint选择方差；SSPO与DPO-10同时改变labeled objective和unpaired机制，不能把整体差异归因于单一分支；逻辑暴露/更新次数对齐不代表token compute-matched。
- 资源与产物：训练rollout分别为7,000与14,000，动态validation额外为280与560；六任务使用独立目录和resolved config。预期回传无样本级聚合指标、配置、manifest/hash、图表与远程证据索引；原始数据/日志、样本级文本/预测、模型、adapter/checkpoint和cache留在执行面。
- 尚未解决：具体Qwen3-1.7B revision、experiment ID、LoRA/序列/optimizer/lr/精度、PE lambda、rollout sampling、dynamic panel namespace/hash、eval-loss tie-break、non-finite policy和SSPO eval-state隔离验收。
- 实际实验目录或取消原因：尚未创建；理论和实验明确通过、代码实现与交接完成并另获执行授权后，才可用新experiment ID创建。
- 对应结果：尚无。

### `cycle-20260818-01` / Round3 resource-constrained unified validation / 设计版本 v0.8 — 2026-08-24

- 状态：设计讨论中，尚未获得本版本实验批准、代码交接或服务器执行授权；不覆盖当前 Qwen3-4B Round2。
- 对应理论版本：`../theory/current_theory.md` v0.2（已通过）。
- 相对 v0.7 的修订：根据本地 `../../../相关工作/SSPO.pdf` 对应的 arXiv v3 及作者公开代码核验，将含混的 `SSPO` 臂重写为 `SSPO-pair-derived`；区分论文 v3 的 KDE/threshold-EMA 算法与早于 v3 的公开仓库 min-chosen-threshold 实现。
- SSPO 数据单位：从 7,000 个已随机换位的公开无标签 pair 中，以独立确定性 hash 每 pair 选恰好一个 response，生成 7,000 个 unpaired single responses；不使用 private label，不把 A/B 同时拆成 14,000 个样本，不冒充论文的独立 UltraChat/SFT 语料。
- SSPO 每步合同：4 labeled pairs + 28 unpaired responses；SimPO reward/labeled risk 固定 `beta=10, margin=2`；全局 minibatch reward mean/std EMA、Gaussian KDE 200-point Bayes-risk threshold、threshold EMA 均使用 `m=0.95`；prior `0.5`；论文未给 bandwidth，Round3 冻结 per-class Scott rule 为复现决定。
- SSPO 目标：hard pseudo-label 由 normalized reward 与 detached EMA threshold 比较得到，unlabeled risk 为 threshold-centered prior-weighted logistic BCE；`gamma_t=max(0.125, exp(-0.01*t))`，其中 `gamma_min=1000/(1000+7000)`。该 scheduler 是 SSPO 方法本身，不属于 PE lambda 消融。
- 计算与风险：SSPO 每步实际评分 8+28 个 responses，PE 联合臂评分 8+56 个 responses，逻辑暴露对齐但不 compute-matched。SSPO 当前步 KDE 每类仅 4 个样本，必须报告 bandwidth、Bayes risk、threshold、reward 分布与 pseudo-positive rate。
- 解释边界：`SSPO-pair-derived - DPO-10` 同时包含 SimPO-vs-DPO labeled objective 和 unpaired branch 的差异；无 SimPO-only matched arm 时不得将差异单独归因于 SSPO unpaired 信号。本臂只是 adapted SSPO，不是论文数据设置的原样复现。
- 完整性门禁：缺少每步 labeled/unpaired 固定组成、KDE/EMA 非有限、使用 min-chosen heuristic 或固定权重替代论文机制，均属于实现失败而非 SSPO 负面结果。
- 实际实验目录或取消原因：尚未创建；必须在设计通过、代码交接和服务器授权后使用新的 experiment ID 创建。
- 对应结果：尚无。

### `cycle-20260818-01` / Round3 resource-constrained unified validation / 设计版本 v0.7 — 2026-08-24

- 状态：设计讨论中，尚未获得本版本实验批准、代码交接或服务器执行授权；不覆盖当前 Qwen3-4B Round2。
- 对应理论版本：`../theory/current_theory.md` v0.2（已通过）。
- 版本变化：受服务器资源限制，新增 Qwen3-1.7B 单模型、单种子的六任务合并验证轮；静态 MVP 与 rollout 主任务在同一数据和步数合同下顺序执行，但不设置强制 gate。
- 要回答的问题：DPO-10 是否相对训练前模型增长；SSPO 与静态 PE 的相对表现；DPO-100 oracle gap；SFT+rollout 与 rollout-only 的差异。
- 变量、对照和数据：1,000 labeled train + 7,000 unlabeled train + 200 validation + 1,000 independent test；DPO-100 恢复 7,000 条 hidden labels，形成 8,000 fully labeled train pairs。
- 六个任务与顺序：`DPO-10 → SSPO → DPO+PE-static → DPO-100 → DPO+PE-SFT+rollout → DPO+PE-rollout-only`；前一任务失败不自动阻断后续任务。
- Batch 与步数：每任务 250 optimizer steps；DPO-10 每步4 labeled；SSPO/PE每步4 labeled +28 unlabeled；DPO-100每步32 labeled。整数对齐为1000/4=7000/28=8000/32=250。
- 动态采样：SFT+rollout 每步28 prompts各生成1条，共7,000条 rollout；rollout-only 每步构造28 pairs、生成56条回复，共7,000 pairs/14,000 responses。
- 超参数边界：seed42；暂不引入 lambda 变化或 scheduler；其余 LoRA、lr、序列、sampling、SSPO/PE resolved config 待本设计批准前补齐。
- checkpoint：每25 steps保存一次 adapter，即25/50/75/100/125/150/175/200/225/250-final；每任务10个、共60个。每个 checkpoint 做 validation，independent test 只评价各任务 final，共6次。
- 指标与解释边界：Acc、Brier、ECE、confidence/`p_i` 分布、责任质量、labeled/unlabeled loss、动态 rollout 长度/失败率/耗时。单种子只作探索性趋势，不宣称统计显著性；置信度变尖而 Acc 不升或 Brier 恶化不算有效。
- 资源预算与停止条件：六任务独立目录与摘要；不设科学结果 gate，但单任务工程失败必须如实记录。模型/数据/标签隔离/非有限 loss 等完整性故障仍使该任务失败，不能伪装为结果。
- 实际实验目录或取消原因：尚未创建；必须在设计通过、代码交接和服务器授权后使用新的 experiment ID 创建。
- 对应结果：尚无。

### `cycle-20260818-01` / Round2 rollout addendum / 设计版本 v0.6 — 2026-08-23

- 状态：数据与采样补充已获用户明确批准；仍处于 `CODE_IMPLEMENTATION`，尚未完成代码交接或服务器授权。
- 固定锚点：从第一轮 `mvp-v0.5-30k/unlabeled_train.jsonl` 的24,000条公开记录逐行取已随机换位的 `response_a`，生成独立的 `sample_id,prompt,response` JSONL；不读取 private label、不选择 chosen、不覆盖冻结源文件。
- 采样合同：两条 Round2 方法均使用 Qwen3 non-thinking `temperature=0.7`、`top_p=0.8`、`top_k=20`、`min_p=0`。
- 解释边界：该 response 是冻结无标签 pair 中的固定历史回复锚点，不是独立高质量 SFT 语料；核心比较解释为固定锚点+rollout 对 rollout-only。
- 实现要求：派生文件记录源/目标 SHA，重复执行只允许验证后复用；预检必须逐条证明 response 等于公开 response_a，采样四元组必须进入 resolved config 和每步队列证据。

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
- 实际实验目录或取消原因：首次实际运行前创建 `../../exp/exp-20260818-01-standard-mvp/`；服务器操作见 `../../../machine/A800-16/history/A800_standard_MVP_runbook.md`。
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
- 实际实验目录或取消原因：本地 `../../exp/exp-20260818-01-standard-mvp/` 只保存无样本级聚合摘要和远程索引；数据、环境、原始日志、逐样本输出、模型及 checkpoint 留在服务器。服务器操作见 `../../../machine/A800-16/history/A800_standard_MVP_runbook.md` v0.2。
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
- 2026-08-21 第三次 DAG 启动证据：tests、旧版256-length smoke和完整 reference cache成功；DPO array `1501048` 的两个 arm 均失败。`gn014` arm 在 `contribution.backward()` 以 logical microbatch=4、max length2048触发 CUDA OOM；`gn005` arm 在 DDP参数一致性 all-gather触发10分钟NCCL timeout，且该节点处于 draining。经用户明确选择，显存修复冻结为不改变global64/optimizer step/损失的 per-rank backward subbatch≤2，allocator使用 expandable segments，strong smoke升级为2×A800、bf16/2048最长样本压力测试并以同样的subbatch=2作硬门禁，同时默认排除 `gn005,gn021`。这仍是实现修复，不把失败训练当作实验结果。
- 2026-08-21 第四次 DAG 启动证据：commit `a4eef37` 的21项服务器测试全部通过；smoke job `1501724` 在模型加载和训练前，被过严的“每个 split 都必须出现2048截断”fixture门禁阻断，因为 labeled-validation 最长样本不足2048。修正后只要求实际参与有梯度计算的 labeled-train 与 unlabeled-train 达到2048；validation仍记录长度，但无梯度评价不要求人为达到上限。该失败没有产生训练结果。
- 2026-08-21 代码交接与执行授权：上述 smoke 门禁修复形成实现基线 `e047ce7`；用户确认离线 bundle 已同步成功，并明确表示希望挂载任务。当前阶段转入 `SERVER_EXECUTION`，须先归档失败 DAG，再从包含该实现基线与授权记录的 clean checkout 提交完整 Slurm DAG。
- 2026-08-22 节点故障与恢复边界：实现基线 `e047ce7` 的 tests、2×A800 strong smoke 和 reference-cache 先后成功；随后 DPO-10 与 DPO-100 在 `gn014` 上于同一时刻分别出现 CUDA driver initialization failure/no GPUs found 与 unspecified launch failure/NCCL watchdog。两条任务都在有效训练前失败且没有 checkpoint，节点随后自行回到 Slurm `IDLE`，因此归因为节点级瞬时 CUDA/driver 故障而非实验结果或已定位的训练代码缺陷。恢复只复用三个已完成门禁，从头重跑两条 DPO 及下游，并排除 `gn014`；共享账户取消操作必须严格限定于当前 registry，不能影响同事 job。
- 2026-08-22 恢复提交兼容修复：首次从 DPO 恢复时，提交器已通过 `sacct` 验证旧 tests/smoke/reference-cache 为 `COMPLETED`，但 Slurm controller 拒绝对旧 reference job 新建 `afterok` 依赖并返回 `Job dependency problem`；首个 `sbatch` 即失败，未创建任何新 job。恢复入口改为仍然 fail-closed 核验旧 gate 状态、代码差异和 cache 完整性，但首个 DPO 不再引用可能已超出 controller 保留窗口的旧 job ID；DPO 后续依赖关系不变。
- 2026-08-22 执行平台迁移：旧集群排队时间过长，用户明确提供新的无 Slurm 独占服务器并要求隔离脚本。平台适配基线 `e4eb95d` 新增 `scripts/standalone/`，只替换基目录、Python 环境、GPU 映射、后台进程与状态控制；训练 stage body、Python 入口、数据/模型 manifest、两 rank DDP、global batch、八条轨迹与评价顺序保持不变。旧 `cluster/` 作为历史 Slurm 适配保留。standalone 默认串行使用两张至少79000 MiB的卡，实际 SKU 必须进入硬件证据；新增适配仅做本地静态复核，环境/tests/strong smoke 尚待新服务器验证。
- 2026-08-22 设备档位适配：用户回到A800集群执行问题，并明确要求同一代码支持1/2/4卡、只改少量shell配置即可切换。实现基线`cf6bb99`把设备数限定为1/2/4，梯度累积分别解析为16/8/4；DPO始终为global64，joint始终为全局8 labeled +56 unlabeled，backward subbatch仍为2。Slurm入口新增`--formal-gpus`，standalone按`SOPPO_TRAIN_GPU_IDS`数量选择；smoke、registry和DPO恢复门禁绑定同一档位。三卡因无法精确整除冻结合同而拒绝。该变更不新增实验臂；不同档位的数值轨迹可能因分片/归约顺序不同而不逐bit一致，故一次实验的八条轨迹不得混用档位。当前仅完成本地静态检查，1/2/4卡服务器tests/strong smoke仍待验证。
- 对应结果：尚无；当前已获服务器执行授权，等待修复后 DAG 完成并返回白名单摘要或失败证据。

### `cycle-20260818-01` / Round3 data-source revision / 设计版本 v1.1 — 2026-08-25

- 状态：锁定的预讨论草案；数据合同重新打开，未批准、未解锁代码/上传/执行，不覆盖当前Round2。
- 对应理论版本：Round3 `r3-theory-v0.6`。
- 版本变化：补回v1.0正文缺失的DPO/PE精确定义；修正base/DPO-10 headroom；冻结GitHub running-stat初始化；目标执行面记录为3×RTX 4090。用户随后撤回旧统一UltraFeedback pair池，要求靠近SSPO官方数据。
- 官方数据事实：作者通用任务使用`HuggingFaceH4/ultrafeedback_binarized` paired preference与`HuggingFaceH4/ultrachat_200k/train_sft` unpaired single-response；10%示例约为6,113 paired（含validation）+20,786 unpaired。严格零UltraFeedback不是作者同一数据合同。
- 已撤回：冻结30k UltraFeedback后通过hash选1,000/7,000/200/1,000、从hidden pair选A/B single、`pair-derived` method ID、由同一8,000 pair恢复DPO-100、以及由此自动得到的六方法250-step合同。
- 当前resource-scaled候选：1,000 paired labeled +7,000 UltraChat singles +200 paired validation +1,000 paired test；SSPO候选ID为`SSPO-code-loss-stratified-ultrachat@2df9e9a`，4+28仍只是候选适配。
- DPO/PE合同：DPO为response总log-prob reference delta beta .1；PE为mean-response-logp margin-free beta10、L1、epsilon1e-8、denominator不断梯度、完整28-pair population，joint为`(L_DPO+.1L_PE)/1.1`。
- SSPO初始化：momentum .95，mean/var None，首次chosen直接初始化，population variance+1e-8，chosen→rejected→unpaired顺序EMA，clip5，first global_step 0；gamma_min由最终nL/nU解析。
- headroom：frozen base与selected DPO-10以共同raw mean-logp beta10辅助score比较，不参与选点；正式DPO test仍用reference delta。
- 资源预算：机器确定为当前3×RTX 4090服务器；GPU角色、环境、wall-time、磁盘与checkpoint keep尚未冻结，且不得干扰正在运行的Round2。
- 阻塞项：官方双源缩小版还是严格零UltraFeedback；paired source/revision；PE-static第二candidate；DPO-100 high-label source/规模/名称；随后重算batch/steps/checkpoints/gamma/panel。
- 实际实验目录或取消原因：尚未创建；设计与代码门禁未通过。
- 对应结果：尚无。

### `cycle-20260818-01` / Round3 evaluation-scope revision / 设计版本 v1.2 — 2026-08-25

- 状态：锁定的预讨论草案；评价口径局部决定已获用户确认，但完整理论/实验仍未批准，代码、上传和执行未解锁，不覆盖当前Round2。
- 对应理论版本：Round3 `r3-theory-v0.7`。
- 版本变化：训练objective与checkpoint selection分离；六方法用共同labeled DPO validation loss选点。独立1,000-pair fixed-pair test成为Round3唯一final evaluation；AlpacaEval 2.0与MT-Bench只登记为Round4候选，本轮禁止运行。
- validation/test：paired source候选改为1,000 validation +1,000 independent test。若使用`ultrafeedback_binarized/test_prefs`，先按`round3-paired-validation-v2`排序选validation，再排除并按`round3-paired-independent-test-v2`排序选test，保证无交集；严格零UltraFeedback方案必须冻结等价替代view。
- checkpoint selection：六方法对同一1,000 labeled pairs计算冻结reference、response总log-prob、beta .1的DPO NLL `eval_selection_loss`；只按原始`(eval_selection_loss,step)`选finite最小值并在完全并列时选更早step。SSPO/PE objective loss和dynamic panel只作方法内诊断，不参与选点或跨方法排名。
- Round3 final score：六个selected checkpoints与frozen base在同一1,000 pairs上同时计算`dpo_reference_delta_beta_0.1`与`raw_mean_logp_delta_beta_10`；每head分别报告Accuracy/NLL/Brier/ECE、confidence与坍缩诊断，只在相同head内比较，不合成综合分数，test不参与重选。
- Round4登记：候选为AlpacaEval 2.0 length-controlled win rate与MT-Bench平均分。Round3不生成其回答、不安装/运行judge流程、不调用外部API、不运行本地替代judge，生成量与API预算为0；Round4另行冻结版本、模型集合、generation、judge snapshot、费用、缓存/重试、失败与可比性合同。
- 资源与产物：Round3预期产物只含共同selection摘要、可选机制诊断、best pointers及独立1,000-pair双head聚合；不含AlpacaEval/MT-Bench generations、annotations、judge logs或分数。
- 尚待解析：v1.1中的paired source、PE-static、DPO-100、训练规模/steps/checkpoints和3×4090布局仍未解决；Round4生成式评测细节不阻塞Round3。
- 实际实验目录或取消原因：尚未创建；设计与代码门禁未通过。
- 对应结果：尚无。

### `cycle-20260818-01` / Round3 dual-source five-method revision / 设计版本 v1.3 — 2026-08-25

- 状态：锁定的预讨论草案；数据/方法/资源分支已闭合，但理论与实验仍未依次整体批准，代码、上传和执行未解锁，不覆盖当前Round2。
- 对应理论：Round3 `r3-theory-v0.8`。
- 五方法：DPO-1K、`SSPO-code-loss-stratified-ultrachat@2df9e9a`、DPO-8K、DPO+PE-SFT+rollout、DPO+PE-rollout-only。PE-static移到Round5消融候选，Round3不实现或运行。
- 数据合同：8K `ultrafeedback_binarized/train_prefs` paired master，其排序前1K为共同limited-label view；7K `ultrachat_200k/train_sft` singles；`test_prefs`先选1K validation、再从剩余选1K independent test。全部使用独立namespace/seed42/SHA-256排序，dataset full revisions和源parquet hashes在获批服务器preflight解析。
- 训练合同：五方法均1 epoch/250 steps。DPO-1K为4 pairs/step，SSPO与两个动态PE为4+28/step，DPO-8K为32 pairs/step；SSPO保持GitHub初始设置、decay .001与$t=0$，$\gamma_{min}=.125$。
- selection/final：每25步保存并在共同1K labeled validation上计算reference-DPO beta .1 selection loss；不运行checkpoint级SSPO/PE objective diagnostics或dynamic rollout panel。每方法只best checkpoint与frozen base进入独立1K fixed-pair test，同时报告reference-delta/raw mean-logp双head Accuracy/NLL/Brier/ECE-15。
- 3×4090：GPU0单进程训练全部五方法；动态PE时GPU1/2分别运行vLLM replica，以`method_id/optimizer_step/adapter_sha256` ACK屏障防止stale policy。train/rollout使用仓库外独立环境。
- 存储：五方法各10个durable checkpoints全部保留，不启用自动pruner。formal前以production-path strong smoke实测投影peak storage，只有`free_bytes >= 2 * projected_peak_bytes`才能继续；不删除Round2或其他已有产物。
- 延后项：AlpacaEval/MT-Bench只登记到Round4，PE-static只登记到Round5；两者的当前生成量、API调用和执行预算均为0。
- 实际实验目录或取消原因：尚未创建；理论/实验门禁未通过。
- 对应结果：尚无。

### Round3 v1.3 内容批准状态更新 — 2026-08-25

- 用户已明确整体通过`r3-theory-v0.8`与`round3-exp-v1.3`的内容。
- 当前唯一活动阶段仍为Round2 `SERVER_EXECUTION`；在Round2完成`RESULT_HANDOFF`并依流程激活Round3前，本批准不解锁Round3代码实现、上传或执行。

### Round3 v1.3 阶段激活状态更新 — 2026-08-26

- 用户明确要求忽略Round2等待并直接开始Round3；Round2以无新增实时证据的行政性`NO_CONCLUSION`完成结果交接，潜在服务器任务保持不动。
- `round3-exp-v1.3`进入`CODE_IMPLEMENTATION`；仅本地静态实现已解锁，实际Round3 experiment目录仍未创建。
- 代码交接、commit/push、服务器checkout变更、上传、tests、strong smoke和formal execution仍锁定，须在实现完成后另行审阅批准。

### Round3 v1.3 checkpoint round-trip验收补全 — 2026-08-26

- 补回此前只保留“预注册数值容差”占位而缺失的具体门禁：SSPO同一checkpoint两次独立重载的running state、scheduler/global step精确一致；下一batch loss绝对差`<=1e-7`；trainable LoRA更新最大绝对差`<=1e-7`且最大相对差`<=1e-6`。
- 该实现验收不改变五方法、数据、训练objective、selection或final-test合同；任一超限会使strong smoke失败并阻断formal，不能在服务器上临时放宽。

### Round3 `round3-code-candidate-v0.2` 本地静态实现 — 2026-08-26

- 状态：`CODE_IMPLEMENTATION`候选已写完并完成本地允许的静态复核；尚未获用户代码交接确认，未commit/push、未上传、未执行服务器tests/smoke/formal。
- 覆盖：双源full-SHA数据与manifest、Qwen3-1.7B非量化LoRA、DPO-1K/GitHub-loss SSPO/DPO-8K/两个双vLLM动态PE、完整训练态checkpoint、共同1K selection、独立1K双head final test与sample-free聚合。
- 本地证据：全部Round3 shell通过`bash -n`，`git diff --check`通过，可执行位与路径/旧接口/禁止项静态搜索通过；按工作区门禁没有运行Python import/compile、pytest、数据、模型或GPU任务。
- 服务器待证：候选依赖组合、Git ref解析与下载API、数据schema/数量、Qwen特殊token、vLLM per-request SamplingParams/LoRA卸载、实际显存、SSPO真实next-batch round-trip与两倍空间门禁。
- Round2边界：没有获取或声称新的实时状态，没有停止任务、修改服务器checkout或删除checkpoint。

### `cycle-20260818-01` / Round3方案B数据勘误 / 设计版本 v1.4 — 2026-08-26

- 状态：用户明确批准方案B及本地修改；对应理论`r3-theory-v0.9`，唯一活动阶段保持`CODE_IMPLEMENTATION`。
- 服务器事实：冻结UltraFeedback revision的`test_prefs`为2,000 source rows，其中indices 374/595/1846的rejected为空，只有1,997个有效pair；原v1.3的1K+1K构造正确地fail closed。
- 修订：保持SHA namespace选出的1,000 validation；剩余997个有效pair全部进入independent test并精确断言，不动态接受其他N、不从train补3条。所有模型仍在同一test上计算两个预注册score heads并报告`N=997`。
- source审计：冻结`train_prefs`/`test_prefs`/`train_sft`的source、valid、malformed与reason aggregates；畸形行选择前隔离，server-only audit记录source index/ID/reason而不保存重复原始文本。view source manifest固定17,997行，malformed audit固定12,197行。
- 路径：新合同使用仓库外`dual_source_v2`与对应v2 reference cache；失败遗留空`dual_source_v1`保留，不删除、不覆盖。
- 不变项：五方法、训练数据与steps、共同1K validation selection、checkpoint、SSPO/PE定义、GPU角色、评价head、Round4/5延期和两倍磁盘门禁均不变。
- 执行边界：服务器当时只有环境/revision/model证据，data v1为空、strong smoke未开始；服务器GLM留下的单文件dirty补丁不构成实现。修订必须从本地完成并重新交接，未经批准不commit/push或上传执行。

### `cycle-20260818-01` / Round3资源波次与token入口修复 / 设计版本 v1.5 — 2026-08-26

- 状态：用户明确批准；对应理论`r3-theory-v1.0`，唯一活动阶段为`SERVER_EXECUTION`。
- 调度：DPO-1K/GPU0、SSPO/GPU1、DPO-8K/GPU2作为第一波独立单卡任务并发；DPO+PE-SFT+rollout与DPO+PE-rollout-only随后分别以GPU0训练、GPU1/2双vLLM串行独占三卡。
- 等价性：只改变无依赖子实验的资源调度，不改变任一方法的数据、loss、logical batch、seed、optimizer、250 steps、checkpoint selection或final test。
- 新门禁：三卡分别完成确定性checkpoint重放；strong smoke按formal并发拓扑运行；共享输入只读、输出目录隔离；动态方法仍要求双ACK完整population且禁止跨方法复用生成。
- 失败证据：`round3-20260826-03`前三个静态smoke通过，首个动态方法因vLLM文本tokenizer默认special-token行为与训练`add_special_tokens=False`合同不一致而停止；修复必须显式预tokenize、左截断并传递token IDs。
- 授权：用户要求Codex持续修改和服务器测试，全部smoke、checkpoint与两倍空间门禁通过后直接启动formal；任何新失败仍fail closed且保留独立attempt。

### `cycle-20260818-01` / Round3 PE reward extension / 设计版本 v1.6 — 2026-08-27

- 状态：用户明确批准理论方向与实验构成修订；对应`r3-theory-v1.1`。本地进入`CODE_IMPLEMENTATION`，服务器部署仍须完成新code handoff。
- 七方法：保留DPO-1K、GitHub-loss SSPO、DPO-8K及旧两个动态方法；旧动态方法回溯命名为`DPO+PE-SimPO-reward-{SFT+rollout,rollout-only}`，新增`DPO+PE-DPO-reward-{SFT+rollout,rollout-only}`。
- 唯一新增科学变量：SimPO profile使用raw mean-response-logp、beta10；DPO profile恢复原始total-response-logp reference ratio、beta.1。四动态方法的1K labeled/7K UltraChat、4+28、250 steps、rollout sampling、lambda.1、checkpoint/selection/final均一致。
- 执行拆分：`round3-20260826-04`及其commit不可改写；它完成后，新两臂在独立extension experiment和commit中顺序独占GPU0 train +GPU1/2 vLLM。跨运行aggregate须验证相同model/data/reference/test manifests、997 sample/label顺序、score heads与final evaluator依赖源码字节一致。
- 资源：旧formal保留50个checkpoints并生成21K rollouts；extension新增20个checkpoints并生成21K rollouts。extension重新做两臂strong smoke与增量存储投影，门禁失败时不自动删除缓存、Round2或旧Round3证据。
- 延期：Round4 AlpacaEval/MT-Bench不启动；Round5 PE-static仍只登记。

### Round4三方法、容器迁移与生成式评价 / `round4-exp-draft-v0.1` — 2026-09-01

- 状态：`LOCKED_PRE_DISCUSSION_DRAFT`；cycle ID与experiment ID均为`TBD`。当前Round3尚未完成结果交接和下一轮规划，本草案未获理论或实验批准，不解锁代码实现、commit/push、镜像构建、服务器smoke或formal。
- 方法解释：暂按“共三种方法”登记为`DPO-label-only`、`SSPO`、`StaticPE`，不是三个SSPO变体。为了形成主要只差无标签机制的对照，候选建议SSPO使用DPO-base；若选择作者默认SimPO-base，必须将labeled objective差异列为额外实验变量。
- 共同模型：`Qwen/Qwen3-1.7B`同一不可变revision，native non-thinking `qwen3`模板，LoRA r8/all linear、BF16、lr `1e-5`、DPO beta `0.1`、cutoff 1024、epoch 1与seed 42作为候选共同设置；精确依赖lock和模型文件manifest由获批服务器preflight解析。
- 共同数据来源：UltraFeedback Binarized `0.1`提供共享labeled view；UltraChat `0.1`提供SSPO/StaticPE共享unlabeled view。DPO只读取完全相同的labeled IDs，不得把unlabeled空行送入DPO trainer。StaticPE的A是UltraChat原始回答，B是冻结初始化Qwen3的一次性贪心非思考生成；A/B稳定hash换位，空生成和完全重复候选fail closed或进入预注册隔离计数。
- 目标：DPO为labeled DPO mean；StaticPE为`(L_DPO+0.1L_PE)/1.1`；SSPO保留其labeled/unlabeled/gamma三项日志。三方法都必须保存按optimizer step的分项loss、eval loss、adapter与resolved config。
- formal batch决定：用户于2026-09-01选择DPO effective batch16。SSPO和StaticPE为每设备4、2 GPU、梯度累积8，名义effective batch64行；DPO为每设备1、2 GPU、梯度累积8，effective batch16。三方法都保持epoch1，不重复或丢弃labeled数据来强凑相同步数。按预期6,113 labeled、20,785 unlabeled和`val_size=0.1`粗算，joint约379 steps、DPO约344 steps；冻结实际行数后记录精确值和相对误差。
- eval：三方法共享同一held-out labeled eval view及相同batch/order；最终生成式候选为完整805条AlpacaEval 2.0，报告LC和普通win rate。是否同时评价frozen base、是否纳入MT-Bench、Alpaca evaluator从当前原型的`0.6.2`升级到官方较新的release、judge实际model snapshot与费用上限均未批准。
- 4090-3 production-path smoke：三方法分别顺序运行，训练恰好2个optimizer steps；24GB专用physical micro-batch允许小于formal值，但不得改变loss定义。StaticPE和SSPO fixture必须确定性保证两个step合计同时覆盖labeled与unlabeled分支，不能因batch过小只测到PE/unlabeled；随后执行至少一个eval batch、adapter保存、LoRA merge、merged model离线重载、少量固定Alpaca指令生成和一次真实judge请求。所有输出使用独立smoke experiment ID，禁止写入formal目录。
- 镜像与FusionOne候选：exact code commit和dependency lock先完成用户交接；镜像不包含数据、模型、API key或服务器凭据，使用不可变非`latest` tag并记录digest。只有4090-3实时验证Docker权限及仓库外scratch空间充足时才允许在该机build；否则使用另行授权的可联网制作机。FusionOne录入后先验证实际GPU SKU/显存、driver/CUDA、GPU可见性、共享内存/IPC、用户与挂载权限，再运行无数据import和production-path smoke。
- 完整流程门禁：`code review → explicit commit/push approval → exact-commit pull → environment/image tests → 4090 smoke → image digest → FusionOne import → container preflight → A100/A800 smoke → formal`。任何失败保留摘要并返回相应上游阶段；smoke通过不允许跳过formal前数据/model/hash和存储门禁。
- 预期formal产物：三份resolved config、共同model/data manifests、每方法分项loss曲线、eval摘要、merged-model manifest、805条生成的server-only逐样本文件、Alpaca annotations/judge raw日志server-only、LC/win-rate无样本聚合、环境/镜像digest与远程证据索引。
- 回传白名单：配置、版本、hash、无样本聚合Markdown/JSON/CSV、汇总图和远程路径索引。数据、模型、adapter/merged checkpoint、逐样本生成、annotations、原始日志、API信息与平台凭据全部留在服务器执行面。
- 当前阻塞决策：SSPO base；formal三方法并发方式；4090 Docker/scratch；Alpaca evaluator/judge/version/预算；frozen base与MT-Bench是否纳入。用户已决定DPO effective batch16，并明确要求行政关闭Round3、直接切换Round4；FusionOne 8×A100由用户确认已实机验证。

### `cycle-20260901-01` / Round4两卡A100顺序执行修订 / `round4-exp-draft-v0.2` — 2026-09-01

- 状态：`LOCKED_DRAFT_DURING_THEORY_DISCUSSION`；对应`r4-theory-v0.2`尚未获批，不解锁任何实现或执行。
- 不变项：DPO/SSPO/StaticPE三方法、Qwen3-1.7B、双源各0.1、epoch1、DPO effective16、SSPO/StaticPE effective64、StaticPE lambda0.1与AlpacaEval 2.0目标不变。
- 资源修订：用户先占用2张A100；三方法共享同一GPU对顺序执行，删除三组2卡/共6卡并发候选。
- smoke修订：删除4090-3训练smoke；所有2-step train/eval/merge/reload/Alpaca API smoke移到目标2×A100容器。
- 准备链：4090-3构建非`latest`无凭据镜像并配置不自动训练的安全启动入口；下载冻结数据后经SSH直传A100仓库外目录，目标端SHA完全一致才可进入schema/data preflight。
- 前置阻塞：4090-3历史证据仍显示无构建工具、scratch不足；A100侧还需明确模型来自已验证只读挂载还是独立校验传输。
