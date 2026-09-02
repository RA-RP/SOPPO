# 理论变化记录

本文件按时间顺序追加所有理论变化。不要只写“更新了理论”，而要保留变化内容、原因和证据链。

## 记录模板

### <cycle ID> — <日期> — <变化摘要>

- 上一理论版本：
- 新理论版本：`current_theory.md` 或对应归档说明
- 保留的内容：
- 修改/删除的内容：
- 新增的内容：
- 变化原因：
- 证据来源：result/experiment ID、文献或原始想法位置
- 对实验设计的影响：
- 尚未解决：

## 变更记录

### `cycle-20260818-01` — 2026-08-18 — 建立初始可检验理论

- 上一理论版本：目录初始化占位稿，尚无正式理论。
- 新理论版本：`current_theory.md` v0.1。
- 保留的内容：有限真实偏好监督、SFT 数据利用、DPO 隐式偏好得分、方向未知候选对、label-encoding 风险和动态联合优化的原始主线。
- 修改/删除的内容：不再把 on-policy 与 SFT rollout 当作第一轮实验的必要条件；将“LERM 本身作为先验”的宽泛表述收紧为“用群体级 label-encoding 结构监督替代 instance-level pseudo target”。
- 新增的内容：静态无标签偏好对 MVP、四组核心对照、Acc/Brier 指标、分阶段推进条件、四条可证伪假设、标签泄漏约束，以及结构坍缩、校准、梯度路径和数值稳定性等风险。
- 变化原因：先隔离并验证最核心的新损失机制，避免首次实验同时引入在线生成、SFT 数据来源和动态分布变化而无法归因。
- 证据来源：`../../../idea/基于有限偏好监督的动态偏好优化方法.docx`；用户提供的 LERM/MVP 重定位附件，其工作副本位于 `../../../idea/理论与MVP工作草稿.md`。本条不包含实验结果。
- 对实验设计的影响：首轮应使用一个完整标注的 preference dataset 人为构造 $D_L/D_U/D_{\mathrm{test}}$，优先比较 DPO-10%、Pseudo-target、DPO + PE 和 DPO-100%；只有结构方法满足预先约定的有效性条件后，才进入 SFT + online rollout 阶段。
- 尚未解决：相关工作与 novelty 的系统核验、`C_{\gamma}` 的定义和角色、$L_{\mathrm{PE}}$ 的梯度路径与稳定实现、具体数据/模型/预算及统计判据。

### `cycle-20260818-01` / Round3 — 2026-08-24 — 建立 SSPO 论文/源码双 profile 理论草案

- 上一理论版本：`current_theory.md` v0.2 已通过，继续作为当前 Round1/Round2 理论；Round3 尚无独立 SSPO 理论记录。
- 新理论版本：`current_theory.md` 的锁定附录 `r3-theory-v0.1`，状态为讨论稿、未通过。
- 保留的内容：有限偏好监督问题、用户此前确定的 Round3 Qwen3-1.7B 与固定数据 split；不修改正在执行的 Round2。
- 修改/删除的内容：否定“SSPO 论文与公开源码是同一个实现”的含混表述；否定“4 labeled + 28 unpaired 是论文或源码 batch”的旧解释。Round3 SSPO 不再使用该单遍混合解释。
- 新增的内容：分别定义 `SSPO-paper-v3` 与 `SSPO-code-main@2df9e9a` 的数据单位、reward、$R_{D_L}$、threshold、$R_{D_U}$、normalization、scheduler 和 batch 行为；明确论文 v3 使用 KDE/Bayes threshold/EMA，而公开源码使用 min-chosen threshold 且随机混合数据池。
- 变化原因：用户明确要求论文和源码内容均须记录，并要求保持此前模型/数据合同，同时按各自方式处理 SSPO 数据混合。
- 证据来源：本地 `../../../相关工作/SSPO.pdf`（arXiv:2511.00040v3）；作者仓库 commit `2df9e9a1d5fb9202a583cb66eb081e0cb60e873d` 的 `preprocessing_ultrachat.py`、`collator.py`、`trainer.py` 与训练配置。本条不包含本项目实验结果。
- 对实验设计的影响：未来 Round3 候选设计应保持同一个 Qwen3-1.7B、同一 1,000 labeled/7,000 pair-derived single-response 数据；paper profile 使用 two-stream 8+56，code profile 使用 8,000-record 合并池随机 batch，二者均按 total batch 64、2 epochs 对齐为 250 steps。若只运行其中一个 profile，必须由用户在实验设计阶段明确选择。
- 尚未解决：paper-v3 KDE bandwidth、decay、EMA 初始化与数值 fallback；分布式全局统计；两个 profile 是否都进入实际 Round3 任务清单。
- 门禁：用户只授权写入 Round3 theory 文档，未明确通过 `r3-theory-v0.1`；Round3 实验、代码与服务器执行仍锁定，当前 Round2 `SERVER_EXECUTION` 不变。

### `cycle-20260818-01` / Round3 — 2026-08-24 — 选择 GitHub code-main 作为实际 SSPO

- 上一理论版本：Round3 `r3-theory-v0.1` 同时记录 paper-v3 与 code-main，并把是否实际运行两个 profile 留作未决项。
- 新理论版本：Round3 `r3-theory-v0.2`，讨论中、未整体通过。
- 保留的内容：paper-v3 的公式/KDE/EMA 作为独立文献事实完整保留；Qwen3-1.7B 与 1,000/7,000/200/1,000 数据合同不变。
- 修改/删除的内容：不再计划将 paper-v3 作为第七个训练任务；实际 SSPO 唯一命名为 `SSPO-code-main-pair-derived@2df9e9a`。六类实验继续对应六个训练任务。
- 新增的内容：明确源码 physical microbatch 与 effective batch 64 的区别；在 1:7、per-device batch 4 下 $E[K]=0.5,P(K=0)\approx0.586$，no-labeled fallback 是常见源码路径。新增 labeled-only validation 会使源码 joint `eval_loss` 因 empty unlabeled mean 变为 NaN 的边界。
- 变化原因：用户明确表示实际比较采用作者 GitHub 上的 SSPO 实现更合适。
- 证据来源：作者仓库 commit `2df9e9a1d5fb9202a583cb66eb081e0cb60e873d` 的训练配置、`trainer.py`、数据 split 与 callback；本条不包含实验结果。
- 对实验设计的影响：SSPO train 使用 8,000-record 合并池随机 shuffle、2 GPUs×per-device 4×GA8、2 epochs/250 optimizer steps；不强制每步 8+56。checkpoint 选择使用项目共同 200-pair evaluator，不使用源码 mixed-data joint `eval_loss`，也不让 eval 改写 running statistics。
- 尚未解决：服务器 2-rank 布局、empty-unpaired/non-finite failure policy、共同 pairwise NLL scale、LoRA/lr/precision 与 checkpoint tie-break。
- 门禁：本条记录了用户的 SSPO profile 单项选择，不构成对完整 Round3 theory/experiment/code/server 的批准；Round2 `SERVER_EXECUTION` 不变。

### `cycle-20260818-01` / Round3 — 2026-08-24 — 冻结一轮暴露、比例分层与方法内 eval

- 上一理论版本：Round3 `r3-theory-v0.2` 计划照源码随机合并数据、2 epochs，并以独立共同 evaluator 替代 joint `eval_loss`。
- 新理论版本：Round3 `r3-theory-v0.3`，讨论中、未整体通过。
- 保留的内容：实际 SSPO 使用 GitHub commit `2df9e9a` 的 SimPO、normalization、min-chosen threshold、unpaired loss 与 scheduler；paper-v3 只记录不运行；模型和 1,000/7,000/200/1,000 数据不变。
- 修改/删除的内容：删除具体训练机器/并行布局讨论；不再照搬源码随机 sampler、effective batch 64 和 2 epochs；不再用跨方法统一 evaluator loss 选 checkpoint。
- 新增的内容：六方法统一 epoch=1、250 optimizer steps；联合方法固定 4 labeled +28 unlabeled 的 1:7 stratified batch。SSPO 正式命名为 `SSPO-code-loss-stratified-pair-derived@2df9e9a`，明确 sampler 是稳定性适配而非 GitHub 原样复现。新增逐方法 train view、logical batch、每步 response 数和一轮生成量总表，并记录 `DPO-10` 在本轮8,000-pair train pool 中实际对应12.5%标注率的命名边界。
- eval 决定：每种方法用与自身训练 objective 相同、但来源为 validation 的 `eval_loss` 选择 checkpoint；联合方法把200个 validation source pairs 固定为25 labeled +175 unlabeled view。只有各自 selected checkpoint 进入独立1,000-pair test，跨方法比较 Accuracy/Brier/ECE 等共同指标，不横比 `eval_loss` 数值。
- 变化原因：用户要求 method-aligned eval、公平的一轮数据暴露、统一 epoch=1，并选择按比例分层以提高 GitHub loss 在小 batch 下的稳定性。
- 对实验设计的影响：六个任务均为250 steps；DPO-10为4 labeled/step，SSPO与三个PE联合任务为4+28/step，DPO-100为32 labeled/step。validation 使用对应方法的数据 view 与 batch composition，test 只评价selected checkpoint。
- 尚未解决：SSPO eval running-state 隔离、动态 eval sampling、non-finite failure policy、eval-loss tie-break 与其余 resolved optimizer config。
- 门禁：这些是 Round3 theory 单项决定，不构成完整理论/实验/代码/服务器批准；Round2 `SERVER_EXECUTION` 不变。

### `cycle-20260818-01` / Round3 — 2026-08-24 — 将动态 validation 收紧为固定单 batch

- 上一理论版本：Round3 `r3-theory-v0.3` 要求六方法每个 checkpoint 都使用完整200-source validation；动态方法因此每轮需要175或350条 rollout。
- 新理论版本：Round3 `r3-theory-v0.4`，讨论中、未整体通过。
- 保留的内容：method-aligned `eval_loss`、十个 checkpoint、200-source 独立 validation pool、selected-checkpoint-only independent test 与禁止跨方法比较 `eval_loss` 均不变。
- 修改/删除的内容：撤销两个 rollout 方法每个 checkpoint 遍历完整25 labeled +175 unlabeled validation view 的要求。
- 新增的内容：两个动态方法从200-source pool中一次性冻结同一个4 labeled +28 unlabeled轻量 panel，每个 checkpoint 只评价这一 logical batch；IDs不重抽，但候选必须由当前 checkpoint 重新生成。十次 validation 分别只生成280与560条回复，为旧设计的1/6.25。无 online generation 的 DPO、SSPO和static PE仍使用各自完整200-source view。
- 变化原因：用户明确否定完整动态 validation 的额外生成成本，并确认固定单 batch 的轻量方案更合理。
- 对实验设计的影响：Round3 experiment 草案需要同步 checkpoint selection 数据视图、生成预算与评价产物；动态方法的 checkpoint 选择只代表该固定32-source panel，必须记录较小 panel 带来的选择方差风险。
- 尚未解决：动态 rollout 具体采样超参、`eval_loss` tie-break、SSPO eval state 隔离验收和其余 resolved optimizer config。
- 门禁：用户确认的是轻量 eval 单项决定，并表示讨论接近完成、要求同步 theory/exp；尚未明确说“通过 `r3-theory-v0.4`/实验设计”，因此不推进阶段、不实现代码、不上传或执行。

### 文档结构整理 — 2026-08-25 — `current_theory.md` 只保留 Round3 主线

- 理论内容变化：无；Round3 仍为`r3-theory-v0.4`，所有已确认与未决项不变。
- 结构变化：从`current_theory.md`移除已通过且服务于Round1/Round2的v0.2正文，重新编号并只保留当前Round3理论。旧理论没有被否定，只是不再与当前主线混排。
- 历史定位：v0.2完整正文可由Git commit `d338eb5bedef16d83a42790c3faa97f8f404315b` 的 `human_read/theory/current_theory.md` 恢复；其形成、批准与后续变化继续由本文件前述记录索引。
- 原因：用户明确要求`current_*`以当前讨论为主线，任何与当前版本冲突的旧内容只出现在历史记录中。
- 门禁：纯文档重组，不构成`r3-theory-v0.4`整体批准，不改变当前Round2执行状态，也不解锁Round3实验、代码或执行。

### `cycle-20260818-01` / Round3 — 2026-08-25 — 冻结Qwen3 Instruct、2048序列与复现完整性合同

- 上一理论版本：Round3 `r3-theory-v0.4`已冻结六方法、GitHub-loss SSPO、1-epoch比例分层与轻量dynamic validation，但模型variant、LoRA/optimizer/precision、序列、PE lambda、rollout sampling与评价完整性策略仍未明确。
- 新理论版本：`r3-theory-v0.5`，新增配置已获用户明确批准内化；理论版本仍未获整体通过。
- 保留的内容：六任务、1,000/7,000/200/1,000 split、4+28/32 batch、250 steps、GitHub commit `2df9e9a` SSPO code-loss、method-aligned eval与selected-checkpoint fixed-pair test均不变。
- 修改/删除的内容：模型从含混的`Qwen3-1.7B`收紧为ModelScope `Qwen/Qwen3-1.7B` post-trained Instruct/hybrid-thinking版本，明确不使用`-Base`；序列上限从未决改为2048而非照搬SSPO/Zephyr的1024；删除所有尚未冻结超参数占位。
- 新增的内容：完整ModelScope下载命令；native Qwen3 non-thinking template；非量化LoRA r8/alpha16/dropout0与七类Qwen linear targets；统一AdamW/lr $10^{-5}$/cosine/warmup/clip/BF16 mixed precision；总长2048、prompt/completion各1024及截断审计；三个PE方法normalized $\lambda_{PE}=0.1$；dynamic train/eval统一temperature .7/top-p .8/top-k20/min-p0；dynamic panel namespace/hash；finite-only eval selection与更早step tie-break；SSPO running-state序列化和隔离验收。
- 变化原因：用户改选Instruct模型，认为2048对UltraFeedback更稳健，并确认其余调研建议；随后明确批准把讨论结果内化到对应文档。
- 证据来源：ModelScope/Hugging Face Qwen3-1.7B模型说明与配置；UltraFeedback Binarized数据卡；Zephyr DPO recipe；SSPO论文v3及作者仓库commit `2df9e9a`的config/trainer/requirements。本条不包含Round3实验结果。
- 对实验设计的影响：Round3 experiment草案推进到v1.0；两种dynamic PE的训练和validation使用同一采样分布，最终test仍直接评分固定A/B且没有temperature；SSPO旧依赖栈不得原样用于Qwen3，只移植并验收其loss/trainer语义。
- 尚待解析：实际下载revision/文件manifest、Round3 experiment ID、实现后的精确dependency lock/commit；这些是执行前manifest而非可自由调节的科学超参数。
- 门禁：用户批准的是新增内容内化，不等于明确说“整体通过`r3-theory-v0.5`/`round3-exp-v1.0`”；Round3代码和执行仍锁定，当前Round2 `SERVER_EXECUTION`不变。

### `cycle-20260818-01` / Round3 — 2026-08-25 — 补回DPO/PE并因SSPO数据源重开设计

- 上一理论版本：Round3 `r3-theory-v0.5`。
- 新理论版本：`r3-theory-v0.6`，讨论中、未整体通过；对应实验草案升级为`round3-exp-v1.1`。
- 保留的内容：Qwen3-1.7B Instruct、non-thinking、非量化LoRA、统一lr 1e-5与BF16/2048；GitHub code-loss而非paper-v3；normalized PE lambda .1；method-aligned eval、selected-checkpoint fixed-pair test和当前Round2执行边界。
- 补回的内容：从历史v0.2恢复并收紧DPO与PE数学合同。DPO固定response总log-prob、冻结初始化reference和beta .1；PE固定mean-response-logp、margin-free beta10、L1、epsilon 1e-8、denominator不detach和完整28-pair logical-population精确梯度。明确PE labeled branch使用DPO而不是SimPO。
- GitHub初始化确认：momentum .95，running mean/var初始为None，首次chosen batch直接初始化，variance使用`unbiased=False`并加1e-8，随后chosen→rejected→unpaired顺序EMA，clamp[-5,5]；scheduler首步global_step为0。
- headroom修正：训练前policy等于reference时DPO reference-delta恒零，因此base/DPO-10只通过不参与选点的共同raw mean-logp beta10辅助score比较；正式DPO test继续使用reference-delta beta .1。
- 数据变化：用户先确认复用旧冻结30k的hash子集，随后明确后悔并希望放弃UltraFeedback、使用SSPO相同数据；后者覆盖前者。官方核验显示SSPO通用任务实际为`ultrafeedback_binarized` paired +`ultrachat_200k/train_sft` unpaired，而非零UltraFeedback。旧pair-derived SSPO ID与统一hidden-pair合同撤回。
- 实验影响：候选SSPO ID改为`SSPO-code-loss-stratified-ultrachat@2df9e9a`；UltraChat single自然支持SSPO和dynamic anchor，但不能直接提供PE-static第二candidate或DPO-100 hidden-label oracle，因此两臂、数据规模、steps/checkpoints和gamma_min重新打开。
- 服务器：用户指定Round3目标执行面为当前3×RTX 4090服务器；GPU角色、rollout replica、环境、磁盘和checkpoint策略仍须设计批准与服务器smoke，不影响当前Round2。
- 证据来源：SSPO作者仓库commit `2df9e9a`的`preprocessing_ultrachat.py`、`make_yaml.py`与`trainer.py`；SSPO论文v3 §5.1。本条不包含Round3实验结果。
- 尚未解决：官方双源缩小版还是严格零UltraFeedback；paired替代源；PE-static构造；DPO-100定义；相应规模/step/eval/resource合同。
- 门禁：用户确认的是上述局部修改与数据方向重开，不是`r3-theory-v0.6`/`round3-exp-v1.1`整体通过；代码、上传和执行仍锁定，当前Round2 `SERVER_EXECUTION`不变。

### `cycle-20260818-01` / Round3 — 2026-08-25 — 统一选点与双head测试，生成式benchmark延期到Round4

- 上一理论版本：Round3 `r3-theory-v0.6`。
- 新理论版本：`r3-theory-v0.7`，讨论中、未整体通过；对应实验草案升级为`round3-exp-v1.2`。
- 保留的内容：六方法训练objective、GitHub-loss SSPO、Qwen3-1.7B、非量化LoRA、动态训练sampling、独立fixed-pair test及当前数据源未决边界。
- 修改的内容：训练objective与checkpoint selection分离。六方法候选统一在冻结1,000-pair labeled validation view上，以冻结初始reference、response总log-prob和beta .1的DPO NLL作为`eval_selection_loss`；PE/joint、SSPO objective与dynamic panel只作诊断，不参与选点。若采用`ultrafeedback_binarized/test_prefs`，用两个独立SHA-256 namespace按“先validation、排除后test”冻结无交集1,000/1,000 views。
- final-test修正：所有selected checkpoints与frozen base在同一1,000-pair test上同时报告reference-delta beta .1与raw mean-logp beta10两个score heads的Accuracy/NLL/Brier/ECE及confidence诊断；只能在相同head内比较，不合成综合分数，test不参与重选。
- 延期登记：AlpacaEval 2.0 length-controlled win rate与MT-Bench平均分移至Round4候选。Round3不生成相应回答、不调用外部judge API、不运行本地替代judge，生成量和API预算均为0。Round4须另行冻结版本、generation、judge snapshot、API预算、缓存/失败策略与可比性。
- 变化原因：用户认可共同selection/双head评分建议，并明确要求Round3现阶段只保留独立1,000-pair final test，将AlpacaEval/MT-Bench登记后推迟到Round4运行。
- 对实验设计的影响：method-aligned objective loss不再决定best checkpoint；dynamic validation降为可选诊断；Round3产物不包含AlpacaEval/MT-Bench generations、annotations、judge logs或榜单分数。
- 尚未解决：paired source是否仍为UltraFeedback、PE-static第二candidate、DPO-100来源/规模、训练steps/checkpoints和3×4090资源布局；Round4生成式评测细节不构成Round3 blocker。
- 门禁：这些是用户确认的评价范围与口径，不等于`r3-theory-v0.7`或`round3-exp-v1.2`整体通过；不解锁Round3代码、上传或执行，当前Round2 `SERVER_EXECUTION`不变。

### `cycle-20260818-01` / Round3 — 2026-08-25 — 回归SSPO双源数据并收敛为五方法

- 上一理论版本：Round3 `r3-theory-v0.7`。
- 新理论版本：`r3-theory-v0.8`，讨论中、未整体通过；对应实验草案升级为`round3-exp-v1.3`。
- 数据决定：用户明确选择SSPO作者的双源数据类型，仅保留本项目数量缩放。8K paired master来自`ultrafeedback_binarized/train_prefs`，其内嵌1K limited view；7K singles来自`ultrachat_200k/train_sft`；`test_prefs`确定性分为1K validation与1K independent test。旧统一UltraFeedback hidden-pair与严格零UltraFeedback分支退出Round3。
- 方法变化：六方法收缩为五方法：DPO-1K、`SSPO-code-loss-stratified-ultrachat@2df9e9a`、DPO-8K、DPO+PE-SFT+rollout、DPO+PE-rollout-only。PE-static从Round3删除并登记到Round5消融；旧DPO-10/DPO-100名称改为实际数量DPO-1K/DPO-8K。
- 训练与选点：五方法均1 epoch/250 optimizer steps，每25步保存。SSPO固定4+28，$\gamma_{min}=0.125$；DPO-8K固定32 pairs/step。删除checkpoint级SSPO/PE objective diagnostics与dynamic rollout panel，只保留共同1K labeled DPO selection与训练telemetry。
- 执行架构：五方法均在GPU0单卡训练；动态方法在GPU1/2上使用两个独立vLLM replicas，必须通过`method_id/optimizer_step/adapter_sha256` ACK屏障。train/rollout环境隔离，50个durable checkpoints全部保留且不启用自动pruner；formal前使用strong-smoke投影峰值并要求两倍空闲空间，不删除Round2产物。
- 执行时解析：dataset/model full revisions、文件与manifest hashes、experiment ID、精确dependency locks/源码commit、实测显存与存储峰值只能在获批后的服务器preflight/tests/smoke解析，不从本地或旧代码猜测。
- 门禁：用户对数据、PE-static延后与其他局部决定的确认已记录，但仍不等于对`r3-theory-v0.8`和`round3-exp-v1.3`的依次整体批准；Round3代码/上传/执行仍锁定，Round2 `SERVER_EXECUTION`不变。

### Round3 v0.8/v1.3 内容批准记录 — 2026-08-25

- 用户明确表示“我认可了”并询问写入theory/exp后进入code；该表述按当前对话对象记录为对`r3-theory-v0.8`与`round3-exp-v1.3`内容的明确整体批准。
- 批准不自动改变当前全局单活动阶段。Round2仍在`SERVER_EXECUTION`，且尚未完成`RESULT_HANDOFF`；因此Round3阶段激活与`CODE_IMPLEMENTATION`暂缓，本记录不授权并行改动训练代码、服务器checkout或当前运行。

### Round3 `CODE_IMPLEMENTATION` 激活记录 — 2026-08-26

- 用户明确要求“round2先不管了，直接开始round3”，覆盖上一条中等待Round2现场结果后再激活Round3的安排。
- Round2在无新增实时证据的前提下只作行政性`NO_CONCLUSION`结果交接；该记录不声称其服务器任务已结束，也不授权停止潜在任务、修改服务器checkout或删除checkpoint。
- 当前唯一活动阶段切换为Round3 `CODE_IMPLEMENTATION`；批准对象仍是`r3-theory-v0.8`与`round3-exp-v1.3`，没有改变理论或实验合同。
- 本阶段只解锁本地静态源码、配置、脚本与文档实现；代码交接、commit/push、上传、服务器tests/smoke/formal仍未授权。

### Round3 描述性诊断与动态位置的实现口径 — 2026-08-26

- 为避免运行者临时选择final-test描述阈值，补齐$p/c$分位数、near-zero/half/one、confidence和collapse的固定口径；这些量不参与选点、排名或成功判定，因此不改变已批准的科学比较。
- 动态candidate pair的A/B换位冻结为独立`round3-dynamic-ab-swap-v1` SHA-256 namespace；它不改变PE的28-pair logical population，也不改变按draw分配两个vLLM副本的路由。
- 其余`r3-theory-v0.8`与`round3-exp-v1.3`合同不变。

### Round3 checkpoint round-trip验收容差补全 — 2026-08-26

- 补全此前正文只写“预注册数值容差”却误删具体数值的实现门禁：同一SSPO checkpoint独立重载两次后，running state、scheduler/global step必须精确一致，下一batch loss绝对差不超过`1e-7`，trainable LoRA参数更新最大绝对差不超过`1e-7`且最大相对差不超过`1e-6`。
- 该门禁只判断序列化/恢复的工程可复现性，不参与方法得分、选点或科学成功判据；超限会阻断formal并返回代码实现阶段。

### Round3 `r3-theory-v0.9` / 方案B冻结数据勘误 — 2026-08-26

- 触发事实：服务器对冻结`ultrafeedback_binarized` revision的只读审计确认`test_prefs` 2,000行中有3行empty-rejected，只剩1,997个有效pair；原v0.8的1K validation +1K test合同按设计fail closed。
- 用户批准：用户明确表示“我也赞成B，请你本地修改”，批准保持原1,000 validation并将同一`test_prefs` split剩余997个有效pair全部作为independent test，不从`train_prefs`补3条。
- 数据质量合同：所有source在选择前确定性schema审计；畸形行不进入view，也不加入跨view排除集，写入无原始文本的server-only source-index/reason audit并绑定SHA。冻结source totals、valid/malformed totals和reason aggregates变化即fail closed。
- 不变项：五方法、1K/8K paired train、7K UltraChat、250 steps、共同1K selection、两个score heads、GPU布局及Round4/5延期均不变；最终指标必须显式报告`N=997`。
- 门禁：当前仍为`CODE_IMPLEMENTATION`；v0.9/v1.4本地修订不授权commit/push、服务器checkout修改、数据重跑、strong smoke或formal。

### Round3 `r3-theory-v1.0` / 3×4090资源波次批准 — 2026-08-26

- 用户批准：用户明确认可将三个单卡静态方法并发运行，并要求Codex修复已发现问题、持续测试直到formal挂载。
- 资源变化：DPO-1K、SSPO、DPO-8K分别固定到物理GPU0/1/2并发；两个动态PE方法随后分别独占三卡串行运行，GPU0训练、GPU1/2双vLLM生成。
- 不变项：数据、模型、loss、batch、seed、250 steps、checkpoint、selection、997-pair test和两个动态方法的current-policy/ACK合同均不变；并发只改变无依赖任务的墙钟调度。
- 工程门禁：静态并发必须在三张卡上分别通过确定性checkpoint重放和并发production-path strong smoke；共享model/data/reference只读，各run/config/log/checkpoint目录隔离。动态方法不得并发或共享rollout。
- 执行授权：用户已授权本轮修复的commit/push、服务器新attempt和完整门禁，并在全部strong smoke与两倍存储门禁通过后直接挂载formal。

### `cycle-20260818-01` / Round3 PE reward纠正 / `r3-theory-v1.1` — 2026-08-27

- 触发：用户指出原始Theory v0.2的PE probability使用$\pi_\theta/\pi_{\mathrm{ref}}$，而Round3重写把动态PE改成了reference-free raw mean-logp；两者是不同方法，不能把reference未进入PE误解为reference约束失效。
- 恢复的主定义：$r_\theta^{\mathrm{DPO}}=0.1(s_\theta-s_{\mathrm{ref}})$，$p_i=\sigma(r_i^A-r_i^B)$，其中$s$为response-token总log-prob。初始化$\pi_\theta=\pi_{\mathrm{ref}}$时任意candidate pair严格$p_i=0.5$；训练后pair logit仍允许为正或负。
- 回溯命名：旧`round3-exp-v1.5`的两个动态臂不删除、不覆盖，明确登记为`SimPO-reward PE`，即$p_i=\sigma(10[q_\theta^A-q_\theta^B])$、$q$为mean-response-logp；它们只作为reward-definition消融，不能代表原始PE主方法。
- 新增主方法：在相同SFT+rollout与rollout-only candidate合同下分别新增两个`DPO-reward PE`，形成七方法Round3；数据、labeled DPO、PE aggregation、lambda、optimizer、selection与final test不变。
- 执行边界：旧exact-commit formal继续自然结束；新增两臂必须等待旧controller终态，以新experiment ID/new exact commit执行并通过跨运行manifest、sample顺序与final-evaluator等价审计。Round4保持锁定。

### Round4三方法候选登记 / `round4-theory-draft-v0.1` — 2026-09-01

- 状态：`LOCKED_PRE_DISCUSSION_DRAFT`；尚未创建新cycle，当前唯一活动阶段仍是Round3 amendment `CODE_IMPLEMENTATION`。本条不构成Round3结果交接、Round4理论批准、实验批准、代码交接或执行授权。
- 用户输入：下一轮候选暂按三种方法理解为DPO、SSPO、StaticPE；共同Qwen3-1.7B与UltraFeedback/UltraChat各10%来源，DPO只使用labeled数据；StaticPE沿用冻结初始化模型一次性生成第二candidate、`lambda=0.1`和DPO-base labeled branch。
- 理论对照意图：优先把三者labeled branch统一为DPO-base，使比较集中于“无unlabeled机制 / SSPO single-response risk / StaticPE fixed-pair conditional encoding”。SSPO使用DPO-base还是作者默认SimPO-base仍未关闭，不能提前宣称已做到单变量比较。
- 评价候选：AlpacaEval 2.0 LC与普通win rate；MT-Bench从旧Round4默认候选降为未决项。evaluator版本、805条数据revision、base是否参评、generation、judge snapshot、费用和失败策略均待Round4讨论冻结。
- 执行候选：exact commit经用户批准后在4090-3拉取；24GB专用smoke对每种方法训练2个optimizer steps，并覆盖eval、LoRA merge/reload、Alpaca输出和真实judge调用；镜像使用非`latest` tag与digest录入FusionOne。smoke只证明工程闭环，不是科学结果。
- 基础设施：既有快照表明4090-3当时没有可用Docker daemon且数据盘接近满载；用户于2026-09-01确认已实机验证FusionOne存在8张A100。Round4按8×A100目标资源登记，单卡显存、拓扑、容器GPU映射和挂载仍由任务preflight自动采集。
- 安全边界：私有账号、密码、内部地址和镜像仓库实值只从本机原始手册或平台界面由操作者填写，禁止进入Git、配置、日志、实验记录或回答。

### `cycle-20260901-01` / Round4激活与执行拓扑修订 / `r4-theory-v0.2` — 2026-09-01

- 激活：用户明确要求行政关闭Round3并直接覆盖为Round4，创建`cycle-20260901-01`并进入`THEORY_DISCUSSION`；Round3旧五方法结果只读保留，DPO-reward extension取消执行。
- 方法与batch：三方法为DPO-label-only、SSPO与StaticPE；用户选择DPO effective batch16，SSPO/StaticPE保持effective batch64，均为2 GPU、GA8、epoch1。
- 资源事实：用户确认已实机验证FusionOne存在8张A100，并决定先创建/占用其中2张；单卡显存、拓扑、映射和挂载仍由preflight解析。
- 流程修订：撤销4090-3训练smoke和三方法6卡并发候选。4090-3只负责镜像准备、冻结数据下载与SSH中转；三方法smoke和formal共用同一2张A100顺序执行。
- 完整性：数据从4090-3传入A100仓库外目录，以源manifest和目标端SHA-256复核闭环；镜像使用安全启动入口，不自动训练，也不包含数据、模型或凭据。
- 已知风险：4090-3历史快照缺少Docker/BuildKit且scratch紧张。该状态必须实时核验；若未改变，指定build步骤不能执行，不能删除旧实验强行腾空间。
- 门禁：本次是讨论中理论修订，不构成理论整体批准，也不授权commit/push、镜像、下载/传输、A100资源创建、smoke或formal。

### `cycle-20260901-01` / Round4理论与实验批准 / `r4-theory-v1.0` — 2026-09-01

- 上一理论版本：`r4-theory-v0.2`，讨论中。
- 新理论版本：`r4-theory-v1.0`，用户已明确通过；唯一活动阶段推进为`CODE_IMPLEMENTATION`。
- 用户授权：用户明确表示“直接到code阶段，想改什么改什么”，按当前Round4理论与实验对象记录为理论、实验的明确通过及剩余工程选择冻结授权；该授权不等于代码交接或服务器执行授权。
- 方法冻结：三方法共同labeled branch使用DPO-base；SSPO不使用SimPO-base；StaticPE使用跨两卡同步的`physical-microbatch-PE`，不跨8次gradient accumulation形成一个PE population。
- 评价冻结：三方法与frozen base共用`alpaca_eval==0.6.2`、`weighted_alpaca_eval_gpt4_turbo`和LC函数；本轮seed42，MT-Bench退出。
- 执行拓扑：复用FusionOne现有`cuda12.4-cudnn-devel-ubuntu22.04-py312-ssh`镜像；4090-3准备CPython3.12/cu124离线wheelhouse、数据和模型manifest，经校验传至A100仓库外目录；A100镜像内新建venv，不复制旧py310环境。
- 代码影响：更新Round4三方法、离线环境、数据/模型传输与A100安装入口；任何改变目标函数、population、方法臂或评价合同的实现偏离必须退回实验讨论。

### `cycle-20260901-01` / Round4 v2 PE语义重命名与扩展 — 2026-09-02

- 用户明确批准：旧双候选、DPO-base `StaticPE`更名为`FrozenPE`；新`StaticPE`使用SimPO-base和单条无标签回答，不再生成candidate B。
- 新StaticPE固定`beta=10`、`simpo_gamma=2`、lambda0.1、temperature1、detached EMA momentum0.95、clip5和epsilon1e-8；PE仍使用soft responsibility的`c_1/c_2` L1项，但不使用threshold、prior或hard pseudo label。
- 解释边界：StaticPE同时改变labeled base和无标签机制，因此不把其与DPO/SSPO/FrozenPE的差异归因为单一PE因素。
- 评价拓扑改为A100仅训练/生成，4090持久profile调用API并聚合WR/LC；formal顺序冻结为base、DPO、SSPO、StaticPE、FrozenPE。
