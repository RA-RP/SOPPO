# 当前实验设计：Round3 Qwen3-1.7B 五方法合并验证

## 0. 版本、状态与历史边界

- Cycle：`cycle-20260818-01` / Round3
- 设计版本：`round3-exp-v1.4`
- 对应理论：`../theory/current_theory.md` `r3-theory-v0.9`
- 理论状态：用户已于2026-08-26明确批准方案B数据勘误`r3-theory-v0.9`
- 实验状态：用户已于2026-08-26明确批准方案B：1,000 validation +997 independent test、确定性畸形审计、不从train补样
- 当前唯一活动阶段：Round3 `SERVER_EXECUTION`
- 代码与执行：旧exact commit已通过阶段C、data v2和reference cache；两个strong-smoke attempt分别因入口`PYTHONPATH`遗漏与SSPO CUDA backward非确定性停止。deterministic诊断使预注册三项差异均为0；用户于2026-08-26明确授权commit/push v0.4、部署新exact commit、完整strong smoke通过后直接挂载formal
- 当前运行索引：`../../exp/round3-20260826-01/README.md`、`../../exp/round3-20260826-02/README.md`
- Round2边界：服务器只读证据确认其controller已在step590停止，step580/589/590仍保留且pruner未运行；不得删除其run/checkpoint
- 历史设计：Round1/Round2完整基线位于Git commit `d338eb5bedef16d83a42790c3faa97f8f404315b`；设计演化见`experiment_archive.md`，当前执行证据见`../code/ROUND2_LIVE_HANDOFF.md`

本文件只表达当前Round3设计；早期版本的完整内容只从历史索引读取，不在本文混排。

## 1. 实验问题、任务与解释等级

Round3作为ModelScope `Qwen/Qwen3-1.7B`、单种子的exploratory validation，回答：

1. 有限真实偏好监督的DPO是否相对训练前模型产生可见增长；
2. GitHub-loss SSPO与两个动态PE的相对行为如何；
3. 更高paired-label预算的DPO reference与有限标签方法之间有多大差距；
4. 固定公开单回复锚点+current rollout与rollout-only是否不同。

五个任务及正式顺序固定为：

```text
DPO-1K
→ SSPO-code-loss-stratified-ultrachat@2df9e9a
→ DPO-8K
→ DPO+PE-SFT+rollout
→ DPO+PE-rollout-only
```

单模型、单种子结果只能解释为探索性趋势，不能宣称统计显著性或最终理论确认。

`DPO+PE-static`从Round3删除并登记为Round5消融候选。Round3不构造其第二candidate、不实现入口、不创建配置或运行目录；该删除不是工程失败或缺失结果。

## 2. 数据与隔离合同

作者GitHub commit `2df9e9a`使用双源数据：

- paired preference：`HuggingFaceH4/ultrafeedback_binarized`的`train_prefs/test_prefs`；
- unpaired single-response：`HuggingFaceH4/ultrachat_200k`的`train_sft`。

其通用10%设置约为6,113个UltraFeedback paired records（含validation）与20,786个UltraChat singles，而不是把同一批UltraFeedback pairs隐藏方向后派生singles。[GitHub preprocessing](https://github.com/MLAI-Yonsei/SSPO/blob/2df9e9a1d5fb9202a583cb66eb081e0cb60e873d/preprocessing_data/preprocessing_ultrachat.py#L260-L314)，[paper training data](https://arxiv.org/html/2511.00040v3#S5.SS1)。

用户2026-08-25明确采用SSPO官方双源类型，仅保持本项目缩放数量。因此旧“统一UltraFeedback hidden-pair池”、`pair-derived` SSPO和严格零UltraFeedback分支均退出Round3。正式数据规模为：

| role | 固定数量 | 来源与用途 |
| --- | ---: | --- |
| paired train master | 8,000 pairs | `ultrafeedback_binarized/train_prefs`；DPO-8K |
| limited labeled view | 1,000 pairs | 8K master的嵌套子集；DPO-1K、SSPO、两个动态PE |
| unpaired train | 7,000 singles | `ultrachat_200k/train_sft`；SSPO、两个动态PE |
| paired validation | 1,000 pairs | `ultrafeedback_binarized/test_prefs`；五方法共同checkpoint selection |
| paired independent test | 997 pairs | `ultrafeedback_binarized/test_prefs`；1,000 validation之后的全部有效剩余；selected-checkpoint-only test |

这是SSPO双源合同的缩小版，不是GitHub 10%数量原样复现。`DPO-8K`使用额外paired feedback作为高标签reference，不称为全数据100% oracle；`DPO-1K`/`DPO-8K`名称替代旧`DPO-10`/`DPO-100`。

### 2.1 Revision、schema与canonical ID

两个dataset在服务器数据preflight开始时先把仓库引用解析为full commit SHA，随后所有读取都显式使用该SHA；resolved config和manifest中禁止只保存`main`。同时记录源parquet文件名、字节数和SHA-256。当前文档冻结数据ID/split/构造算法，不在本地猜尚未解析的full SHA。

统一定义：

```text
canonical_prompt = NFC(prompt).replace("\r\n", "\n").replace("\r", "\n").strip()
sample_id = SHA256(dataset_id || "\0" || resolved_revision || "\0" ||
                   split || "\0" || prompt_id || "\0" || source_row_index)
```

UltraFeedback row必须包含`prompt,prompt_id,chosen,rejected`；chosen/rejected的首条user content经canonicalization后必须等于`prompt`，各自必须存在一条非空assistant response。UltraChat row必须满足`messages[0].role=user,messages[1].role=assistant`，且`messages[0].content`与`prompt` canonical相等；与SSPO GitHub预处理一致，只取`messages[1].content`作为公开single response，后续轮次不进入Round3。不满足这些条件的source row不得进入选择池，也不得因单行畸形终止整个冻结数据构造；它必须进入无原始文本的server-only `malformed_source_rows.jsonl`，记录dataset/split、source index、sample/source ID、可用prompt hash与排序后的reason codes。

本版绑定服务器已解析的冻结revision事实：`train_prefs`为61,135 source/81 malformed/61,054 valid，reason counts为empty-chosen 20、empty-rejected 72；`test_prefs`为2,000/3/1,997，三个畸形source indices固定为374、595、1846且均为empty-rejected；`train_sft`为207,865/12,113/195,752，reason counts为empty-prompt 1、message0-prompt-mismatch 12,092、empty-response 21。reason counts允许同一畸形行贡献多个reason。上述source totals、valid/malformed totals或reason aggregate任一变化都fail closed。

Round3只做上述exact canonical-prompt隔离，不宣称语义近重复过滤；这是已知局限。任何canonical prompt在train master、unpaired、validation、test之间重复都必须在确定性选择前排除并计数。

### 2.2 确定性子集与隔离顺序

所有排序key均为

```text
SHA256(UTF8(namespace || "\0" || seed || "\0" || sample_id)), seed=42
```

按以下顺序冻结，不能调换后接受另一批样本：

1. 先隔离`test_prefs`的3条畸形行；对剩余1,997个有效pair用`round3-paired-validation-v3`排序选1,000 validation，再对剩余记录以`round3-paired-independent-test-v3`排序并要求恰好得到997 test。两者canonical prompt必须唯一且无交集；不得动态接受其他数量，也不得从`train_prefs`补3条。
2. 从`train_prefs`排除validation/test prompt后，用`round3-uf-paired-train-8k-v1`排序选8,000 master pairs；排序前8,000构成DPO-8K，前1,000同时构成所有limited-label方法共享的DPO-1K view。
3. 从`ultrachat_200k/train_sft`排除全部paired master/validation/test prompt并按canonical prompt去重，再用`round3-uc-unpaired-train-7k-v1`排序选7,000 singles。SSPO与两个动态PE共享这7,000 source IDs和顺序。
4. paired A/B位置使用独立`round3-paired-ab-swap-v1` hash最低bit确定；A为chosen时$z=1$，A为rejected时$z=0$。同一pair在所有方法、selection和test中的A/B位置一致。

动态candidate pair的A/B位置使用`SHA256("round3-dynamic-ab-swap-v1\0" || method_id || "\0" || optimizer_step || "\0" || sample_id)`摘要最低bit确定。该换位只消除固定位置约定，不能改变28-pair population、跨副本路由或复用生成文本。

最终在仓库外`dual_source_v2`输出8K master、1K limited、7K unpaired、1K validation和997 test五份canonical views及各自SHA-256，并输出17,997行view source manifest、12,197行malformed-source audit、source aggregate和排除/去重计数。失败遗留的空`dual_source_v1`保留作证据，不删除、不覆盖。test使用不嵌入数量的`test.public.jsonl`与`test.private_labels.jsonl`；公开入口不能读取private labels，test label不进入训练或selection入口。

## 3. 模型、序列与统一优化合同

### 3.1 模型与模板

- 模型ID：ModelScope `Qwen/Qwen3-1.7B`，即post-trained Instruct/hybrid-thinking版本，不使用`Qwen/Qwen3-1.7B-Base`；
- 下载命令：`modelscope download --model Qwen/Qwen3-1.7B --local_dir ./dir`；不得在model ID后追加`README.md`；
- 五方法共享同一本地模型目录、resolved revision、模型/tokenizer/chat-template文件SHA-256与special-token manifest；
- 所有固定数据编码和动态生成均使用模型原生Qwen3 chat template，显式`enable_thinking=false`并使用response-only loss mask；
- Qwen3兼容环境必须满足`transformers>=4.51.0`。SSPO仓库原始`transformers==4.46.1`环境不得直接复用；实现阶段移植commit `2df9e9a`的loss/trainer语义并在Qwen3兼容依赖栈上验收。

实际resolved revision与文件manifest只能在执行面下载后冻结；五方法中途不得改变revision、template或special tokens。

### 3.2 DPO与PE loss合同

DPO-1K、DPO-8K及两个动态PE方法的labeled分支统一使用冻结初始Qwen3-1.7B作为reference。令$s_\theta$和$s_{\mathrm{ref}}$为response tokens上的**总**log-prob，$z_i=1$表示A更优：

$$
d_i^{\mathrm{DPO}}
=0.1\left[(s_\theta^A-s_{\mathrm{ref}}^A)-(s_\theta^B-s_{\mathrm{ref}}^B)\right],
$$

$$
L_{\mathrm{DPO}}
=\operatorname{mean}_i[-z_i\log\sigma(d_i^{\mathrm{DPO}})-(1-z_i)\log\sigma(-d_i^{\mathrm{DPO}})].
$$

PE candidate pair使用reference-free mean-response-logp：

$$
p_i=\sigma\left(10[q_\theta(x_i,y_{iA})-q_\theta(x_i,y_{iB})]\right),
\qquad \mathbf q_i=[p_i,1-p_i]^\top.
$$

固定$\epsilon=10^{-8}$、L1距离且denominator不detach：

$$
\widehat{\mathbf e}_+=\frac{\sum_i p_i\mathbf q_i}{\sum_i p_i+\epsilon},
\qquad
\widehat{\mathbf e}_-=\frac{\sum_i(1-p_i)\mathbf q_i}{\sum_i(1-p_i)+\epsilon},
$$

$$
L_{PE}=\frac12\left(\lVert\widehat{\mathbf e}_+-[1,0]^\top\rVert_1+\lVert\widehat{\mathbf e}_--[0,1]^\top\rVert_1\right).
$$

$L_{PE}$必须对一个optimizer step完整的28个candidate pairs精确计算；物理microbatch或顺序subbatch不得改变population或梯度。两个动态PE方法使用

$$
L_{\mathrm{joint}}=\frac{L_{\mathrm{DPO}}+0.1L_{PE}}{1.1}.
$$

### 3.3 LoRA、optimizer与precision

| 配置 | 五方法固定值 |
| --- | --- |
| finetuning | LoRA；不使用QLoRA或其他量化训练 |
| rank / alpha / dropout | 8 / 16 / 0 |
| target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`；排除embedding与`lm_head` |
| learning rate | `1.0e-5` |
| optimizer | `adamw_torch` |
| AdamW | weight decay 0；betas `(0.9,0.999)`；epsilon `1.0e-8` |
| scheduler | cosine；warmup ratio 0.1 |
| clipping | max grad norm 1.0 |
| precision | BF16 forward/autocast；`bf16=true, fp16=false, pure_bf16=false`；trainable LoRA与optimizer state为FP32 |

不做method-specific lr sweep。统一`1e-5`是本轮五方法公平性与预算选择，不冒充SSPO论文分别调参的复现结果。

### 3.4 序列、截断与长度审计

五方法统一：

```yaml
max_length: 2048
max_prompt_length: 1024
max_completion_length: 1024
```

在原生chat template后执行长度合同：prompt超过1024时从左侧截断、保留末端1024 tokens；固定response超过1024时从右侧截断、保留前端1024 tokens；任一prompt+response序列不得超过2048。chosen/rejected、SSPO single、UltraChat固定anchor与rollout一律使用同一规则，loss只作用于response tokens。

动态生成不先超长生成再静默切片。每条rollout在生成前设置：

$$
\texttt{max\_new\_tokens}=\min\{1024,\,2048-L_{\mathrm{prompt}}\}.
$$

达到EOS自然结束；达到上限的回复保留并标记`finish_reason=length`。训练、validation分别汇总：prompt截断条数/比例与丢弃token数，fixed-response截断条数/比例，rollout cap-hit率、EOS率、response-token均值/分位数。禁止静默截断或因超长丢样本后仍声称完整epoch。

### 3.5 PE固定权重

两个动态PE方法均固定§3.2的`lambda_pe=0.1`，有效权重约为`0.9091/0.0909`。该`lambda_pe`与SSPO的`sspo_gamma_decay=0.001`不是同一参数；本轮不做lambda sweep或scheduler。

## 4. epoch、batch与step

- seed：42，单种子
- epoch：五个任务全部为1
- optimizer steps：五个任务全部为250

| # | 方法 | train view | 每个optimizer step | 一轮使用/生成 |
| ---: | --- | --- | --- | --- |
| 1 | DPO-1K | 1,000 paired labeled | 4 labeled pairs | 1,000 pairs |
| 2 | SSPO code-loss | 同一1,000 labeled +7,000 UltraChat singles | 4 labeled +28 singles | 1,000 pairs +7,000 singles |
| 3 | DPO-8K | 8,000 paired master pool | 32 labeled pairs | 8,000 pairs |
| 4 | DPO+PE-SFT+rollout | 同一1,000 labeled +7,000 UltraChat anchors/prompts | 4 labeled +28 candidate pairs | 生成7,000条rollout |
| 5 | DPO+PE-rollout-only | 同一1,000 labeled +7,000 UltraChat prompts | 4 labeled +28 candidate pairs | 生成14,000条rollout |

整数对齐固定为：

$$
1000/4=7000/28=8000/32=250.
$$

除DPO-8K的paired-label预算外，其余四方法共享同一1K labeled view；三个unpaired方法共享同一7K UltraChat source IDs与顺序。五方法optimizer updates均为250，但response数与生成成本不同，不能宣称token-compute matched。

## 5. 五个方法的训练目标

### 5.1 DPO-1K

使用8K master pool排序前1,000个paired preference records，每步4 pairs，只计算§3.2的reference-based DPO loss，250步完整遍历一次。

### 5.2 `SSPO-code-loss-stratified-ultrachat@2df9e9a`

采用作者GitHub commit `2df9e9a1d5fb9202a583cb66eb081e0cb60e873d` trainer的实际loss和相同UltraChat unpaired来源，但用本项目固定1:7 sampler：

- 每步固定4 labeled pairs +28 unpaired singles；禁止no-labeled fallback；
- $q=\mathrm{mean\_response\_logp}$；
- labeled risk：

$$
R_L=\operatorname{mean}[-\log\sigma(\beta(q_w-q_l)-\Delta)],
\qquad \beta=10,\ \Delta=2;
$$

- 初始化严格为GitHub语义：`running_mean=None,running_var=None,reward_norm_momentum=0.95,reward_clip_range=5.0`；
- 每次非空调用计算`batch_var=var(unbiased=False)+1e-8`；首次chosen调用直接初始化mean/var，随后按chosen、rejected、unpaired顺序分别以0.95/0.05 EMA更新同一份state；normalized values clamp到$[-5,5]$；
- threshold：$\delta_{\mathrm{code}}=\min_i z_{w,i}$，不使用KDE、Bayes grid search或threshold EMA；
- $d_k=\beta(z_{u,k}-\delta_{\mathrm{code}})$，按$d_k$正负分别使用prior-weighted logistic risk，$p_+=p_-=0.5$；
- 联合目标：

$$
L_t=\gamma_tR_L+(1-\gamma_t)R_U,
\qquad
\gamma_t=\max\left(\frac{n_L}{n_L+n_U},e^{-0.001t}\right),
\quad t=\texttt{state.global_step},\ t_\text{first}=0.
$$

`code-loss`、`stratified`和`ultrachat`三个限定必须进入method ID和resolved config；不得混入论文v3 KDE机制，也不得宣称GitHub sampler或数据规模原样复现。固定$n_L=1{,}000,n_U=7{,}000$，因此$\gamma_{\min}=0.125$；保持decay 0.001与首步$t=0$，250步内floor不会触发。

### 5.3 DPO-8K

使用全部8,000 paired master records，每步32 pairs，只计算§3.2的reference-based DPO loss，250步完整遍历一次。其前1,000条与DPO-1K完全相同；该方法只解释额外paired supervision headroom，不称为全量oracle。

### 5.4 DPO+PE-SFT+rollout

每步4 paired labeled +28 UltraChat prompts。每个prompt使用该UltraChat row的固定公开assistant response作为锚点和一条本方法current-policy rollout，形成28个方向未知candidate pairs并计算§3.2联合loss；250步共生成7,000条rollout。

### 5.5 DPO+PE-rollout-only

每步4 paired labeled +28 UltraChat prompts。每个prompt由本方法current policy独立生成两条回复，形成28个方向未知candidate pairs并计算§3.2联合loss；250步共生成14,000条回复，不得复制同一生成结果或共享另一方法的生成结果。

### 5.6 Dynamic rollout sampling

两个动态方法的训练rollout使用完全相同的采样分布：

```yaml
enable_thinking: false
do_sample: true
temperature: 0.7
top_p: 0.8
top_k: 20
min_p: 0.0
repetition_penalty: 1.0
presence_penalty: 0.0
max_new_tokens: min(1024, 2048 - formatted_prompt_tokens)
eos_token_id: [151645, 151643]
pad_token_id: 151643
```

special-token IDs必须与resolved tokenizer manifest一致，不一致则预检失败而不是静默覆盖。rollout-only的两条回复使用相同参数、不同`draw_index`和独立随机流；不得复制同一回复。训练随机种子由`base_seed=42, optimizer_step, sample_id, draw_index`确定。两个动态方法共享sampling合同和source IDs，但分别由自己的current policy生成，禁止跨方法或跨step复用文本。

其余三种方法训练时不生成回复，因此不存在训练temperature。当前最终test也是固定pair评分，没有test rollout或temperature。

## 6. Checkpoint与共同selection eval

五个任务固定在以下step保存durable checkpoint并计算共同`eval_selection_loss`：

```text
25, 50, 75, 100, 125, 150, 175, 200, 225, 250/final
```

step250与final是同一checkpoint，不保存重复副本。训练objective和checkpoint-selection objective严格分离；五个方法都用§6.1同一个`eval_selection_loss`选择checkpoint。

### 6.1 共同labeled validation view

五方法共享同一冻结的1,000 labeled pairs、相同A/B换位、pair顺序、Qwen3 chat template、截断和样本权重。eval batch统一为4 pairs，即每个checkpoint 250 batches。令§3.2的DPO logit为$d_i^{\mathrm{DPO}}$：

$$
L_{\mathrm{eval\_select}}
=\frac1{1000}\sum_i[-z_i\log\sigma(d_i^{\mathrm{DPO}})-(1-z_i)\log\sigma(-d_i^{\mathrm{DPO}})].
$$

所有方法统一使用冻结初始Qwen3-1.7B reference、$\beta_{\mathrm{DPO}}=0.1$和response总log-prob。该loss只负责选点，不改变SSPO或PE的训练loss。reference log-prob允许对冻结validation view预计算一次，但必须绑定model/tokenizer/template/data manifest并在服务器交接时通过直接复算抽查。

Round3不计算checkpoint级`eval_sspo_loss`、`eval_pe_loss`或`eval_joint_loss`，也不运行dynamic rollout diagnostic panel。训练过程仍按step记录loss components、$p$熵/极端比例、SSPO threshold/pseudo-positive rate和rollout长度等无样本级聚合telemetry；这些是训练完整性证据，不是eval、不参与选点或方法排序。

### 6.2 Selection与non-finite policy

每个方法只在finite checkpoints中按原始未四舍五入的二元组

```text
(eval_selection_loss, checkpoint_step)
```

做字典序最小化；loss完全相同时选择更早step。禁止依据显示时的四舍五入值判定并列。

- 任一selection batch或aggregate为NaN/Inf：该checkpoint无效，记录原因后继续其他checkpoint；
- 十个checkpoint全部无效：该方法为工程失败，不进入independent test；
- 任一train loss或gradient为NaN/Inf：立即终止该方法；
- 禁止把非有限值替换为0、静默跳过optimizer update后继续，或使用test补选checkpoint；
- 训练telemetry中任一应有component非有限必须单独记录，不能用其他量替代共同selection loss。

### 6.3 SSPO state与checkpoint round-trip验收

SSPO checkpoint必须显式保存并恢复`running_mean`、`running_var`、`global_step`及其他通用训练态。共同selection evaluator只读policy/reference log-prob，不调用SSPO loss，不更新running statistics。代码交接前必须在服务器验收：

1. selection前后SSPO checkpoint state序列化SHA-256完全相同；
2. save/load round-trip恢复相同`running_mean/running_var/global_step`，并使同一下一训练batch的loss、state与参数更新在预注册数值容差内相同：两次从同一checkpoint独立重载并执行同一batch，running state、scheduler step与global step必须精确一致，loss绝对差不超过`1e-7`，全部trainable LoRA参数更新后的最大绝对差不超过`1e-7`且最大相对差不超过`1e-6`；任一超限即strong-smoke失败，不得放宽后继续formal；
3. 人工移除SSPO running state时fail closed，不得静默重置。

## 7. Round3唯一final test：独立997 pairs

每个方法只有最低共同`eval_selection_loss`选出的一个checkpoint进入独立997-pair test；frozen base也作为共同参照进入同一test。Test不默认使用step250，也不得用于重选checkpoint、超参或方法。

Test直接评分冻结的有标签A/B preference pairs，不为动态方法再次生成rollout，因此没有test temperature或sampling配置。所有selected checkpoints与frozen base都必须在同一批pairs上同时计算两个score head：

$$
p_i^{\mathrm{ref}}
=\sigma\left(0.1[(s_\theta^A-s_{\mathrm{ref}}^A)-(s_\theta^B-s_{\mathrm{ref}}^B)]\right),
$$

$$
p_i^{\mathrm{raw}}
=\sigma\left(10[q_\theta(x_i,y_{iA})-q_\theta(x_i,y_{iB})]\right).
$$

两个head分别输出：

- Accuracy；恰好$p_i=0.5$的样本计0.5 credit；
- NLL；仅在报告时将概率clamp到$[10^{-12},1-10^{-12}]$以避免`log(0)`，并报告clamp数；
- Brier；
- ECE-15；在$[0,1]$上用15个等宽bin，除最后bin右闭外都左闭右开，空bin贡献为0；
- $p_i$的均值、标准差、分位数及接近0/0.5/1的比例，以及$c_i=\max(p_i,1-p_i)$的confidence分布；
- `sum(p_i)`与`sum(1-p_i)`；
- selected step、本方法best `eval_selection_loss`和`score_type`。

描述性诊断的实现阈值不再留给运行者选择：$p$与$c$的分位数均为`{.01,.05,.25,.50,.75,.95,.99}`；接近0/0.5/1分别为$p\le.01$、$|p-.5|\le.01$、$p\ge.99$；confidence比例使用$c\ge\{.6,.7,.9,.99\}$；collapse同时报告极端概率比例$P(p\le.01\lor p\ge.99)$与A/B预测多数侧比例$\max\{P(p>.5),1-P(p>.5)\}$。这些阈值只用于无样本级描述，不进入selection、综合分数或方法排序。

对任一head的$p_i$与标签$z_i$，固定

$$
\mathrm{NLL}=-\frac1N\sum_i[z_i\log p_i+(1-z_i)\log(1-p_i)],
\qquad
\mathrm{Brier}=\frac1N\sum_i(p_i-z_i)^2,
$$

$$
\mathrm{ECE15}=\sum_{m=1}^{15}\frac{|B_m|}{N}
\left|\frac1{|B_m|}\sum_{i\in B_m}z_i-
\frac1{|B_m|}\sum_{i\in B_m}p_i\right|,
$$

其中空bin的项定义为0，NLL使用上述report-only clamp后的$p_i$。

输出分别标记`dpo_reference_delta_beta_0.1`与`raw_mean_logp_delta_beta_10`。跨方法只能在相同`score_type`内比较，两个head不合成单一综合分数；method-specific diagnostic loss不得作为final score。frozen base的reference-delta恒为0.5是定义上的基线事实，不作为失败。

### 7.1 Frozen-base / DPO-1K辅助headroom

训练前base不能只用恒为0.5的DPO reference-delta回答before/after。raw head因此同时承担不参与选点和调参的辅助headroom：对frozen base与selected DPO-1K在同一paired validation/test view上报告Accuracy、NLL、Brier、ECE-15和样本数。禁止把raw score冒充DPO objective或用于重选checkpoint。

### 7.2 Round4生成式benchmark登记：只登记、不运行

AlpacaEval 2.0 length-controlled win rate与MT-Bench平均分登记为Round4候选最终生成式benchmark。Round3明确排除：

- 不生成AlpacaEval或MT-Bench prompts对应的模型回答；
- 不安装或运行对应evaluator/judge流程；
- 不申请、读取或调用OpenAI或其他外部judge API；
- 不运行本地开源judge替代品；
- 不把任何上述指标列入Round3成功标准、产物清单或方法排序。

该登记不构成Round4实验批准。Round4必须另行冻结benchmark/version与revision、待评模型集合、prompt/chat template、generation参数、judge provider/model/snapshot、API可用性与费用上限、reference outputs、缓存/重试/失败策略、统计口径和历史榜单可比性；不得默认继承已deprecated的历史judge名称。

### 7.3 Round5 PE-static消融登记：只登记、不实现

`DPO+PE-static`登记为Round5消融候选。Round5必须另行冻结第二条固定candidate的来源和matched source/generator control；Round3不创建其data view、配置、代码入口或运行目录。该登记不构成Round5实验批准。

## 8. 预先判断、完整性失败与解释边界

- DPO-1K相对训练前模型的变化只按§7.1共同raw-score辅助口径解释；
- GitHub-loss SSPO、两个动态PE与DPO-1K的test差异分别报告；
- DPO-8K只报告paired-supervision gap，不称为全量oracle或同源hidden-label oracle；
- SSPO与DPO-1K同时改变labeled objective和unpaired机制，整体差异不能只归因于unpaired分支；
- 置信度变尖但Accuracy不升、Brier恶化或发生预测坍缩，不能解释为结构监督有效；
- SFT+rollout与rollout-only只支持“固定历史单回复锚点 vs纯在线rollout”的解释；
- 单种子只作探索性趋势，不宣称统计显著性。

以下属于实现完整性失败，而不是科学负面结果：

- 数据源/revision/split/hash未冻结、标签泄漏或split交叉；
- 任一训练step不满足预注册batch组成或不是epoch=1/250 steps；
- SSPO没有使用sequential running statistics、clamp、min-chosen threshold、prior-weighted$R_U$与scheduler，或混入论文KDE机制；
- SSPO checkpoint没有保存/恢复running state，或共同selection改写该state；
- 任一vLLM副本未ACK当前`method_id/optimizer_step/adapter_sha256`、使用stale adapter或跨step复用rollout，或rollout-only复制同一回复；
- loss/gradient非有限、selected checkpoint缺失或test参与选择；
- Round3生成AlpacaEval/MT-Bench回答、调用judge API或运行本地替代judge。

前一方法工程失败不自动取消后续方法，但必须独立记录失败阶段、错误摘要和已消耗规模。

## 9. 执行架构、规模、存储与产物

五个任务各为250 optimizer steps，各保留10个durable checkpoints，共50个。SFT+rollout训练生成7,000条，rollout-only训练生成14,000条；不存在checkpoint级diagnostic rollout额外生成。Round3的AlpacaEval/MT-Bench生成量与API预算均为0。

### 9.1 3×RTX 4090固定职责

- 五个方法都由GPU0上的单训练进程负责forward/backward/optimizer，不使用DDP或跨卡loss分片；
- DPO-1K、SSPO与DPO-8K运行时GPU1/2保持空闲，不为占满资源而改变方法语义；
- 两个动态PE方法运行时，GPU1运行vLLM replica 0，GPU2运行vLLM replica 1；GPU0顺序收集完整28-pair logical population，再以两遍或代数等价方式计算精确PE梯度；
- 物理subbatch大小只能由production-path strong smoke在不改变logical batch的前提下解析；DPO/selection在完整logical batch上精确聚合sample mean后才更新，SSPO必须在完整4 chosen、4 rejected、28 unpaired上各计算一次statistics并只按chosen→rejected→unpaired各更新running state一次。可用两遍重算或代数等价聚合，不得将物理subbatch当成新SSPO batch或中途optimizer step；
- rollout样本按`SHA256("round3-rollout-replica-v1\0" || method_id || "\0" || optimizer_step || "\0" || sample_id || "\0" || draw_index)`的摘要末位bit确定副本；每条文本只由一个副本生成，两副本不共享已生成文本；
- 每个optimizer step生成前，两副本都必须显式ACK同一`(method_id, optimizer_step, adapter_sha256)`。任一ACK缺失、不匹配或使用stale adapter都使该step fail closed，不得混合返回值继续训练。

训练前对五个方法共享的冻结reference所需log-prob统一预计算；cache放在Git仓库外，严格绑定model/tokenizer/template/data manifest。

### 9.2 环境、目录与磁盘门禁

- 训练环境固定在`<SERVER_BASE>/envs/round3-train/`，rollout环境固定在`<SERVER_BASE>/envs/round3-rollout/`；两者都在Git仓库外，不复用或修改Round2环境；
- train env使用Qwen3兼容的Transformers/PEFT/PyTorch栈，rollout env独立锁定vLLM栈；精确版本只能在获批后的服务器tests/strong smoke通过后写入lock与environment manifest；
- 五个任务使用独立Round3 run directory、resolved config和fact summary，不覆盖Round2或其他方法产物；
- 每个durable checkpoint至少保存adapter、optimizer/scheduler state、RNG state和global step；SSPO额外保存running statistics。全50个durable checkpoints都保留，不启用keep-N或其他自动pruner；
- vLLM每step的current-policy发布是带`adapter_sha256`的临时staging artifact，不冒充durable checkpoint。其原子发布和生命周期必须在代码交接时单独验收；本文不授权对当前服务器任何已有adapter/checkpoint执行删除或清理；
- formal前先用production-path strong smoke实际写出代表性checkpoint、reference cache和dynamic policy staging，据此计算`projected_peak_bytes`。仅当独立Round3执行路径的`free_bytes >= 2 * projected_peak_bytes`时允许formal；否则fail closed，不自动删除Round2或任何已有产物。

### 9.3 预期产物与回传边界

预期产物：

- 五份resolved config；
- source/派生view/reference-cache/checkpoint manifest与SHA-256；
- 每个checkpoint的共同`eval_selection_loss`与训练telemetry摘要；
- 五个best-checkpoint指针；
- 五份selected-checkpoint及frozen-base的独立997-pair双head聚合指标；
- SSPO state/checkpoint完整性摘要、dynamic rollout/replica synchronization聚合摘要；
- 失败、终止或无结论证据。

本地回传白名单仅包括无样本级内容的Markdown/JSON/CSV聚合摘要、汇总图、resolved config、manifest/hash、环境摘要和远程证据索引。原始数据、样本级prompt/response/rollout、逐样本预测、模型、adapter/checkpoint、cache、原始日志和平台日志留在执行面，不得复制到本地仓库。

## 10. 已闭合设计、执行时解析项与门禁

本版已关闭数据源/数量/确定性抽样、五方法构成、PE-static延后、DPO-1K/8K定义、统一250-step/checkpoint、$\gamma_{\min}$、共同selection、唯final test、GPU角色、环境隔离和存储门禁等设计分支。实现者不得从旧代码猜测或恢复已删除分支。

以下是只能在获批后的服务器preflight/tests/strong smoke中解析的运行事实，不是未决的科学设计：

1. 两个dataset repository的resolved full commit SHA、源parquet SHA-256、构造后manifest SHA-256和交叉隔离审计；
2. ModelScope实际下载的Qwen3-1.7B resolved revision及model/tokenizer/template manifest；
3. Round3 experiment ID、当时3×RTX 4090实时硬件/磁盘证据、精确dependency locks、最终源码commit和数值验收摘要；
4. strong-smoke实测的峰值显存、checkpoint/cache/staging尺寸、`projected_peak_bytes`及两倍空闲空间门禁结果。

Round4和Round5登记项不阻塞Round3设计闭合，也不能在Round3代码阶段顺手实现。2026-08-25批准的`r3-theory-v0.8`/`round3-exp-v1.3`假设`test_prefs`有2,000个有效pair；服务器冻结数据审计使该假设fail closed。2026-08-26用户明确表示“我也赞成B，请你本地修改”，据此批准`r3-theory-v0.9`/`round3-exp-v1.4`的数据勘误：保持1,000 validation、使用997 independent test、确定性隔离并审计畸形行、保持split级独立性且不从train补样。同日用户授权服务器推进至formal挂载；strong smoke先后暴露入口环境与CUDA非确定性实现缺陷，依照预注册门禁未放宽容差。v0.4修复现已获commit/push和重新执行授权，仍须完整strong smoke通过后才能启动formal。
