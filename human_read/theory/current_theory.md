# 当前理论：Round3 的 reference-relative PE 与 GitHub-loss SSPO

## 0. 版本、状态与历史边界

- Cycle：`cycle-20260818-01` / Round3
- 当前理论版本：`r3-theory-v1.1`
- 状态：**用户已于2026-08-27明确纠正：原始Theory v0.2的DPO implicit reward才是PE主方法；现有reference-free mean-logp动态臂只保留为`SimPO-reward PE`对照，并新增两个`DPO-reward PE`主方法**
- 当前唯一活动阶段：Round3 amendment `CODE_IMPLEMENTATION`；旧合同`round3-exp-v1.5`的formal仍按其exact commit自然运行，不允许热修改
- Round2边界：2026-08-26服务器只读证据确认controller已在step590停止、step580/589/590仍保留、两个pruner均未运行；不得删除其run/checkpoint
- Round3下游状态：旧五方法formal `round3-20260826-04`绑定`r3-theory-v1.0`/`round3-exp-v1.5`与exact commit `b1beef5550ac47c9c78b98c1729014cc153b1251`，作为不可改写的历史执行继续到终态。修订后的七方法合同为`round3-exp-v1.6`；只新增两个DPO-reward动态方法，不重跑或覆盖旧五方法
- 模型：ModelScope `Qwen/Qwen3-1.7B`（post-trained Instruct/hybrid-thinking版本，不是`-Base`）；七个任务共享同一冻结模型revision、初始化checkpoint与manifest
- 证据类型：SSPO论文/源码事实、官方数据预处理事实与本项目实验前设计，不包含Round3实验结果
- 历史理论：Round1/Round2 v0.2完整正文位于Git commit `d338eb5bedef16d83a42790c3faa97f8f404315b`；变化索引见`theory_changelog.md`

本文件只表达当前Round3主线；早期版本的完整内容只从历史索引读取，不在本文混排。

## 1. Round3研究问题与方法定位

Round3在同一个ModelScope `Qwen/Qwen3-1.7B`、同一冻结数据合同和单种子资源约束下，比较七种训练任务：`DPO-1K`、GitHub-loss SSPO、`DPO-8K`高标签reference、两个回溯登记的SimPO-reward动态PE对照，以及两个恢复原始理论定义的DPO-reward动态PE主方法。

核心问题是：在1,000条paired preference监督与7,000条独立UltraChat single-response数据下，SSPO的单回复伪标注机制和reference-relative动态PE的群体结构机制能否利用无标签数据；DPO implicit reward能否避免raw policy likelihood把响应来源当成偏好方向；固定公开单回复锚点是否相对于纯rollout提供额外信息；8,000条paired监督能给出多大的label-budget headroom。

`DPO+PE-static`从Round3删除。UltraChat一条记录只提供一个公开assistant response，强行补造第二条固定candidate会引入新的生成器或数据源混杂；该方法登记到Round5消融阶段重新设计，当前不实现、不运行，也不作为Round3缺失结果。

本轮SSPO仍只计划运行一个GitHub code-loss profile，不运行paper-v3 profile。由于unpaired数据从旧pair-derived方案改回作者仓库的UltraChat来源，旧method ID

`SSPO-code-loss-stratified-pair-derived@2df9e9a`

已撤回，不得进入resolved config。本版数据分支已关闭，实际method ID固定为

`SSPO-code-loss-stratified-ultrachat@2df9e9a`。

名称的三个限定不能省略：

- `code-loss`：采用作者GitHub trainer的实际loss，而不是论文v3的KDE/threshold-EMA机制；
- `stratified`：采用本项目固定1:7的two-stream batch，而不是仓库随机合并池sampler；
- `ultrachat`：unpaired singles来自作者仓库相同的`HuggingFaceH4/ultrachat_200k`来源，不再从UltraFeedback hidden pairs派生。

## 2. 共同数据、模型与公平性合同

作者GitHub commit `2df9e9a`的通用任务不是单一数据集，而是双源合同：paired preference来自`HuggingFaceH4/ultrafeedback_binarized`的`train_prefs/test_prefs`，unpaired single-response来自`HuggingFaceH4/ultrachat_200k`的`train_sft`。预处理脚本分别按`--fb`和`--ch`保留子集，再合并shuffle；论文通用设置报告10% UltraFeedback约6,113个paired records（含validation）和10% UltraChat约20,786个unpaired records。[GitHub preprocessing](https://github.com/MLAI-Yonsei/SSPO/blob/2df9e9a1d5fb9202a583cb66eb081e0cb60e873d/preprocessing_data/preprocessing_ultrachat.py#L260-L314)，[paper training data](https://arxiv.org/html/2511.00040v3#S5.SS1)。

2026-08-25用户明确关闭数据分支：Round3保持SSPO官方双源**类型**，只缩放数量。paired preference固定来自`HuggingFaceH4/ultrafeedback_binarized`，unpaired single-response固定来自`HuggingFaceH4/ultrachat_200k/train_sft`；不再考虑严格零UltraFeedback，也不再从UltraFeedback hidden pair派生unpaired数据。

旧“从冻结30k UltraFeedback抽取统一hidden-pair池”的方案永久退出Round3。当前规模合同为：

| role | 固定数量 | 来源与用途 |
| --- | ---: | --- |
| paired train master | 8,000 pairs | `ultrafeedback_binarized/train_prefs`；`DPO-8K`使用全部 |
| limited labeled view | 1,000 pairs | 上述8,000的确定性前缀子集；DPO-1K、SSPO与四个动态PE共享 |
| unpaired train | 7,000 singles | `ultrachat_200k/train_sft`；SSPO与四个动态PE共享source IDs |
| paired validation | 1,000 pairs | `ultrafeedback_binarized/test_prefs`；七方法共同checkpoint selection |
| paired independent test | 997 pairs | `ultrafeedback_binarized/test_prefs`；从同一冻结split的1,997个有效pair中排除1,000 validation后取全部剩余；selected-checkpoint-only test |

服务器对冻结revision `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`的只读审计确认：`test_prefs`共2,000行，其中source row indices `374/595/1846`的rejected assistant response为空，因此只有1,997个有效pair。用户明确选择方案B，不从`train_prefs`补3条，以保留final test完整的split级独立性。该变化只把最终评价样本数从1,000修正为997，不改变训练数据、七方法、selection或评分定义。

冻结源的畸形行在任何选择/去重前按确定性schema规则隔离，并以无原始文本的server-only audit记录dataset/split、source row index、sample/source ID、canonical prompt hash（若存在）和reason codes。实现必须精确核对冻结revision下的聚合事实：`train_prefs` 61,135/81 malformed/61,054 valid，`test_prefs` 2,000/3/1,997，`train_sft` 207,865/12,113/195,752；数量或reason aggregate变化即fail closed。未进入任何view的畸形prompt不加入跨view排除集，但其审计文件、aggregate与SHA-256必须写入data manifest。

`DPO-1K`与`DPO-8K`替代含混的`DPO-10`/`DPO-100`名称：1K是8K master pool的真子集，8K只是本轮高标签reference，不冒充UltraFeedback全量100%。数据仓库完整revision在获批后的服务器preflight解析为不可变full commit SHA并写入manifest；在此之前不从本地猜SHA，也不允许`main`漂移进入正式运行。

### 2.1 模型、模板与依赖边界

模型来源固定为ModelScope `Qwen/Qwen3-1.7B`，完整下载命令为：

```bash
modelscope download --model Qwen/Qwen3-1.7B --local_dir ./dir
```

命令中不得追加`README.md`，否则语义变成单文件下载。该模型是Qwen3经过post-training的对话模型，支持thinking/non-thinking切换；Round3统一使用原生Qwen3 chat template并显式设置`enable_thinking=false`，以匹配通用assistant回复并控制rollout长度。[Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B)。

所有方法必须共享同一resolved model revision、模型文件SHA-256、tokenizer/chat-template文件SHA-256和special-token manifest。Qwen3要求`transformers>=4.51.0`，而SSPO仓库原始requirements固定到`transformers==4.46.1`；因此“采用GitHub SSPO实现”解释为移植并保持commit `2df9e9a`的loss/trainer语义，不能原封不动沿用其不兼容依赖栈。[Qwen config](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json)，[SSPO requirements](https://github.com/MLAI-Yonsei/SSPO/blob/2df9e9a1d5fb9202a583cb66eb081e0cb60e873d/requirements.txt)。

### 2.2 DPO精确定义

对应用chat template后的response tokens定义序列总log-prob

$$
s_\theta(x,y)=\sum_{t\in\mathrm{response}}\log\pi_\theta(y_t\mid x,y_{<t}).
$$

reference固定为与七方法共同初始化完全相同、从不更新的Qwen3-1.7B。对随机A/B位置和标签$z_i\in\{0,1\}$（$z_i=1$表示A更优），定义

$$
d_i^{\mathrm{DPO}}
=
\beta_{\mathrm{DPO}}
\left[
(s_\theta(x_i,y_{iA})-s_{\mathrm{ref}}(x_i,y_{iA}))
-(s_\theta(x_i,y_{iB})-s_{\mathrm{ref}}(x_i,y_{iB}))
\right],
\qquad \beta_{\mathrm{DPO}}=0.1,
$$

$$
L_{\mathrm{DPO}}
=
\operatorname{mean}_i\left[
-z_i\log\sigma(d_i^{\mathrm{DPO}})
-(1-z_i)\log\sigma(-d_i^{\mathrm{DPO}})
\right].
$$

DPO-1K、DPO-8K和四个动态PE方法的labeled分支都使用这一相同定义；不得把SSPO的reference-free SimPO labeled loss代入PE方法。

### 2.3 PE精确定义与两个reward profile

原始Theory v0.2及本轮主方法对方向未知candidate pair使用DPO implicit reward。单响应reward与pair probability固定为

$$
r_\theta^{\mathrm{DPO}}(x,y)
=\beta_{\mathrm{PE}}
\left[s_\theta(x,y)-s_{\mathrm{ref}}(x,y)\right],
\qquad \beta_{\mathrm{PE}}=0.1,
$$

$$
p_i^{\mathrm{DPO}}
=\sigma\!\left(
r_\theta^{\mathrm{DPO}}(x_i,y_{iA})
-r_\theta^{\mathrm{DPO}}(x_i,y_{iB})
\right).
$$

这里$s_\theta$与$s_{\mathrm{ref}}$均为§2.2定义的response-token总log-prob。初始化时$\pi_\theta=\pi_{\mathrm{ref}}$，因此任意candidate pair都有$p_i^{\mathrm{DPO}}=0.5$；在完全对称的28-pair population上，初始PE分支也不提供破坏对称性的方向，第一步由同一个joint objective中的labeled DPO分支推动$\theta$离开reference。随后DPO-reward PE才响应各candidate相对reference的变化，正负号表示相对上调幅度差异，而不是policy对自身样本的raw熟悉程度。

旧`round3-exp-v1.5`已经执行的两个动态臂使用了另一个profile，现回溯明确命名为`SimPO-reward PE`对照：

$$
q_\theta(x,y)=\operatorname{mean\_response\_logp}_\theta(y\mid x),
\qquad
p_i^{\mathrm{SimPO}}
=\sigma\!\left(10[q_\theta(x_i,y_{iA})-q_\theta(x_i,y_{iB})]\right).
$$

该profile是reference-free raw mean-logp，不是原始PE主方法。它在SFT+rollout中可能通过“current-policy rollout与policy同源、SFT anchor异源”的来源捷径产生极端分离；因此只能作为reward-definition消融，不能再简称为未限定的`DPO+PE`。

令$\mathbf q_i=[p_i,1-p_i]^\top$、$\mathbf e_+=[1,0]^\top$、$\mathbf e_-=[0,1]^\top$，则

$$
\widehat{\mathbf e}_+
=\frac{\sum_i p_i\mathbf q_i}{\sum_i p_i+\epsilon},
\qquad
\widehat{\mathbf e}_-
=\frac{\sum_i(1-p_i)\mathbf q_i}{\sum_i(1-p_i)+\epsilon},
\qquad \epsilon=10^{-8},
$$

$$
L_{PE}
=\frac12\left(
\lVert\widehat{\mathbf e}_+-\mathbf e_+\rVert_1
+\lVert\widehat{\mathbf e}_--\mathbf e_-\rVert_1
\right).
$$

denominator不detach；$p_i$同时作为soft responsibility和编码分量，但不是instance-level target。$L_{PE}$必须在一次optimizer step完整的28个candidate pairs上精确计算，不能分别在物理microbatch计算后取平均。若执行使用多卡或顺序subbatch，必须通过两遍或代数等价方式得到同一个logical-population梯度。

四个动态PE方法统一使用

$$
L_{\mathrm{joint}}
=
\frac{L_{\mathrm{DPO}}+0.1L_{PE}}{1.1}.
$$

### 2.4 共同优化与PE权重

七个任务统一使用非量化LoRA，不使用QLoRA：

| 项目 | Round3固定值 |
| --- | --- |
| LoRA | rank 8，alpha 16，dropout 0 |
| target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`；排除embedding与`lm_head` |
| optimizer | `adamw_torch`，weight decay 0，betas $(0.9,0.999)$，epsilon $10^{-8}$ |
| scheduler | cosine，warmup ratio 0.1，max grad norm 1.0 |
| learning rate | 七方法统一$10^{-5}$ |
| precision | BF16 forward/autocast；`fp16=false`、`pure_bf16=false`，LoRA trainable weights与optimizer state保留FP32 |

LoRA rank 8、all-linear target、$10^{-5}$、cosine、warmup 0.1与clip 1来自SSPO仓库设置；alpha 16、dropout 0和AdamW其余默认值来自其固定LLaMA-Factory/Transformers语义。七方法统一学习率是本项目在不做method-specific sweep时的公平性选择，不冒充SSPO论文分别调参的原始方案。[SSPO config generator](https://github.com/MLAI-Yonsei/SSPO/blob/2df9e9a1d5fb9202a583cb66eb081e0cb60e873d/examples/train/make_yaml.py)。

四个动态PE方法固定§2.3的normalized $\lambda_{PE}=0.1$，有效权重约为$0.9091/0.0909$。两个reward profile只改变构造$p_i$的单响应reward、归一化与$\beta_{PE}$；labeled DPO、PE aggregation、联合权重、数据和rollout均不变。该值是本项目预注册选择，不是SSPO、Qwen、UltraFeedback或UltraChat给出的参数；它与SSPO的scheduler decay $\lambda=0.001$是两个完全不同的量。

### 2.5 序列与rollout支持

七方法统一使用：

$$
L_{\max}=2048,\qquad L_{\mathrm{prompt}}\leq1024,\qquad L_{\mathrm{completion}}\leq1024.
$$

`2048`是本项目为降低固定UltraFeedback chosen/rejected截断率所作的稳健性选择。当前官方UltraFeedback Binarized数据卡定义数据构造、字段和split，但不规定最大序列长度；SSPO仓库示例和Zephyr DPO recipe均使用1024，因此不能把本轮2048写成其官方设定。[UltraFeedback Binarized card](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)，[Zephyr DPO recipe](https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-beta/dpo/config_full.yaml)。

固定chosen/rejected、SSPO single和UltraChat固定anchor均在应用同一chat template后遵守总长2048、prompt 1024、completion 1024的上限，loss只作用于response tokens。prompt超限时保留末端1024 tokens，completion超限时保留前端1024 tokens；两类截断必须分别计数。

动态rollout不允许先超长生成再静默裁剪。生成前先把formatted prompt限制为1024，再设置

$$
\texttt{max\_new\_tokens}=\min\{1024,\,2048-L_{\mathrm{prompt}}\}.
$$

遇到EOS时自然结束；达到长度上限的回复仍可进入loss，但必须标记`finish_reason=length`。训练与selection分别汇总prompt truncation、completion cap-hit、EOS和实际response-token统计；长度上限相同不代表七方法token compute相同。

## 3. 必须分别记录的两种SSPO

### 3.1 共用符号

定义

$$
D_L=\{(x^{(i)},y_w^{(i)},y_l^{(i)})\}_{i=1}^{n_L},
\qquad
D_U=\{(x_u^{(j)},y_u^{(j)})\}_{j=1}^{n_U}.
$$

SSPO使用reference-free、长度归一化的policy log-prob：

$$
q_\theta(x,y)=\operatorname{mean\_response\_logp}_\theta(y\mid x).
$$

$q_\theta$或由它缩放得到的$r_\theta$是单条response得分；$R_{D_L}$和$R_{D_U}$分别是labeled与unpaired数据上的平均risk，不能把risk和reward混称。

### 3.2 `SSPO-paper-v3`：只作论文事实记录

论文labeled risk为SimPO：

$$
R_{D_L}(f_\theta)
=
\frac{1}{n_L}\sum_i
-\log\sigma\!\left(
r_\theta(x^{(i)},y_w^{(i)})
-r_\theta(x^{(i)},y_l^{(i)})
-\Delta
\right),
$$

其中$r_\theta=\beta q_\theta$。论文Algorithm 1每步分别抽取$B_L\subset D_L$和$B_U\subset D_U$，但没有规定固定的数字比例；因此Round3的`4+28`不是论文原始设定。[SSPO v3 Eq. (4)](https://arxiv.org/html/2511.00040v3#S3.E4)，[Algorithm 1](https://arxiv.org/html/2511.00040v3#A3.SS4)。

论文v3对当前labeled winning/losing和unpaired rewards使用joint batch mean/std并做EMA；在标准化winning/losing rewards上分别做Gaussian KDE，通过prior-weighted Bayes risk选择threshold，再对threshold做EMA。unpaired response以

$$
\widetilde s_k=\mathbb I\{z_k>\widehat\delta_t\}
$$

得到hard pseudo-label，并计算

$$
R_{D_U}(f_\theta)
=
\frac1{n_U}\sum_k
\left[
\widetilde s_kp_+[-\log\sigma(d_k)]
+(1-\widetilde s_k)p_-[-\log\sigma(-d_k)]
\right],
$$

其中$d_k=z_k-\widehat\delta_t$。论文联合目标为

$$
\mathcal L_t
=
\gamma_tR_{B_L}+(1-\gamma_t)R_{B_U},
\qquad
\gamma_t=\max\{\gamma_{\min},\gamma_0e^{-\lambda t}\}.
$$

论文机制来源：[practical threshold](https://arxiv.org/html/2511.00040v3#S4.SS2)，[adaptive scheduler](https://arxiv.org/html/2511.00040v3#S4.SS3)。本轮不运行该paper-v3 profile，不能把它的KDE、Bayes threshold或threshold EMA混入实际SSPO臂。

### 3.3 `SSPO-code-main@2df9e9a`：源码事实

公开仓库commit `2df9e9a1d5fb9202a583cb66eb081e0cb60e873d`早于本地论文v3。仓库数据管线把labeled rows和unpaired rows合并shuffle，随机batch再由collator按`data_types`拆分；它不保证固定labeled/unpaired数量。[preprocessing](https://github.com/MLAI-Yonsei/SSPO/blob/2df9e9a1d5fb9202a583cb66eb081e0cb60e873d/preprocessing_data/preprocessing_ultrachat.py#L292-L343)，[collator](https://github.com/MLAI-Yonsei/SSPO/blob/2df9e9a1d5fb9202a583cb66eb081e0cb60e873d/src_sspo/llamafactory/data/collator.py#L184-L259)。

源码default SimPO labeled risk为

$$
R_{B_L}^{\mathrm{code}}
=
\operatorname{mean}\left[
-\log\sigma\bigl(\beta(q_w-q_l)-\Delta\bigr)
\right].
$$

其unpaired路径与论文v3不同：

1. 按chosen、rejected、unpaired调用顺序分别更新同一份running mean/variance，而非一次joint batch更新；
2. 使用running variance开方，并把normalized values clamp到$[-5,5]$；
3. threshold为当前normalized chosen values的最小值

$$
\delta_{\mathrm{code}}=\min_i z_{w,i};
$$

4. 不使用KDE、Bayes grid search或threshold EMA；
5. 对unpaired response定义

$$
d_k^{\mathrm{code}}
=
\beta(z_{u,k}-\delta_{\mathrm{code}}),
$$

并计算

$$
\ell_{U,k}^{\mathrm{code}}
=
\begin{cases}
p_+[-\log\sigma(d_k^{\mathrm{code}})],&d_k^{\mathrm{code}}>0,\\
p_-[-\log\sigma(-d_k^{\mathrm{code}})],&d_k^{\mathrm{code}}\leq0.
\end{cases}
$$

最终目标仍为

$$
\mathcal L_t^{\mathrm{code}}
=
\gamma_tR_{B_L}^{\mathrm{code}}
+(1-\gamma_t)R_{B_U}^{\mathrm{code}}.
$$

GitHub初始化与更新语义现明确冻结为：`reward_norm_momentum=0.95`、`running_mean=None`、`running_var=None`、`reward_clip_range=5.0`。第一次非空调用用该调用的`batch_mean`和`batch_var`直接初始化；`batch_var=var(unbiased=False)+1e-8`。随后按chosen→rejected→unpaired的调用顺序分别执行

$$
m\leftarrow0.95m+0.05m_B,
\qquad
v\leftarrow0.95v+0.05v_B,
$$

并以$\sqrt v$标准化、clamp到$[-5,5]$。scheduler使用trainer进入当前loss时的`state.global_step`，因此第一optimizer update前$t=0$、$\gamma_0=1$。不得用零初始化、预热batch、joint初始化、unbiased variance或不同调用顺序替换。源码事实见[trainer initialization](https://github.com/MLAI-Yonsei/SSPO/blob/2df9e9a1d5fb9202a583cb66eb081e0cb60e873d/src_sspo/llamafactory/train/dpo/trainer.py#L87-L103)与[trainer loss lines 183–261](https://github.com/MLAI-Yonsei/SSPO/blob/2df9e9a1d5fb9202a583cb66eb081e0cb60e873d/src_sspo/llamafactory/train/dpo/trainer.py#L183-L261)。仓库通用示例使用$\beta=10,\Delta=2,p_+=p_-=0.5,\gamma_0=1,\lambda=0.001$。

## 4. Round3实际SSPO合同

### 4.1 数据来源

SSPO的$D_U$单位是单个prompt–response，不是方向未知response pair。Round3不再从本项目UltraFeedback pair中选择A/B，而是固定采用作者仓库相同的`HuggingFaceH4/ultrachat_200k` `train_sft`来源。GitHub预处理取每条记录的`prompt`及首条assistant content，形成独立unpaired response。

正式数据合同仍须冻结dataset revision、split、canonical row schema、去重/交叉污染规则和确定性子集SHA-256。公开训练入口只能读取`sample_id,prompt,response`，不得包含任何伪造的chosen/rejected字段或方向标签。

### 4.2 比例分层适配

每个optimizer step固定使用

$$
4\ \text{labeled pairs}+28\ \text{unpaired singles}.
$$

一个labeled pair含两条response，一个unpaired record含一条，因此每步共有$2\times4+28=36$条response sequences参与loss。`4+28`是本项目固定的resource-scaled stratified sampler，不是论文或GitHub仓库的原始固定数字；在1,000/7,000规模下恰好产生250 steps。

固定4个labeled winners确保每个step的min-chosen threshold有定义，并消除仓库随机小batch可能进入的no-labeled fallback。代价是该臂不能称作GitHub sampler原样复现。

### 4.3 loss与scheduler

实际SSPO严格使用§3.3的code loss：sequential running statistics、clamp、min-chosen threshold、prior-weighted$R_U$和curriculum scheduler。采用仓库示例的

$$
\beta=10,\quad \Delta=2,\quad p_+=p_-=0.5,
$$

以及GitHub规则中的

$$
\gamma_{\min}=\frac{n_L}{n_L+n_U},
\qquad
\gamma_t=\max\{\gamma_{\min},e^{-0.001t}\},
\qquad t=\texttt{state.global_step},\ t_\text{first}=0.
$$

本轮固定$n_L=1{,}000,n_U=7{,}000$，因此$\gamma_{\min}=0.125$，不再是候选值。保持GitHub decay $0.001$与首步$t=0$；在250步短程中floor不会实际触发，这是忠实保留GitHub scheduler后可预期的行为，不得为了让unpaired权重更大而偷偷重标度decay。

任何KDE、Bayes threshold、threshold EMA或joint reward-statistics替换都会产生另一个hybrid方法，不得仍命名为本profile。

## 5. 七个训练任务与统一250-step合同

| # | 方法 | train view | 每个optimizer step | epoch | steps | 一轮总暴露/生成 |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 1 | DPO-1K | 1,000 paired preference pairs | 4 labeled pairs | 1 | 250 | 1,000 pairs |
| 2 | SSPO code-loss | 同一1,000 labeled +7,000 UltraChat singles | 4 labeled pairs +28 singles | 1 | 250 | 1,000 pairs +7,000 singles |
| 3 | DPO-8K | 8,000 paired master pool | 32 labeled pairs | 1 | 250 | 8,000 pairs |
| 4 | DPO+PE-SimPO-reward-SFT+rollout | 同一1,000 labeled +7,000 UltraChat anchors/prompts | 4 labeled +28 candidate pairs | 1 | 250 | 生成7,000条rollout |
| 5 | DPO+PE-SimPO-reward-rollout-only | 同一1,000 labeled +7,000 UltraChat prompts | 4 labeled +28 candidate pairs | 1 | 250 | 生成14,000条rollout |
| 6 | DPO+PE-DPO-reward-SFT+rollout | 与方法4相同 | 4 labeled +28 candidate pairs | 1 | 250 | 独立生成7,000条rollout |
| 7 | DPO+PE-DPO-reward-rollout-only | 与方法5相同 | 4 labeled +28 candidate pairs | 1 | 250 | 独立生成14,000条rollout |

整数对齐为$1000/4=7000/28=8000/32=250$。七个任务共享模型初始化、同一1K labeled子集与同一顺序；DPO-8K使用包含该1K子集的完整master pool，因此只改变paired-label预算。四个动态方法两两形成matched reward-profile comparison：candidate构造相同时，只改变§2.3的`SimPO-reward`或`DPO-reward`。不同方法的rollout仍由各自current policy独立生成，因此这是训练轨迹级对照，不是逐样本paired test。SSPO与DPO-1K仍同时存在SimPO-vs-reference-DPO的labeled-objective差异，其总体差异不能只归因于unpaired分支。

## 6. 共同checkpoint selection

训练objective和checkpoint-selection objective必须分离。七个方法固定在step 25、50、75、100、125、150、175、200、225、250保存durable checkpoint并计算共同`eval_selection_loss`；step250与final是同一对象，不重复保存。

七个方法统一在同一份冻结的1,000-pair labeled validation view上计算同一个`eval_selection_loss`：

$$
L_{\mathrm{eval\_select}}
=\frac1N\sum_i\left[-z_i\log\sigma(d_i^{\mathrm{DPO}})-(1-z_i)\log\sigma(-d_i^{\mathrm{DPO}})\right],
$$

其中$d_i^{\mathrm{DPO}}$严格采用§2.2的冻结初始Qwen3-1.7B reference、response总log-prob和$\beta_{\mathrm{DPO}}=0.1$。所有方法共享相同pairs、A/B换位、截断、batch顺序和样本加权；该loss只用于选点，不把SSPO或PE的训练目标改写成DPO。

`ultrafeedback_binarized/test_prefs`固定2,000条source rows；先隔离3条空rejected畸形行，再在1,997个有效pair上按namespace `round3-paired-validation-v3`的SHA-256排序选择1,000条validation，剩余997条按独立namespace `round3-paired-independent-test-v3`排序并精确断言全部进入test。精确canonicalization、审计、去重与A/B换位合同写在experiment v1.4。

Round3不运行checkpoint级SSPO/PE objective diagnostic，也不运行dynamic rollout diagnostic panel。训练时仍记录loss components、$p$熵/极端比例、SSPO threshold/pseudo-positive rate和rollout长度等聚合telemetry；SFT+rollout还记录方向对齐后的rollout-vs-anchor硬胜数/率与软正例质量，但不保存逐样本分数。这些都不是eval、不参与选点。该删除使Round3除共同selection外只保留§7的独立997-pair final test。

SSPO checkpoint仍必须显式序列化并恢复`running_mean`与`running_var`；共同selection evaluator只能读取policy/reference log-prob，不调用SSPO loss、不更新running state。selection前后state hash必须完全相同，missing state在checkpoint round-trip验收时fail closed。下一batch round-trip由同一checkpoint独立重载两次：running state、scheduler/global step精确一致，loss绝对差`<=1e-7`，trainable LoRA更新后的最大绝对差`<=1e-7`且最大相对差`<=1e-6`；超限按工程失败处理，不构成调整科学合同的理由。

每个方法只在finite checkpoint中选择最低共同`eval_selection_loss`；排序键固定为原始未四舍五入的`(eval_selection_loss, step)`，因此loss完全相同时选择更早step。任一selection batch或aggregate出现NaN/Inf，该checkpoint无效但继续评价其他checkpoint；若某方法全部checkpoint均无效，则该方法记为工程失败且不进入test。训练loss或gradient非有限时立即终止该方法；禁止把非有限值替换为0、跳过后继续更新或用test补选checkpoint。诊断量非有限必须记录并判定对应诊断失败，但不能用另一个诊断量替代共同selection loss。

## 7. Round3唯一final test与Round4延期登记

Round3唯一final evaluation是独立997-pair fixed-pair test。每个方法只有按§6共同`eval_selection_loss`选出的一个checkpoint进入test；frozen base作为共同参照也在同一view上评分。Test直接评分冻结的有标签A/B preference pairs，不为动态方法再次生成rollout，也不参与checkpoint或超参选择。

因此当前test没有`temperature`或其他generation sampling参数。“PE rollout与test rollout保持一致”不适用于本轮；需要一致的是四个动态PE训练的sampling合同。

每个模型都必须在同一批997 pairs上同时输出两个score head，不能再让DPO臂只用reference-delta、SSPO/PE臂只用raw mean-logp后横向比较：

$$
p_i^{\mathrm{ref}}
=\sigma\!\left(0.1[(s_\theta^A-s_{\mathrm{ref}}^A)-(s_\theta^B-s_{\mathrm{ref}}^B)]\right),
$$

$$
p_i^{\mathrm{raw}}
=\sigma\!\left(10[q_\theta(x_i,y_{iA})-q_\theta(x_i,y_{iB})]\right).
$$

每个head分别报告Accuracy、NLL、Brier、ECE、confidence distribution、$\sum_i p_i$、$\sum_i(1-p_i)$与坍缩诊断；恰好$p_i=0.5$的accuracy计0.5 credit。实现口径固定为：$p$与$c=\max(p,1-p)$都报告分位数`{.01,.05,.25,.50,.75,.95,.99}`；`near-zero/near-half/near-one`分别定义为$p\le.01$、$|p-.5|\le.01$、$p\ge.99$；confidence阈值固定为`{.6,.7,.9,.99}`；collapse diagnostics固定为$p\le.01$或$p\ge.99$的比例，以及A/B预测多数侧比例$\max\{P(p>.5),1-P(p>.5)\}$。这些只作预定义描述性诊断，不参与选点、排名或成功判定。跨方法比较只能在相同`score_type`内进行，不合成单一综合分数，也不使用test重选checkpoint、超参或方法。frozen base的reference-delta恒为0.5是定义上的基线事实，不作为失败。

训练前base与DPO-1K不能只使用DPO reference-delta做before/after，因为当policy等于reference时该delta恒为0、概率恒为0.5。Round3的raw head因此同时承担base/DPO-1K辅助headroom：在同一冻结paired validation/test view上，对frozen base和selected DPO-1K使用

$$
p_i^{\mathrm{raw}}=\sigma\!\left(10[q_\theta(x_i,y_{iA})-q_\theta(x_i,y_{iB})]\right)
$$

计算相同口径的Accuracy/NLL/Brier/ECE。两个head都对所有方法报告并分别命名；辅助headroom不得用于重选checkpoint或超参。

### 7.1 Round4生成式benchmark登记（本轮禁止执行）

AlpacaEval 2.0 length-controlled win rate与MT-Bench平均分登记为Round4候选生成式benchmark。Round3不生成其prompts对应的模型回答，不安装或调用评测器，不申请、读取或使用judge API key，不发起OpenAI或其他外部裁判请求，也不运行本地替代judge。

该登记仅保存后续研究意图，不是Round4设计批准或执行授权。Round4必须重新讨论并冻结：benchmark/version与数据revision、待评模型集合、prompt/chat template、generation参数、judge provider/model/snapshot、API可用性与费用上限、reference outputs、缓存/重试/失败策略、统计口径及与历史榜单的可比性。旧`gpt-4-1106-preview`或`gpt-4`配置不得在未核验可用性时被默认继承。

### 7.2 Round5 PE-static消融登记（本轮禁止实现）

`DPO+PE-static`登记到Round5消融候选。Round5必须单独回答第二条冻结candidate来自何处，并设置matched source/generator control；Round3不创建其数据view、配置、代码入口或运行目录。本登记不构成Round5设计批准。

## 8. 可证伪预测与解释边界

| 编号 | 预测 | 必须观察 | 削弱或反驳条件 |
| --- | --- | --- | --- |
| S1 | code-loss threshold受每步4个winning rewards中的最小值控制 | min chosen、threshold、pseudo-positive rate | 实际threshold不等于源码定义 |
| S2 | 固定4+28消除no-labeled fallback，但不能消除4-sample极值方差 | batch composition、fallback count、threshold variation | 任一步不是4+28或仍触发fallback |
| S3 | labeled separation未建立时，不能把unpaired分支解释为可靠偏好监督 | labeled reward accuracy/separation、$R_L$、$R_U$ | 无labeled separation却宣称可靠pseudo signal |
| S4 | 共同labeled DPO selection loss可公平选点，但不能替代独立test | best step/selection loss与selected-test metrics | 用train-objective或test重选checkpoint |
| S5 | 两个rollout副本必须消费同一current-policy版本 | optimizer step、adapter hash、replica ACK | 任一副本使用旧adapter、跨step混合或缺失hash证据 |
| P1 | SimPO-reward SFT+rollout可能利用响应来源捷径，而非偏好质量 | source-aligned rollout胜率、$p$熵/极端率、PE loss | 初始化附近不出现来源分离或DPO-reward同样立即饱和 |
| P2 | DPO-reward在初始化时精确给出$p_i=0.5$，随后只响应相对reference的策略变化 | step-0/step-1 reference delta、finite与对称性审计 | policy=reference时任一pair不为0.5 |
| P3 | 在相同candidate构造下，DPO-reward PE应比SimPO-reward PE减少source-aligned饱和 | 两profile的source-aligned hard/soft win、entropy、extreme rate与test | DPO-reward仍从首步由来源完美分离且无实现错误 |

Round3是单模型、单种子的探索性比较，不能宣称统计显著性。置信度变尖而Accuracy不升、Brier恶化或预测坍缩，不支持结构监督有效。SFT+rollout与rollout-only的比较只解释为“固定历史单回复锚点 vs纯在线rollout”，不能外推为一般SFT语料收益；SimPO-reward与DPO-reward的比较只解释reward定义，不把独立生成轨迹冒充逐样本配对因果效应。

本轮不运行paper-v3与code-main直接对照，因此结果不能回答KDE是否优于min-chosen threshold。

## 9. 已固定、执行时解析与批准边界

已固定：

- 七任务研究问题骨架、ModelScope `Qwen/Qwen3-1.7B` Instruct共同初始化与native non-thinking template；
- SSPO官方双源类型：UltraFeedback Binarized paired +UltraChat 200k unpaired；8K paired master、其内嵌1K limited view、7K singles、1K validation与997-pair independent test；
- `DPO-1K`与`DPO-8K`命名及嵌套关系；PE-static移至Round5；
- DPO与PE的完整数学定义；DPO-reward PE为原始主方法，SimPO-reward PE为回溯命名对照；PE exact logical-population梯度和normalized $\lambda_{PE}=0.1$；
- SSPO GitHub code-loss、running statistics初始化/更新、min-chosen threshold与$t=0$ scheduler语义；
- 七方法全部1 epoch/250 optimizer steps，checkpoint steps固定25至250每25步一次，$\gamma_{\min}=0.125$；
- 训练objective与selection objective分离；七方法共同使用冻结1,000-pair labeled DPO validation loss，Round3不运行checkpoint级SSPO/PE或dynamic diagnostic；
- LoRA r8/alpha16/dropout0/all Qwen linears、非量化BF16 mixed precision、AdamW/cosine/warmup/clip与七方法统一lr $10^{-5}$；
- 总序列2048、prompt/completion各1024上限及rollout截断审计；
- non-thinking train rollout sampling与seed角色；
- finite-only checkpoint selection、较早step tie-break、non-finite policy与SSPO checkpoint-state round-trip验收；
- selected-checkpoint-only independent 997-pair fixed-pair test；所有模型同时报告reference-delta与raw mean-logp两种score head，并只在同head内比较；
- AlpacaEval 2.0与MT-Bench只登记为Round4候选，Round3禁止生成、judge API调用和本地替代judge；
- PE-static只登记为Round5消融候选，Round3禁止实现或运行；
- Round3目标执行机器为当前3×RTX 4090服务器；第一资源波次在三个独立单卡进程中并发运行DPO-1K/GPU0、SSPO/GPU1和DPO-8K/GPU2，各自只见逻辑`cuda:0`且共享输入只读、运行目录隔离；并发波次必须通过三卡确定性重放与production-path smoke；
- 四个动态方法各自独占三卡并严格串行：GPU0训练，GPU1/2运行两份独立vLLM replica并由step/adapter hash/ACK屏障同步；四个方法不得共享current-policy rollout。旧SimPO-reward两臂在`round3-20260826-04`运行；新增DPO-reward两臂只能在旧controller终态后以新experiment ID与exact commit运行；
- 独立train/rollout环境、全部十个durable checkpoints保留、无自动pruner；formal前按strong-smoke实测尺寸计算projected peak，空闲空间不足两倍projected peak时fail closed且不删除Round2产物。

执行前解析：

- 两个dataset仓库的resolved full commit SHA、源parquet SHA-256、构造后manifest与交叉隔离审计；
- ModelScope下载实际resolved revision与模型/tokenizer文件manifest；
- Round3 experiment ID、服务器实测3×4090硬件证据、精确dependency lock、源码commit与projected storage peak。

用户已明确确认补回DPO/PE合同、共同checkpoint-selection与双score-head final test口径、保持GitHub SSPO初始化并指定3×4090目标服务器；又明确采用SSPO官方双源类型和本项目缩放数量、从Round3删除PE-static并登记到Round5。2026-08-25批准的`r3-theory-v0.8`/`round3-exp-v1.3`因冻结数据实际只有1,997个有效held-out pairs而触发fail-closed。2026-08-26用户明确表示“我也赞成B，请你本地修改”，据此形成`r3-theory-v0.9`/`round3-exp-v1.4`：保持1,000 validation、改用997 independent test、畸形行确定性审计且不从train补样。同日用户进一步明确认可三个静态方法并发、两个动态方法三卡串行的资源安排，形成`r3-theory-v1.0`/`round3-exp-v1.5`并挂载五方法formal。2026-08-27用户发现动态PE不应使用reference-free mean-logp，明确要求恢复早期获批Theory v0.2的$\pi_\theta/\pi_{\mathrm{ref}}$定义；现有两臂回溯登记为SimPO-reward PE，并新增两个DPO-reward PE，形成`r3-theory-v1.1`与七方法实验修订。Round4保持锁定，不得在本修订完成前开启。
