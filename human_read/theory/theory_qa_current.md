# Round3 Theory QA：当前决定与执行时解析项

## 状态

- 理论版本：`r3-theory-v0.9`
- 实验草案：`round3-exp-v1.4`
- 日期：2026-08-26
- 状态：用户已于2026-08-26明确批准方案B数据勘误与本地修改；当前唯一活动阶段为Round3 `CODE_IMPLEMENTATION`
- 完整理论：`current_theory.md`
- 历史QA：`theory_qa_archive.md`及Git commit `d338eb5bedef16d83a42790c3faa97f8f404315b`

本文只做Round3快速问答索引，不重复完整推导。

## 已讨论清楚

| 问题 | 当前决定 |
| --- | --- |
| SSPO论文与源码是否相同 | 不同；paper-v3使用KDE/Bayes threshold/threshold EMA，GitHub commit `2df9e9a`使用sequential running statistics、clamp和min-chosen threshold |
| Round3实际运行哪个SSPO | 只运行`SSPO-code-loss-stratified-ultrachat@2df9e9a`；不运行paper-v3或旧`pair-derived` profile |
| 数据源与数量 | 8K `ultrafeedback_binarized/train_prefs` paired master，其内嵌1K limited view；7K `ultrachat_200k/train_sft` singles；冻结`test_prefs`先隔离3条空rejected行，再确定性分为1K validation +997 independent test |
| 畸形行如何处理 | 选择前按schema确定性隔离；精确核对source/valid/malformed/reason aggregate，server-only audit记录source index和SHA但不含原始文本；数量变化fail closed，不从train补test |
| 如何冻结子集 | full dataset revision在获批服务器preflight解析；使用独立namespace、seed 42和SHA-256排序；1K labeled是8K master的真子集 |
| `4 labeled +28 unpaired`来自哪里 | 是本项目Round3的固定分层sampler，不是论文或GitHub原始固定数字 |
| SSPO GitHub初始化 | momentum .95，running mean/var为`None`；首次chosen batch直接初始化，batch variance用`unbiased=False`并加$10^{-8}$；之后chosen→rejected→unpaired顺序EMA，clamp $[-5,5]$；首步$t=0$ |
| SSPO scheduler | $\gamma_{min}=1000/(1000+7000)=0.125$，decay .001；250步内floor不会触发，但不为增大unpaired权重而修改GitHub语义 |
| Round3有几个方法 | 五个：DPO-1K、GitHub-loss SSPO、DPO-8K、DPO+PE-SFT+rollout、DPO+PE-rollout-only |
| epoch与steps | 五方法均1 epoch /250 optimizer steps；每25步保存，各保留10个durable checkpoints |
| PE-static | 从Round3删除，登记到Round5消融；本轮不设计第二candidate、不实现、不运行 |
| DPO精确定义 | response总log-prob、冻结初始Qwen reference、$\beta_{DPO}=0.1$；PE方法labeled branch也用该DPO，不用SimPO替换 |
| PE精确定义 | mean-response-logp、margin-free $\beta_{PE}=10$、L1、epsilon $10^{-8}$、denominator不detach；完整28-pair logical population精确梯度；normalized $\lambda_{PE}=0.1$ |
| checkpoint如何选择 | 五方法都在同一冻结1K-pair validation上用reference-DPO beta .1的`eval_selection_loss`选点；不用PE/joint/SSPO objective loss |
| 是否运行方法内checkpoint diagnostics | 不运行SSPO/PE objective eval或dynamic rollout panel；只保留训练telemetry，不参与选点 |
| Selection loss并列/非有限 | 只在finite checkpoints中按原始`(eval_selection_loss,step)`排序，完全并列选较早step；全无效或train loss/gradient非有限是工程失败 |
| Round3最终评价 | 只使用independent 997-pair fixed-pair test；五个selected checkpoints与frozen base同时报reference-delta/raw mean-logp两个head的Accuracy、NLL、Brier、ECE-15等，只在同head内比较 |
| Test是否再次rollout | 不；直接评分冻结的有标签A/B preference pairs |
| base/DPO-1K headroom | 不能用初始时恒为0.5的DPO reference-delta；辅助比较使用共同raw mean-logp delta beta10，不参与选点 |
| AlpacaEval/MT-Bench | Round3只登记不运行；不生成回答、不调用judge API、不运行本地替代judge；延后到Round4重新设计和审批 |
| Qwen3与优化 | ModelScope `Qwen/Qwen3-1.7B` post-trained版本，native non-thinking template；五方法共享resolved revision/manifest；非量化LoRA r8/alpha16/dropout0、lr $10^{-5}$、BF16 mixed precision |
| 3×4090如何分工 | 五方法都在GPU0单卡训练；两个动态PE时GPU1/2分别运行独立vLLM replica，并以`method_id/step/adapter_sha256` ACK屏障防止stale policy |
| 存储策略 | 五方法共50个durable checkpoints全部保留，不启用自动pruner；formal要求空闲空间至少为strong-smoke投影峰值的2倍，不删除Round2产物 |

## 只能在获批服务器阶段解析

- UltraFeedback/UltraChat repository的resolved full commit SHA、源parquet SHA-256、构造后manifest SHA-256与隔离审计；
- ModelScope实际下载的Qwen3-1.7B resolved revision与model/tokenizer/template manifest；
- Round3 experiment ID、当时3×4090实时硬件/磁盘证据、精确dependency locks、最终源码commit和数值验收摘要；
- strong-smoke实测的显存峰值、产物尺寸、projected storage peak与两倍空闲空间门禁。

除已由只读审计冻结的数据计数外，其余仍是执行时resolved facts，不应由实现者猜测，也不需要在本地伪造结果。`r3-theory-v0.9`和`round3-exp-v1.4`的方案B批准及Round3 `CODE_IMPLEMENTATION`状态均已记录；当前允许本地静态实现，仍禁止修订代码交接前上传或进入服务器执行。Round2已停止且checkpoint保留，仍不得删除其产物。
