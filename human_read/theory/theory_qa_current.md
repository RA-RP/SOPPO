# Round3 Theory QA：DPO-reward PE修订

## 状态

- 理论版本：`r3-theory-v1.1`
- 实验版本：`round3-exp-v1.6`
- 日期：2026-08-27
- 状态：用户已明确要求恢复早期Theory v0.2的reference-relative PE，并把既有两臂登记为SimPO-reward对照；当前唯一活动阶段为Round3 amendment `CODE_IMPLEMENTATION`
- 旧执行：`round3-20260826-04`仍绑定`r3-theory-v1.0`/`round3-exp-v1.5`与exact commit `b1beef5550ac47c9c78b98c1729014cc153b1251`，只允许自然运行到终态
- 完整定义：`current_theory.md`

## 已冻结决定

| 问题 | 当前决定 |
| --- | --- |
| 正确PE reward是什么 | 对每个response使用total response log-prob，$r_\theta(x,y)=0.1[\log\pi_\theta(y\mid x)-\log\pi_{ref}(y\mid x)]$，$p_i=\sigma(r_\theta(x,y_i^A)-r_\theta(x,y_i^B))$ |
| $\pi_{ref}$是什么 | 同一冻结Qwen3-1.7B初始化模型；动态候选每步生成后，以adapter-disabled、no-grad路径现场评分，不用固定训练pair cache替代 |
| rollout还能低于SFT吗 | 能。若rollout相对reference的改善小于SFT相对reference的改善，则DPO logit为负且rollout方向概率低于0.5；初始化$\pi_\theta=\pi_{ref}$时所有pair恰为0.5 |
| 旧两个动态run是什么 | 它们实际使用raw mean-response-logp、$\beta=10$，现登记为`SimPO-reward PE`消融，不删除、不重命名既有method ID、不冒充原始PE |
| Round3有几个方法 | 七个：三个静态方法、两个旧SimPO-reward动态方法、两个新增DPO-reward动态方法 |
| 新方法ID | `dpo_pe_dpo_reward_sft_rollout`与`dpo_pe_dpo_reward_rollout_only` |
| 新旧动态方法如何匹配 | 除reward profile外，数据、candidate构造、sampling、4+28 logical batch、labeled DPO、$\lambda_{PE}=0.1$、250 steps、checkpoint、selection与final test全部一致 |
| 如何接在当前执行之后 | 不热改旧controller。旧run终态后，新exact commit/new experiment只运行两个新增方法，再以跨运行证据审计组合七方法结果 |
| epoch与checkpoint | 七方法均1 epoch/250 optimizer steps；每25步保存。旧run 50个、新extension 20个，共70个durable checkpoints |
| selection与final | 七方法共享冻结1K-pair reference-DPO selection；selected checkpoints与frozen base在同一997-pair test上报告两个既有score heads，只做同head比较 |
| 数据 | 8K `ultrafeedback_binarized/train_prefs` master及其1K子集；7K `ultrachat_200k/train_sft`；`test_prefs`隔离3条畸形行后分1K validation+997 test |
| SSPO | 保持GitHub commit `2df9e9a`：sequential running statistics、min-normalized-chosen threshold、无threshold EMA/KDE |
| GPU波次 | 旧五方法按既有三静态并发、两动态三卡串行执行，11:31快照第五方法仍在185/250；extension必须等其完整终态，再让两个DPO-reward方法各自三卡串行，GPU0训练、GPU1/2双vLLM |
| 延后项 | Round4 AlpacaEval/MT-Bench仍锁定且不运行；PE-static仍登记到Round5，不在本修订实现 |

## 只能在获批服务器阶段解析

- 新extension experiment ID、exact reviewed commit与commit-bound reference-cache目录；
- server-only tests、两条production-path strong smoke、显存/耗时、存储投影与两倍free门禁；
- 旧controller的真实终态、新旧模型/数据/test/reference outputs及final evaluator等价证据；
- 两条新增formal轨迹、selection、997-pair final test和sample-free跨运行聚合。

本地只完成静态源码、配置、shell和文档复核。未经用户审阅代码交接，不得commit/push、部署、测试或挂载extension；Round4也不因本次修订而解锁。
