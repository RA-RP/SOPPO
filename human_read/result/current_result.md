# Round3结果交接：五方法结果保留、amendment行政关闭

## 0. 版本与状态

- Cycle：`cycle-20260818-01` / Round3
- Formal experiment：`round3-20260826-04`
- 执行设计与代码：`r3-theory-v1.0` / `round3-exp-v1.5` / exact commit `b1beef5550ac47c9c78b98c1729014cc153b1251`
- 未执行修订：`r3-theory-v1.1` / `round3-exp-v1.6` 的两个DPO-reward PE extension
- 结果版本：`round3-result-administrative-close-v0.1`
- 状态：`COMPLETE_FORMAL_WITH_UNEXECUTED_AMENDMENT`；五方法formal有完整聚合，amendment没有新增实验结果
- 用户决定：2026-09-01明确要求不再保留Round4为锁定候选，直接覆盖进入Round4
- 交接语义：保留并如实登记Round3五方法结果，不运行两个DPO-reward extension，不用缺失extension结果补写结论

## 1. 执行完整性

- controller于2026-08-27 12:56达到`completed/all_methods`，exit 0；五方法均完成250/250 optimizer steps，各保留10个durable checkpoints。
- 共同1K validation选点、独立997-pair双head final test与sample-free aggregate完成。
- selected steps：DPO-1K 225、SSPO 25、DPO-8K 250、SimPO-reward PE SFT+rollout 250、SimPO-reward PE rollout-only 225。
- `r3-theory-v1.1`后来要求的两个DPO-reward PE没有进入服务器执行，不存在对应checkpoint、test或Alpaca结果。
- 旧run、失败attempt、Round2产物和全部checkpoint保持原位；本次行政切换未发起服务器命令、未停止任务、未删除产物。

## 2. 已观察事实

独立997-pair的DPO reference-delta head accuracy（tie计0.5）为：

| 方法 | Accuracy |
| --- | ---: |
| frozen base | 0.5000 |
| SSPO | 0.5155 |
| DPO-1K | 0.6479 |
| SimPO-reward PE SFT+rollout | 0.6459 |
| SimPO-reward PE rollout-only | 0.6349 |
| DPO-8K | 0.6790 |

同head下，DPO-8K比DPO-1K高3.11pp；SSPO比DPO-1K低13.24pp；两个SimPO-reward PE分别比DPO-1K低0.20pp和1.30pp。raw mean-logp beta10 head的各方法accuracy集中在约0.589–0.594且置信度高度饱和，区分力较弱。

## 3. 解释与结论边界

- 这些结果支持“本次单种子设置下，1K DPO已学习到明显偏好信号，8K labeled预算进一步提高”；不支持统计显著性或跨模型外推。
- 本次GitHub-loss SSPO在预注册1:7、250-step、统一lr设置下没有达到DPO-1K表现；不能据此否定所有SSPO定义或调参设置。
- 两个历史动态臂实际是SimPO raw mean-logp reward profile；其PE责任在训练中近乎饱和。它们不能代表reference-relative StaticPE，也不能用于否定未执行的DPO-reward extension。
- Round3没有AlpacaEval/MT-Bench结果。Round4必须重新生成和评价，不能从paired test推断生成式benchmark得分。
- 用户选择直接进入Round4意味着不再为Round3 amendment补实验；这是研究路线行政决定，不是DPO-reward PE的科学负面结果。

## 4. 证据与交接确认

- 本地白名单摘要：`../../../exp/round3-20260826-04/README.md`
- 服务器原始证据索引、aggregate路径与日志SHA见上述README；逐样本预测、模型和checkpoint留在服务器。
- 完整结果快照已追加到`result_archive.md`。
- 用户明确行政交接：2026-09-01“直接覆盖成round4”。
- 结果交接状态：`COMPLETE_BY_EXPLICIT_ADMINISTRATIVE_TRANSITION`
- 下一入口：`../nextCycle/current_plan.md`，随后新cycle进入Round4 `THEORY_DISCUSSION`。
