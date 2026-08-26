# round3-20260826-03：Round3 v0.4完整strong-smoke attempt

## 元信息与状态

- Cycle：`cycle-20260818-01` / Round3
- 执行时设计：`r3-theory-v0.9` / `round3-exp-v1.4`
- 服务器exact commit：`14c0292cba2e0322d93a62330bd99d1f8471f174`
- 状态：失败并保留；formal未启动
- 前序attempt：`../round3-20260826-01/README.md`、`../round3-20260826-02/README.md`
- 复用冻结data v2；reference cache以相同数值输出重新绑定本exact commit

## 已通过事实

- 服务器合同测试：6 passed，两个环境`pip check`干净。
- DPO-1K、GitHub-loss SSPO和DPO-8K均完成一个production-path optimizer step并写出完整smoke checkpoint。
- 三个checkpoint均验证FP32 LoRA、392项optimizer state、scheduler/global step和RNG恢复。
- SSPO两次独立next-batch重放的loss、running state、scheduler/global step和全部LoRA更新差异均为精确0，满足`1e-7/1e-6`预注册容差。
- 三个方法峰值allocated显存约8.49/8.49/8.82 GB；失败后所有GPU清理为空闲。

## 停止原因

首个动态方法`dpo_pe_sft_rollout`的两个vLLM replica均完成启动、adapter加载和生成，但在训练接收结果前触发：

```text
ValueError: vLLM effective prompt length differs from the Round3 truncation contract
```

服务器安装的vLLM 0.9.2源码核对确认：文本prompt路径把`add_special_tokens`留给tokenizer默认值，而训练编码器明确使用`add_special_tokens=False`。因此这属于tokenizer入口实现不一致，不是数据、loss、1024长度或sampling合同失败。第二动态方法、storage projection和formal均未开始。

## 后续获批修复

用户于2026-08-26明确批准`r3-theory-v1.0`/`round3-exp-v1.5`资源波次并要求持续修复到formal挂载。`round3-code-candidate-v0.5`将：

- 在worker中显式复用训练chat template、`add_special_tokens=False`和末端1024 token左截断；
- 通过vLLM `TokensPrompt`传递已冻结IDs并逐ID核对返回值；
- 并发运行DPO-1K/GPU0、SSPO/GPU1、DPO-8K/GPU2，随后串行运行两个三卡动态方法。

这些变化不修改数据、loss、logical batch、seed、optimizer、sampling、长度上限或数值容差。

## 远程证据索引

主日志：

```text
/data/youchang/youchang/platform_logs/round3-20260826-03/codex_validation/14c0292cba2e/03_strong_smoke.log
```

SHA-256：`ef2ddf990d43655f55636009dbe17509f61e006041269f59081f18497f56bd9e`。原始日志、queue request/response、模型、cache和checkpoint只留在服务器。
