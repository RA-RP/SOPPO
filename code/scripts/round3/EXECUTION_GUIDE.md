# Round3 DPO-reward PE extension：3×4090执行手册

本手册对应`r3-theory-v1.1`、`round3-exp-v1.6`与`round3-code-candidate-v0.6`。当前仅处于`CODE_IMPLEMENTATION`；下列命令是代码交接通过后的服务器流程，不是现有执行授权。旧`round3-20260826-04`必须按exact commit自然结束，禁止热改checkout、停止controller或删除任何checkpoint。Round4不启动。

## 1. 执行对象

旧run的五个方法保持不可变，其中`dpo_pe_sft_rollout`和`dpo_pe_rollout_only`按实际代码登记为SimPO-reward PE。extension只新增：

```text
dpo_pe_dpo_reward_sft_rollout
dpo_pe_dpo_reward_rollout_only
```

两条均在GPU0训练、GPU1/2运行双vLLM replica，并分别独占三卡串行。PE使用total response `log pi_theta-log pi_ref`、beta .1；reference通过同一模型临时禁用LoRA adapter计算。其他训练、rollout、selection与final-test合同不变。

## 2. 不可跳过的门禁

1. 用户已审阅v0.6 diff并明确批准一个exact commit可部署；Codex/GLM不得自行commit或push。
2. 只读证据确认旧controller为`completed/all_methods`；如果仍在运行或失败，停止，不创建extension证据。
3. 服务器checkout可安全切换且目标commit clean；三卡无其他compute进程。
4. 明确提供全新的extension experiment ID和不同于旧run的commit-bound reference-cache目录；不得覆盖旧cache、run、logs或export。
5. 两条production-path strong smoke、checkpoint验证与extension-only存储投影全部成功。
6. formal前真实满足`free_bytes >= 2 * projected_peak_bytes`；脚本没有删除或覆盖豁免。

## 3. 运行者必须显式设置

以下只是变量形状，不包含可猜测的现场值：

```bash
export SOPPO_ROUND3_BASELINE_EXPERIMENT_ID='<completed-five-method-experiment-id>'
export SOPPO_ROUND3_EXPERIMENT_ID='<new-extension-experiment-id>'
export SOPPO_ROUND3_REFERENCE_DIR='<new-absolute-commit-bound-reference-cache-dir>'
```

`SOPPO_ROUND3_EXPERIMENT_ID`必须不同于baseline。reference目录也必须不同于baseline resolved config中的目录；preflight会fail closed。模型、data v2和两个既有venv只读复用，不重新下载、不重新处理数据、不重装环境。

## 4. 获批后的服务器验证

在clean reviewed commit上依次执行：

```bash
bash code/scripts/round3/00_reuse_baseline_revisions.sh
bash code/scripts/round3/03_extension_strong_smoke.sh
```

第一步只在旧controller完整终态后复制其冻结source revision证据，并记录controller、baseline config与source文件SHA。第二步只解析两个extension配置，运行server-only tests，构建新commit绑定的reference cache，分别完成一个真实4+28 optimizer step、双replica ACK、adapter-disabled reference scoring与checkpoint重载，再只对新增两方法做存储投影。

检查至少包括：

- server tests全部通过，新增断言覆盖初始化概率恰为0.5、负DPO logit可出现和VJP系数只使用一次；
- 两个`smoke_complete.json`、两个`checkpoint_verified.json`齐全；
- DPO-reward telemetry声明`dpo_reference_logratio_total`/beta .1，step0 `policy_minus_reference_abs_max <= 1e-6`；
- 两个vLLM ready/ACK与adapter SHA一致，无stale response；
- `storage_projection.json`只列两个extension methods，且没有自动删除。

任一失败立即停止并回传无样本摘要；不得在服务器现场编辑源码、放宽容差、减少logical batch/长度或删除已有产物后擅自重试。

## 5. 正式extension

只有用户再次审阅strong-smoke与存储证据并明确授权后，才能执行：

```bash
bash code/scripts/round3/start_extension.sh
```

控制器顺序为：解析两个formal configs → 一次性存储门禁 → SFT+rollout DPO-reward → rollout-only DPO-reward → 共同1K selection → 两个selected checkpoints的997-pair双head final test → 新旧七方法跨运行sample-free聚合。终态为`completed/all_extension_methods`。

extension状态可使用相同experiment ID调用只读入口：

```bash
bash code/scripts/round3/status_all.sh
```

它对缺失的旧方法目录只显示missing，不会读取或修改baseline。停止入口`stop_all.sh`默认仅预览；只有用户针对该extension experiment明确授权时才可加`--execute`，且不得作用于baseline。

## 6. 跨运行可比性与产物边界

`aggregate_extension.py`只在以下条件全部满足时合并：旧controller终态、model/data/revisions/test manifests一致、reference output SHA一致、997个sample ID与private label顺序完全一致、关键final-evaluator依赖在两个commit间字节等价。输出只含聚合指标和同head差值，不产生综合分数。

本地只允许保存sample-free aggregate、resolved config、环境摘要、checksum和远程证据索引。数据、模型、reference cache、checkpoint、逐样本预测、private labels、原始日志与rollout queue内容必须留在服务器。AlpacaEval/MT-Bench与任何judge API在Round3仍为零调用；PE-static仍不实现。
