# exp-20260902-02-round4-formal：Round4 四方法正式对比

## 元信息

- Experiment ID：`exp-20260902-02-round4-formal`
- Cycle ID：`cycle-20260901-01`
- 状态：运行中（正式计划尚待服务器生成）
- 对应实验设计：`../../human_read/exp/current_experiment.md` `round4-exp-v2.0`
- 对应理论：`../../human_read/theory/current_theory.md` `r4-theory-v2.0`
- 理论/实验设计用户确认日期：2026-09-02
- 代码版本：`71b0a6cf347a86bed9ac1a0ae6b68f0ee3dc2500`
- 代码交接与服务器执行授权：用户于2026-09-02明确授权持续提交、同步、smoke通过后自动开始formal。
- 执行位置：A100-2负责训练/生成，4090-3负责API judge；本地未运行项目代码、数据、模型或评价。

## 问题与冻结合同

在相同Qwen3-1.7B、冻结UltraFeedback/UltraChat 0.1视图、epoch1与共同AlpacaEval输出上，对比DPO、SSPO、StaticPE和FrozenPE。DPO仅使用labeled数据；其余方法使用各自冻结的labeled/unlabeled合同。目标指标为同一primary API judge产生的WR、LC与长度统计；不以中间结果选择超参数或改变未启动臂。

## 实际运行与证据索引

- A100代码路径：`/root/SOPPO`，exact commit同上。
- A100正式根目录：`/root/runs/round4/formal/formal-71b0a6c-v1`。
- A100导出根目录：`/root/exports/round4/formal/formal-71b0a6c-v1`。
- A100数据视图：`/root/data/round4-v2/formal/formal-71b0a6c-v1`。
- 入口：服务器执行`code/scripts/round4/05_run_formal_a100.sh prepare|base|dpo|sspo|staticpe|frozenpe formal-71b0a6c-v1`。
- 4090将只拉取每臂不可变输出及manifest，调用持久primary profile并保存不含密钥、端点和逐样本内容的聚合结果。

## 产物与回传边界

| 产物 | 服务器位置 | 本地保留 |
| --- | --- | --- |
| 计划与最终配置文本 | A100正式根目录 | 可回传文本/校验值 |
| 训练日志、checkpoint、adapter、合并模型 | A100运行/导出根目录 | 否 |
| 805条生成与API原始标注 | A100/4090导出根目录 | 否 |
| WR、LC、长度等聚合摘要 | 4090 judge结果目录 | 是，完成后回传 |

## 事实摘要

- 已完成的前置证据：`cb2d54c`四臂2-step full-chain smoke通过；此处不把2条smoke评价作为正式结果。
- 正式运行、API评价及任何异常尚未在本记录中填入；只在服务器聚合摘要生成后追加。

## 结果交接

尚未进入`RESULT_HANDOFF`。所有四臂完成或有终止摘要并形成远程证据索引后，才可更新本节。
