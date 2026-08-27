# `human_read/exp/`：实验设计

本目录维护三个内容文件：

- `current_experiment.md`：最新实验设计；阶段未解锁时只能保存明确标记的锁定草案。
- `useful_experiments.md`：经复盘后认为对论文有用的历史实验设计。
- `experiment_archive.md`：所有版本的实验设计，按 cycle/experiment ID 追加。

`CODE_HANDOFF.md`记录Round3代码交接，并在后半保留Round2历史交接。旧五方法formal绑定`b1beef5`继续自然运行；2026-08-27用户要求把动态PE恢复为DPO implicit reward并新增两臂，当前为修订后的`CODE_IMPLEMENTATION`，新代码尚未获提交或服务器执行授权。

## 阶段门禁

- 只有 `../theory/current_theory.md` 已记录用户明确通过后，才能进入 `EXP_DISCUSSION`。在此之前，现有设计只能作为锁定的预讨论草案。
- 本阶段只与用户讨论、修订和冻结实验问题、变量、对照、数据、指标、预先判断标准、资源与风险；普通讨论、沉默或未反对都不构成通过。
- 只有用户明确通过具体设计版本后，才能进入 `CODE_IMPLEMENTATION`，在 `../../code/` 编写实现。实验设计通过不等于代码通过，也不构成 SFTP 上传或服务器执行授权。
- 退出本阶段时，`current_experiment.md` 必须记录理论通过状态、实验设计通过状态和确认日期，并把下一入口指向 `../../code/CODE_OVERVIEW.md`。若设计发生实质变化，代码阶段及其后续授权立即失效并重新讨论。

## 更新规则

1. 设计必须从 `../theory/current_theory.md` 的具体假设或预测出发，而不是先有运行再补目的。
2. `current_experiment.md` 至少写明：问题、假设、变量与对照、数据、指标、预先判断标准、消融、资源预算、风险和预期产物。
3. 设计定稿或发生实质修改时，把带版本标识的完整设计追加到 `experiment_archive.md`，再保持 `current_experiment.md` 为最新版本。
4. 每次实际运行都在 `../../exp/<experiment_id>/` 建独立目录；这里必须链接过去。
5. 结果复盘后，只有能支撑论文论点、关键消融、重要负面发现或方法选择的设计才进入 `useful_experiments.md`，并写明入选理由。
6. 预讨论草案可以归档以保留思路，但归档条目必须明确写“未批准、代码阶段未解锁”，不能把“已写完”当作“已定稿”。

## 执行位置要求

- 实验设计阶段只进行纯文本设计；获批后，源码编辑属于独立的 `CODE_IMPLEMENTATION`，SFTP 上传属于后续 `SERVER_EXECUTION`。
- 依赖/环境、单元与集成测试、数据、模型、smoke、pilot、训练、评价和聚合全部写成服务器任务，并给出服务器产物路径。
- 设计必须列出本地回传白名单与服务器保留黑名单，禁止使用“可在本地分析”等可选表述。
