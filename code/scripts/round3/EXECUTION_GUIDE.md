# Round3 3×4090服务器执行手册（当前未授权）

本手册对应`r3-theory-v0.9`、`round3-exp-v1.4`和五方法Round3方案B实现。它只定义代码交接后的执行顺序；当前仍处于`CODE_IMPLEMENTATION`，未经用户明确确认修订代码版本并另行授权`SERVER_EXECUTION`，不得运行下列入口、上传代码或改动服务器checkout。

## 0. 必须先满足的门禁

1. 用户已审阅当前未提交diff，并明确确认可形成服务器候选commit；commit/push不得由实现者自行执行。
2. 已有2026-08-26只读证据确认Round2在step590停止、step580/589/590保留、第二方法未启动、两个pruner未运行；每次新的服务器动作前仍须重新核验controller/process/GPU/df，且不得删除Round2产物。
3. 用户基于实时证据明确授权Round3服务器阶段；三个GPU必须空闲，磁盘门禁必须通过。不得修改潜在运行中的Round2 checkout，不得删除任何Round2或Round3 checkpoint。
4. 服务器`<SERVER_BASE>/SOPPO`是唯一Git仓库并处于用户确认的clean commit；环境、cache、data、models、runs、exports和platform logs全部位于仓库外。

## 1. 运行者必须显式解析的值

以下变量不能从本地文档、旧运行或示例猜测：

```bash
export SOPPO_ROUND3_EXPERIMENT_ID='<new-path-safe-experiment-id>'
export SOPPO_ROUND3_MODEL_REF='<explicit-refs/heads/...-or-refs/tags/...>'
export SOPPO_ROUND3_ULTRAFEEDBACK_REF='<explicit-refs/heads/...-or-refs/tags/...>'
export SOPPO_ROUND3_ULTRACHAT_REF='<explicit-refs/heads/...-or-refs/tags/...>'
```

可选路径覆盖必须仍指向仓库外的服务器目录。`SOPPO_ROUND3_PHYSICAL_PAIR_SUBBATCH`初始可为`1`；只有五条production-path strong smoke均通过后，才可使用有服务器证据的更大值。依赖`.in`文件只是候选范围，成功环境的精确`pip freeze`由服务器保存在环境目录，不得提前宣称为lock。

## 2. 获批后的准备顺序

从clean、reviewed Round3 checkout执行：

```bash
bash code/scripts/round3/00_setup_envs.sh  # 仅环境不存在时；支持Conda或Python 3.10 venv
bash code/scripts/round3/00_resolve_revisions.sh
bash code/scripts/round3/00_download_model.sh
bash code/scripts/round3/00_prepare_data.sh
bash code/scripts/round3/03_strong_smoke.sh
```

已存在并通过`pip check`的两个Python 3.10 venv不得为形式统一而重装；此时跳过`00_setup_envs.sh`并重新记录版本/freeze即可。revision脚本只对运行者显式给出的公开Git ref执行`ls-remote`并原子保存full SHA；已有experiment的`source_revisions.json`不得覆盖或重跑。数据脚本使用新的`dual_source_v2`，确定性隔离冻结revision的畸形行，生成8K/1K/7K/1K/997 views、17,997行source manifest和12,197行无原始文本malformed audit；失败遗留空v1目录保留，不删除。strong smoke依次完成配置解析、server-only CPU tests、输入/GPU前置门禁、v2 reference cache、五方法各一个完整logical step、真实checkpoint重载和空间投影。任一步失败立即停止，不自动删目录或改参重试。

strong smoke后只读检查至少包括：

- 五个`smoke_complete.json`与`checkpoint_verified.json`；
- 双vLLM方法的两个ready/ACK、request/response和adapter卸载日志；
- 五方法finite loss/gradient、峰值显存和最长样本截断telemetry；
- `storage_projection.json`中的组成项与`formal_required_free_bytes`；
- 两个环境的`pip check`与外部`environment.freeze.txt`。

任何依赖修改、OOM、vLLM接口不兼容、数值或checkpoint round-trip失败都返回`CODE_IMPLEMENTATION`。不能把失败的strong smoke当作正式方法结果，也不能静默降低逻辑batch、长度、方法数或评价范围。

## 3. 正式长链（需要再次明确授权）

只有用户审阅strong-smoke证据、确认代码交接并明确授权正式运行后，才可启动：

```bash
bash code/scripts/round3/start_all.sh
```

控制器依次完成formal config解析、一次性`free >= 2 × projected_peak`门禁、五方法串行训练、共同1K validation选点、独立997-pair双head final test和sample-free聚合。方法顺序固定为：

1. `dpo_1k`
2. `sspo_code_loss_stratified_ultrachat_2df9e9a`
3. `dpo_8k`
4. `dpo_pe_sft_rollout`
5. `dpo_pe_rollout_only`

所有方法在GPU0训练；只有两个动态方法在GPU1/2各启动一个vLLM副本。每方法保留steps 25–250的十个durable checkpoints；dynamic staging adapter也在结果保留决策前全部保留。没有keep-N pruner或自动清理。

## 4. 只读状态与精确停止

状态入口不会发送信号：

```bash
bash code/scripts/round3/status_all.sh
```

它显示controller、五方法status/state/best、metrics尾部、pruner禁用声明、`nvidia-smi`和`df`。停止入口默认只预览：

```bash
bash code/scripts/round3/stop_all.sh
```

只有用户对该Round3 experiment明确授权停止后才可加`--execute`。脚本会重新核对记录的PID、PGID和`/proc` starttime，只向匹配的Round3 controller进程组发送TERM，不触碰Round2，也不删除checkpoint。

## 5. 允许回传与必须留在服务器的产物

允许回传本地的只有sample-free聚合JSON、聚合Markdown/CSV/图、resolved config文本、环境摘要、manifest/checksum和远程证据索引。数据视图、source manifest、reference cache、模型、adapter/checkpoint、逐样本预测、private labels、原始日志和queue request/response必须留在服务器。

Round3不运行AlpacaEval或MT-Bench（登记到Round4），不实现PE-static（登记到Round5），不启用QLoRA，也不把候选依赖版本的成功与否当作已知事实。
