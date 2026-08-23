# Round2 3×4090 执行指南

> 当前状态：`LOCKED / CODE REVIEW`。本指南对应尚未提交、尚未服务器验证的 TP2/vLLM 实现。用户确认本次代码交接、SFT corpus、temperature 和 top-p 以前，只阅读命令，不上传或启动。

## 1. 固定资源和目录

以下全部是 3×4090 独占服务器命令，不在 Mac 本地执行：

```bash
export SERVER_BASE=/data/youchang/youchang
cd "$SERVER_BASE/SOPPO/code/scripts/round2"
```

固定资源布局：

```text
GPU 0,1 = Qwen3-4B Transformers native TP=2 LoRA training
GPU 2   = one persistent vLLM rollout worker
```

Round2 使用仓库外的两个隔离环境：

```text
<SERVER_BASE>/envs/round2-train
<SERVER_BASE>/envs/round2-rollout
```

第一轮的 `envs/youc`、`scripts/cluster/` 和 `scripts/standalone/` 不被修改，也不能和本入口混用。

## 2. 正式运行前必须先确定的三项

必须由用户预先确认：

```bash
export SOPPO_ROUND2_SFT_DATA_FILE='<24k单回复SFT JSONL绝对路径>'
export SOPPO_ROUND2_TEMPERATURE='<获批temperature>'
export SOPPO_ROUND2_TOP_P='<获批top-p>'
```

SFT JSONL 每行只能含 `sample_id,prompt,response`（可含值为 `round2.sft.v1` 的 `schema_version`）。它必须与 frozen `unlabeled_train.jsonl` 的24,000个 ID/prompt 精确连接，且不能含 label、chosen/rejected 或 response pair。

配置缺少上述任一项都会立即失败。这是研究门禁，不是环境故障。

## 3. 同步经确认的 clean commit

本次代码交接通过以后再执行。服务器必须位于用户确认的完整 commit，且工作区为空：

```bash
git -C "$SERVER_BASE/SOPPO" rev-parse HEAD
git -C "$SERVER_BASE/SOPPO" status --short
```

`status --short` 必须没有输出。resolved config 会写入该完整 commit，后续 preflight 再次核对；运行期间不要 `git pull`、checkout 或编辑服务器仓库。

## 4. 建立两个隔离环境

首次执行一次：

```bash
cd "$SERVER_BASE/SOPPO/code/scripts/round2"
export SOPPO_CONDA_EXE='<conda绝对路径；若command -v conda可用则可不填>'
bash 00_setup_envs.sh
```

该步骤安装依赖并运行 `pip check`，但不启动 GPU 训练。不要把 Round2 包覆盖安装进第一轮 `envs/youc`。

## 5. 启动前实时检查三张卡

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv
```

preflight 要求三张卡均为 RTX 4090、物理显存至少23GiB，并且没有 compute process。不要按进程名直接杀任务；先确认占用者，只有自己的已知进程才能停止。

## 6. 一次挂完整长链

每次必须使用新的 experiment ID：

```bash
export ROUND2_EXPERIMENT_ID='exp-YYYYMMDD-NN-round2-tp2'
export SOPPO_ROUND2_TRAIN_GPU_IDS='0,1'
export SOPPO_ROUND2_ROLLOUT_GPU_IDS='2'
export SOPPO_ROUND2_SFT_DATA_FILE='<已确认绝对路径>'
export SOPPO_ROUND2_TEMPERATURE='<已确认值>'
export SOPPO_ROUND2_TOP_P='<已确认值>'

cd "$SERVER_BASE/SOPPO/code/scripts/round2"
bash start_all.sh
```

后台控制器按顺序执行：

```text
resolve 两条 formal config
  → 全量 server pytest
  → 两条方法各一个 production-path strong smoke
  → SOPPO-PE-sft-rollout-exp formal
  → SOPPO-PE-rollout-only-exp formal
  → 两条 validation-selected adapter 的独立3000 test评价
  → Round2 sample-free aggregate/export
```

strong smoke 保持完整8+56 population、bf16/2048和真实 TP2，并强制 rollout 生成512 token。它比轻量单测慢，但会在 formal 前验证24GB显存、adapter交接与在线PE反传。任何阶段失败都会阻断后续。

正式每条方法为 `floor(24000/56) × 2 = 856` 个 optimizer step，与第一轮 joint trainer 的 drop-last 口径一致；每个 epoch 的32条余数不进入该 epoch。因而 SFT+rollout 正式生成47,936条回复，rollout-only正式生成95,872条回复。实际 wall time 不预猜，由两条 strong smoke 的日志分别外推。

## 7. 查看状态和日志

```bash
cd "$SERVER_BASE/SOPPO/code/scripts/round2"
bash status_all.sh
```

持续看控制器日志：

```bash
tail -f "$SERVER_BASE/runs/$ROUND2_EXPERIMENT_ID/controller.log"
```

查看某一方法：

```bash
bash status_method.sh soppo_pe_sft_rollout_exp.yaml
bash status_method.sh soppo_pe_rollout_only_exp.yaml
```

关键文件：

```text
runs/<experiment>/controller.json
runs/<experiment>/strong_smoke/complete.json
runs/<experiment>/<method>/controller_status.json
runs/<experiment>/<method>/state.json
runs/<experiment>/<method>/tp_evidence.json
runs/<experiment>/<method>/logs/tp_train.log
runs/<experiment>/<method>/logs/vllm_worker.log
runs/<experiment>/<method>/logs/metrics.jsonl
```

## 8. 精确停止

先预览，只查看本 experiment 的进程组：

```bash
bash stop_all.sh
```

确认 PID/PGID 和命令都属于本次 Round2 后：

```bash
bash stop_all.sh --execute
```

不要使用 `pkill python`、`killall` 或按用户批量结束，因为服务器上可能还有其他服务。

## 9. 手工分阶段（排错时）

只有在不使用 `start_all.sh` 且 experiment 目录尚不存在时使用：

```bash
bash 01_resolve_all.sh

bash 00_server_tests.sh

bash dry_run.sh soppo_pe_sft_rollout_exp.yaml
bash dry_run.sh soppo_pe_rollout_only_exp.yaml

bash 02_strong_smoke.sh

bash run_method.sh soppo_pe_sft_rollout_exp.yaml
bash run_method.sh soppo_pe_rollout_only_exp.yaml

bash 03_evaluate.sh
bash 04_aggregate.sh
```

失败目录默认不复用、不覆盖。当前 checkpoint 只有 LoRA 参数，没有 optimizer/scheduler state，因此没有 bit-exact 热恢复；排错后应保留失败证据并换新的 experiment ID。
