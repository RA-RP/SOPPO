# Round2 3×4090 执行指南

> 当前状态：`LOCKED / CODE REVIEW`。本指南对应尚未提交、尚未服务器验证的 TP2/vLLM 实现。24k固定单回复锚点与采样参数已确认；用户明确确认完整代码交接并手工形成 clean commit 以前，只阅读命令，不上传或启动。

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

## 2. 已冻结的数据和采样合同

源数据固定为：

```text
/data/youchang/youchang/data/ultrafeedback/mvp-v0.5-30k/unlabeled_train.jsonl
```

`00_prepare_sft_anchor.sh` 不读取 private label，只逐行取已经随机换位的公开 `response_a`，生成：

```text
/data/youchang/youchang/data/round2/mvp-v0.5-30k/sft_anchor_response_a/sft_anchor.jsonl
```

派生 JSONL 每行只含 `schema_version,sample_id,prompt,response`；预检要求24,000条 response、ID和prompt都与公开源精确一致。冻结源文件不会被覆盖，已有派生目录只有完全通过 manifest/SHA/逐行验证才能复用。

两条方法统一使用 Qwen3 non-thinking 官方采样配置：

```text
temperature=0.7, top_p=0.8, top_k=20, min_p=0
```

这些值已写入配置并由 validator 锁定，不需要在 shell 中重复填写。

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

随后显式生成或复核一次固定锚点：

```bash
bash 00_prepare_sft_anchor.sh
```

`start_all.sh` 内还会再执行同一验证，因此手工执行这一步是为了在占用 GPU 前更早发现数据错误，不会重复生成或覆盖有效文件。

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

cd "$SERVER_BASE/SOPPO/code/scripts/round2"
bash start_all.sh
```

后台控制器按顺序执行：

```text
生成/复核24k固定单回复锚点
  → resolve 两条 formal config
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
