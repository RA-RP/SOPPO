# SOPPO Round2：3×4090 TP 训练与在线 rollout 流水线

> 状态：代码实现中，尚未提交、尚未通过服务器 strong smoke，也未获本次代码交接确认。

## 1. 三张卡怎么分

Round2 不是三卡 DDP，而是两个独立执行器：

```text
GPU 0 ─┐
       ├─ Qwen3-4B TP=2 LoRA training（一个模型横跨两卡）
GPU 1 ─┘

GPU 2 ─── vLLM rollout worker（独立的一份冻结 base + 当前 LoRA）
```

训练侧 `TP=2, PP=1, DP=1`。两张卡共同完成同一个 forward/backward，不各自处理不同数据副本。GPU 2 只生成候选，不计算 PE loss、不更新参数。

实现不再需要外部 Megatron entrypoint：

- `src/round2/tp_trainer.py`：项目内置训练循环；
- `src/round2/tp_backend.py` / `run_tp.py`：TP 命令、版本和真实分片证据；
- `src/round2/run_rollout.py`：常驻 vLLM worker；
- `src/round2/queue_protocol.py`：训练与 rollout 的原子文件队列；
- `src/round2/sft_schema.py`：单回复 SFT corpus 的隔离门禁。

## 2. 一个 optimizer step 怎么流动

```text
训练 rank 0/1 发布 current LoRA adapter（step t + READY/SHA）
                  │
                  ▼
rank 0 写入 56 个 prompt 的 rollout request
                  │
                  ▼
GPU 2 加载 step t adapter 并生成候选
                  │
                  ▼
GPU 2 原子写 response；训练两 rank 共同读取并校验
                  │
                  ▼
8 labeled pairs + 56 dynamic pairs 的完整群体前向
                  │
                  ▼
求 labeled / PE 对每个 response score 的精确一阶系数
                  │
                  ▼
每次只 materialize 1 pair，逐 response 反向累计
                  │
                  ▼
clip → 一个 optimizer step → 发布 step t+1 adapter
```

物理 pair subbatch=1 是显存执行方式，不是把 logical batch 改成 1。一次 optimizer step 仍严格消费 8 个 labeled pair 与 56 个 dynamic pair；PE 仍在完整 56-pair population 上求值。

## 3. 为什么两种方法生成数不同

两条方法共享相同的 24,000 prompt、SFT corpus、采样配置和 current-policy 定义，只替换候选构造：

| 方法 | GPU 2 每 prompt 生成 | PE pair |
| --- | ---: | --- |
| SFT+rollout | 1 条 | `SFT response` vs `rollout_0` |
| rollout-only | 2 条 | `rollout_0` vs `rollout_1` |

rollout-only 不能复用同一条回答作为两侧。若 A=B，则 mean-logp delta 恒为 0、`p_i=0.5`，PE 没有方法所需的候选差异。两种方法也不能跨完整训练轨迹共享实际 rollout，因为做过第一次更新后，它们的 current policy 已经不同。rollout-only 仍用 SFT manifest 校验共同的24k prompt universe，但发往训练/worker的 request 不含 SFT response 文本。

## 4. 不可变 handoff

训练端不会让 vLLM 读取正在写的 checkpoint。每个 adapter 先写入 `.partial` 目录；两个 TP rank 都完成 PEFT gather/save 后，rank0 再写：

- `adapter_model.safetensors`；
- `adapter_config.json`；
- `checkpoint_meta.json`；
- `READY.json`（含 adapter SHA-256）。

最后以原子目录 rename 发布。rollout worker 只接受完成目录，并在每个 request 后卸载当步 adapter。

## 5. 配置和数据门禁

resolved config 是运行时唯一真源。它冻结：

- 完整 Git commit；
- Qwen3 路径和 manifest；
- 30k 数据路径；
- SFT corpus 路径和 SHA；
- GPU `0,1` / `2`；
- TP=2、LoRA r8/alpha16、bf16/2048；
- 8+56、2 epochs、lr1e-5、paper `gamma_t`；
- temperature、top-p、512 max new tokens。

SFT corpus 必须是 24,000 行 `sample_id,prompt,response`，与冻结 unlabeled split 的 ID 和 prompt 精确一一对应；任何 label、chosen/rejected 或 pair 字段都会被拒绝。

当前尚未由用户确认的正式参数是：

1. 单回复 SFT corpus 的来源与绝对路径；
2. rollout `temperature`；
3. rollout `top_p`。

因此代码保持 null / 环境变量必填，不自行选择经验默认值。

## 6. strong smoke 与正式长链

`02_strong_smoke.sh` 对两个方法各跑一个真实生产 step，不使用缩小到几条样本的假 batch。每条 smoke 都覆盖：

- TP=2 真分片证据，不允许两份模型复制；
- 8+56 完整 population；
- 最长真实 labeled 样本达到 2048 的反向；
- rollout 强制生成 512 token；
- current adapter 发布、SHA、vLLM 加载/卸载；
- 两种候选构造、PE、有限 loss/gradient、optimizer step；
- 更新后 adapter 再发布和最长8条 validation。

`run_all.sh` 前台依次执行：

```text
resolve 两条 formal config
  → 全量 server pytest
  → SFT+rollout strong smoke
  → rollout-only strong smoke
  → SFT+rollout formal
  → rollout-only formal
  → independent test evaluation
  → Round2 aggregate/export
```

`start_all.sh` 用独立 session 在后台运行同一长链，使 SSH 断开不影响进程。任一步失败都阻断后续；不会覆盖失败证据，也不会自动猜测恢复点。

正式每条方法按第一轮相同的 drop-last 口径运行 `floor(24000/56)×2=856` 个 optimizer step。两条方法不能用同一个 smoke 耗时相互代替估算：SFT+rollout 每步生成56条，rollout-only每步生成112条。

## 7. 状态与停止边界

- 全链状态：`runs/<experiment>/controller.json`；
- 全链 PID/日志：`controller.pid`、`controller.log`；
- 单方法状态：`<method>/controller_status.json` 和 `state.json`；
- 训练日志：`<method>/logs/tp_train.log`；
- rollout 日志：`<method>/logs/vllm_worker.log`；
- 指标：`<method>/logs/metrics.jsonl`；
- TP 证据：`<method>/tp_evidence.json`。

`status_all.sh` 只读这些文件。`stop_all.sh` 默认只预览；带 `--execute` 后也只向本 experiment 记录的进程组发送 TERM，不使用 `pkill python` 或按账号批量结束。

## 8. 当前限制

- 尚未在真实 3×4090 上验证 Transformers TP、PEFT TP adapter 保存、DTensor optimizer/clip 或 vLLM adapter round-trip。
- 每步在线 rollout 与 adapter 发布会显著增加 wall time 和存储；实际耗时由 strong smoke 和首个 formal step 校准。
- 当前保留每步 LoRA adapter，但不保存 optimizer/scheduler state，因此不能声称 bit-exact 热恢复。
- 第一轮 A800 DDP pipeline 与第二轮 4090 TP pipeline 是两个独立入口；本实现不会让第一轮任务自动迁移或重跑。
