# 训练流程（中文阅读版）

```yaml
view_type: stable_methodology_ch
source_file: training_protocol.md
last_organized: 2026-06-16
```

> 本文件是 `training_protocol.md` 的中文阅读版，内容与源文件保持一致。

## 训练分支

| 分支 | 内容 | 关键配置 |
|---|---|---|
| `theta0`（cold start） | 学生模型在 `n_cold=256` 条 prompt 上做 OPD-like 训练（`lmbda=1.0`），作为所有 OPD arm 的共同起点，也是几何漂移的对照基线。 | `max_steps = n_cold // cold_steps_div`（256//4=64），`grad_accum=4` → 上报 `DataSize=256`。 |
| `opd_lmbda1` / `opd_lmbda05` | 从 `theta0` 的 merge 后 checkpoint 继续做 OPD-like 训练，`lmbda∈{1.0,0.5}`，`seed=42`。 | `opd_steps=200`，`opd_grad=4` → `DataSize=800`。两个 arm 只差 `lmbda`。 |
| `sft_n{128,256,512,1024}` | 从 `theta0` merge 后 checkpoint 继续做监督微调（continued SFT），监督样本数 ∈ {128,256,512,1024}（chat 渲染后的 teacher 补全）。 | `learning_rate=1e-5`（OPD arm 是 3e-5），`num_train_epochs=3`，cosine schedule，`warmup_ratio=0.05`，按 held-out `eval_loss` 选最优 checkpoint。 |

所有分支共用同一套 LoRA 形状（`r=16, alpha=32, dropout=0.05`，`q/k/v/o/gate/up/down_proj`，见 `model_and_data-ch.md`），保证几何/能力对比不被 adapter 容量差异混淆。

## OPD-like 训练机制

- 通过 TRL 的 on-policy-distillation trainer 实现（`trl_runner.py`），每个 arm 由生成的 YAML 配置驱动。
- `max_length=4096`，`max_completion_length=512`，`per_device_train_batch_size=1`，`bf16=True`，优化器 `adamw_torch`，`gradient_checkpointing=False`。
- **vLLM colocate 加速**：非 smoke 运行默认开启。学生 rollout（OPD 训练中的采样补全）走同进程内 vLLM 引擎（continuous batching），不走 HF `generate`：
  - `vllm_mode: colocate`，`vllm_gpu_memory_utilization=0.30`，`vllm_max_model_length=4096`，`vllm_tensor_parallel_size=1`。
  - `vllm_enable_sleep_mode=True`：optimizer 步时让 vLLM 引擎休眠，把显存让给 teacher 模型和反向传播。
  - `vllm_sync_frequency=1`：每个 optimizer step 把 LoRA 权重重新同步进 vLLM 引擎（保证 rollout 始终反映最新策略）。
  - 单卡 48G 运行，Cycle 04 未观察到 OOM。
- **进程隔离**：每个 OPD arm（含 `theta0`）都在**独立子进程**中训练（通过 `subprocess.run` 调 `run_trl_distill_smoke.py`），不在主脚本进程内直接跑。原因：vLLM colocate 会初始化 `torch.distributed`（NCCL）；同一进程内串行启动第二个 colocate 训练会撞上已销毁/复用的进程组，且 distributed 初始化状态下加载 PEFT adapter 会撞 `EmbeddingParallel`。子进程退出即清理进程组；LoRA adapter 的合并放在父进程做（父进程从不初始化 distributed），两步都保持干净。

## Continued-SFT 机制

- 用 TRL 标准 `SFTTrainer`/`SFTConfig` 实现（不是 distillation trainer）——也就是说 continued SFT 是对 chat 渲染后的 teacher 补全做普通监督 next-token 训练，不是 teacher-feedback 方法。
- 从 `theta0` 的 merge checkpoint 起训，LoRA 配置与 OPD arm 相同。
- `gradient_checkpointing=True`（OPD arm 是 `False`），`max_grad_norm=0.5`，`lr_scheduler_type=cosine`，`dataset_text_field="text"`，`packing=False`。
- 用同一统一数据池里固定的 64 条 held-out 切片算 `eval_loss` 来选 checkpoint（`load_best_model_at_end=True`，`metric_for_best_model=eval_loss`）——这正是 `evaluation_protocol-ch.md` 中"matched-gain"选择所依赖的底层机制。

## 未确认事项

- Cycle 03 训练是否用了同样的 vLLM-colocate / 子进程隔离模式，还是更早期的非 colocate 路径，尚未对 `run_opd_minimal_closure.py`（v1）逐行核对。v1/v2 的差异只在 `run_opd_minimal_closure_v2.py` 模块 docstring（5 项方法学修复）里有概述，本文档未做完整 diff。在核对之前，**Cycle 03 的流程细节置信度低于 Cycle 04/05**。
