# 模型与数据（中文阅读版）

```yaml
view_type: stable_methodology_ch
source_file: model_and_data.md
last_organized: 2026-06-16
```

> 本文件是 `model_and_data.md` 的中文阅读版，内容与源文件保持一致；事实依据见源文件 `grounded_in` 列表。

## 模型身份

| 角色 | Checkpoint | 说明 |
|---|---|---|
| Teacher | `Qwen3-4B`（instruct） | 用于 OPD teacher feedback，以及生成 `theta0`/cold-start 的 S 探针 rollout。 |
| Student / base | `Qwen3-1.7B`（**instruct 版**，不是官方纯 pretrain 的 `Qwen3-1.7B-Base`） | 所有"未训练 base"行都是这个 instruct checkpoint 的未微调版本。这一点在跟 Qwen 官方 base 模型分数对比时非常关键（见 `FINDING_05_gsm8k_chat_template_mismatch.md`）。 |
| Adapter | LoRA `r=16, alpha=32, dropout=0.05, bias=none`，`target_modules=[q,k,v,o,gate,up,down_proj]` | OPD 训练和 continued-SFT 训练用同一套 LoRA 配置，避免几何/训练对比被 adapter 容量差异混淆。 |

## Chat / Think 格式（导致 Cycle04→05 测量 bug 的关键点）

- 所有训练数据和 rollout 都通过 tokenizer 的 chat template 构造。任何跳过 chat template 的生成/训练都是**协议 bug**，不是合法的 ablation——Cycle 04 的 GSM8K/OOD-lite eval 正是踩了这个坑（见 `evaluation_protocol-ch.md`）。
- 监督目标数据（`train_sft.jsonl`，既是 continued-SFT 训练数据，也是产生 `theta0` 的 teacher 补全目标）**完全没有** `<think>` 标签：每条 teacher 补全都是"题目→逐步推理→`\boxed{...}`"，没有 thinking block。
- 因此本项目所有模型都必须在**非 think 模式**（`enable_thinking=False`）下训练和评测。Qwen3 chat template 默认 `enable_thinking=True`；评测时不关掉会重新引入训练/评测格式不匹配（已有实证：只加 chat template 不关 thinking，GSM8K 仍卡在 ~0.39-0.43，同时加上 `enable_thinking=False` 才回升到 ~0.70-0.72）。
- **固定规则**：本项目新增任何评测任务，都必须显式设置 `apply_chat_template=True` 且 `enable_thinking=False`，不能假设任务默认行为已经满足——GSM8K/MATH500 的 bug 都是栽在沉默的默认值上。

## 训练/探针数据

| 数据池 | 来源 | 作用 |
|---|---|---|
| `train.parquet`（NuminaMath-1.5） | 训练侧 prompt 总来源 | cold-start(`theta0`)、OPD rollout prompt、continued-SFT 监督目标都来自这里。通过 `build_unified_pool` 一次性切分成统一种子的数据池，cold-start/OPD/SFT/held-out-eval/GetSlice-probe 各取互不重叠的切片，避免分别采样带来的 bias。 |
| `test.jsonl`（NuminaMath-1.5） | ID 留出评测集 | 已核实与训练行 ID 不重叠（`train_row_ids ∩ test_row_ids = 0`）。这是本项目主 ID 轴（见 `evaluation_protocol-ch.md`）。 |
| SFT 监督格式 | chat `messages`（`user`/`assistant`），经 `apply_chat_template(..., add_generation_prompt=False)` 渲染 | assistant 轮是 teacher 的完整非 think 补全；这正是 continued-SFT 训练以及 theta0 cold-start 训练所对齐的目标文本字面值。 |

## 未确认事项

- OPD-like distillation loss（`lmbda`/`beta`/`loss_top_k` 的精确语义）来自 TRL 的 `DistillationConfig`/`DistillationTrainer`（见 `trl_runner.py`），尚未在本工作区里转写成显式公式。**需要在后续代码对话中确认**，在此之前不要在论文中给出该 loss 的数学描述。
