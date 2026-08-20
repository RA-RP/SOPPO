# 评测协议（中文阅读版）

```yaml
view_type: stable_methodology_ch
source_file: evaluation_protocol.md
last_organized: 2026-06-18
```

> 本文件是 `evaluation_protocol.md` 的中文阅读版，内容与源文件保持一致。本文件存在的直接原因：GSM8K 漏加 chat template、MATH500 抽取器损坏这两个真实 bug，都是因为协议从未被完整写在一处文档里才迟迟没被发现。读任何新分数前先看这份文档，协议一变就立刻更新。

## 三轴协议框架（cycle05 建立，2026-06-17 QA 确认）

每个评测任务都必须沿以下三个轴明确指定。未核查全部三轴是 GSM8K 和 MMLU 失败的直接原因。

| 轴 | 含义 | 影响 |
|---|---|---|
| **think 轴** | `enable_thinking=True/False` | 生成任务：False 防止 think-block；loglikelihood 任务：True + think_end_token 或切换 generative |
| **chat format 轴** | `--apply_chat_template` 开/关 | instruct 模型的生成任务需要；WinoGrande 是已确认例外（无 chat template——field standard）；loglikelihood 多选任务开 chat template 会扭曲 loglikelihood 比较 |
| **token cutoff 轴** | 生成任务的 `max_gen_toks` | CoT + boxed 答案需要 3072；loglikelihood 任务不涉及 |

**任务类型分类：**
- **生成任务**（GSM8K、MATH500、NuminaMath-test）：三轴都要管
- **loglikelihood 任务**（TruthfulQA-MC1、WinoGrande、MMLU、ARC-challenge）：token cutoff 不相关；think 轴和 chat 轴影响 loglikelihood 质量分布

**每任务协议（2026-06-17 QA 确认，MMLU 决定 2026-06-17 最终确定，ARC 2026-06-17 完成）：**

| 任务 | think | chat | token/fewshot | 抽取器 | 验证状态 |
|---|---|---|---|---|---|
| NuminaMath-test | `False` | ✅ | 3072 tok，0-shot | last_boxed + math_verify | ✅ cycle04 |
| GSM8K | `False` | ✅ | 3072 tok，0-shot | extract_pred → last_boxed | ✅ 8/8 模型（cycle05）|
| MATH500 | `False` | ✅ | 3072 tok，0-shot | last_boxed + math_verify | ✅ 8/8 模型（cycle05）|
| MMLU | `False` | ❌ **无 chat template** | loglikelihood，5-shot | lm_eval MC 默认 | ✅ 8/8 模型（cycle05，base model 模式，0.60-0.61）|
| TruthfulQA-MC1 | `False` | ✅ | loglikelihood，0-shot | lm_eval MC 默认 | ✅ 8/8 模型（cycle05，chat template）|
| WinoGrande | `False` | ❌ **无 chat template** | loglikelihood，0-shot | lm_eval MC 默认 | ✅ 8/8 模型（cycle05，无 chat template）|
| ARC-challenge | `False` | ❌ **无 chat template** | loglikelihood，25-shot（任务默认值）| lm_eval acc_norm | ✅ 8/8 模型（cycle05，0.4565–0.4710）|

> **MMLU format 决定（2026-06-17）：** MMLU 使用 **base model 模式**（不加 chat template），与 Qwen3 官方技术报告及 lm_eval 社区标准一致。原因：(1) lm_eval loglikelihood MMLU 对 A/B/C/D token 做直接对数概率比较，不涉及文本生成，chat template 的格式前缀对 loglikelihood 比较有干扰而非帮助；(2) `--apply_chat_template` 在未设置 `enable_thinking=False` 时将 MMLU 崩溃到随机基线 0.2295（FINDING_05_mmlu_chat_template_collapse.md）；(3) Qwen3-4B 官方 MMLU 72.99% 与 no-chat-template lm_eval 运行对齐（0.7014）；(4) 社区 lm-eval-harness Issue #3405、#3576 均确认 no-chat-template 是该任务的正确协议。全部 8 模型已在 base model 模式下完成重测（cycle05，2026-06-17），得分范围 0.6004-0.6086（`mmlu_protocol_version=2`）。

> **ARC-challenge 协议说明：** 25-shot 是 lm-eval arc_challenge 任务的默认 fewshot 数（未显式传 `--num_fewshot`；日志记录"Overwriting default num_fewshot of arc_challenge from None to 25"）。无 chat template，与 MMLU 和 WinoGrande 协议一致。`enable_thinking=False` 通过 model_args 传入。

## 任务-轴映射

| 任务 | 轴 | 状态 |
|---|---|---|
| NuminaMath-test（1024 行留出集） | **ID**（domain 内，主轴） | 从一开始协议就是对的（`component/numina/runner.py` 已应用 chat template + `enable_thinking=False`）。 |
| GSM8K（full，N=1319） | **OOD/迁移** | Cycle 04 跑的时候没加 chat template（协议 bug）。**全部 8 个模型已在 cycle05 完成修正重测。** |
| MATH500（hendrycks_math500，N=500） | OOD/迁移 | Cycle 04 全部模型 `0.000` 是抽取器 bug（$-span 抽取横跨整个 CoT 推理链）。**全部 8 个模型已在 cycle05 完成修正重测。** |
| OOD-lite：MMLU、TruthfulQA-MC1、WinoGrande、ARC-challenge | OOD-lite，次要稳定性轴 | **全部完成（cycle05，2026-06-17/18）。** MMLU: base model 模式，no chat template，5-shot，得分 0.6004-0.6086。TruthfulQA-MC1: chat template + loglikelihood，0-shot，得分 0.3317-0.3439。WinoGrande: no chat template，0-shot，得分 0.6172-0.6322。ARC-challenge: no chat template，25-shot（任务默认），acc_norm，N=1172，得分 0.4565-0.4710。OOD-lite 可引用，但注意：4 项任务 8 模型间 spread 均极小（<1.5%），仅可作为能力保持检验，不可作为 OPD vs SFT 判别轴。 |

## GSM8K 和 MATH500 为什么错了（Cycle 04），Cycle 05 怎么修的

- **GSM8K**：走 `lm_eval` 默认 `gsm8k` 任务，没加 chat template，却用在一个指令微调过的 chat 模型上。每个模型的 strict-match 都是 `0.000`。修复：专门的 `component/gsm8k/runner.py`，加 chat template、`enable_thinking=False`、`max_gen_toks=3072`、采样 `temp=0.7/top_p=0.8/top_k=20`——不是 `lm_eval` 自带的 flexible-extract 路径。
- **MATH500**：`lm_eval` 的抽取器通过截取生成全文中第一个和最后一个字面 `$` 字符之间的内容来提取答案——对于夹杂大量中间 LaTeX 的 CoT 回答，这会切到推理过程中段，几乎必然打 0 分。修复：专门的 `component/math500/runner.py`，对模型自己的回答用 `last_boxed_only_string`/`remove_boxed` + `math_verify` 符号等价判断。
- **两个修复叠加后**：GSM8K 从 ~0.41 升到 ~0.61-0.73，MATH500 从 `0.000` 升到 ~0.54-0.73（全部 8 个模型）。

## 固定规则：抽取有效性审计

任何近零分，或者某个轴上所有模型分数都诡异一致，都必须先排查抽取/prompting 假象，**之后**才能解读为能力结论。最小审计应检查：是否加了 chat template、`enable_thinking` 是否匹配训练格式、生成预算是否足够、抽取器是否拿至少一条人工核对过的正确生成验证过。

## 采样/生成设置

| 设置 | ID 轴（NuminaMath） | GSM8K/MATH500（Cycle05 修正后） | OOD-lite（MMLU/TruthfulQA/WinoGrande/ARC）|
|---|---|---|---|
| Chat template | 已加 | 已加 | MMLU: ❌ 无。TruthfulQA: ✅ 有。WinoGrande: ❌ 无。ARC: ❌ 无。 |
| `enable_thinking` | `False` | `False` | `False`（所有四个任务通过 model_args 传入）|
| 生成预算 | 3072 token（v2 clean） | 3072 token | 任务默认（loglikelihood，不涉及生成长度）|
| 采样 | temp 0.7/top_p 0.8/top_k 20 | temp 0.7/top_p 0.8/top_k 20 | 确定性评分（loglikelihood 多选任务）|
| Scorer | `math_verify` + 取最后一个 `\boxed{}`（`scorer_v2.py`） | 同样的 `math_verify`/boxed 模式（专用脚本） | `lm_eval` 任务默认 |

## OPD vs SFT 对照匹配规则

| 规则 | 定义 | 用途 |
|---|---|---|
| Base ceiling | 训练后变体 vs 未训练 base，带不确定性 | 任何稳定性结论前的强制 headroom 门 |
| Equal GSM8K | 按最近 `GSM8K_gain` 匹配（Cycle03/04 原规则） | 仅作连续性诊断——已知对 `theta0` 饱和度敏感 |
| Equal ID | 按 NuminaMath-test 分数在容差内匹配 | 主要的公平对照比较 |
| Equal budget | 训练/更新/样本预算可比 | 主要的公平对照比较 |

## 已解决事项（cycle05 关闭，2026-06-18）

原先所有待确认或未完成的事项现已全部解决：

- **OOD-lite 套件**：MMLU + TruthfulQA + WinoGrande + ARC-challenge 全部 8 个模型在正确协议下完成。数字可引用；注意判别力警告（4 项任务 8 模型间 spread <1.5%）。
- **抽取器实现**：`component/gsm8k/runner.py` 先调用 `scorer.py` 的 `extract_pred()`（丢弃 `<think>` 块，优先取最后一个 `\boxed{...}`，否则退到"answer is X"正则，再否则取最后一行非空文本），然后只在收窄后的片段上跑数字正则。`component/math500/runner.py` 用同一个 `extract_pred()` 片段但用 `scorer_v2.score()`（math_verify 符号等价）。脚本在 `Eval/component/` 下（cycle05 eval stack 统一化，2026-06-18）。
- **全部 8 个模型已修正**：`opd_lmbda1`、`sft_n256`、`sft_n512`、`sft_n1024` 均已在 cycle05 完成修正 GSM8K/MATH500 重测。完整 8 个模型结果见 `../../local_experiment_results/cycle_05_matched_control_id_ood/run_01/tables/id_ood_trajectory.csv`。
