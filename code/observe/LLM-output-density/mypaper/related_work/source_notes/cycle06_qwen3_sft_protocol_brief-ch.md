# Cycle 06 来源笔记：Qwen3-1.7B / Qwen3-1.7B-Base SFT 协议简报

```yaml
artifact_type: related_work_source_note_ch
cycle: cycle_06_sft_feasibility_and_degradation
status: active
created: 2026-06-18
scope: protocol_and_baseline_grounding_before_experiment_design
write_owner: related_work
```

本笔记记录 Cycle 06 进入实验设计前需要知道的工程来源背景。它不是实验设计，也不是预注册协议。它的作用是避免再次出现训练格式、评测格式或 scorer 不匹配。

---

## 来源集合

### Qwen 官方来源

1. **Qwen3 Technical Report**
   - URL: https://arxiv.org/abs/2505.09388
   - 相关点：
     - Qwen3-1.7B-Base 在 edge-side base models 中评测。
     - 技术报告 Table 8 给出 Qwen3-1.7B-Base 的参考分数，包括 MMLU 62.63、GSM8K 75.44、MATH 43.50（均依赖 Qwen 官方协议）。
     - 报告描述了 post-training 多阶段流程，并说明小模型使用 strong-to-weak distillation。

2. **Qwen3 官方博客 / 使用说明**
   - URL: https://qwenlm.github.io/blog/qwen3/
   - 相关点：
     - Qwen3 post-training 包含 long-CoT cold start、reasoning RL、thinking-mode fusion、general RL。
     - Post-trained Qwen3 支持 thinking 与 non-thinking 两种模式。
     - 官方示例使用 `tokenizer.apply_chat_template(..., enable_thinking=True/False)`。

3. **Qwen/Qwen3-1.7B model card**
   - URL: https://huggingface.co/Qwen/Qwen3-1.7B
   - 相关点：
     - 这是 post-trained checkpoint，不是 pretraining-only base checkpoint。
     - `apply_chat_template` 中 `enable_thinking=True` 是默认值。
     - `enable_thinking=False` 进入 non-thinking mode。
     - 生成输出可能在 `</think>` 前包含 thinking content，之后包含 final content。

4. **Qwen/Qwen3-1.7B-Base model card**
   - URL: https://huggingface.co/Qwen/Qwen3-1.7B-Base
   - 相关点：
     - Training stage 是 `Pretraining`。
     - 模型有 1.7B 参数、28 层、GQA 16/8 heads、32k context。
     - 评测细节指向 Qwen 官方报告/博客。

### 社区工程来源

1. **HuggingFaceTB/qwen3-1.7b-gsm8k-sft**
   - URL: https://huggingface.co/HuggingFaceTB/qwen3-1.7b-gsm8k-sft
   - 相关点：
     - 从 Qwen/Qwen3-1.7B fine-tune 得到，目标是 GSM8K-style math reasoning。
     - 使用 247,467 个样本，来源为 GSM8K train 与 MetaMathQA GSM-related examples。
     - 训练格式是带 `<think>...</think>` reasoning 的 conversational messages。
     - 报告 77.2% GSM8K 与 55.2% MATH-500，并声称 base comparison 为约 20% GSM8K 与 62.0% MATH-500。
   - 注意：
     - 这是社区 model card，不是 Qwen 官方 benchmark。
     - 其 base comparison 与某些官方/本地分数 regime 不一致，因此最好作为协议敏感性警告，而不是直接目标分数。

2. **Unsloth Qwen3 run/fine-tune guide**
   - URL: https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune
   - 相关点：
     - 总结 Qwen3 thinking/non-thinking mode 的实际处理。
     - 指出 non-thinking mode 使用空 `<think></think>` framing。
     - 提醒 thinking mode 不应使用 greedy decoding。
   - 注意：
     - 可作为实现参考，但权威性低于 Qwen 官方来源。

---

## 对 Cycle 06 的含义

1. **把 Qwen3-1.7B 与 Qwen3-1.7B-Base 当作不同实验对象。**
   - Qwen3-1.7B 是 post-trained，有 chat-template 和 thinking-mode 行为。
   - Qwen3-1.7B-Base 是 pretraining-only，除非 Cycle 06 明确将其 instruction-tune 到该格式，否则不应继承 chat/think 假设。

2. **对 post-trained Qwen3-1.7B，训练/评测格式必须显式写清楚。**
   - 决定 SFT target 是否包含 `<think>...</think>` reasoning。
   - 决定评测使用 `enable_thinking=True`、`enable_thinking=False`，还是二者作为不同轴。
   - 在 artifact 中保留完整 chat template 和空 think block 行为。

3. **对 Qwen3-1.7B-Base，baseline 评测必须协议匹配。**
   - 官方 base-model 分数只有在本地协议与官方设置一致时才可比较。
   - 本地分数若偏离，先触发 protocol audit，再讨论能力差异。

4. **社区 SFT baseline 是 warning light，不是权威标准。**
   - HuggingFaceTB GSM8K SFT card 暗示：目标数学 SFT 可能提升 GSM8K，同时降低 MATH-500。
   - 这支持 Cycle 06 同时测目标增益和相邻数学/OOD 退化。
   - 它不能定义本项目 pass/fail threshold。

5. **不要混淆本地路径名和官方模型身份。**
   - 本地 unfinetuned `/root/autodl-tmp/model/Qwen/Qwen3-1.7B` 不应直接写成 `Qwen3-1.7B-Base`，除非 checkpoint 身份已核实。
   - Experiment Design 应记录精确 checkpoint path、Hugging Face ID、tokenizer 文件和 chat-template 行为。

---

## 当前项目格式：已经知道什么

本节总结 `exp/experimental_methodology/` 中当前已经稳定记录的协议。把它放在这里，是因为 Cycle 06 设计必须知道哪些部分已经验证，哪些部分在切换到 `Qwen/Qwen3-1.7B-Base` 后会改变。

### 模型身份与训练格式

| 模型对象 | 当前工作区状态 | 训练格式含义 |
|---|---|---|
| 本地 `/root/autodl-tmp/model/Qwen/Qwen3-1.7B` | Post-trained / instruct Qwen3-1.7B，不是官方 pretraining-only `Qwen3-1.7B-Base`。当前结果表里的 "base" 行指的是这个本地 student 未 finetune 的状态。 | 历史训练样本是经 Qwen tokenizer chat template 渲染的 `messages`（`user` / `assistant`）。Assistant target 包含 step-by-step reasoning 和 `\boxed{...}`，但 **没有 `<think>` tag**。 |
| `Qwen/Qwen3-1.7B` 官方 HF ID | Post-trained model，有 hybrid thinking/non-thinking 行为和 `enable_thinking` 控制。 | 如果 Cycle 06 使用它，必须记录 chat template 行为和 `enable_thinking`。本项目历史选择对应 chat template + **non-thinking** target。 |
| `Qwen/Qwen3-1.7B-Base` 官方 HF ID | Pretraining-only causal LM，默认不是 chat/instruct model。 | 不能假设 chat template 或 `<think>` 行为。Cycle 06 必须决定是 plain prompt/completion 训练，还是引入 chat-style SFT 格式。官方 base 分数只有在协议匹配时才可比较。 |

当前历史 SFT 格式：

```text
input object: chat messages
rendering: tokenizer.apply_chat_template(..., add_generation_prompt=False)
assistant target: teacher completion with reasoning -> final \boxed{...}
think tags in target: none
therefore expected eval mode: apply chat template + enable_thinking=False
```

关键结论：对当前项目使用的 post-trained Qwen3-1.7B 来说，评测时 `enable_thinking=True` 是 train/eval mismatch，因为训练 target 没有 `<think>...</think>` wrapper。若 Cycle 06 改变 target format，这条规则需要重新审查。

### 当前 post-trained Qwen3-1.7B stack 的已验证评测格式

下表是 Cycle 05 修正后当前本地已验证协议。它适用于 post-trained / instruct-style 本地 Qwen3 stack，除非 Experiment Design 明确改变模型身份或训练 target format。

| 任务 | 评测类型 | Chat template | `enable_thinking` | token / few-shot | extractor / scorer | 当前解释 |
|---|---|---:|---:|---|---|---|
| NuminaMath-test | generation | yes | `False` | 3072 tokens, 0-shot | last boxed answer + `math_verify` | Primary ID axis。 |
| GSM8K | generation | yes | `False` | 3072 tokens, 0-shot | `extract_pred` -> last boxed / final answer | OOD / transfer math axis。Cycle 04 漏 chat template 是 bug。 |
| MATH500 | generation | yes | `False` | 3072 tokens, 0-shot | last boxed answer + `math_verify`；避免 lm-eval dollar-span extractor | OOD / transfer math axis。旧 MATH500=0 是 extractor bug。 |
| MMLU | loglikelihood multiple choice | no | `False` | 5-shot | lm-eval MC default | 按 Qwen report / lm-eval community 采用 base-model mode。 |
| TruthfulQA-MC1 | loglikelihood multiple choice | yes | `False` | 0-shot | lm-eval MC default | Capability-preservation check；当前结果中不具备判别力。 |
| WinoGrande | loglikelihood multiple choice | no | `False` | 0-shot | lm-eval default | Capability-preservation check；本地稳定协议是不加 chat template。 |
| ARC-challenge | loglikelihood multiple choice | no | `False` | 25-shot task default | lm-eval `acc_norm` | Capability-preservation check；当前不是 OPD/SFT 判别轴。 |

实际读法：

- 当前 post-trained stack 的数学生成任务使用 **chat template + non-thinking mode**。
- MMLU、WinoGrande、ARC 使用 **no chat template**，因为它们在当前本地协议中是 loglikelihood/base-model-style multiple-choice 评测。
- TruthfulQA-MC1 是 OOD-lite 中的例外：使用 chat template + loglikelihood scoring。
- 四个 OOD-lite 任务当前都只作为 capability-preservation check，因为 Cycle 05 发现 8 个模型间 spread 均 <1.5%。

### 如果 Cycle 06 使用 `Qwen3-1.7B-Base`，什么会改变

上表不能直接照搬到 `Qwen/Qwen3-1.7B-Base`。对 base model：

1. Baseline evaluation 应优先复现或 sanity-check 官方 base-model protocol。
2. 如果 SFT 引入 chat format，那么 post-SFT 评测可能需要单独的 chat-format protocol；但 pre-SFT base row 仍应保留自己的 protocol identity。
3. 如果 SFT 采用 plain prompt/completion，那么评测时加 chat template 很可能是格式 artifact。
4. Experiment Design 必要时应记录两套 protocol row：`base_model_protocol` 和 `post_sft_protocol`，不要假装一个协议同时适用于两者。

---

## 留给 Experiment Design 的开放协议问题

1. Cycle 06 是同时训练 `Qwen/Qwen3-1.7B` 与 `Qwen/Qwen3-1.7B-Base`，还是只选择一个主 student？
2. 如果训练 post-trained model，target format 是 chat + `<think>` reasoning、chat + 空 think block，还是 no-think final answer？
3. 如果训练 base model，SFT 是否引入 chat format，还是使用 plain prompt/completion？
4. 哪个官方/社区分数最适合作为每个模型身份的 sanity-check baseline？
5. 在解释 GSM8K、MATH500、NuminaMath、MMLU 前，必须运行哪些 scorer/extractor audit？
