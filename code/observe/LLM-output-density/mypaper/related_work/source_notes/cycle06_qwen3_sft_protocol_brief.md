# Cycle 06 Source Note: Qwen3-1.7B / Qwen3-1.7B-Base SFT Protocol Brief

```yaml
artifact_type: related_work_source_note
cycle: cycle_06_sft_feasibility_and_degradation
status: active
created: 2026-06-18
scope: protocol_and_baseline_grounding_before_experiment_design
write_owner: related_work
```

This note records engineering-facing source context for Cycle 06. It is not an
experiment design and should not be treated as a pre-registered protocol. Its
purpose is to prevent another format/scorer mismatch before Experiment Design.

---

## Source Set

### Official Qwen Sources

1. **Qwen3 Technical Report**
   - URL: https://arxiv.org/abs/2505.09388
   - Relevant points:
     - Qwen3-1.7B-Base is evaluated among edge-side base models.
     - The report's Table 8 gives Qwen3-1.7B-Base reference scores including MMLU 62.63, GSM8K 75.44, and MATH 43.50 under Qwen's official setup.
     - The report describes post-training as a multi-stage process and notes that smaller models use strong-to-weak distillation.

2. **Qwen3 official blog / usage guide**
   - URL: https://qwenlm.github.io/blog/qwen3/
   - Relevant points:
     - Qwen3 post-training includes long-CoT cold start, reasoning RL, thinking-mode fusion, and general RL.
     - Post-trained Qwen3 supports both thinking and non-thinking modes.
     - Official examples use `tokenizer.apply_chat_template(..., enable_thinking=True/False)`.

3. **Qwen/Qwen3-1.7B model card**
   - URL: https://huggingface.co/Qwen/Qwen3-1.7B
   - Relevant points:
     - This is the post-trained checkpoint, not the pretraining-only base checkpoint.
     - `enable_thinking=True` is the default in `apply_chat_template`.
     - `enable_thinking=False` switches to non-thinking mode.
     - Generated outputs may contain thinking content before `</think>` and final content after it.

4. **Qwen/Qwen3-1.7B-Base model card**
   - URL: https://huggingface.co/Qwen/Qwen3-1.7B-Base
   - Relevant points:
     - Training stage is `Pretraining`.
     - Model has 1.7B parameters, 28 layers, GQA 16/8 heads, and 32k context.
     - Evaluation details are delegated to Qwen's official report/blog.

### Community Engineering Sources

1. **HuggingFaceTB/qwen3-1.7b-gsm8k-sft**
   - URL: https://huggingface.co/HuggingFaceTB/qwen3-1.7b-gsm8k-sft
   - Relevant points:
     - Fine-tuned from Qwen/Qwen3-1.7B for GSM8K-style math reasoning.
     - Uses 247,467 examples from GSM8K train and MetaMathQA GSM-related examples.
     - Training format is conversational messages with `<think>...</think>` reasoning.
     - Reports 77.2% GSM8K and 55.2% MATH-500, with a stated base comparison of ~20% GSM8K and 62.0% MATH-500.
   - Caution:
     - This is a community model card, not an official Qwen benchmark.
     - The reported base comparison conflicts with some official/local score regimes, so it is best used as a protocol-sensitivity warning rather than a direct target.

2. **Unsloth Qwen3 run/fine-tune guide**
   - URL: https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune
   - Relevant points:
     - Summarizes practical thinking/non-thinking mode handling for Qwen3.
     - Notes that non-thinking mode uses empty `<think></think>` framing.
     - Warns against greedy decoding in thinking mode.
   - Caution:
     - Useful implementation reference, but lower authority than Qwen official sources.

---

## Cycle 06 Implications

1. **Treat Qwen3-1.7B and Qwen3-1.7B-Base as separate experimental objects.**
   - Qwen3-1.7B is post-trained and has chat-template + thinking-mode behavior.
   - Qwen3-1.7B-Base is pretraining-only and should not inherit chat/think assumptions unless Cycle 06 explicitly instruction-tunes it that way.

2. **For post-trained Qwen3-1.7B, train/eval format must be explicit.**
   - Decide whether SFT targets include `<think>...</think>` reasoning.
   - Decide whether evaluation uses `enable_thinking=True`, `enable_thinking=False`, or both as separate axes.
   - Preserve the exact chat template and any empty think block behavior in artifacts.

3. **For Qwen3-1.7B-Base, baseline evaluation must be protocol-matched.**
   - Base-model official scores are useful only when the local protocol matches the official setup.
   - Local differences should trigger protocol audit before capability claims.

4. **Community SFT baselines are useful as warning lights, not authorities.**
   - The HuggingFaceTB GSM8K SFT card suggests a targeted math SFT can improve GSM8K while lowering MATH-500.
   - This supports Cycle 06's plan to measure both target gain and adjacent math/OOD degradation.
   - It does not define a pass/fail threshold for this project.

5. **Do not mix local path names with official model identities.**
   - The local unfinetuned `/root/autodl-tmp/model/Qwen/Qwen3-1.7B` should not be described as `Qwen3-1.7B-Base` unless the checkpoint identity is verified.
   - Experiment Design should record exact checkpoint paths, Hugging Face IDs, tokenizer files, and chat-template behavior.

---

## Current Project Format: What Is Already Known

This section summarizes the current workspace protocol from
`exp/experimental_methodology/`. It is included here because Cycle 06 design
must know which parts are already validated and which parts change if the
student switches to `Qwen/Qwen3-1.7B-Base`.

### Model Identity And Training Format

| Model object | Current status in this workspace | Training format implication |
|---|---|---|
| Local `/root/autodl-tmp/model/Qwen/Qwen3-1.7B` | Post-trained / instruct Qwen3-1.7B, not official pretraining-only `Qwen3-1.7B-Base`. This is the "base" row in current results, meaning unfinetuned local student. | Current historical training examples are chat-rendered `messages` (`user` / `assistant`) through Qwen tokenizer chat template. Assistant targets contain step-by-step reasoning and `\boxed{...}`, but **zero `<think>` tags**. |
| `Qwen/Qwen3-1.7B` official HF ID | Post-trained model with hybrid thinking/non-thinking behavior and `enable_thinking` control. | If used in Cycle 06, chat template behavior and `enable_thinking` must be explicitly logged. The local historical choice corresponds to chat template + **non-thinking** targets. |
| `Qwen/Qwen3-1.7B-Base` official HF ID | Pretraining-only causal LM. It is not a chat/instruct model by default. | Do not assume chat template or `<think>` behavior. Cycle 06 must decide whether to train plain prompt/completion or introduce a chat-style SFT format. Official base scores are comparable only under matching base-model protocols. |

Current historical SFT format:

```text
input object: chat messages
rendering: tokenizer.apply_chat_template(..., add_generation_prompt=False)
assistant target: teacher completion with reasoning -> final \boxed{...}
think tags in target: none
therefore expected eval mode: apply chat template + enable_thinking=False
```

Key consequence: for the post-trained Qwen3-1.7B currently used in the project,
`enable_thinking=True` at evaluation is a train/eval mismatch, because the
training target did not include `<think>...</think>` wrappers. This is not
necessarily true for a new Cycle 06 run if the target format is changed.

### Current Validated Evaluation Format For The Post-Trained Qwen3-1.7B Stack

This table is the current local validated protocol after Cycle 05 correction.
It applies to the post-trained / instruct-style local Qwen3 stack unless
Experiment Design explicitly changes the model identity or training target
format.

| Task | Evaluation type | Chat template | `enable_thinking` | Token / few-shot | Extractor / scorer | Current interpretation |
|---|---|---:|---:|---|---|---|
| NuminaMath-test | generation | yes | `False` | 3072 tokens, 0-shot | last boxed answer + `math_verify` | Primary ID axis. |
| GSM8K | generation | yes | `False` | 3072 tokens, 0-shot | `extract_pred` -> last boxed / final answer | OOD / transfer math axis. Missing chat template was a Cycle 04 bug. |
| MATH500 | generation | yes | `False` | 3072 tokens, 0-shot | last boxed answer + `math_verify`; avoid lm-eval dollar-span extractor | OOD / transfer math axis. Old MATH500=0 was an extractor bug. |
| MMLU | loglikelihood multiple choice | no | `False` | 5-shot | lm-eval MC default | Base-model mode per Qwen report / lm-eval community. |
| TruthfulQA-MC1 | loglikelihood multiple choice | yes | `False` | 0-shot | lm-eval MC default | Capability-preservation check; not discriminative in current results. |
| WinoGrande | loglikelihood multiple choice | no | `False` | 0-shot | lm-eval default | Capability-preservation check; no chat template is the established local protocol. |
| ARC-challenge | loglikelihood multiple choice | no | `False` | 25-shot task default | lm-eval `acc_norm` | Capability-preservation check; not an OPD/SFT discriminator in current results. |

Practical reading:

- Generation math tasks for the current post-trained stack use **chat template + non-thinking mode**.
- MMLU, WinoGrande, and ARC use **no chat template** because they are loglikelihood/base-model-style multiple-choice evaluations in the current local protocol.
- TruthfulQA-MC1 is the current exception among OOD-lite tasks: it uses chat template with loglikelihood scoring.
- All four OOD-lite tasks are currently treated as capability-preservation checks because Cycle 05 found <1.5% spread across all 8 models.

### What Changes If Cycle 06 Uses `Qwen3-1.7B-Base`

The above task table should **not** be copied blindly to `Qwen/Qwen3-1.7B-Base`.
For the base model:

1. Baseline evaluation should first reproduce or sanity-check official
   base-model protocols where possible.
2. If SFT introduces a chat format, then post-SFT evaluation may need a
   separate chat-format protocol, but the pre-SFT base row should still be
   reported with its own protocol identity.
3. If SFT stays plain prompt/completion, then adding chat template at evaluation
   is likely a format artifact.
4. Experiment Design should record two separate rows when necessary:
   `base_model_protocol` and `post_sft_protocol`, instead of pretending one
   protocol applies to both.

---

## Open Protocol Questions For Experiment Design

1. Will Cycle 06 train both `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-1.7B-Base`, or select one primary student?
2. If training the post-trained model, is the target format chat + `<think>` reasoning, chat + empty think block, or no-think final answer only?
3. If training the base model, will SFT introduce a chat format, or use plain prompt/completion?
4. Which official/community score should be treated as the closest sanity-check baseline for each model identity?
5. Which scorer/extractor audit must run before interpreting GSM8K, MATH500, NuminaMath, and MMLU numbers?
