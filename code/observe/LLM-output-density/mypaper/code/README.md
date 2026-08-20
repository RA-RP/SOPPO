# code/ — 代码对话当前工作面

```yaml
view_type: current_workspace_section
section: code
status: active
```

本目录是代码编写对话的入口和工作面，取代了原来的 `code_handoffs/` 和 `code_handins/`。

## 目录文件说明

| 文件 | 类型 | 说明 |
|---|---|---|
| `current_code_structure.md` | 持续维护 | 当前代码库结构的完整说明（脚本、包结构、运行方式、协议）。每 cycle 追加变更节，不全文重写。 |
| `code_evolution.md` | append-only | 跨 cycle 演进记录：每轮新增/删除/修改了什么代码，以及原因。 |
| `current_code_brief.md` | 覆盖维护 | 当前 cycle 的具体代码任务说明（要跑什么、输入输出路径、copyback 规则）。每次 Experiment Design 对话更新。 |
| `QA_<cycle>.md` | 每 cycle 一份 | 代码对话开始前的 QA 讨论记录。讨论确认后才开始写代码；结论内化到其他文件后保留原文，不删除。 |
| `cycle09_reviewer_robustness_handoff.md` | 当前专项转交 | 将最新 reviewer 的稳健性问题冻结为 RR0--RR6 可执行任务；先做零训练 preflight，任何新前向需 Theory GO。 |

## 代码对话启动顺序

```
readme.md（工作区整体规则）
↓
code/current_code_brief.md（当前任务说明）
↓
code/current_code_structure.md（当前代码库完整说明）
↓
exp/experimental_methodology/（4 文件，稳定协议层）
↓
exp/current_experiment_design.md（本轮设计规格）
↓
local_experiment_results/<上一 cycle>/（先验结果，如需要）
```

## 每 cycle 工作流

1. **Experiment Design 对话**写 `current_code_brief.md`，列出本轮任务、输入路径、输出要求、copyback 规则。
2. **代码对话启动前**，在 `code/` 下新建 `QA_<cycle>.md`，提出需要与用户确认的问题（协议参数、路径、不确定的设计决定等）。
3. **用户回答**后，双方讨论，达成共识后将结论内化回 `current_code_brief.md`（或 `current_code_structure.md` 的稳定内容），再开始执行代码。
4. 执行完成后：
   - 将关键 artifact copyback 到 `local_experiment_results/<cycle>/`。
   - 将实际执行情况（实际命令、路径偏差、发现的问题）追加到 `code_evolution.md` 的本 cycle 节的 "handin" 子节。
   - 在 `current_code_structure.md` 末尾追加本 cycle 新增/修改的代码说明。

## 维护规则

- `current_code_structure.md`：持续维护，每 cycle 只追加变更节，不重写历史内容。
- `code_evolution.md`：append-only，不改写历史条目。
- `current_code_brief.md`：每 cycle 覆盖重写，旧版本内容已在 `code_evolution.md` 中保留摘要。
- `QA_<cycle>.md`：讨论完成后保留原文，不删除，不改写。
