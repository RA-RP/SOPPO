# 当前结果

```yaml
view_type: current_workspace_section_ch
section: result
status: active
last_organized: 2026-06-16
```

本目录是最高层、最可读的结果入口，使用增量形式，每个文件回答一个问题。详细数字和证据在 `../local_experiment_results/<cycle_name>/` 中。

## 当前文件

| 文件 | 回答的问题 |
|---|---|
| `incremental_log-ch.md` | 每个 cycle 加了什么、改变了什么？ |
| `current_picture-ch.md` | 当前所有可用实验结论综合说明了什么？ |
| `claims_allowed-ch.md` | 哪些结论有支撑、哪些结论不允许声称？ |

## 维护规则

- `incremental_log-ch.md` 只追加，不改写历史小节。
- `current_picture-ch.md` 反映当前聚合状态：新结果落地或旧结果被修正时更新。
- `claims_allowed-ch.md` 是活跃边界：结论随结果变化时更新。
- 不要抹除负面结果。后续 cycle 可以修订，但历史 claim 边界应可追溯。
- Gate 指标和探索性指标必须分开。
- 引用 `../local_experiment_results/` 中的原始表格和图，不在这里直接嵌入大表。
