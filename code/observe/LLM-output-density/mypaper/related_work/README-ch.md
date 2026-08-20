# 相关工作

```yaml
view_type: current_workspace_section_ch
section: related_work
status: active
last_updated: 2026-06-18
```

本目录是项目的文献库。按以下四个文件维护：

## 当前文件

| 文件 | 职责 | 维护规则 |
|---|---|---|
| `related_work.md` / `related_work-ch.md` | **详述**：每篇论文的具体信息（标题、作者、年份、URL、内容概括、与本项目的关联） | 每轮新增条目；确认不再使用时才删除（谨慎）|
| `source_matrix.md` / `source_matrix-ch.md` | **综述**：分类介绍论文间关系，说明为什么是这一组论文；含综述矩阵表格 | 每轮新增分类或更新综述文字；table 随详述同步 |
| `incremental_related_work.md` / `incremental_related_work-ch.md` | **增量记录**：每轮新增/删除了哪些论文及原因 | append-only，每轮 cycle 收尾追加一节 |
| `source_notes/` | 特殊情况的扩展笔记（超出标准条目格式的详细设计笔记） | 按需新增，不覆盖已有文件 |

## 维护规则

- **Related Work 对话**的写入范围仅限本目录。
- 本目录由 Related Work 对话维护；Experiment Design 对话如采纳新协议来源，可直接追加到 `source_matrix.md` 的 Eval Protocol 区块。
- 删除论文前须确认：(1) 该论文不在任何 paper_drafts 引用中；(2) 在 `incremental_related_work.md` 中注明删除原因。
- `-ch.md` 中文版文件与对应英文版保持同步。

## Related Work 对话启动顺序

（按此顺序读取，再开始写作）

1. `../readme.md`（整体规则和读写边界）
2. `../next_cycle/current_open_questions.md` ← 指明下一步方向，是搜索新文献的起点
3. `../result/current_picture.md`
4. `../result/claims_allowed.md`
5. `../theory/current_theory_update.md`
6. 本目录：`related_work.md` → `source_matrix.md` → `incremental_related_work.md`
