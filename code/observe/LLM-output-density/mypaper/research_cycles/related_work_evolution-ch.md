# 相关工作演进（Cycle 03 → 05，中文阅读版）

```yaml
view_type: cross_cycle_evolution_log_ch
source_file: related_work_evolution.md
last_organized: 2026-06-16
```

> 本文件是 `related_work_evolution.md` 的中文阅读版，内容与源文件保持一致。本文件记录的是**每轮变了什么、为什么变**，不是逐轮全文复述——要看全文请跳到每条记录给出的路径。当前稳定的跨 cycle 文献定位（为什么是这组论文，不是某一轮的快照）在 `../related_work/current_source_matrix-ch.md`（"为什么是这组论文"一节）。

## Cycle 03 —— 起点

全文来源：`cycle_03_trl_opd_like/01_related_work/`（`related_work_before_experiment_03.md`、`related_work_search_03.md`、`source_notes/ood_id_positioning_with_ppl.md`）。

Cycle 03 建立了本项目至今最宽的文献基础，围绕三条至今仍是项目定位的主线（见 `../related_work/current_source_matrix-ch.md`）：OPD 机制类论文（GKD、MiniLLM、OPD 综述、Lightning-OPD、NPD、On-Policy Context Distillation）、SFT 退化/条件性论文（SFT Doesn't Always Hurt、Rethinking Generalization in Reasoning SFT）、几何/谱机制论文（The Path Not Taken、SVD-LLM/ASVD/FWSVD、LoRA vs Full Fine-tuning）。它还带着第四条、**不属于**当前 OPD-vs-SFT 主线的支线：一个用 PPL + 自定义几何分数为 synthetic math SFT 数据做筛选的设计（`source_notes/ood_id_positioning_with_ppl.md`）。这是用户自己更早期的工作，研究几何分数能否筛选出缓解 OOD 退化的 SFT 数据——它被保留为合法的 ID/OOD 相关工作和未来可能的扩展方向，不是被丢弃的废稿，只是还没并入当前的 K1/OPD-vs-SFT 主线。

## Cycle 03 → Cycle 04：收窄到可执行核心

全文来源：`cycle_04_opd_stability_gain/01_related_work/related_work_update_04.md`、`source_matrix_04.md`（= 当前 `../related_work/current_related_work_update-ch.md`、`current_source_matrix-ch.md`）。

变了什么：Cycle 03 那张横跨数十篇论文、分 must-cite/discussion/appendix 三档的大矩阵，被收窄成直接支撑 Cycle 04 实际跑的 matched-gain OPD-vs-SFT-with-geometry 设计的 7 篇核心文献。为什么：Cycle 04 是同一个 K1 问题的更严格重跑，是 feasibility 导向的，不是文献扩展轮——Cycle 03 那些只服务于 discussion 的更宽文献（数据筛选、calibration、loss landscape 替代解释）被搁置为本轮范围外，不是被判定为错。Cycle 04 还新增了一张"来自 Cycle 03 的本地证据"表，把 Cycle 03 的实证发现（matched pair gap、OOD-lite/drift 对比、theta0 饱和警告）显式带进 Cycle 04 的设计约束。

## Cycle 04 → Cycle 05：没有新检索，多了一份问题清单

全文来源：`cycle_05_matched_control_id_ood/01_related_work/new_questions_from_cycle04.md`（= 当前 `../next_cycle/current_open_questions-ch.md`）。

变了什么：Cycle 05 没有跑新的文献检索。相反，Cycle 04 的结果生成了一份结构化的问题清单（base-ceiling 构造、对照构造、ID/OOD 测量、几何/机制、论文定位），点名了*未来*某次相关工作轮该去找什么——例如在微调对比前先报告 base model baseline 的先例文献，以及 baseline 已经部分饱和时如何定义"matched"对照。截至本文撰写，这次后续检索还没有跑；它仍是留给恢复相关工作的那一轮的开放项。
