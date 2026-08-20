# 当前理论目录

```yaml
view_type: current_workspace_section_ch
section: theory
status: active
source_file: README.md
last_organized: 2026-06-16
```

本目录是当前 theory 和 metric-definition 入口。它应自包含：读者不需要打开旧 cycle 就能理解当前 hypothesis 或 geometry metrics。

## 当前文件

| 文件 | 作用 | 主要来源 |
|---|---|---|
| `paper_thesis_framework.md` | **论文宪法**：冻结论点句（2026-07-13）+ claim 树（C1–C8，证据等级 ★/◐/○）+ 相关工作对位 + 未决依赖账本 + 论文骨架映射。写作阶段只从此树取主张。 | Cycle 09 论点讨论 |
| `current_theory_update.md` / `current_theory_update-ch.md` | 分层假设：base-aware K1（入场条件）→ Cycle 06 SFT-feasibility 前置条件 → Cycle 07 轨迹机制 → Cycle 08 active H-OPD test → Cycle 09 被抢发后的 Tier-0 贡献重定位（分层 framing）+ mini-round 裁决与 claim 边界 1–20。 | Cycle 05–09 theory updates |
| `geometry_metric_definitions.md` / `geometry_metric_definitions-ch.md` | GetSlice 指标定义与解释边界；Cycle 07 新增轨迹/子空间稳定性指标；Cycle 08 OverlapLift matched-LoRA 相对信号（后撤回，见事后注）；Cycle 09 激活实测旋转 θ_r（UV 开启）、层级化（L18-headline）读法、对 2605.30524 的双 ER 构念、探针电池 v2.1、adapter-first fp32(B@A) 权重测量修正。 | Cycle 03 definitions + Cycle 04–09 usage |

## 维护规则

- 稳定定义应跨 cycle 保留。
- 如果新实验改变指标解释，更新 interpretation notes，而不是删除指标。
- 子空间稳定性：γ_r = σ_r − σ_{r+1}（Davis–Kahan）是容量代理；Cycle 09 起实测旋转 θ_r 也可用（UV 保存开启）。禁止把 γ_r 写成 θ_r。
- Cycle 07 把 OOD-drop 重构为 checkpoint / 优化阶段属性；不要写成方法的固有属性。
- OverlapLift 作为 on/off-principal 信号已撤回（Cycle 08 事后注）；权重方向指标是 ρ（`result/cycle08_rho_metric_spec.md`），Cycle 09 修正为 adapter-first fp32(B@A) 计算。A09 出结果前不做任何 on/off-principal 声明。
- 任何"压缩"声明必须注明构念：白化输出相关谱（本项目，SVD-LLM lens）vs raw 表示性 ER（2605.30524）。
