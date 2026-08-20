# 结果解读演进（Cycle 03 → 05，中文阅读版）

```yaml
view_type: cross_cycle_evolution_log_ch
source_file: result_interpretation_evolution.md
last_organized: 2026-06-18
```

> 本文件是 `result_interpretation_evolution.md` 的中文阅读版，内容与源文件保持一致。当前稳定的结果视图在 `../result/current_picture-ch.md`、`../result/claims_allowed-ch.md`、`../result/incremental_log-ch.md`。

## Cycle 03 —— 起点

结论：工程可行性通过，K1 得到**狭窄、初步**的支持。在找到的唯一一对有效 matched pair（`opd_lmbda1`(800) vs `sft_n256`，`GSM8K_gain_gap=0.010`）下，OPD 的 OOD-lite penalty 更低、worst OOD-lite drop 更低，在测量的全部 7 个模块上 layer-14 spectral drift 也比 matched SFT 更低。明确**不允许**的结论：完整 OPD 复现、GSM8K 提升声称、任何 principal-angle 声称。标注的局限：`eval_limit=200` 噪声大、只有一次 OPD 跑、几何只测了第 14 层。

## Cycle 03 → Cycle 04：更严格的重跑推翻了稳定性结论

变了什么：在新的预注册 gate 下，Gate A/B 通过但 C/D **失败**——"Cycle 03 的稳定性信号没有复现"。失败分析文档明确说明根因不是"OPD 变差了"，而是一种**对照锚点失效模式**：更不饱和的 `theta0` 把最近的 matched-gain SFT 逼到了 `sft_n128`，它几乎没离开 `theta0`，因此"稳定"得没有意义——OPD 输给了一个几乎没动的对照。同一轮还产出了两个独立的重大发现：(1) `MATH500=0`（所有模型）是**抽取假象**（scorer 的严格路径在这种输出格式上系统性失败），这促成了引入 NuminaMath-test 作为干净的 ID 轴；(2) 未训练 base 在这个新 ID 轴上**持平 `theta0`**，还高于两个 OPD arm——一个暗示当前设置下没有任何方法展现出净能力提升的 base-ceiling 发现。正式结论："工程上成功，预注册声称失败"——被重新定性为关于对照锚点对 `theta0` 饱和度敏感的方法学教训，不是对 OPD 价值本身的判决。

## Cycle 04 → Cycle 05（已关闭，2026-06-18）：base-ceiling 判断本身被修正

关键产物：`../local_experiment_results/cycle_05_matched_control_id_ood/FINDING_05_gsm8k_chat_template_mismatch.md`、`FINDING_05_math500_dollar_span_extraction_bug.md`、`FINDING_05_mmlu_chat_template_collapse.md`。完整结果视图：`../result/current_picture-ch.md`、`../result/claims_allowed-ch.md`、`../result/incremental_log-ch.md`。详细数字表格：`../local_experiment_results/cycle_05_matched_control_id_ood/run_01/tables/id_ood_trajectory.csv`。

**变了什么：** Cycle 05 从 reuse audit 开始。Cycle 04 的 GSM8K base-ceiling 发现最初被读作确认——同一组 GSM8K 数字（~0.41 全模型）说明 base ≈ theta0，无 headroom。一项预注册的 pre-pivot capability check（在下载 non-instruct base model 之前触发）把这些数字拿去和 Qwen 官方基准对比，发现差了约 35 分——追查后发现是 Cycle 04 的 GSM8K eval 一直没加 chat template（和一轮之前的 `MATH500=0` 是**同一种假象形态**）。三个叠加的评测协议 bug 在全部 8 个模型上被识别并修正：

1. **Finding 05a（GSM8K）：** 缺少 chat template + enable_thinking=False + 3072-token 预算。GSM8K: ~0.41 → ~0.61-0.73。推翻了 GSM8K 轴上的 base-ceiling / 无 headroom 读数。
2. **Finding 05b（MATH500）：** `$`-span 抽取器横跨 CoT 中的中间 LaTeX，不是最终 `\boxed{}`。MATH500: 0.000 → ~0.54-0.73。
3. **Finding 05c（MMLU）：** `--apply_chat_template` + 未设置 enable_thinking=False 将所有模型崩到 0.2295（4 选项随机猜测）。解决：base model 模式（无 chat template，5-shot，标准 loglikelihood）——Qwen3 官方协议（arXiv 2505.09388）。得分 0.60-0.61。

**完整 8 个模型修正后结果（cycle05）：**

| 模型 | NuminaMath ID | GSM8K | MATH500 | MMLU | TQA-MC1 | WinoGrande | ARC acc_norm |
|---|---|---|---|---|---|---|---|
| base | 0.5516 | 0.7005 | 0.7140 | 0.6061 | 0.3341 | 0.6172 | 0.4565 |
| theta0 | 0.5527 | 0.7180 | 0.7260 | 0.6019 | 0.3415 | 0.6235 | 0.4590 |
| opd_lmbda05 | 0.5392 | 0.7180 | 0.7080 | 0.6004 | 0.3390 | 0.6251 | 0.4650 |
| opd_lmbda1 | 0.5348 | **0.7346** | 0.7140 | 0.6061 | 0.3427 | 0.6235 | 0.4616 |
| sft_n128 | 0.5370 | 0.7096 | 0.7260 | 0.6049 | 0.3439 | 0.6290 | 0.4599 |
| sft_n256 | 0.5516 | 0.6839 | 0.7080 | 0.6033 | 0.3415 | 0.6235 | 0.4667 |
| sft_n512 | 0.5179 | 0.6732 | 0.7140 | 0.6086 | 0.3378 | 0.6322 | 0.4710 |
| sft_n1024 | 0.3890 | 0.6141 | 0.5400 | 0.6065 | 0.3317 | 0.6322 | 0.4667 |

**相比 Cycle 04 的关键结果转变：**
- **Headroom 恢复：** theta0 在 GSM8K（+0.0175）和 MATH500（+0.0120）两个轴上都领先 base。Cycle 04 的"无 headroom"读数完全是协议假象。
- **opd_lmbda1 GSM8K 最高：** 0.7346，+0.0341 over base，+0.0166 over theta0。但 NuminaMath ID 低于 base（0.5348 vs 0.5516）。
- **sft_n1024 异常值：** 数学轴严重退化（NuminaMath ID −0.163，GSM8K −0.086，MATH500 −0.174 vs base），OOD-lite 稳定。与 1.7B 模型在 n=1024 SFT 样本下的灾难性遗忘一致。
- **OOD-lite 非判别性：** 全部 4 项任务（MMLU、TruthfulQA、WinoGrande、ARC-challenge）在全 8 个模型间 spread <1.5%。Gate C04 形式上确认 FAIL，但这是非信号：判别力不足，不是 OPD 损害了泛化。
- **ARC-challenge 加入** 作为第 4 项 OOD-lite 任务（25-shot，无 chat template，acc_norm，N=1172）。Spread = 0.0145 ≈ 1 SE（±0.0146）。确认 OOD-lite 能力保持格局。

**Gate 重分析（cycle05 修正数字）：**
- Gate B04：确认 PASS。匹配基准是 NuminaMath-test ID；sft_n128 仍是最近 SFT 锚点。
- Gate C04：确认 FAIL。OPD 未在任何 OOD-lite 指标上击败 sft_n128。非信号（spread <1.5%）。
- Gate D04：确认 FAIL。依赖 Gate C04。

**Eval stack：** 全部 8 项评测任务统一到 `Eval/run_eval.py` 单一入口，`--task` 和 `--backend {custom,lm_eval}` 参数控制。自定义脚本移入 `Eval/component/{gsm8k,math500,numina}/runner.py`（2026-06-18）。
