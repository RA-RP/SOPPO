# round3-20260826-04：Round3 v0.5全绿strong-smoke与formal正式挂载

## 元信息与状态

- Cycle：`cycle-20260818-01` / Round3
- 执行时设计：`r3-theory-v1.0` / `round3-exp-v1.5`
- 服务器exact commit：`b1beef5550ac47c9c78b98c1729014cc153b1251`（v0.5）
- 状态：**formal已完成**（2026-08-27 12:56 Asia/Shanghai，`completed/all_methods`，exit 0）
- 前序attempt：`../round3-20260826-01/README.md`、`../round3-20260826-02/README.md`、`../round3-20260826-03/README.md`
- 执行者：Codex完成v0.5部署与完整strong-smoke（全绿）后因额度耗尽停止；用户于2026-08-26深夜授权ZCode接手，完成存储门禁处置与formal挂载
- 数据/模型：复用冻结`dual_source_v2`（8K/1K/7K/1K/997）与绑定`b1beef5`的reference cache；`source_revisions.json`沿用attempt-03冻结值

## strong-smoke通过事实（Codex执行，2026-08-26下午至夜间）

- 服务器合同测试：8 passed（新调度合同测试），两个环境`pip check`干净，checkout clean。
- reference cache按commit provenance独立重建，四个view逐行顺序/finite/SHA审计通过。
- Wave 1并发smoke：DPO-1K/GPU0、GitHub-loss SSPO/GPU1（含两次独立重载确定性校验，差异精确为0）、DPO-8K/GPU2全部通过checkpoint验证。
- 两个动态方法双vLLM smoke通过：显式token-ID prompt路径（v0.5修复）在生产路径验证，无长度不一致错误。
- `03_strong_smoke.exit_code=0`；五方法`smoke_complete`+`checkpoint_verified`齐全；storage projection写出。

## 存储门禁与用户手动批准（ZCode执行，2026-08-27凌晨）

- smoke投影：`projected_peak_bytes=55,717,692,774`，`formal_required_free_bytes=111,435,385,548`（2×峰值，门禁为纯算术比较，代码无覆盖参数）。
- 挂载前实测`/data` free=`102,056,132,608`，缺口`9,379,252,940` B；用户明确手动批准通过该存储门禁并承担磁盘风险。
- 实际处置：未修改代码、投影或门禁阈值；仅删除可再生缓存共`12,348,178,432` B——pip下载缓存（`cache/pip`）、HF datasets `downloads`暂存、未被manifest引用的`openbmb___ultra_feedback` arrow缓存。manifest按SHA-256绑定的18个源文件（4,090,098,911 B）全部完好复核。
- 门禁真实通过：`free_bytes_at_gate=114,404,261,888 ≥ required`，证据`formal_storage_gate.json`状态`passed`。

## formal挂载（ZCode执行）

- 入口：`code/scripts/round3/start_all.sh`，环境`SOPPO_ROUND3_EXPERIMENT_ID=round3-20260826-04`、`SOPPO_ROUND3_REFERENCE_DIR=/data/youchang/youchang/cache/round3/reference_qwen3_1.7b_dual_source_v2_b1beef5550ac`。
- 控制器：PID=PGID=22905（isolated process group，无自动checkpoint pruner）。
- 挂载后顺序：resolve formal configs → storage gate passed → Wave 1三静态并发训练。
- 2026-08-27 00:22快照：`dpo_1k` step 25（loss 0.710）、`sspo` step 12（loss 2.048）、`dpo_8k` step 7（loss 0.701），均250步合同；GPU0/1/2持续负载；`/data`剩余107G。
- 预期后续：Wave 2 `dpo_pe_sft_rollout`、Wave 3 `dpo_pe_rollout_only`（各独占三卡），随后共同1K validation选点、独立997-pair双head final test与sample-free聚合。

## 2026-08-27 11:31只读快照与后续解释边界

- controller仍为`running/formal_dpo_pe_rollout_only`，exact commit仍为`b1beef5550ac47c9c78b98c1729014cc153b1251`；未停止、未改checkout。
- `DPO-1K`、SSPO、`DPO-8K`与`dpo_pe_sft_rollout`均已完成250/250并各保留10个durable checkpoints；`dpo_pe_rollout_only`为185/250，三卡均由该方法占用。
- 旧SFT+rollout的raw mean-logp telemetry在step249/250仍显示rollout hard wins 28/28、soft probability约0.999998/0.999988和PE extreme rate 1.0；该事实促成本轮reward定义复核，但在最终test完成前不是方法效果结论。
- `/data`约79GiB可用、使用率98%，明显低于挂载后107G快照。旧formal已有一次性门禁证据且继续自然运行；任何后续extension必须重新做独立storage projection和两倍free门禁，不能沿用旧门禁。
- 用户于2026-08-27要求恢复Theory v0.2的DPO implicit reward。这个run及其两个动态method ID据实解释为SimPO-reward PE历史对照；不得回写其config、日志、checkpoint或把它冒充DPO-reward。新增两臂属于新experiment/new exact commit，必须等待本controller终态。

## formal完成与最终测试结果（2026-08-27 12:56终态）

- 终态：controller `completed/all_methods`，exit 0；五方法全部250/250，各保留10个durable checkpoints；final test（独立997-pair、双head）与sample-free聚合完成；产物`exports/round3/round3-20260826-04/round3_aggregate.json`（84KB，`single_seed_exploratory=true`）；三卡释放；`/data`剩余约74G。
- 选点：dpo_1k step225、sspo step25、dpo_8k step250、dpo_pe_sft_rollout step250、dpo_pe_rollout_only step225。
- 主head（DPO reference-delta β=0.1，accuracy含tie半分）：frozen_base 0.5000（全tie）、sspo 0.5155、dpo_1k 0.6479、dpo_pe_sft_rollout 0.6459、dpo_pe_rollout_only 0.6349、dpo_8k 0.6790。
- 同head差值：DPO-1K−base +14.79pp；DPO-8K−DPO-1K +3.11pp；SSPO−DPO-1K **−13.24pp**；PE-SFT+rollout−DPO-1K −0.20pp；PE-rollout-only−DPO-1K −1.30pp；PE-SFT+rollout−rollout-only +1.10pp。
- 次head（raw mean-logp β=10）：各方法accuracy均挤在0.589–0.594，frozen_base 0.5898；该head基本不区分方法（confidence_mean≈0.94、ECE≈0.35，高度饱和），与训练期发现的raw likelihood来源捷径一致。
- 初步读法（待RESULT_HANDOFF正式讨论）：1K标注即可学到稳定偏好信号，8K给出label-budget增益；GitHub-loss SSPO在1:7分层、250步、统一lr设置下基本未学到偏好（仅略高于base）；两个SimPO-reward PE臂相对DPO-1K无增益——与PE软责任饱和的机制诊断一致，正是DPO-reward extension要检验的问题。
- 解释边界：单种子探索性结果；两个动态臂按`simpo_mean_logp` reward profile登记为对照，不能据此否定reference-relative PE本身；AlpacaEval/MT-Bench延后Round4。

## 远程证据索引

```text
/data/youchang/youchang/runs/round3-20260826-04/controller.json
/data/youchang/youchang/runs/round3-20260826-04/formal_storage_gate.json
/data/youchang/youchang/runs/round3-20260826-04/storage_projection.json
/data/youchang/youchang/platform_logs/round3-20260826-04/codex_validation/b1beef5550ac/03_strong_smoke.log
/data/youchang/youchang/platform_logs/round3-20260826-04/controller.log
```

| 证据 | SHA-256 |
| --- | --- |
| `03_strong_smoke.log` | `c3a57e65d1bf3001c027991df5ec52c4fcb61322b1e6f0c6a9a24e9adb219cc4` |
| `controller.log`（挂载初期快照） | `8d6608f91a8ee15cecb870b95fb2ace42e9558bc9091931671aee8b262bf1baf` |

## 边界

- 未删除或修改任何Round2 runs/checkpoint、Round3失败attempt证据、manifest绑定源文件；服务器checkout保持clean `b1beef5`。
- 本次本地仅新增/更新Markdown记录；未commit/push（formal挂载不需要代码变更）。
- formal完成或失败后保存白名单摘要与远程证据索引，进入`RESULT_HANDOFF`；停止只能经用户对该experiment明确授权后`stop_all.sh --execute`。
