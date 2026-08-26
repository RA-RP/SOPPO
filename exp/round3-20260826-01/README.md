# round3-20260826-01：Round3 Qwen3-1.7B五方法验证

## 元信息

- Experiment ID：`round3-20260826-01`（服务器既有ID；沿用已冻结source revision与model manifest，不另建同义ID）
- Cycle ID：`cycle-20260818-01`
- 状态：失败并保留（阶段A/C、data v2与reference cache完成；strong smoke在首方法checkpoint verifier入口失败）
- 对应实验设计：`../../human_read/exp/current_experiment.md` `round3-exp-v1.4`
- 对应理论：`../../human_read/theory/current_theory.md` `r3-theory-v0.9`
- 理论批准：用户于2026-08-26明确批准`r3-theory-v0.9`
- 实验设计批准：用户于2026-08-26明确批准`round3-exp-v1.4`
- 代码版本：`ed1bfca002799f11ea1bad29f6f06e2e15fdd565`
- 代码交接与服务器执行授权：用户于2026-08-26明确要求Codex亲自执行已说明的阶段C；授权只覆盖环境与CPU合同测试
- 执行位置：4090-3服务器，仓库`/data/youchang/youchang/SOPPO`
- 服务器运行手册：`../../code/scripts/round3/GLM_VALIDATION_GUIDE.md`（由Codex按相同阶段边界执行）

## 当前实际执行

- 阶段A：2026-08-26完成只读现场核验；Round2 controller已停止，第一方法最后step 590、第二方法未启动、两个pruner PID失效；Round2 run/checkpoint未修改
- 阶段C：2026-08-26 16:46:10–16:46:18（Asia/Shanghai）执行
- 入口：`code/scripts/round3/02_server_tests.sh`
- train Python：3.10.12，`pip check`为`No broken requirements found.`
- rollout Python：3.10.12，`pip check`为`No broken requirements found.`
- 合同测试：`6 passed in 2.92s`，入口退出码0
- Git：执行前后HEAD均为上述exact commit，`git status --short`为空
- GPU：三张RTX 4090执行前后均为1 MiB/0%且无compute process；阶段C未运行GPU工作
- 磁盘：`/data`总计3.6T、已用3.4T、剩余96G、使用率98%；这是后续data/strong-smoke/formal门禁风险
- data v2：8K/1K/7K/1K/997/997 views、17,997行source manifest与12,197行malformed audit全部通过
- reference cache：8K train、1K train、1K validation与997 test全部完成并发布manifest
- strong smoke：DPO-1K单步训练/checkpoint成功；随后`03_strong_smoke.sh`调用verifier时因未继承`PYTHONPATH`报`ModuleNotFoundError: src`，attempt立即停止，其他四方法和formal未启动

## 远程证据索引

证据根目录：

```text
/data/youchang/youchang/platform_logs/round3-20260826-01/codex_validation/ed1bfca00279
```

| 证据 | SHA-256 |
| --- | --- |
| `stage_c_precheck.log` | `1c0a8ee521e73283b90d44bc3fce8eb22d806523b956143720338ab1328323e9` |
| `environment_check.log` | `1f45efeb5d25b4a7ee9b3dcc74dc5e7ebcb4b81125d88bcc062678d4ea720f5b` |
| `02_server_tests.log` | `358c985485d132cc6c3a4949a005266701023ecb788bf2542f1906ccf611f25f` |
| `02_server_tests.exit_code` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `stage_c_postcheck.log` | `93eba60dffd2ebb0090341e7a99f2abd28de7b6d868e0a6ca743633d433410d9` |

服务器已用`sha256sum -c SHA256SUMS`复核以上五项全部为`OK`。原始日志只保留在服务器，本地不复制样本、数据、模型、cache或运行环境。

## 尚未执行与授权边界

- `dual_source_v2`与v2 reference cache已生成并保留
- DPO-1K strong-smoke checkpoint已生成并保留，但本attempt没有完整checkpoint verification
- 未运行formal、评价或聚合
- 未停止进程、未修改服务器checkout、未删除或覆盖Round2 checkpoint以及失败遗留`dual_source_v1`
- 后续每个写产物阶段仍需用户明确授权；不得从阶段C成功推定生产路径正确
