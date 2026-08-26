# round3-20260826-02：Round3 strong-smoke修复前重试

## 元信息与状态

- Cycle：`cycle-20260818-01` / Round3
- 设计：`round3-exp-v1.4` / `r3-theory-v0.9`
- 服务器代码：`ed1bfca002799f11ea1bad29f6f06e2e15fdd565`
- 状态：失败并保留；formal未启动
- 前序attempt：`../round3-20260826-01/README.md`
- 复用证据：`source_revisions.json`与前序SHA同为`bd0ec0aaedbfd02205e65ecc774dcc32dcf35f580c7b5b1d0fbd60645eb29f11`；复用冻结data v2和reference cache

## 观察事实

- DPO-1K：production one-step训练成功；checkpoint verifier通过，FP32 trainable LoRA、392项optimizer state、scheduler step 1与RNG恢复均正确；峰值allocated/reserved为8,468,885,504/10,993,270,784 bytes
- SSPO：4 labeled +28 unpaired完整logical population训练与checkpoint成功，finite loss `42.55149841308594`、finite pre-clip grad norm `796.4470825195312`；峰值allocated/reserved为8,468,886,528/10,993,270,784 bytes
- SSPO verifier：两次独立重载的loss、running state、scheduler/global step完全一致，但LoRA更新最大绝对差`1.3404528544924688e-6`、最大相对差`1.99992835521698`，超过预注册`1e-7`/`1e-6`，因此fail closed
- 后续DPO-8K、两个动态PE、storage projection与formal均未启动
- 失败后GPU全部空闲；`/data`约95G可用；服务器checkout仍clean

## 诊断结论与修复边界

同一checkpoint在服务器启用`CUBLAS_WORKSPACE_CONFIG=:4096:8`、PyTorch deterministic algorithms并关闭TF32后，loss差、LoRA最大绝对差、最大相对差均为`0.0`，running state仍精确一致。这说明失败来自CUDA backward非确定性，不是checkpoint状态缺失。

本地`round3-code-candidate-v0.4`据此统一trainer/verifier确定性后端，并修复strong-smoke入口`PYTHONPATH`。未调整SSPO loss、batch、长度、数据、方法或预注册容差；用户于2026-08-26已明确允许Codex commit/push、部署并以新experiment attempt从头验证，通过后直接挂载formal。

服务器主日志：

```text
/data/youchang/youchang/platform_logs/round3-20260826-02/codex_validation/ed1bfca00279/03_strong_smoke.log
```

该日志SHA-256为`fcb9a6125fbac3aca08c65205dc15db27963a0c3cddf7de16b0cd7b1331c6341`；退出码文件记录1。原始checkpoint、日志和逐步产物只保留服务器，不复制本地。
