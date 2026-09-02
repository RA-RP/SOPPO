# 当前实验设计：Round4 FrozenPE 与 StaticPE

## 0. 版本与门禁

- Cycle：`cycle-20260901-01` / Round4
- 实验版本：`round4-exp-v2.0`
- 理论来源：`../theory/current_theory.md` `r4-theory-v2.0`，2026-09-02用户明确通过。
- 当前唯一活动阶段：`CODE_IMPLEMENTATION`
- 实验状态：`APPROVED`
- 用户确认：2026-09-02，用户确认四方法合同、StaticPE SimPO-base loss、原StaticPE改名FrozenPE、超参数、A100/4090评价流水线及方法顺序，并要求开始正式执行。
- 代码状态：用户已授权自行提交、同步与迭代直至 full-chain smoke 通过。`67ebed0`越过Trainer batch接口并进入Qwen3模型前向，却在自定义梯度检查点包装器处理`functools.partial`时停止，未完成优化step。当前候选为`round4-code-v2.0.4`，继续沿恢复型 smoke 边界执行。

`round4-exp-v1.0`保留为历史设计。旧执行环境在任何训练前因无效偏好行校验停止，相关证据位于`../../exp/exp-20260901-01-round4-server-prep/README.md`。

## 1. 方法臂与数据

| 方法 | train数据 | labeled目标 | unlabeled目标 | reference | rollout |
| --- | --- | --- | --- | --- | --- |
| DPO | UltraFeedback labeled | DPO | 无 | 是 | 否 |
| SSPO | labeled + UltraChat | DPO | 既有SSPO threshold风险 | 是 | 否 |
| StaticPE | labeled + UltraChat单回答 | SimPO | 单回答连续PE | 否 | 否 |
| FrozenPE | labeled + UltraChat固定A/B | DPO | 双回答PE | 是 | 是，仅B一次 |

- UltraFeedback/UltraChat各按冻结源的`0.1`确定性抽样，再应用已记录的无效行过滤；最终有效行数、来源revision和SHA由新manifest决定。
- DPO、SSPO、StaticPE、FrozenPE共享labeled IDs；SSPO/StaticPE共享single-response UltraChat IDs；FrozenPE由同一IDs生成A/B paired view。
- 共同eval是held-out labeled view。训练中不使用最终AlpacaEval或API结果选择checkpoint。

## 2. 冻结超参数

| 项目 | DPO | SSPO | StaticPE | FrozenPE |
| --- | ---: | ---: | ---: | ---: |
| `pref_beta` | 0.1 | 0.1 | 10.0 | 0.1 |
| `simpo_gamma` | — | — | 2.0 | — |
| PE lambda | — | — | 0.1 | 0.1 |
| PE temperature | — | — | 1.0 | — |
| EMA momentum / clip | — | 0.95 / 5 | 0.95 / 5 | — |
| epsilon | — | — | 1e-8 | 1e-8 |
| per-device train | 1 | 4 | 4 | 4 |
| GPU / GA / effective batch | 2 / 8 / 16 | 2 / 8 / 64 | 2 / 8 / 64 | 2 / 8 / 64 |
| per-device eval | 4 | 4 | 4 | 4 |

共同设置：Qwen3-1.7B固定revision、Qwen3 non-thinking、LoRA r8/all linear、BF16、lr `1e-5`、cosine、warmup `0.1`、cutoff1024、epoch1、seed42。SSPO继续使用 `prior=0.5`、`gamma_0=1.0`、`gamma_decay=0.001`、`gamma_min=0.2273`。

## 3. 训练、生成与评价顺序

### 3.1 smoke

每个训练臂在A100-2上用2卡运行独立2-step smoke：step1/2各eval一次，保存adapter，merge，新进程reload，生成冻结Alpaca前2条。A100仅写GPU-phase request bundle；4090以同一judge profile完成每臂1条API smoke与WR/LC，回传后才算full-chain smoke通过。

### 3.2 formal

full-chain smoke和用户formal授权后，按下列顺序形成连续结果：

1. frozen base生成805条并经4090评价；
2. DPO训练、merge、生成805条、立即送4090评价；
3. SSPO训练、merge、生成805条、立即送4090评价；
4. StaticPE训练、merge、生成805条、立即送4090评价；
5. 生成FrozenPE candidate B，随后训练FrozenPE、merge、生成805条、立即送4090评价。

每个训练臂独占两张A100并从同一base独立初始化；4090评价前一臂可与A100训练下一臂重叠。GPU结果和API结果均不得反向调整未开始臂的参数。

## 4. A100/4090职责与结果合同

| 机器 | 职责 | 不做的事 |
| --- | --- | --- |
| A100-2 | preprocessing、candidate B、2-GPU训练/eval、merge/reload、两GPU分片生成、request bundle | 不联网调用judge API，不保存API key |
| 4090-3 | 拉取并核验request、读取私有judge profile、API annotations、本地WR/LC、回传aggregate result | 不训练、不保存checkpoint、不改写A100输出 |

request bundle包含相对路径、输出SHA、行数、冻结instruction/baseline SHA、generation合同、exact commit和环境绑定。result bundle绑定request manifest SHA，包含profile fingerprint、judge model、版本、annotation计数、WR/LC及去敏远程证据索引；不含key、完整endpoint、原始回答或原始API响应。

一个正式评价只使用一个primary profile。额外profile可以重评同一不可变outputs，但单独形成新的`evaluation_id`，不与主结果混合。

## 5. 预先规定的验收与解释边界

- StaticPE必须验证：长度归一化、SimPO margin位置、EMA不回传梯度、eval不污染EMA、至少2个global unlabeled、lambda=0退化为SimPO、DDP与单进程拼接一致、resume恢复一致。
- FrozenPE必须验证：DPO-base双回答数值/梯度与旧实现一致、A/B等量、candidate空/重复fail closed。
- 所有方法必须验证：两卡、GA、sampler、finite梯度、adapter、merge/reload、输出数量/顺序/SHA、API annotation数量、WR/LC有限值。
- StaticPE和FrozenPE的loss尺度不同，训练分项不横向排名；最终比较以共同AlpacaEval WR、LC和长度为主，并报告训练诊断与token compute。
- 由于没有SimPO-label-only arm，StaticPE结果只回答该整体方法在本合同下的表现；不单独宣称PE或SimPO任一因素的因果收益。

## 6. 服务器执行门禁

1. 完成v2实现、静态复核和文档映射；
2. 用户明确确认新的exact code commit可以提交服务器；
3. 4090构建新项目wheel与judge环境，复用并复核已有冻结资产；
4. A100创建新的commit-bound venv，保留旧失败prepared目录，重跑filtered preprocessing；
5. 完成服务器单元/合同测试与full-chain smoke；
6. 用户确认smoke摘要后启动formal；
7. 结果回传并进入`RESULT_HANDOFF`。
