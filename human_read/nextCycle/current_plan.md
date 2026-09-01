# Round4激活规划：三方法StaticPE比较与AlpacaEval 2.0

## 0. 状态

- 来源：`cycle-20260818-01` / Round3
- 目标：`cycle-20260901-01` / Round4
- 规划版本：`round4-activation-plan-v0.2`
- 对应结果：`../result/current_result.md` `round3-result-administrative-close-v0.1`
- 用户决定：2026-09-01明确要求直接覆盖进入Round4，确认FusionOne存在已实机验证的8×A100资源，并决定先占用2张A100；4090-3只准备镜像/数据，全部smoke转到A100
- 规划状态：`APPROVED_BY_EXPLICIT_USER_TRANSITION`
- 新cycle活动阶段：Round4 `THEORY_DISCUSSION`

## 1. 证据摘要与理论处理

Round3五方法formal已完成，DPO-1K和DPO-8K在paired test上形成有效label-budget基线；GitHub-loss SSPO与两个SimPO-reward动态PE未超过DPO-1K。DPO-reward extension没有执行，不能从Round3推断reference-relative StaticPE效果。

Round4不补跑Round3 extension，改为更紧凑的三方法比较：label-only DPO、SSPO、StaticPE。它沿用同一Qwen3-1.7B和SSPO官方10%双源数据类型，并新增完整AlpacaEval 2.0生成式评价。

## 2. 选定行动

1. Round4理论先冻结三方法的唯一差异，尤其SSPO采用DPO-base还是SimPO-base、StaticPE的micro-batch PE统计语义；
2. 实验设计冻结共同数据IDs、1 epoch、两卡训练、DPO effective batch16、SSPO/StaticPE effective batch64、eval和Alpaca合同；
3. 复核现有`SSPO/SSPO` legacy原型，补齐DPO labeled-only、SSPO matched配置、2-step smoke、merge/reload、三方法统一Alpaca入口和镜像交付；
4. 经用户代码交接批准后commit/push，在4090-3拉取exact commit、构建无凭据镜像、下载冻结数据并生成manifest/SHA；
5. 通过SSH把数据直接传到A100仓库外目录，优先创建/占用2张A100；镜像启动后先做硬件、挂载与数据SHA preflight；
6. DPO、SSPO、StaticPE依次在同一2张A100上完成2-step全链smoke，获单独formal授权后再顺序训练；
7. 三方法最终都生成完整805条AlpacaEval 2.0回答并报告LC与普通win rate。

## 3. 资源与安全边界

- 4090-3当前历史快照显示Docker权限和scratch空间存在阻塞；build前必须实时核验，不删除旧实验释放空间。
- FusionOne目标资源为用户已实机验证的8×A100，本轮先占用2张且三方法顺序共享；单卡显存、拓扑、容器映射和挂载仍由preflight采集。
- 私有账号、密码、内部地址、registry实值和API key不进入Git、镜像、配置、日志或研究记录。
- Round4当前只解锁`THEORY_DISCUSSION`；实验、代码、commit/push、服务器smoke、镜像和formal仍依次受门禁约束。

## 4. 规划交接

- 完整规划已归档：是，见`plan_archive.md`
- 用户明确通过目标切换：是，2026-09-01“直接覆盖成round4”
- 新Cycle ID：`cycle-20260901-01`
- 下一入口：`../theory/current_theory.md` `r4-theory-v0.2`
