# SOPPO Round2 基础设施流水线草稿

> 状态：协作草稿
> 范围：帮助理解 round2 如何把配置、训练、rollout、资源、日志与失败处理串起来
> 相关文档：[human_read/AGENTS.md](../AGENTS.md)、[current_experiment.md](../exp/current_experiment.md)、[CODE_OVERVIEW.md](../../code/CODE_OVERVIEW.md)

## 0. 为什么要写这份文档

这个项目很适合学习 infra，因为 round2 不只是“跑一个模型”，它还要同时处理：

- 训练后端（`Megatron` / `Megatron-Core`）
- rollout 后端（`vLLM`）
- 训练与 rollout 之间严格的 GPU 隔离
- 明确的外部 entrypoint
- 清晰的状态、日志和 launch record
- 缺少条件时必须 fail-closed

这份文档的目标，是把 round2 当成一个**系统**来理解，而不是把它当成一个单脚本任务。

## 1. 心智模型

可以把 round2 理解成一个围绕两个执行器的小型控制平面：

- **训练执行器**：消费训练配置，启动 Megatron 侧工作
- **rollout 执行器**：消费 rollout 配置，启动 vLLM 侧工作

SOPPO 本身主要负责编排：

1. 校验配置
2. 解析路径
3. 检查资源
4. 拼接命令
5. 写入启动/状态记录
6. 把执行交给外部 entrypoint
7. 收集日志和结果

理解每个文件时，可以先问自己：

> 这个文件是在描述策略、在生成命令，还是在真正执行工作？

## 2. 核心组成

### 2.1 配置层

主要位置：

- `code/configs/round2/base.yaml`
- `code/configs/round2/*.yaml`

这一层应该回答：

- round2 的实验标识是什么
- 训练 GPU 集合是什么
- rollout GPU 集合是什么
- 模型 / 数据 / 输出根目录在哪里
- 是否需要外部安装
- entrypoint 和 working dir 是否已经提供

建议重点思考：

- 哪些值是默认值？
- 哪些值必须从环境变量补齐？
- 哪些值必须保证训练与 rollout 不重叠？
- 哪些值属于 round1，哪些属于 round2？

### 2.2 后端适配层

主要位置：

- `code/src/round2/megatron_backend.py`
- `code/src/round2/rollout_backend.py`
- `code/src/round2/rollout_schema.py`

这一层应该回答：

- round2 如何把配置映射成命令
- 哪些字段在启动前是必填的
- 外部 worker 的契约是什么
- rollout 产物如何被校验

### 2.3 启动器

主要位置：

- `code/src/round2/run_megatron.py`
- `code/src/round2/run_rollout.py`

这一层应该回答：

- GPU 可见性是怎么设置的
- launch record 是怎么写的
- 什么时候会提前退出
- 什么内容会在真正执行前就被写入磁盘

### 2.4 可观测性

常见产物：

- `status.json`
- Megatron 日志
- rollout 日志
- launch record
- resolved config
- smoke 输出

这些产物告诉我们系统处于以下哪种状态：

- 未启动
- 被阻塞
- 正在运行
- 失败
- 部分完成
- 已准备好进入下一阶段

## 3. round2 流水线，按步骤看

### 第 1 步：读取并解析配置

流水线从读取 round2 YAML 开始。

预期动作：

- 加载 base config
- 应用 override
- 解析路径
- 分离训练和 rollout 的设置
- 确认 round2 的 experiment ID / 输出根目录

这里要理解：

- 配置继承是怎么工作的
- override 是如何叠加的
- 哪些环境变量可以覆盖 YAML

### 第 2 步：校验硬约束

如果关键条件不满足，round2 应该 fail-closed。

常见硬约束：

- entrypoint 路径缺失
- working directory 缺失
- 模型路径缺失
- GPU 列表重叠
- 依赖未安装
- 输出根目录与 round1 产物冲突
- rollout 路径未配置

这里要区分：

- 哪些是配置错误
- 哪些是资源错误
- 哪些是应该立即阻断执行的错误

### 第 3 步：构建外部命令

round2 不直接拥有 Megatron 或 vLLM 的内部实现，而是为外部 worker 生成命令。

训练命令构建应该说明：

- Python 解释器
- 外部 entrypoint
- 模型路径
- manifest 路径
- 数据路径
- 输出路径
- GPU IDs
- TP / PP / DP 参数
- seed / learning rate / batch 等参数

rollout 命令构建应该说明：

- Python 解释器
- 外部 entrypoint
- base model
- checkpoint / adapter 路径
- artifact 路径
- GPU IDs
- rollout 长度和显存参数

这里要理解：

- 命令行契约是怎么形成的
- 哪些属于 SOPPO，哪些属于 worker
- 如何让训练和 rollout 的契约平行但彼此独立

### 第 4 步：启动前写记录

好的 infra 系统会在真正启动前先写一些东西。

预期记录包括：

- 启动元数据
- resolved config 快照
- 状态标记
- 可能还有 PID 或进程记录

这里要理解：

- 哪些记录用于重启和调试
- dry-run 是只预览还是也会产生真实记录
- 如何避免“看起来像启动了，但其实没有”的假象

### 第 5 步：启动训练

训练启动应该回答：

- 哪些 GPU 是可见的
- 使用哪个 working directory
- 训练过程是外部的还是本地的
- 失败如何被暴露出来

需要重点问的 infra 问题：

- 缺少 entrypoint 时是否立即硬失败？
- 训练 GPU 与 rollout GPU 是否通过结构保证不重叠？
- 输出是否写到 round2 专用路径？

### 第 6 步：等待可用输出

训练输出必须达到一个可用于 rollout 的 checkpoint / artifact 边界。

要问的问题：

- 什么叫“可以 rollout 了”？
- round2 如何避免读取未完成的 checkpoint？
- 有没有明确的 handoff 文件或状态转换？

### 第 7 步：启动 rollout

rollout 只有在 checkpoint 稳定后才应该发生。

要问的问题：

- rollout 读取的是同一个 checkpoint 路径，还是一个拷贝出来的快照？
- rollout 是否允许和其他工作并发？
- rollout 产物的契约是什么？

### 第 8 步：记录结果和失败状态

这个流水线即使失败，也应该留下足够的痕迹。

可能的终态包括：

- 启动前被阻塞
- 启动了但立刻失败
- 训练完成但 rollout 失败
- rollout 完成但产物无效
- round2 完整成功

这里要理解：

- 如何定位失败边界
- 如何区分环境问题和代码问题
- 如何避免删除有价值的失败证据

## 4. 资源模型

### 4.1 GPU 划分

round2 当前把训练和 rollout 视为两个独立资源池。

一个很重要的不变量：

- 训练 GPU 不能和 rollout GPU 重叠
- 可见 GPU 集合必须和后端契约一致
- GPU “空闲”并不代表可用；只要还有别的进程占着显存或 compute，就不能当成空闲

需要关注：

- `CUDA_VISIBLE_DEVICES`
- 训练 GPU IDs
- rollout GPU IDs
- GPU 利用率与显存占用
- 外部后台进程

### 4.2 不需要 GPU 也能做的工作

即使 GPU 被占用，下面这些仍然可以先做：

- 配置解析
- 路径校验
- 命令构建
- 静态代码检查
- 依赖检查
- launch record 格式检查
- 日志结构审阅

这一点很重要，因为它把**准备**和**执行**区分开了。

## 5. 失败矩阵

这部分我们可以后续一起补全。

| 现象 | 可能原因 | 安全动作 | 不安全动作 |
| --- | --- | --- | --- |
| entrypoint 是 null | 外部 worker 配置缺失 | 停止并报告阻塞 | 自己猜一个路径 |
| `vllm` 缺失 | 依赖未安装 | 在正确环境中安装 | 伪造导入成功 |
| GPU 被其他用户占用 | 资源冲突 | 等待并再次检查 | 杀掉对方进程 |
| status 存在但任务没推进 | 启动或 worker 失败 | 查看日志 | 覆盖状态 |
| rollout 读取到不完整 checkpoint | handoff 竞态 | 增加明确的就绪契约 | 直接启动 |
| dry-run 写了正式记录 | 契约问题 | 修代码或修文档 | 当作无伤大雅 |

## 6. 我们接下来要补的内容

建议按四轮来补这份文档：

1. **路径轮**：精确目录、文件、环境变量
2. **启动轮**：训练与 rollout 的精确命令顺序
3. **状态轮**：status 文件的含义和变化时机
4. **失败轮**：哪些情况阻塞、哪些情况重试、哪些必须人工介入

## 7. 待回答的问题

请帮我补这几个点：

- round2 的 experiment ID 命名规则是什么？
- 哪个文件是 round2 状态转换的权威来源？
- 这份文档你更想侧重哪一类内容：
  - 命令流
  - GPU / 进程编排
  - 配置契约
  - 故障恢复
- 要不要下一步加一张极简 ASCII 流程图？

## 8. 下一步可以加的内容

可能的下一批补充：

- round2 训练 → checkpoint → rollout 的流程图
- 环境变量及其含义的表格
- smoke vs full run 的检查清单
- 一个“为什么这就是 infra”的小节，把每个概念对应到真实工程能力

---

草稿说明：这份文档故意没有写完，目的是和你一起迭代，而不是当作最终规范。
