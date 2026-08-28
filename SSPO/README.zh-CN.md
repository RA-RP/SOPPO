# SSPO（半监督偏好优化）

本仓库包含 SSPO 以及多种基线方法（DPO、ORPO、SimPO、KTO、SSRM、SPA）的实现。

## 安装

1. 使用 Python 3.10 或更高版本创建名为 `sspo` 的虚拟环境：

```bash
conda create -n sspo python==3.10.0
conda activate sspo
```

2. 安装所需依赖包：

```bash
cd SSPO
pip install -r requirements.txt
```

## 运行

### SSPO 训练

1. 预处理数据：

```bash
python preprocessing_data/preprocessing_ultrachat.py --fb [feedback_ratio] --ch [chat_ratio]
python preprocessing_data/preprocessing_medical.py --fb [feedback_ratio] --ch [chat_ratio]
python preprocessing_data/preprocessing_business.py --fb [feedback_ratio] --ch [chat_ratio]
```

2. 生成 YAML 配置和训练命令：

```bash
python examples/train/make_yaml.py
python examples/train/make_yaml_medical.py
python examples/train/make_yaml_business.py
```

3. 执行训练：

```bash
# 从 make_yaml.py 的输出中复制生成的命令
# 将其粘贴到 examples/train/train.sh 中
bash examples/train/train.sh
```

### DPO、ORPO、SimPO、KTO 训练

按照与 SSPO 相同的步骤操作，但需要修改 `examples/train/train.sh` 中的方法。

### SPA 训练

我们遵循 [SPA 仓库](https://github.com/kingdy2002/SPA)中的实现。详细说明请参阅该仓库。

### SSRM 训练

1. 生成额外的未标注响应：

```bash
python examples/SSRM/generate_responses.py
```

2. 使用预训练奖励模型进行伪标注：

```bash
python examples/SSRM/pseudo_label.py
```

3. 根据置信度阈值筛选数据：

```bash
python examples/SSRM/conf_threshold.py
```

4. 合并反馈数据：

```bash
python examples/SSRM/merge_json.py
```

5. 执行完整的 SSRM 训练流程：

```bash
# 在 examples/SSRM/train-ssrm.sh 中配置迭代次数
# 该脚本会按照指定的迭代次数执行步骤 1～4
bash examples/SSRM/train-ssrm.sh
```

## 注意事项

- 请务必调整由 `make_yaml.py` 生成的 YAML 配置文件中的超参数。
- 对于 SSRM，可以通过修改 `train-ssrm.sh` 中的命令来控制迭代次数。
