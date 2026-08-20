# 深圳超算集群执行指南

**Cycle**: cycle-20260818-01  
**Experiment**: v0.3 MVP  
**基础路径**: `/home-ssd/Users/nsgm_jiangwh/youchang`

---

## 节点类型说明

| 节点类型 | 地址 | 用途 | 限制 |
|---------|------|------|------|
| **登录节点** | 10.32.48.56 (mn006) | SSH 登录，提交作业，查看结果 | 禁止运行程序 |
| **编译节点** | gn001 (10.32.34.65) | 安装软件，下载数据，编译代码 | 需从登录节点 ssh 跳转 |
| **计算节点** | 自动分配 | 运行训练任务 | 通过 sbatch 提交 |

---

## 阶段执行清单

### 前置准备

1. **连接 VPN**
   ```
   地址: https://dsjvpn.nsccsz.cn:4443
   使用微信小程序"腾讯身份验证器"获取安全码
   ```

2. **登录到登录节点**
   ```bash
   ssh nsgm_jiangwh@10.32.48.56
   ```

3. **上传 ICLR 文件夹**
   ```bash
   # 本地执行
   sftp nsgm_jiangwh@10.32.48.56
   put -r /path/to/ICLR /home-ssd/Users/nsgm_jiangwh/youchang/
   ```

---

### 阶段 -1：环境准备

**在编译节点执行**

```bash
# 1. 从登录节点跳转到编译节点
ssh gn001

# 2. 进入代码目录
cd /home-ssd/Users/nsgm_jiangwh/youchang/ICLR/work/code/scripts/cluster

# 3. 运行环境设置脚本
bash 00_server_setup.sh
```

**完成标志**：
- ✓ 创建 conda 环境 `youc`
- ✓ 安装 PyTorch 2.4.0 + CUDA 12.1
- ✓ 安装所有依赖
- ✓ 生成 `project_config.json`

**预计时间**：10-20 分钟

---

### 阶段 0：单元测试

**在编译节点执行**

```bash
# 1. 激活环境
source /home-ssd/Users/nsgm_jiangwh/youchang/activate_env.sh

# 2. 运行测试
cd /home-ssd/Users/nsgm_jiangwh/youchang/ICLR/work/code/scripts/cluster
bash 01_server_tests.sh
```

**完成标志**：
- ✓ L_PE 数值正确性测试通过
- ✓ 梯度路径测试通过
- ✓ DPO loss 测试通过

**预计时间**：2-5 分钟

---

### 阶段 1：数据准备

**在编译节点执行**（需要网络代理下载数据）

```bash
# 1. 激活环境（如未激活）
source /home-ssd/Users/nsgm_jiangwh/youchang/activate_env.sh

# 2. 运行数据准备
cd /home-ssd/Users/nsgm_jiangwh/youchang/ICLR/work/code/scripts/cluster
bash 02_prepare_data.sh
```

**完成标志**：
- ✓ 下载 UltraFeedback 10k 样本
- ✓ 生成 6 个 JSONL 文件
- ✓ 位置随机化审计通过
- ✓ 跨集合泄漏检查通过

**预计时间**：10-30 分钟（取决于网络）

---

### 阶段 2-4：预实验和主实验

**在登录节点提交作业**

```bash
# 1. 返回登录节点
exit  # 从编译节点退出

# 2. 准备训练脚本
cd /home-ssd/Users/nsgm_jiangwh/youchang/ICLR/work/code/scripts/cluster

# 3. 提交预实验作业
sbatch 03_preexperiment.sh

# 4. 查看作业状态
squeue -u nsgm_jiangwh

# 5. 预实验完成后，提交主实验
sbatch 05_run_main.sh
```

**完成标志**：
- ✓ 预实验锁定 (ε, β, lr)
- ✓ 8 次训练完成
- ✓ 每个方法生成 checkpoints 和 logs

**预计时间**：1-2 天

---

### 阶段 5-7：观测和评估

**在编译节点或登录节点执行**

```bash
# C_ε 观测
bash 06_c_epsilon.sh

# 测试评估
bash 07_evaluate.sh

# 结果聚合
bash 08_aggregate.sh
```

**完成标志**：
- ✓ C_ε 轨迹生成
- ✓ 测试指标计算
- ✓ 白名单结果导出

**预计时间**：半天

---

## 常用命令

### 环境管理

```bash
# 激活环境
source /home-ssd/Users/nsgm_jiangwh/youchang/activate_env.sh

# 查看 conda 环境
conda env list

# 检查 Python 和包
python --version
pip list | grep torch
```

### 作业管理

```bash
# 提交作业
sbatch script.sh

# 查看队列
squeue -u nsgm_jiangwh

# 取消作业
scancel <job_id>

# 查看作业详情
scontrol show job <job_id>

# 查看已完成作业
sacct -u nsgm_jiangwh
```

### 存储管理

```bash
# 查看存储使用情况
lfs quota -u nsgm_jiangwh /home-ssd/ -h

# 查看目录大小
du -sh /home-ssd/Users/nsgm_jiangwh/youchang/*

# 清理缓存
rm -rf /home-ssd/Users/nsgm_jiangwh/youchang/cache/*
```

### 文件传输

```bash
# 从本地上传
sftp nsgm_jiangwh@10.32.48.56
put -r /local/path /home-ssd/Users/nsgm_jiangwh/youchang/

# 下载到本地
get -r /home-ssd/Users/nsgm_jiangwh/youchang/export_local /local/path
```

---

## 目录结构

```
/home-ssd/Users/nsgm_jiangwh/youchang/
├── ICLR/                    # Git 仓库（轻量）
│   └── work/code/          # 源码、脚本、配置
├── envs/                    # Conda 环境
│   └── youc/
├── data/                    # 数据集
│   └── ultrafeedback/
├── models/                  # 模型
├── exp/                     # 实验结果（重量级）
│   └── cycle-20260818-01/
├── logs/                    # 日志
├── export_local/            # 可回传的结果
├── cache/                   # 缓存
├── project_config.json      # 项目配置
└── activate_env.sh          # 环境激活脚本
```

---

## 故障排查

### 问题 1：conda 命令不可用

```bash
# 初始化 conda
eval "$(conda shell.bash hook)"
```

### 问题 2：网络代理加载失败

```bash
# 手动加载（仅在编译节点）
source /home-ssd/Soft/modules/bashrc
module load proxy/proxy
```

### 问题 3：CUDA 不可用

```bash
# 检查 GPU
nvidia-smi

# 检查 PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### 问题 4：作业失败

```bash
# 查看作业输出
cat job.<job_id>.out

# 查看作业错误
sacct -j <job_id> --format=JobID,State,ExitCode
```

---

## 注意事项

1. ✅ **编译节点**：安装软件、下载数据
2. ✅ **登录节点**：提交作业、查看结果
3. ❌ **禁止在登录节点运行程序**（会被管理员警告）
4. ✅ **检查存储配额**：避免超额导致作业失败
5. ✅ **保留失败作业现场**：便于技术支持分析

---

## 支持联系

- **服务台邮箱**: service@nsccsz.cn
- **工单系统**: https://www.nsccsz.cn/tms/login.html
- **投诉必须明确包含**："投诉"字样

---

**最后更新**: 2026-08-20
