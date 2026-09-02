# Round4 v2 服务器脚本

所有命令仅在代码交接获用户明确确认后的服务器执行阶段运行。

## 责任边界

- 4090-3：构建离线wheelhouse、下载/校验冻结资产、维护私有 judge profile，并对从A100传来的输出调用API judge。
- A100-2：离线安装exact-commit环境、预处理、候选B（仅FrozenPE）、2-GPU训练/eval、merge/reload和输出生成。
- A100不读取API key或base URL；4090不训练、不保存checkpoint。

## v2 2-step smoke

`03_prepare_smoke.py`和`03_run_smoke_a100.sh`运行四个独立臂：DPO、SSPO、
StaticPE、FrozenPE。DPO的effective batch为16；后三者为64。每臂恰好2个optimizer
step、每step eval、adapter、merge和新进程生成前2条冻结AlpacaEval instructions。

`03_run_smoke_a100.sh`最终写`SMOKE_SUMMARY.json`和不含回答内容的`JUDGE_REQUEST.json`。
将请求及其校验过的模型输出传到4090后，使用：

```bash
export ROUND4_JUDGE_PYTHON='<4090 judge venv>/bin/python'
export ROUND4_JUDGE_API_KEY='...'
export ROUND4_JUDGE_BASE_URL='.../compatible-mode/v1'
bash code/scripts/round4/04_run_api_judge_4090.sh \
  primary <model_outputs.json> <reference_outputs.json> <new_output_dir>
```

judge profile保存在4090的`~/.config/soppo/judge_profiles.json`（权限0600），格式从
`SSPO/examples/evaluation/judge_profiles.example.json`复制。profile只保留环境变量名、
模型和解码合同；key与base URL从4090进程环境读取。结果只写WR、LC、profile fingerprint
和judge model，逐样本annotation留在4090。

## v2 执行门禁

`00_*`、`01_*`、`02_*`的旧离线环境/资产脚本仍可复用，但必须用新的exact commit重新
构建项目wheel和commit-bound venv。保留旧`af6dac4` prepared失败目录。完成四臂GPU smoke
与4090 API smoke且用户审阅汇总后，才能依次进行formal：base、DPO、SSPO、StaticPE、FrozenPE。
