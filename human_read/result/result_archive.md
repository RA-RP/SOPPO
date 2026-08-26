# 所有实验结果归档

本文件按时间顺序追加所有已完成、失败、中断或无结论实验的结果快照。历史条目只允许追加勘误或后续解释，不应覆盖原始观察。

## 归档模板

### <cycle ID> / <experiment ID> — <日期> — <状态>

- 对应设计版本：
- 原始证据目录：
- 代码与配置版本：
- 执行完整性与偏差：
- 关键观察及不确定性：
- 与预先判断标准的比较：
- `C_{\gamma}` 观察：
- 当前解释：
- 替代解释/混杂：
- 结论边界：
- 后续勘误或复现情况：

## 归档记录

### `cycle-20260818-01` / Round2 experiment ID未知 — 2026-08-26 — `COMPLETE_WITHOUT_RESULT_EVIDENCE`

- 对应设计版本：Round2 v0.6历史合同，见`../exp/experiment_archive.md`与`../code/ROUND2_LIVE_HANDOFF.md`
- 原始证据目录：未回传；服务器experiment ID、commit、step和PID未知
- 代码与配置版本：未从服务器`controller.json`解析，不从本地HEAD猜测
- 执行完整性与偏差：只有2026-08-24中途快照；两条formal、evaluation、aggregate与keep-20 pruner的最终状态均未知
- 关键观察及不确定性：没有可核验的final metrics；用户于2026-08-26指示“Round2先不管了，直接开始Round3”
- 与预先判断标准的比较：证据不足，所有预测均无法判断
- `C_{\gamma}`观察：Round2未预注册该阶段，也没有回传数据
- 当前解释：这是用户选择不再等待证据的行政关闭，不是科学成功/失败判定
- 替代解释/混杂：服务器任务可能已完成、失败或仍在运行；本地没有实时证据区分
- 结论边界：Round2不向论文主张提供可用实证结论；Round3是用户独立审阅并批准的新方案
- 后续勘误或复现情况：若未来取得Round2白名单证据，只追加新记录；不覆盖本次`NO_CONCLUSION`快照
- 服务器安全边界：本次未运行停止、修改checkout或删除checkpoint操作

### `cycle-20260818-01` / `exp-20260824-05-round2-tp2` — 2026-08-26 — 后续执行状态勘误

- 勘误范围：只补充原归档后取得的只读执行证据，不改变`NO_CONCLUSION`科学结论。
- 执行状态：controller在第一方法step590以exit 143停止；第二方法未启动，independent evaluation与aggregate未运行。
- 保留状态：第一方法共20个checkpoint，含step580/589/590，best指向step480；两个keep-20 pruner PID为陈旧记录、对应进程不存在。
- 资源快照：核验时三张GPU空闲；`/data`约96G可用、使用率98%。这是带时间证据，不是永久实时状态。
- 安全边界：核验没有发送停止信号、修改服务器checkout或删除checkpoint；旧Round2环境后来已由外部操作删除，runs/checkpoints仍须保留。
- 结论边界：没有final metrics，不能判断任一方法效果，也不能把step590或exit 143解释为训练完成。
