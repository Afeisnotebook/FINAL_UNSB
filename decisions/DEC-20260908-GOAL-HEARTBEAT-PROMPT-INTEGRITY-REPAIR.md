# Goal heartbeat prompt完整性修复

日期：2026-09-08

`final-unsb-goal`自动化仍为`ACTIVE`，但其持久化名称和prompt已经乱码，并且4090A恢复说明
停留在e63旧证据。这没有改变任何训练或监督器，却会使后续Codex唤醒无法可靠理解北极星、
deleted-inode恢复边界和当前PID关系，属于“运行还在但监控语义可能静默失效”的真实风险。

通过Codex自动化接口完整重写该heartbeat，而不是直接编辑自动化文件。保留原两小时频率、
当前线程、`failed_runs_only`通知策略和`ACTIVE`状态；新prompt不保存任何SSH密码，并包含：

- 多算法full-data e200、matched controls、合法外部基线、统一评估与长期因果审计的北极星；
- 4090A deleted-inode进程不得主动重启、只读relay不是完整环境、111项依赖和e64恢复门；
- 最新直接核验的AM-TNC e65、ST-CGR e82、Proposal e90、CycleGAN e188、matched plain e32
  与DCLGAN e52状态；
- 每条训练、export、import和统一评估的权威守护关系；
- confirmation20封存、不得用中间paired指标调度、不得重跑健康训练等硬边界；
- 只有真实故障或重要交付事件才修复、提交并通知，健康等待保持安静。

更新后重新读取持久化TOML，名称和完整prompt均为可读UTF-8，`updated_at`已经变化，状态、
频率和通知策略保持不变。本修复没有触碰GPU、checkpoint、训练PID或队列。
