# DEC-20260912：Goal heartbeat 同步 DDSB watcher 恢复状态

DDSB source watcher 已从旧 PID `20388` 的 Windows 状态文件写入故障恢复到 detached control
commit `40c4b37`，新 source/health PID 为 `22104/16220`。持久 heartbeat 若仍只写“DDSB
reproduction incomplete”而没有新的恢复身份，会在后续故障中继续检查已死亡 PID，构成长程
任务静默停止风险。

现已原位更新同一 `final-unsb-goal` automation，保持 ACTIVE、两小时间隔、仅失败通知和目标
线程不变。prompt 绑定主仓 `83fdf5b`、新 DDSB control commit/PID/output，并明确候选命中只能
进入人工作者身份、公式与实现审查，永不自动授权 DDSB 训练。其它训练、后继与科学边界没有
改变；配置复读确认不含任何凭据。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_GOAL_HEARTBEAT_DDSB_RECOVERY_SYNC_20260912T060920.json`。
