# DEC-20260912：持久 Goal 同步到 AM-TNC e168 与 DDSB 恢复监督链

两小时 `final-unsb-goal` heartbeat 已原位更新，继续保持 ACTIVE 和仅失败通知。提示词现在明确：

- 4090A AM-TNC 已连续到 e168；健康的 deleted-inode 进程不得为了消除标签被重启或覆盖，
  未来恢复只认已验签的隔离训练 runtime 与外层 guard。
- DDSB 公开来源守望以 source PID `22104`、fail-closed recovery supervisor PID `26164`
  和 combined health PID `30196` 为权威；已退役 PID `16220/31124` 不得恢复。
- DDSB 候选命中仍只触发人工来源与公式审查，不能自动授权训练。

自动化频率、通知策略、科学队列与训练协议均未改变，配置中不含服务器凭据。
