# DEC-20260911：5090B matched plain e64 停滞恢复

## 决策

5090B matched plain 不是健康慢跑：heartbeat、checkpoint 和训练 trace 在 e64 后约 66.9 小时
没有任何写入，progress watcher 已累计 2596 次
`ALERT_LIVE_PROCESS_COMPUTE_WITHOUT_IO_PROGRESS`，两次 GPU 采样均为 0%，且 8 秒内进程 I/O
完全不变。因此本次只终止精确匹配的停滞 child PID 534984，保留 successor、supervisor、
exporter、run 目录和 e64 full-state，由原 supervisor 从相同 checkpoint 精确恢复。

## 恢复门与当前状态

- 恢复前确认训练仓 commit `e4a5eed9fe14e671e07329a970d93cd9828240ac` 且工作区干净；manifest、
  protocol fingerprint、PID parent/command、checkpoint SHA 和 CPU load 均通过。
- e64 checkpoint step 为 547392，SHA256 为
  `dcaa116e26e341189ce99ed9a389510adca49a36aa0324d82861fd5c31caf2b8`。
- 原 supervisor PID 534983 已启动新 trainer PID 76358；恢复后 GPU 利用率 53%，I/O 活跃，
  supervisor failure count 为 1。只有 e65 新 full-state 和 scientific hash 出现后才升级为
  exact-resume closure。
- 已完成 epoch 没有损失，runtime cohort 和科学协议没有改变。

## 调度影响

原 9 月 12 日 e200 预测失效。按 e64 前 GPU 独占实测速率重新估算约为 9 月 15 日 15:25，
5090B 租期需要立即复核，可能至少延长两天。matched delta 在 e200 导出前继续不可用。

## 监控缺口

progress watcher 正确报警，但上层 health watcher 没有把其 alert 状态传播为故障。先等待 e65
闭环，再在不触碰训练的前提下修复健康汇总语义，避免再次静默停滞。

本次未读取性能值、未打开 confirmation20，也未改变算法、batch、数据顺序或评估协议。

证据：
`evidence/paper_aio/PAPER_AIO_5090B_MATCHED_PLAIN_E64_STALL_RECOVERY_STARTED_20260911T204955.json`
