# DEC-20260907：5090B matched plain使用progress-aware健康监控

## 裁决

5090B matched plain没有停滞。旧健康监视器PID 524155仅按raw heartbeat的7200秒阈值报警，
但该阈值已经短于实际完成的共驻e25（7292秒），因此其`ALERT_STATE_STALE`是假阳性，不能
触发训练恢复。

同一时刻，正式progress watcher PID 524156确认trainer PID 534984仍在运行并持续产生CPU和
读取I/O；30秒内CPU tick增加13059、读取约49.6 MB。随后原进程完成e26、222378 steps，耗时
8984.00秒，直接证明它一直在推进。

## 非破坏性替换

新增健康监视器PID 667507。它不直接把长epoch当故障，而是检查：

- successor PID 523993及其持续更新的控制状态；
- progress watcher PID 524156及其综合PID、CPU、I/O和checkpoint判定；
- 实际磁盘余量。

新监视器连续三次轮询健康、零告警，并在SSH断开后确认PPID为1。之后才核对旧PID命令并向
旧监视器发送SIGTERM。plain和CycleGAN的supervisor/trainer/exporter均未改变或中断。

e26 checkpoint重算SHA与sidecar一致，隔离CPU加载完整的G/F/D/E、optimizers、schedulers、
方法状态、四类RNG和双sampler。此次调整仅修正观测层，不改变runtime cohort、训练协议、
数据顺序或matched关系；没有读取性能值，confirmation20继续封存。

完整回执：
`evidence/paper_aio/PAPER_AIO_5090B_MATCHED_PLAIN_E26_PROGRESS_AWARE_HEALTH_REPLACEMENT_20260907T162554.json`。
