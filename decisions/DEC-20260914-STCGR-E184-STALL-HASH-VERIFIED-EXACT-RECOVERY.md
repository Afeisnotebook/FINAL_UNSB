# ST-CGR e184 工程停滞的哈希恢复

5090A 的 ST-CGR 在 e184 后超过冻结的 7,200 秒无完整 epoch、checkpoint 或 I/O 进展；实际心跳年龄约 34,587 秒，连续五次 GPU 采样均为 0%。trainer 仍消耗 CPU，但 progress watcher 已明确报告 `ALERT_LIVE_PROCESS_COMPUTE_WITHOUT_IO_PROGRESS`。这是与历史 e69/e125 相同的工程停滞类别，不是算法性能裁决。

恢复前核验了唯一 supervisor/trainer、父子关系、训练 commit、协议 fingerprint、manifest、candidate authorization 以及 e184 full state 和 sidecar 哈希，并把完整 e184 状态复制到本机异机备份。随后只向精确匹配的停滞 trainer PID 357237 发送 SIGTERM。原 supervisor PID 357236 和 outer guard PID 942101 均保持不变；supervisor 从同一 e184 full state 启动 trainer PID 42095，恢复后 GPU 采样为 33%–56%，progress watcher 回到 `HEALTHY_WITHIN_EPOCH_BOUND`。

没有丢失任何完整 epoch，最多重放 e184 之后尚未提交的一个 epoch 内工作；没有更改 batch、AMP、TF32、算法、数据顺序、优化器或 runtime cohort，也没有读取性能指标。当前状态是“exact resume 已启动”，必须等新 trainer 完成 e185 并核验 full-state/scientific hash 后，才关闭本次连续性门。
