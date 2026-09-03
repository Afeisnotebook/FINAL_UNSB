# ST-CGR live-process progress sentinel

5090A 的 ST-CGR 在 e12 后出现过一次特殊工程停滞：trainer PID 存活且 CPU 时间继续
增加，但输入 I/O 不再变化，GPU 也持续空闲。普通健康监控只检查 PID 和 epoch 级
heartbeat，因此会把这类状态延迟数小时才报告。

新增 `operations/paper_aio_progress_watch.py` 作为只诊断的补充哨兵。它从 supervisor 的
Linux children 状态动态识别当前 trainer，并结合 heartbeat 年龄、child 启动时间及固定
窗口内的 `/proc/PID/io` 与 CPU ticks 变化进行分类。精确恢复会产生新 child；因此有效
停滞年龄取 heartbeat 年龄与 child 年龄的较小值，不会因旧 heartbeat 对刚恢复的 child
产生假告警。超过两小时但仍有 I/O 进展也不会报警。

该哨兵没有发送 signal、启动训练或修改 checkpoint 的代码路径，也不读取任何性能结果。
告警只提供给现有每小时 Goal 做人工/Codex 工程裁决；实际恢复仍必须核对 checkpoint、
milestone 阶段和进程证据后，只终止卡死 child，再由既有 supervisor 精确恢复。

单元测试覆盖 child 重启时钟、过期但仍推进、纯计算无 I/O 停滞、近期 heartbeat 以及
Linux stat 解析。部署到 5090A 后应使用 7200 秒阈值、30 秒采样与 60 秒轮询，且不得
替换或重启现有 ST-CGR supervisor/trainer。

实现 commit `88e68d6` 的全量测试为 `637 passed`。服务器直连 GitHub 首次因 TLS 中断
失败，因此使用校验通过的完整历史 Git bundle 建立固定 checkout；这没有触碰训练仓。
哨兵 PID `435350`、PPID 1。首个真实 30 秒探针识别当前 trainer PID `431032`，用新 child
年龄把旧 e12 heartbeat 的 9406 秒折算为 1712 秒有效年龄，期间读取字符增加约 45.9 MB、
实际块读取增加约 26.4 MB，状态为 `HEALTHY_WITHIN_EPOCH_BOUND`、零告警。

部署后的恢复性复核发现 v1 把哨兵自身 PID 放进冻结合同，导致同目录重启时合同比较必然
失败。v2 将 PID 仅写入动态 state，冻结合同只保留宿主、supervisor、路径和阈值；因此
相同命令可以在不改变监控语义的情况下恢复。v2 验证后替换 v1，训练进程不受影响。
