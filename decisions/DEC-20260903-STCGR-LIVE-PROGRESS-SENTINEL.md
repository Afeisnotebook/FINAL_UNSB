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
