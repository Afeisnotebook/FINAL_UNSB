# DEC-20260912：4090A误清理后的恢复与本地控制链闭环

## 裁决

4090A 的误清理确实删除了原 `unsb_cov` 项目路径和完整 Conda 环境，正在运行的
AM-TNC supervisor/trainer 仍由 deleted inode 提供解释器。因此不能把原路径当作可恢复
环境，也不能覆盖或盲目重启当前进程。

当前训练没有丢失：原 PID 3446757/3446758 未被触碰，已自然推进到 e165。唯一训练恢复
权威仍是只读隔离 runtime，其 Python、111 项映射依赖、full-state 加载和一次真实 CPU
续步均已验证；外机保存 e164 full-state 与 runtime archive。进程级故障通常最多损失当前
未完成 epoch，整机故障至少可从异机 e164 恢复。我们不宣称隔离 runtime 与仍存活的
deleted-inode 进程在 CUDA 下一步逐位一致。

本次审计同时发现 Proposal 的只读增量证据 relay PID 14528 已停止，但 Proposal 长训未受
影响，e100/e150 的 checkpoint、sidecar、export receipt 和 import set 哈希均完整。修复共享
JSON 读取的 Windows 短锁重试后，使用原冻结合同启动恢复监督器 PID 5112，并由其启动唯一
relay PID 31764。恢复前后既有 import set 与 lane receipt 哈希不变，未读取性能值。

本地 DCLGAN export/push、terminal audit/pathology 的存活 child 已由锁容错监督器原位收养；
新的组合 health PID 9960 对十个状态源报告零告警。仅退役重复或指向旧死 PID 的 health/control
进程，所有训练 PID、checkpoint、RNG、sampler 和协议均未改变。

## 后续约束

- AM-TNC 健康时继续保持原进程，严禁为了“修复路径”主动重启。
- AM-TNC 仅在原进程真实故障时，由 guard 在重新验签隔离 runtime 与最新 full-state 后恢复。
- Proposal relay 只传递冻结的 e100/e150/e200 target-blind 证据，不得读取指标或参与调度。
- confirmation20 继续封存；不改变 batch、数据顺序、算法或 runtime cohort。

证据：`evidence/paper_aio/PAPER_AIO_POST_CLEANUP_CONTROL_CHAIN_RECOVERY_20260912T035000.json`。
