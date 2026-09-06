# 5090C Proposal e75 stall exact-resume decision

5090C Proposal 在完成 e75 后没有继续写出 epoch heartbeat。冻结的 progress
watcher 在 14,400 秒门限处将它判为
`ALERT_LIVE_PROCESS_COMPUTE_WITHOUT_IO_PROGRESS`：训练 PID 9734 仍以 CPU
计算状态存活，但 GPU 连续采样为 0，checkpoint、heartbeat 和 I/O 均未前进。
日志与内核没有 OOM、CUDA error、NaN 或进程被系统杀死的证据。

恢复前重新计算了 e75 checkpoint 与 sidecar SHA-256，并在同一 5090C runtime
以 CPU 完整加载 checkpoint。状态包含 step 641475、模型/优化状态、Python、NumPy、
CPU/CUDA RNG、两个独立 sampler 和冻结协议元数据。因此仅向失去进度的子 PID 9734
发送一次 SIGTERM；没有终止 supervisor PID 9729、迁移 lane、修改 runtime、算法、
batch 或协议。原 supervisor 按其冻结的 `--resume` 命令，在 30 秒后启动 PID 366768。

新 PID 随后恢复 33--55% GPU 利用率，并在 4675.53 秒后原子写出 e76 / 650028
updates。e76 checkpoint 和 sidecar重新哈希，且再次通过 CPU full-state 加载；progress
watcher与主health watcher均回到健康状态。由此接受“从最后完整 e75 full state 恢复并
重新执行部分 e76”的工程闭环。没有完成的 epoch 丢失，但由于不存在未中断的反事实
e76，本裁决不声称二者逐字节相同。

恢复不读取paired性能值，confirmation20继续封存，5090C与既有matched-runtime关系
不变。按恢复后的完整epoch速度，e200粗略预计为9月13日23:10，距登记的9月15日租期
边界仍约24.8小时；当前继续运行，并在e80或下一次实质恢复事件重新估算。

证据：
`evidence/paper_aio/PAPER_AIO_5090C_PROPOSAL_E75_STALL_EXACT_RESUME_E76_CLOSURE_20260907T060850.json`
