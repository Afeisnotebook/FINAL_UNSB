# DEC-20260911：AM-TNC 与 Proposal e150 固定里程碑闭环

4090A AM-TNC和5090C Proposal均按各自冻结协议自然完成e150 / 1,282,950 updates。
两条lane的固定checkpoint、sidecar和指标文件均只做逐文件SHA256，不读取性能值；增量导出
receipt亦已产生。两套e150状态、指标哈希、heartbeat与guard状态已复制到本机独立目录，复核
源端哈希后设置为只读。

AM-TNC原deleted-inode supervisor/trainer PID 3446757/3446758没有被触碰，隔离恢复guard
PID 439182保持零重启；Proposal supervisor/trainer PID 9729/766090及outer guard PID 431648
同样保持健康、零guard重启。两条训练继续至e200。

本次里程碑不产生算法性能裁决，不改变matched plain关系，不选择最佳checkpoint，也不打开
confirmation20。ST-CGR与DCLGAN同步只读核查健康，未修改其进程。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_AMTNC_AND_PROPOSAL_E150_HASH_OFFHOST_CLOSURE_20260911T124000.json`。
