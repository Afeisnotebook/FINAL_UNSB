# DEC-20260908：AM-TNC 清理事故后自然完成 e80

4090A 的 AM-TNC supervisor/trainer PID `3446757/3446758` 在未收到信号、未迁移、未重启、
未覆盖旧环境的条件下自然完成 e80 / 684,240 updates。活动解释器继续显示 deleted inode；
这仍是已知清理事故的可见结果，不是重启理由。外层隔离恢复 guard/health PID
`439182/439447` 持续健康，restart count 为 0，恢复命令继续直接绑定隔离 runtime，而不是
已失去权威的旧 prefix。

e80 full-state、sidecar、heartbeat 和静态 lane authorization 已复制到本机只读目录
`E:/UNSB_Expl/recovery_backups/4090A_AMTNC_20260908_E80`。核心文件 SHA256 与4090A源文件一致，
因此整机故障时的异机恢复锚点由 e76 前移到 e80；当前训练和所有后继队列均未触碰。

本里程碑不重复执行恢复探针：e76已经完成隔离 runtime 111/111哈希复核和真实CPU单步恢复，
当前没有身份漂移、进程故障或其他触发新探针的证据。仍保留“隔离恢复后的下一次CUDA更新
不能在不中断健康进程时证明逐位等价”的披露边界。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_AMTNC_E80_POST_CLEANUP_MILESTONE_20260908T161300.json`。
