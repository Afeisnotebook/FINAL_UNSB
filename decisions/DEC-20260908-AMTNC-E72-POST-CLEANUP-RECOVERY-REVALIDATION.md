# DEC-20260908：4090A 错误清理后的 AM-TNC e72 恢复复核

## 裁决

错误清理造成了真实且不可原地掩盖的影响：原 `unsb_cov` 完整环境已经不存在，AM-TNC
trainer PID 3446758 仍从 deleted inode 运行。旧 prefix 现在只是若干入口脚本和三个
fail-closed Python relay，不是可被当作 Conda 环境的新进程基础，也不得覆盖重建。

当前已完成训练成果没有丢失，后续也不再被该路径阻塞。AM-TNC 在原 PID 上自然推进到
e72 / 615,816 updates；最新 full-state、算法状态、优化器、调度器、RNG 和双 sampler
均已原子保存。隔离 runtime 对全部 111 项映射重新验签，并从 e72 状态在独立目录真实执行
一项 CPU update：`615816 -> 615817`，以 `ENGINEERING_PAUSE` 正常退出，stderr 为 0。

## 已采取的非破坏性保护

- 未向 PID 3446757/3446758 发送信号，未迁移、重启或修改当前训练。
- 训练 guard PID 439182 继续监控原 supervisor，重启计数为 0；若原 supervisor 和 trainer
  同时消失，它会先验签隔离 Python、111 项依赖、代码、协议和最新 full-state，再使用隔离
  runtime 的绝对路径恢复。
- terminal source export、incremental export、AM-TNC evaluation、unified evaluation、
  final delivery 和 DCLGAN evaluation 均有从独立评估 runtime 运行的恢复监督器；当前恢复
  计数均为 0。仍持 deleted inode 的旧 child 允许自然存活，但不再是后继恢复的唯一入口。
- e72 原始 checkpoint、sidecar、heartbeat 和 lane authorization 已复制到
  `E:/UNSB_Expl/recovery_backups/4090A_AMTNC_20260908_E72`，逐文件哈希与远端源一致并设为只读；
  隔离 runtime 的异机归档也继续保留。

## 损失边界

如果当前 trainer 正常存活到 e200，事故不造成训练损失。如果它意外退出，可从最近完整
data epoch 精确恢复，最多损失尚未保存的当前 epoch，而不是数日累计进度。不能诚实证明的
唯一一点，是隔离 runtime 的下一次 CUDA update 与仍存活的 deleted-inode 进程逐位相同；
证明它需要中断健康训练，因此没有执行。这个限制不等于恢复不可用，但必须在论文运行回执中
保留披露。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_AMTNC_E72_POST_CLEANUP_RECOVERY_REVALIDATION_20260908T075200.json`。
