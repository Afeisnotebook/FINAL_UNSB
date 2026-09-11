# DEC-20260912：4090A 误清理影响与 AM-TNC e167 恢复闭环

## 结论

这次清理造成了真实但已经被限定住的工程损坏：`/home/yc/unsb_cov` 和原完整
`unsb_cov` Conda 环境不再存在，活动 supervisor/trainer PID `3446757/3446758` 仍依赖
deleted inode。旧 prefix 下的三个 Python 文件只是只读兼容 relay，不能称为原环境，也不能
作为恢复权威。为了消除 deleted 标签而主动重启，反而会把当前可工作的内存态变成不必要风险。

截至 e167（1,428,351 updates），没有已完成 epoch、checkpoint 或科学状态损失。训练仍在原
PID 上连续运行，guard 零重启、零告警，未观察到 NaN、OOM、保存失败；本次没有向健康训练
发送信号，也没有覆盖旧环境或修改算法协议。

## 已执行的补救

- 在隔离路径重新逐文件验签训练恢复 runtime，manifest 中 111/111 个映射文件的大小、只读
  属性和 SHA256 均匹配；隔离 Python SHA256 为
  `66fccc990e49c33a1f06249128136ef063f16883ef06550960ce79943c5d13ab`。
- 用该隔离 Python 在 CPU 上完整加载 e167 full-state，确认 G/F/D/E、4 个 optimizer、4 个
  scheduler、AM-TNC 方法状态、Python/NumPy/CPU/CUDA RNG、primary/secondary sampler、step
  和 epoch 均在；加载前后 checkpoint SHA256 保持
  `5576708a8c97de43ba1aa6a2fb5c0ab7d6895a0c59ec730b8a29c540c35c35d4`。
- 既有 e154 隔离 runtime 的真实一步恢复探针仍提供执行级证明；不能在不中断当前进程的前提下
  证明下一次 CUDA update 与 deleted-inode 进程逐位一致，因此不作这一额外声明。
- 外层 guard PID `439182` 直接绑定隔离 runtime，在任何恢复启动前重验 runtime 和最新
  full-state，身份不一致、重复进程或无进展预算耗尽时 fail closed。health PID `439447`
  当前健康且 restart count 为 0。
- e200 terminal exporter 的现有等待 child 仍持有 deleted inode，但其 recovery supervisor
  已直接绑定并重验固定 evaluator runtime；incremental exporter 当前 child 已使用该隔离 runtime。
  evaluator Python SHA256 已复核为
  `40dce5922786c10046c262c038a9c74fc7d6bb0ae3fdc189449a07687269b0fd`，因此旧等待 child 即使退出
  也不会迫使后继从残缺 prefix 启动。清理没有阻断 source-bound 导出或统一评估链。
- 将 e167 full-state、sidecar、heartbeat 和 lane authorization 复制到本机异盘只读目录
  `E:/UNSB_Expl/recovery_backups/4090A_AMTNC_20260912_E167`；传输前后远端源哈希不变，逐文件
  与本地一致。隔离 runtime 的 5.49 GB 异机归档继续保留。

## 损失上界与后续动作

若仅当前进程退出，恢复通常最多重做尚未原子保存的当前 epoch；若 4090A 主机与本地盘同时
丢失，可从异机 e167 full-state 和隔离 runtime 恢复。当前最安全的动作是让健康 PID 自然运行
到 e200，由既有 source-bound exporter 和统一评估链接管；不得盲目从旧路径新起进程。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_AMTNC_E167_RECOVERY_IMPACT_AND_OFFHOST_CLOSURE_20260912T052145.json`。
