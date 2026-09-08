# DEC-20260908：AM-TNC e76 清理事故后的可执行恢复闭环

## 事故影响

4090A 的错误清理确实造成了不可忽略的环境损坏：原 `unsb_cov` Conda 环境已经不存在，
运行中的 AM-TNC supervisor/trainer 仍持有 deleted inode。旧路径目前只有三个只读 relay 和
少量残留文件，不能被称作恢复后的 Conda 环境，也不得在原路径覆盖重建。

健康训练保持 PID 3446757/3446758 连续运行，未收到信号、未迁移、未重启。截至本次核查，
它已经自然推进到 e76 / 650,028 updates，最新完整 checkpoint、sidecar 和科学状态仍完整。

## 本次处置

- 直接使用隔离恢复目录，而不借用旧 prefix 语义；对 Python、manifest 和 111 项映射依赖做
  全量只读哈希复核，并同时核验训练/控制 commit、协议 fingerprint 与 e76 full-state，结果为
  `PASS_RUNTIME_AND_CHECKPOINT_RECOVERY_PREFLIGHT`。
- 将 e76 full-state 复制到独立 probe 目录，隐藏 CUDA 并使用 `gpu=-1`，成功执行
  650,028 → 650,029 的真实训练更新；stderr 为 0。主 lane checkpoint 哈希、原 PID 和 GPU
  进程均未变化。
- 将 e76 full-state、sidecar、heartbeat 和 lane authorization 复制到本机只读目录
  `E:/UNSB_Expl/recovery_backups/4090A_AMTNC_20260908_E76`，逐文件 SHA256 与远端一致。
  已有的 5.49 GB 隔离 runtime 归档继续在本机只读保存。
- 外层 guard PID 439182 与健康 watcher PID 439447 继续监视原进程，restart count 为 0；只有
  原 supervisor 和 trainer 都消失且运行时、源码、协议和最新 full-state 全部重新验签后，
  才允许从隔离 Python 启动恢复，否则 fail closed。

## 裁决

事故没有造成已完成 epoch、模型/优化器/调度器、AM-TNC 方法状态、RNG、sampler 或 checkpoint
丢失；后续训练恢复、source-bound export 和统一评估目前均未被阻塞。即使当前进程现在退出，
可从最近完整 epoch 恢复，最坏计算损失限定为尚未原子保存的当前 epoch；若整台4090A主机
损坏，e76状态和精确 runtime 也已有异机副本。

仍不能声称事故“毫无影响”：不打断健康训练就无法证明隔离 runtime 的下一次 CUDA update
与 deleted-inode 进程逐位一致。因此论文 provenance 必须继续披露该工程事故，旧 prefix 也
永久失去恢复权威地位。当前最安全行动是保持训练连续运行，由隔离 guard 承担后继恢复。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_AMTNC_E76_POST_CLEANUP_EXECUTABLE_RECOVERY_CLOSURE_20260908T120600.json`。
