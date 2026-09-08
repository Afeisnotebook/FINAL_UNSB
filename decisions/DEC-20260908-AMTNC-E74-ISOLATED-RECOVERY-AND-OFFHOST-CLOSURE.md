# DEC-20260908：AM-TNC e74 隔离恢复与异机保护闭环

## 裁决

4090A 的错误清理造成了真实环境损坏：原 `unsb_cov` 完整 Conda 环境已经不存在，运行中的
AM-TNC supervisor/trainer 仍持有 deleted inode。当前旧 prefix 中出现的 326 字节 Python
入口只是 fail-closed relay，加上少量残留文件；它既不是原解释器，也不是完整环境，因此不得
覆盖、补齐后宣称“原环境已恢复”。

健康训练保持原 PID 3446757/3446758，不重启、不迁移。截至 2026-09-08 10:16 +08，训练已
自然推进到 e74 / 632,922 updates，GPU 正常计算，outer guard 与全部关键后继均为零重启。

## 本次恢复核验

- 从 guard 的冻结命令直接使用隔离 Python，而不是旧 prefix；Python SHA256 与存活 deleted
  inode 捕获值一致。
- 对隔离 runtime 的 111 项映射、训练/控制 commit、协议 fingerprint 和 e74 full-state 重新
  执行 verify-only，返回 `PASS_RUNTIME_AND_CHECKPOINT_RECOVERY_PREFLIGHT`。
- e74 full-state、sidecar 和 heartbeat 在远端传输前后哈希不变，并复制到本机只读目录
  `E:/UNSB_Expl/recovery_backups/4090A_AMTNC_20260908_E74`；最新完整科学状态因此已有异机保护。
- terminal/incremental export、AM-TNC evaluation、unified evaluation、final delivery 与
  DCLGAN evaluation 的恢复监督器仍从独立隔离评估 runtime 运行，未依赖被删环境。

## 损失边界

已经完成的 epoch、checkpoint、G/F/D/E、optimizers、schedulers、AM-TNC 状态、RNG 和 sampler
均未丢失，后续训练恢复、导出和统一评估均未被阻塞。若当前 deleted-inode trainer 意外退出，
guard 会先重验隔离 runtime 和最新 full-state，再从最近完整 epoch 恢复；最坏计算损失仅为尚未
原子保存的当前 epoch。

仍不能诚实证明隔离 runtime 的下一次 CUDA update 与存活进程逐位一致，因为这需要中断当前
健康训练。事故因此不是“完全无影响”，但已经从可能丢失全部 AM-TNC 进度，收敛为一次可披露、
可恢复且有界的工程 provenance 风险。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_AMTNC_E74_ISOLATED_RECOVERY_AND_OFFHOST_CLOSURE_20260908T101700.json`。
