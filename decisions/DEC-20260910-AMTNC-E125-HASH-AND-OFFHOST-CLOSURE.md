# DEC-20260910：AM-TNC e125 固定里程碑与异机保护闭环

## 裁决

4090A 上的 AM-TNC 在 supervisor/trainer PID 3446757/3446758 不变、外层恢复 guard 零重启的条件下，按冻结协议自然完成 e125（1,069,125 updates）。运行解释器仍是 deleted inode，原 `unsb_cov` prefix 仍不是恢复权威；本次没有为消除标签而重启、迁移或覆盖任何健康进程。

e125 固定 checkpoint、sidecar 和指标产物已逐文件计算 SHA256，并复制到 `E:/UNSB_Expl/recovery_backups/4090A_AMTNC_20260910_E125` 后设为只读。指标文件只复制和哈希，没有读取数值。该 checkpoint 现在是最新的异机 full-state 锚点；隔离 runtime 的执行级恢复验证仍诚实保留在 e113 的 111/111 映射重哈希与 CPU 单步，不把本次里程碑写成新的 CUDA 逐位等价证明。

## 后继

- 保持 deleted-inode 健康训练连续运行，不发送信号、不原地补环境。
- 若活进程退出，guard 仍必须重新核验隔离 runtime、训练代码、协议、manifest 与当时最新 full-state 后才允许恢复。
- e150 由既有 source-bound incremental exporter 处理；e125 本身不触发新的性能裁决。
- matched control、主表和 sustained 裁决均按冻结协议等待，不使用此次中间指标控制训练，confirmation20 继续封存。

机器可读证据：`evidence/paper_aio/PAPER_AIO_AMTNC_E125_HASH_AND_OFFHOST_CLOSURE_20260910T123500.json`。
