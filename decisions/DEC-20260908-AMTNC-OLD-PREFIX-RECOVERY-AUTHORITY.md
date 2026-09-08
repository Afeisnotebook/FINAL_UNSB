# DEC-20260908：AM-TNC 旧路径不是恢复权威

## 裁决

4090A 的错误清理确实删除了原 `unsb_cov` 完整 Conda 环境；正在运行的 AM-TNC
supervisor/trainer 仍映射 deleted inode，这一事实不能被改写为“环境已原地恢复”。但后续恢复
也不再依赖该路径：训练 guard 的冻结命令直接以只读隔离 runtime 的绝对 Python 启动
supervisor，而 supervisor 使用同一 `sys.executable` 启动 trainer。

截至 2026-09-08 08:47 +08，PID 3446757/3446758 仍健康，最新完整状态为 e73 / 624,369
updates。重新执行的 verify-only 门以退出码 0 验签全部 111 项 runtime 映射、代码 commit、协议
fingerprint 和 e73 full-state。旧路径中的 Python 只是 fail-closed relay；禁用 CUDA 的导入探针
证明它当前能转交到隔离 runtime，但它不构成完整环境，也不得成为恢复权威或被覆盖重建。

## 损失与后继影响

- 已完成 epoch、checkpoint、优化器、方法状态、RNG 和 sampler 均未丢失。
- 若活进程退出，outer guard 从最新完整 epoch 使用隔离 runtime 精确恢复；最大计算损失为当前
  未完成 epoch。
- terminal/incremental export、AM-TNC evaluation、unified evaluation、final delivery 和
  DCLGAN evaluation 的恢复监督器均从独立评估 runtime 运行，重启计数为 0。
- 隔离 runtime 与 e73 full-state 已有异机哈希备份。

唯一不能在不中断健康训练的条件下证明的是：隔离 runtime 的下一次 CUDA update 与仍存活的
deleted-inode 进程逐位一致。因此事故不能声称“完全没有风险”，但已经从“旧路径一删即无法
恢复”降为“最坏回退到最近完整 epoch，并保留运行时披露”。当前训练不重启、不迁移、不覆盖
旧 prefix。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_AMTNC_OLD_PREFIX_AUTHORITY_AUDIT_20260908T083700.json`。
