# DEC-20260912：AM-TNC e169 清理影响复核与恢复保护更新

## 裁决

4090A 的错误清理确实造成了工程损坏，而不是显示问题：原始 `/home/yc/unsb_cov`
和完整 `unsb_cov` Conda 环境仍不存在，活动 supervisor/trainer PID
`3446757/3446758` 的可执行文件仍显示为 deleted inode。因此不得为了消除该标签而重启
健康进程，也不得把旧 prefix 下的兼容入口当作完整环境或恢复权威。

截至 e169（1,445,457 updates），未发现科学状态损失。活动训练连续运行，checkpoint 与
sidecar 哈希一致，guard PID `439182` 和 health PID `439447` 健康、零重启、零告警；未观察
到 NaN、OOM 或保存失败。本次没有向训练进程发送信号，没有修改算法或训练协议。

## 本次复核与补强

- 重新逐文件计算隔离训练 runtime 的哈希：manifest 中 111/111 个文件、共
  4,198,514,872 bytes 全部匹配；隔离 Python SHA256 仍为
  `66fccc990e49c33a1f06249128136ef063f16883ef06550960ce79943c5d13ab`。
- 使用该隔离 Python 在 CPU 上完整加载 e169 full-state，确认 G/F/D/E、optimizers、
  schedulers、AM-TNC 方法状态、Python/NumPy/CPU/CUDA RNG、双 sampler、step 与 epoch 均在；
  加载前后 checkpoint SHA256 均为
  `7c2d5c03c6d4adff8b71be7cb0d5e3437fc0176f24b5aac1922b2e3f4c887af6`。
- guard 的冻结恢复命令直接使用隔离 Python，并在真正启动前重验 runtime 与最新 full-state；
  身份不匹配、重复进程或连续无进展时 fail closed。e200 exporter 的恢复 supervisor 也使用
  独立 evaluator runtime，因此原环境删除不会阻断导出和统一评估。
- 将 e169 full-state、sidecar、heartbeat 和 lane authorization 复制到本机异盘目录
  `E:/UNSB_Expl/recovery_backups/4090A_AMTNC_20260912_E169`，传输后再次核对远端源与本地
  SHA256 完全一致，并把本地文件设为只读。e175 的永久 milestone 异机备份仍按既有任务执行。

## 可避免的损失与剩余边界

若只有活动进程退出，服务器上的 e169 full-state 与隔离 runtime 可恢复，最多重做尚未原子
保存的当前 epoch；若整台4090及其存储同时丢失，本机异盘现也可从 e169 恢复。因而错误清理
不再意味着从头重训，也不阻断 e200 或论文评估链。

仍不能在不打断健康进程的前提下证明：隔离 runtime 恢复后的下一次 CUDA update 与当前
deleted-inode 进程逐位相同。若真正发生恢复，必须将其作为工程 provenance 边界披露，不能
虚构为同一运行时逐位连续。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_AMTNC_E169_POST_CLEANUP_RECOVERY_REAUDIT_20260912T071106.json`。
