# DEC-20260911：4090A 清理事故后的 AM-TNC e154 隔离恢复复核

## 裁决

错误清理确实删除了原完整 `unsb_cov` 环境，当前 AM-TNC supervisor/trainer 仍由
deleted inode 执行；不能把旧目录的外观当作已经恢复，也不能覆盖或重启健康进程。
不过，这次复核确认该事故目前没有造成已完成训练状态损失，也不再阻塞后续恢复。

旧 Conda prefix 只保留三个只读 relay，不是恢复权威。真正的恢复权威位于独立只读目录
`/home/yc/recovery_snapshots/unsb_cov_runtime_candidate_exact_3446758_20260906T090000`，
持久 guard 的冻结命令直接调用该路径，不依赖已删除 inode 或不存在的
`/home/yc/unsb_cov`。

## 本次实际核验

- AM-TNC 原 supervisor/trainer PID 3446757/3446758 未收到信号、未重启、未迁移，已自然
  完成 e154 / 1,317,162 updates。
- 新的 verify-only 门重新验签 Python、manifest、全部 111 个运行时文件、训练代码、
  协议、lane authorization 及 e154 full-state，结果为
  `PASS_RUNTIME_AND_CHECKPOINT_RECOVERY_PREFLIGHT`。
- 从 e154 checkpoint 复制出的隔离分支在 CPU 上真实执行一项 update：
  `1,317,162 -> 1,317,163`，以 `ENGINEERING_PAUSE` 正常退出，stderr 为 0；
  主 checkpoint 哈希及活动 PID 均未改变。
- 旧 prefix 的 relay 也实测可启动新进程并导入匹配的 PyTorch/NumPy/Pillow/CUDA；
  但它仍只作为兼容入口，不作为恢复证明或权威路径。
- e154 full-state、sidecar、heartbeat 和 lane authorization 已复制到本机异盘，
  逐文件哈希与来源一致并设为只读；5.49 GB 隔离运行时异机归档继续存在且只读。

## 损失边界

如果当前 deleted-inode trainer 自然运行到 e200，清理事故不会造成训练结果损失；如果它
意外退出，guard 会先重新验签隔离运行时和最新完整状态，再从完整 data-epoch 精确恢复。
最坏只需重做退出时尚未保存的一个 epoch，而不是从头训练。

唯一不能在不中断健康进程的前提下证明的是：隔离 runtime 的下一次 CUDA update 与原
deleted-inode 进程逐位相同。该边界保留在论文运行回执中，不被夸大为跨运行时逐位等价。
本次没有读取性能值、没有改变算法或协议，也没有打开 confirmation20。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_AMTNC_E154_ISOLATED_RECOVERY_REVALIDATION_20260911T165734.json`。
