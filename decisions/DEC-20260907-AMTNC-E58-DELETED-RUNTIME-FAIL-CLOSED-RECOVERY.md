# DEC-20260907：AM-TNC deleted-runtime 风险收紧为 fail-closed

## 结论

4090A 的错误清理确实删除了原完整 `unsb_cov` 环境；PID 3446758 仍由 deleted
inode 执行。这个事实不能通过覆盖旧路径或主动重启来“修好”。本次保持 supervisor
3446757 和 trainer 3446758 原样连续运行，实时进度已到 e58 / 496,074 steps，完成态和
checkpoint 没有损失。

旧路径当前只有三个只读 relay，它们不是重建环境，也不是训练恢复的可信根。可信恢复根是
独立目录
`/home/yc/recovery_snapshots/unsb_cov_runtime_candidate_exact_3446758_20260906T090000`。
活动 deleted inode 与隔离解释器 SHA256 相同；从活动进程映射捕获的 111 个依赖重新逐项
校验后，缺失、哈希不符和可写文件均为零。隔离解释器还通过了 RTX 4090 CUDA 分配探针。

## 当前 checkpoint 恢复证明

e58 最新完整状态被复制到隔离输出后，在隐藏 CUDA、`--gpu -1` 条件下从 496,074 精确
推进到 496,075。完整网络、优化器、调度器、AM-TNC 方法状态、Python/NumPy/CPU/CUDA
RNG 和两个 sampler 均由正常训练入口恢复。主 lane checkpoint 在探针前后保持
`a708af3c...0d37`，未使用主训练 GPU。

第一次把 CUDA 隐藏却仍传入 `--gpu 0` 的调用按预期在 checkpoint 加载前拒绝，随后修正为
CPU 参数；这次控制参数错误没有修改主 lane，也没有重启训练。

## 新的自动接管边界

恢复 guard 已升级到 commit `c4eed76e1e55e02e7508a5c57722430448307ae6`。新 guard PID
439182 和健康监视器 PID 439447 均已脱离 SSH、PPID=1、健康且零重启。旧 guard 及其健康
监视器只在新代码通过一次性 preflight 后退役，训练 PID 未变化。

以后只有在原 supervisor 和 trainer 同时消失时才考虑接管；接管前必须同时通过代码、Git、
协议、lane authorization、最新 checkpoint、隔离 Python、manifest 以及 111 个运行时依赖的
完整哈希门。任何一项不符都会 fail-closed，不会沿已删除环境或猜测环境启动。

因此，本次清理造成了环境损坏和“当前 epoch 以内”的剩余运行风险，但没有造成已完成计算
损失，也没有阻断后续 e200。最安全策略仍是保持当前 deleted-inode 训练不动，由隔离运行时
只承担真正故障后的 full-state 恢复。

部署后在关闭 SSH 前再次观察到原训练已连续推进到 e59 / 504,627 steps；新 guard 与健康
监视器均保持 PPID=1、零重启、健康，进一步确认这次控制层替换没有扰动主轨迹。

完整回执：
`evidence/paper_aio/PAPER_AIO_AMTNC_E58_DELETED_RUNTIME_FAIL_CLOSED_RECOVERY_20260907T184945.json`。
