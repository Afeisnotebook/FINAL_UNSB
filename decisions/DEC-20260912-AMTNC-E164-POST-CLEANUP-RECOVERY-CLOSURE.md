# DEC-20260912：4090A 清理事故后的 AM-TNC e164 恢复闭环

## 裁决

用户指出的风险事实仍然成立：原完整 `unsb_cov` 环境已经不存在，活动 AM-TNC
supervisor/trainer 仍使用 deleted inode，不能覆盖旧路径或为消除该标记而重启健康进程。
但这不再等于“当前没有后继恢复能力”。训练恢复权威已经与旧路径分离，固定在只读隔离
runtime；持久 guard 的冻结命令直接调用该解释器，旧 Conda prefix 只保留兼容 relay。

## 本次实际处理

- 只读确认原 PID 3446757/3446758 未变化，训练自然完成 e164 / 1,402,692 updates；
  checkpoint、sidecar 与 heartbeat 一致，GPU 正常计算。
- 重新逐项验签隔离 runtime 清单，111/111 个文件通过 SHA256；隔离 Python 版本和关键
  依赖仍为 Python 3.10.20、Torch 2.5.1+cu121、NumPy 2.2.6、Pillow 12.2.0。
- 由隔离 Python 在 CPU 上重新加载 e164 full-state，确认模型、方法状态、metadata、RNG、
  A/B sampler、step 与 data epoch 均可读取。e154 已完成的一次真实 update 恢复探针继续作为
  执行级证明；本次没有为重复证明而打扰主 GPU。
- 外层 guard PID 439182 与健康监视器 PID 439447 均健康、零重启。guard 合同直接绑定隔离
  Python，并规定每次真实恢复前重新验签 runtime 和最新 full-state，任何差异均 fail closed。
- 将 e164 full-state、sidecar、heartbeat 和 lane authorization 复制到本机异盘目录
  `E:/UNSB_Expl/recovery_backups/4090A_AMTNC_20260912_E164`，传输前后源状态未变化，逐文件
  哈希一致且本地文件设为只读。此前 5.49 GB runtime 异机归档继续复用。

## 损失边界

清理没有损失任何已完成 epoch、checkpoint、优化器、scheduler、方法、RNG 或 sampler 状态。
如果 deleted-inode 进程故障，guard 可从最新完整 epoch 经隔离 runtime 恢复，通常最坏重做
当前未完成 epoch；如果整台 4090A 的本地存储同时丢失，现有异机灾备至少可恢复到 e164。

仍不能在不中断健康进程的情况下证明隔离 runtime 的下一次 CUDA update 与 deleted-inode
进程逐位相同，因此不作该项声明，也不把隔离恢复误写成新的 runtime cohort。本次没有读取
性能值、改变算法或协议，也没有打开 confirmation20。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_AMTNC_E164_POST_CLEANUP_RECOVERY_CLOSURE_20260912T030445.json`。
