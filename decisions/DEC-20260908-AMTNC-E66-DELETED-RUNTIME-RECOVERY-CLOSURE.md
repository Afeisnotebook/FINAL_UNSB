# DEC-20260908：4090A 错误清理后的 AM-TNC e66 恢复闭环

## 裁决

错误清理造成了真实影响：原 `unsb_cov` Conda 环境已不存在，运行中的 AM-TNC trainer
PID 3446758 仍持有 deleted inode。不能把旧路径下三个 326 字节的 Python 文件描述成完整
环境，也不能为了消除 `(deleted)` 标记而主动重启进程。

但这次事故目前**没有造成训练进度、checkpoint 或科学状态损失，也没有阻断后续训练、导出
或统一评估**。保持原 PID 连续运行仍是风险最低的选择。

## 本次现场闭环

- 原 supervisor/trainer PID 3446757/3446758 持续健康，已自然推进至 e66 / 564,498 updates；
  外层 guard 439182 与 health 439447 均为零重启、零告警。
- 重新读取 4.20 GB 的 111 个活动映射依赖并逐项计算 SHA256：0 缺失、0 不匹配、0 可写。
  隔离运行时继续保持只读，Python SHA 与活动 deleted inode 相同。
- 从旧路径 Python relay 新建解释器，实际落到隔离运行时，Python/Torch/NumPy/Pillow 版本均
  与恢复合同一致。因此“按原路径一定失败”不再成立；成立的是“原完整环境不可用，只有受控
  relay 可用”。relay 不能被覆盖或扩建。
- 将当时最新的 e65 full state 复制到独立目录，在隐藏 GPU 的 CPU 分支真实执行一个 update，
  从 555,945 到 555,946，stderr 为零；主轨迹未被修改。第一次调用误选宿主不允许的 CPU
  28–31，在 Python 启动前由 `taskset` 拒绝；改用允许的 24–27 后通过，不构成恢复失败。
- 原训练随后正常完成 e66。e66 checkpoint 与 sidecar 哈希闭环，并额外复制到本机只读目录
  `E:/UNSB_Expl/recovery_backups/4090A_AMTNC_20260908_E66`；本地与远端 SHA256 完全相同。
  隔离运行时 5.49 GB 归档也重新验证了本地/远端相同哈希且均只读。
- e66 full state 已由隔离 Python 在 CPU 上加载，确认网络、优化器、scheduler、Python/NumPy/
  CPU/CUDA RNG、两个 sampler 与方法状态都在。e200 exporter、恢复监督器以及使用独立只读
  evaluator runtime 的四条统一评估 waiter 仍在线。

## 可避免的损失与剩余限制

如果活动进程之后异常退出，guard 会在代码、协议、checkpoint、解释器和 111 项依赖全部重新
验签后，从最近完整 epoch 恢复；任一不一致都会 fail closed。当前最多损失正在计算、尚未原子
保存的一个 data epoch。即使4090主机磁盘再遭误操作，本机已有 e66 完整状态和精确运行时副本。

仍不能声称“恢复后的下一次 GPU update 与存活进程逐位相同”，因为没有中断健康进程做这种
反事实测试。我们严格声称的是：运行时身份匹配、完整状态可加载、真实一更新可执行、后继链
不依赖已删除的完整环境。旧环境禁止原地覆盖，AM-TNC 继续由原 PID 训练到 e200。

机器可读回执：
`evidence/paper_aio/PAPER_AIO_AMTNC_E66_DELETED_RUNTIME_RECOVERY_CLOSURE_20260908T015200.json`。
