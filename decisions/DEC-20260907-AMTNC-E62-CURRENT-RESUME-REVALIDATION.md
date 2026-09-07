# DEC-20260907：4090A 清理影响与 AM-TNC e62 恢复复验

## 裁决

错误清理确实破坏了原完整 `unsb_cov` 环境。运行中 AM-TNC 的 PID 3446758 仍从
deleted inode 执行；旧路径下的三个 `python*` 文件只是只读中继脚本，并不代表原环境
已被重建。这个损坏不能通过覆盖旧目录或主动重启来消除。

不过，已完成的训练进度没有丢失，后续 e200 也没有被阻断。现场复查时原 supervisor
3446757 和 trainer 3446758 保持连续，AM-TNC 已完成 e62 / 530,286 updates。e60 固定
里程碑及 e62 最新 full-state 均存在，当前 GPU 正常工作。

## 最新恢复证明

可信恢复根保持为独立只读目录
`/home/yc/recovery_snapshots/unsb_cov_runtime_candidate_exact_3446758_20260906T090000`，
而不是旧 `unsb_cov` 路径。隔离 Python 与活动 deleted inode 的 SHA256 相同；现场重新
检查 111 个由活动进程捕获的关键依赖，缺失、哈希不符和可写项均为零。该运行时可导入
当前 PyTorch/CUDA 栈并识别 RTX 4090。

本次又从当前 e62 full-state 创建了完全隔离、隐藏 CUDA 的 CPU 分支。它从 530,286
精确恢复并执行到 530,287，stderr 为零；主 lane checkpoint 在探针前后保持
`5c85b8fd...0756e`，没有占用主训练 GPU，也没有触碰主训练进程。

fail-closed guard PID 439182 与健康监视器 PID 439447 均保持 PPID=1、健康、零重启。
guard 看到的当前 step 和 checkpoint hash 与主 lane 一致；只有原 supervisor/trainer
消失并且代码、协议、latest full-state、隔离解释器、manifest及111项依赖全部重验通过
时才允许恢复。任一不符都会拒绝启动。

## 影响边界

- 没有已完成 epoch、checkpoint、数据或科学状态损失；当前连续训练不重启。
- 如果活动 deleted-inode 进程意外死亡，最坏损失为尚未完成的一个 data epoch；最近完整
  epoch 可以用隔离运行时恢复。
- 无法也不声称新的进程与仍在内存中的旧进程“下一 GPU step 位级一致”；可证明的是完整
  状态可加载、同一训练入口可继续推进、关键二进制身份逐项匹配。
- e200 后的 source-bound export 和统一评估有独立恢复监督器，不依赖把旧环境伪装成完整
  Conda 环境。

因此最安全的处理不是重建或覆盖旧路径，而是保持健康训练连续，把隔离运行时和
fail-closed guard 作为唯一恢复入口。完整回执见
`evidence/paper_aio/PAPER_AIO_AMTNC_E62_CURRENT_RESUME_REVALIDATION_20260907T223230.json`。
