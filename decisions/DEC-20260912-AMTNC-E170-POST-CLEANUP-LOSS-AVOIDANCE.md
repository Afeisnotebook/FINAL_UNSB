# DEC-20260912：AM-TNC e170 清理影响与损失规避裁决

## 结论

4090A 的错误清理造成了真实但目前可控的工程损坏：原始
`/home/yc/unsb_cov` 与完整 Conda 环境仍不存在，活动 AM-TNC trainer PID
`3446758` 仍运行在 deleted inode 上。因此不得重启健康进程，也不得在旧路径盲目重建并把它
冒充原运行时。

截至 e170（1,454,010 updates），没有发现已完成训练状态或论文交付链损失。活动
supervisor/trainer PID `3446757/3446758` 连续健康，guard/health PID `439182/439447`
零重启；e200 export recovery PID `706411` 继续等待终点。GPU、磁盘、checkpoint 保存均正常。

## 已完成的非破坏性补强

- 重新验证活进程与隔离恢复 Python 的 SHA256 完全相同：
  `66fccc990e49c33a1f06249128136ef063f16883ef06550960ce79943c5d13ab`。
- 低优先级逐文件验签隔离 runtime：111/111 项、4,198,514,872 bytes 全部匹配。
- 用隔离 Python 在 CPU 上完整加载最新 e170 full-state，并确认 G/F/D/E、四个 optimizer、
  四个 scheduler、AM-TNC 状态、全部 RNG、双 sampler、step 与 epoch 都在；加载前后 checkpoint
  SHA256 均为 `d11ca70d550897eb4a1b7d1b93f58f2e0d0dc76940f326561df544682575c24e`。
- 将 e170 full-state、sidecar、heartbeat、lane authorization、恢复回执和 guard 状态复制到
  `E:/UNSB_Expl/recovery_backups/4090A_AMTNC_20260912_E170`，逐文件与远端源重验 SHA256 后设为只读。
- 保留既有 e175 永久 milestone 异机备份动作；没有向任何训练进程发送信号，也没有修改算法、
  数据顺序或训练协议。
- 两小时 Goal heartbeat 已原位更新到 e170 回退点和提交 `c5815af`，保持仅失败通知；未写入凭据。

## 影响边界

如果只有活动进程退出，guard 会在重新验签隔离 runtime 和最新 full-state 后，从最近完整 epoch
精确恢复；最多损失尚未原子保存的当前 epoch。若 4090A 整机及存储同时丢失，本机异盘可从
e170 恢复。因此本次清理不会迫使从头重训，也不阻断 e200 导出和统一评估。

唯一仍无法在不中断健康训练的前提下证明的是：恢复后的第一个 CUDA update 与当前
deleted-inode 进程逐位相同。若真实恢复发生，论文 provenance 必须如实披露这一边界，不能把
它写成同一进程的逐位连续。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_AMTNC_E170_POST_CLEANUP_LOSS_AVOIDANCE_20260912T081011.json`。
