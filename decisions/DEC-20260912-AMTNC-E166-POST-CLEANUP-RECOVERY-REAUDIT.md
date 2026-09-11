# DEC-20260912：4090A 误清理后的 AM-TNC e166 恢复再审

## 裁决

错误清理造成的事实没有被掩盖：原完整 `unsb_cov` Conda 环境已经丢失，活动
supervisor/trainer PID `3446757/3446758` 仍持有 deleted inode。不能为消除该标签而重启
健康训练，也不能把旧 prefix 称为原环境恢复。

但截至本次再审，错误清理不会阻断 AM-TNC 继续训练、full-state 恢复、e200 source-bound
导出或统一评估。训练已经在原 PID 上自然完成 e166（1,419,798 updates），无 NaN、OOM、
保存失败或 guard 告警；本次没有向训练发送信号、迁移 checkpoint 或修改协议。

## 已执行的补救与核验

- 重新验签只读隔离训练 runtime：从活进程映射保全的 111/111 项均逐字节匹配；Python、
  Torch、NumPy、Pillow分别为3.10.20、2.5.1+cu121、2.2.6、12.2.0。
- 旧路径的三个 Python 入口只是固定 relay；直接入口和旧路径入口均落到同一个隔离 runtime。
  真正恢复命令由 guard 合同直接绑定隔离 Python，不依赖旧 prefix。
- 使用隔离 Python 在 CPU 上加载最新 e166 full-state，确认network、optimizer、scheduler、
  AM-TNC方法状态、Python/NumPy/CPU/CUDA RNG、primary/secondary sampler、step和epoch齐全；
  加载前后 checkpoint SHA256 保持不变。e154 的真实一步恢复探针仍提供执行级证明。
- 外层 guard PID `439182`、health PID `439447` 健康且零重启；真实恢复前会重新验签 runtime
  和最新full-state，不匹配时fail closed。
- 把e166 full-state、sidecar、heartbeat与lane authorization复制到本机异盘只读目录
  `E:/UNSB_Expl/recovery_backups/4090A_AMTNC_20260912_E166`，复制前后远端源哈希不变，逐文件
  与本地一致。5.49 GB隔离runtime归档仍在异机只读保存。
- e200 terminal exporter、增量exporter及统一评估均已有独立、哈希固定的 evaluator runtime
  和恢复监督器；旧日志中的一次 `latin1` 失败已经被恢复监督器接管，当前增量集已闭合e100/e150，
  不是持续故障。

## 当前损失上界

没有丢失任何已完成epoch、checkpoint或训练状态。若只有deleted-inode进程退出，通常最多
重做当前未完成epoch；若4090A整机或本地盘同时丢失，至少可从异机e166恢复。仍不能在不中断
活进程的情况下证明隔离runtime的下一次CUDA update与原deleted-inode进程逐位一致，因此不
作该项声明，并在论文工程记录中保留该边界。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_AMTNC_E166_POST_CLEANUP_RECOVERY_REAUDIT_20260912T045420.json`。
