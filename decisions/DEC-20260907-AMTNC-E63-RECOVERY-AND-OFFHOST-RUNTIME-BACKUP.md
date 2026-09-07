# DEC-20260907：4090A AM-TNC e63 恢复闭环与异机运行时备份

## 裁决

4090A 的错误清理确实删除了原完整 `unsb_cov` 环境，活动 trainer PID 3446758 仍从
deleted inode 执行。旧路径下的 `python*` 是只读 relay，不应被误认为已恢复的 Conda
环境，也不得在健康训练存活时原地覆盖或主动重启。

该事故目前没有造成已完成训练进度损失，也不再阻塞后续恢复。原 supervisor 3446757、
trainer 3446758、外层恢复 guard 439182 均保持连续；AM-TNC 已推进到 e63 / 538,839
updates。当前 full-state 文件完成全哈希复验，guard 同时重新检查隔离 Python、manifest
及 111 个活动映射依赖，结果全部通过，仍为零重启、零告警。

## 为什么恢复能力成立

隔离运行时位于
`/home/yc/recovery_snapshots/unsb_cov_runtime_candidate_exact_3446758_20260906T090000`。
它的 Python SHA256 与活动 deleted inode 一致。此前 e62 当前状态已经在隐藏 CUDA 的隔离
分支中完成一次真实更新（530,286 到 530,287）；本次 e63 预检又对新的 latest full-state
执行了完整文件哈希，并逐项复验 111 个运行时依赖。guard 只有在原 supervisor 与 trainer
同时消失后才可能恢复，而且恢复前必须重新通过上述全部身份门；任一差异都会 fail closed。

为了避免同一台 4090 再次误清理导致运行时与其恢复副本同时丢失，本次把精确隔离运行时
以最低 I/O 优先级只读归档，并复制到本机独立于 Git 的目录。远端和本地 5,487,298,560
字节归档 SHA256 均为 `6396a1f2...b2a52`。因此运行时现在有跨宿主副本，而不是只依赖
4090A 单机目录。

## 损失边界

- 已完成到 e63 的 checkpoint、优化器、sampler、RNG 与方法状态均已保存并验签。
- 健康训练没有被中断、迁移或修改；备份和核验不读取 paired 指标。
- 若活动进程未来意外死亡，最多损失当前尚未完成的一个 data epoch，然后从最近完整
  full-state 进入受控恢复。
- 不声称隔离运行时恢复后的下一 GPU step 与仍存活的 deleted-inode 进程逐位一致；可证明
  的边界是代码/解释器/依赖身份一致、完整状态可加载并可继续执行更新。
- e200 source-bound export 与后续统一评估由独立恢复监督器保护，当前同样零重启、零告警。

因此正确动作仍是保持当前进程连续，把隔离运行时、fail-closed guard 和异机只读归档作为
唯一恢复链，不在原路径重建或覆盖环境。完整回执见
`evidence/paper_aio/PAPER_AIO_AMTNC_E63_RECOVERY_AND_OFFHOST_RUNTIME_BACKUP_20260907T232900.json`。
