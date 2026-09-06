# DEC-20260906：4090A训练恢复与统一评估运行时分离

## 裁决

AM-TNC 的隔离恢复环境只用于训练进程真实退出后的 full-state resume，不再被描述为
4090A 上所有后继任务的通用运行时。错误清理同时影响了数个长期等待的评估进程，但它们
尚未生成任何指标，也尚未加载 NumPy、Pillow 或 Torch 数值模块，因此当前没有论文结果
被污染，健康等待进程继续保留。

## 运行时边界

- AM-TNC trainer PID 3446758 使用原始解释器的 deleted inode；其精确依赖保全和恢复
  relay 已由上一裁决闭环。
- AM-TNC matched evaluator 与 DCLGAN evaluator 同样保留原解释器 inode；动态 first-wave、
  ST-CGR 和 final-delivery waiter 则来自错误清理时的另一解释器。所有这些进程当前都只是
  metric-blind 等待器，completed evaluations 为零。
- 若任一评估 waiter 消失，不允许机械调用 AM-TNC 训练 relay。必须先冻结一个专用统一
  评估运行时并验证 Python、NumPy、Pillow、Torch 及评估代码身份；若已经产生部分评估
  结果，则只能保持同一运行时继续，或者从只读 checkpoint 重建整个评估 cohort。

## 当前影响

所有健康训练保持原 PID，队列、算法和协议均未改变。最新只读刷新中，ST-CGR 已到 e57、
Proposal 到 e63、CycleGAN 到 e136、5090B matched plain 保持 e2 完整状态并继续 e3，
本地 DCLGAN 保持 e16 完整状态并继续训练。没有读取中间 paired 性能，没有打开
confirmation20。

本裁决防止后续把“训练可恢复”错误扩张为“任意评估进程可以用同一 relay 重启”。它不要求
当前重启任何 waiter，也不改变已经完成 checkpoint 的结论：已完成科学状态没有损失；若
AM-TNC 真实退出，最多损失当前未完成的一个 data epoch。
