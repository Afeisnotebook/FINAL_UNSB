# DEC-20260912：4090A 清理后重新验证 e200 评估与交付命令

## 结论

4090A 原 `unsb_cov` 环境被误删没有阻断 e200 后的统一评估和论文交付链。AM-TNC、统一评估、
ST-CGR 裁决、最终交付和 DCLGAN 评估的恢复 supervisor 均存活、restart count 为 0；组合 health
分别监视 8 项和 2 项状态，当前全部健康、零告警。

## 执行级复核

所有冻结 child command 都明确使用隔离 evaluator Python：
`/home/yc/recovery_snapshots/unsb_cov_unified_evaluator_candidate_20260906T094000/bin/python`，
SHA256 为 `40dce592...69b0fd`。在各自真实 `cwd` 中逐一导入 torch、NumPy、Pillow 和
`research.paper_aio.unified` 均通过，checkout commit 分别匹配 `0492097`、`2f57f22` 与
`82ca28a`；ST-CGR authority、统一 import root 和 DCLGAN manifest 均存在。

在用户主目录裸运行时 `research` 不在 `sys.path`，导入会失败；这不是 runtime 损坏，因为冻结
命令从不以该目录为 `cwd`。本次未加载 checkpoint、未读取性能值、未申请 GPU 锁，也未改变
任何训练进程。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_4090A_POST_CLEANUP_EVALUATOR_COMMAND_REVALIDATION_20260912T064213.json`。
