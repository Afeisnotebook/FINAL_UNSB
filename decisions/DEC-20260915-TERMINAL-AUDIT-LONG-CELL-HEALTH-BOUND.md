# Terminal audit 长单元监控边界修正

恢复后的terminal audit没有停滞：Proposal三个固定target-blind单元已完成，AM-TNC e100随后完成并进入e150。GPU活动和审计状态都证明科学任务持续推进。

旧health watcher `2988`将审计child状态阈值设为600秒，短于单个固定审计单元，因此产生`ALERT_STATE_STALE`误报。仅停止该监控器并以同一冻结health实现启动`13656`；audit child阈值改为7200秒，supervisor和pathology阈值仍为600秒。审计进程`3912`、supervisor `20436`及其任何科学状态均未重启或修改。

这项修改只修复监控语义，不改变算法、数据、审计单元或性能边界。paired性能未读取，confirmation20继续封存。

权威证据：`evidence/paper_aio/PAPER_AIO_TERMINAL_AUDIT_LONG_CELL_HEALTH_BOUND_20260915T192342.json`。
