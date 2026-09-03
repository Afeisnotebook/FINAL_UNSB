# ST-CGR e13：精确恢复事件正式闭环

日期：2026-09-03

5090A ST-CGR此前在e12后出现live-process停滞，经同一supervisor从完整e12状态精确
恢复。现在同一trainer PID 431032已完成e13（111,189 updates），完整checkpoint的
实算SHA256与sidecar一致，sidecar与heartbeat的scientific-state hash也一致。

e13用时4,974.04秒，与恢复前正常epoch范围一致。随后supervisor保持`CHILD_RUNNING`，
trainer进入e14；progress sentinel更新到step 111189、心跳年龄约133秒，30秒I/O与
CPU继续增加且零告警。因此本事件从“精确恢复已启动”升级为“跨完整data epoch证明
恢复有效”，不再属于未决工程风险。

5090A的plain trainer、resume successor仍不存在，ST-CGR继续独占该卡。此次闭环未
读取任何性能结果、未改训练配置、未打开confirmation20，也不把绝对轨迹写成matched
gain。
