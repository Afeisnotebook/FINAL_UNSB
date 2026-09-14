# DEC-20260914：5090B克隆端点再次不可达，恢复本地交付链但不猜测远端训练状态

## 当前裁决

`connect.weste.seetacloud.com:43172` 在2026-09-14 11:49再次拒绝TCP连接。由于不能读取远端PID、heartbeat或最新full-state，本次只写作“provider endpoint unavailable / remote state unknown”，不能写作训练失败，也不能声称checkpoint丢失。Git中最后严格封存的是e134；异机增量导入最后为e100。

用户提供的`44804`此时同样不可达，因此不能把它当作替代算力或直接迁移matched plain。

## 已执行恢复

本地端点watcher和两条5090B克隆交付relay的旧进程也已消失，原因尚未证明。按照冻结contract重新启动其恢复supervisor、动态child和独立health watcher。三条新链均为`HEALTHY`，只等待网络或合法source-bound export。4090A上的克隆relay、恢复supervisor和统一评估等待器始终存活，无需重启。

其他科学训练未触碰：Proposal为e199、ST-CGR为e184、本地DCLGAN为e187，监督器/训练器继续运行。

## 下一步

需在租赁平台恢复43172实例或SSH映射。端点恢复后先只读读取GPU UUID、远端进程和最新完整checkpoint；若冻结训练supervisor/trainer仍在则不干预，若确实不存在才从最新完整full-state精确恢复。不得启动替代lane、不得从e134之外猜测恢复点、不得使用任何中间paired指标做调度。
