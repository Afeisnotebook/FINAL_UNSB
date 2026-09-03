# 5090B现有长训增加非侵入式连续性守护，并降低Codex轮询频率

## 决定

5090B的CUT和CycleGAN继续使用原始scientific checkout、原始supervisor和原始trainer。
在外层增加两个source-bound continuity guard，仅当同一lane的supervisor和trainer均消失时，
才从已验证的latest full-state调用冻结的原supervisor命令恢复。现有训练不因部署而重启。

部署完成后，将`final-unsb-goal` heartbeat从每小时一次降低为每两小时一次，并把正常唤醒
压缩成metric-blind健康巡检。工程故障、固定里程碑、successor切换、runtime关系待审和租期
风险仍触发完整审计。

## 证据

- control commit：`605ccc564751ad10c49957b3b2eeea3d41c3b1c0`；远端通过完整Git bundle建立干净checkout。
- CUT guard PID `159133`收养supervisor `3739`和trainer `5771`。
- CycleGAN guard PID `159134`收养supervisor `11845`和trainer `11846`。
- 两条guard均为`MONITORING_EXISTING_SUPERVISOR`，`total_restarts=0`，stderr为空，未产生恢复日志。
- health PID `159190`在第二轮轮询仍为`HEALTHY`且零告警；三个新控制进程均已归属PID 1。
- 部署期间CUT从e81推进到e82；CycleGAN保持正常epoch内计算并已完成e50。
- 既有matched-plain与DCLGAN successor/health/progress进程全部存活，顺序未变。
- 4090A plain、5090A ST-CGR、5090C Proposal和本地审计链同时完成轻量健康核验。

完整回执见
`evidence/paper_aio/PAPER_AIO_5090B_CONTINUITY_GUARDS_AND_BIHOURLY_GOAL_20260903T220753.json`。

## 边界

本决定只改变工程连续性和Codex轮询频率，不改变算法、数据顺序、训练协议、优化器、
checkpoint或后继顺序。guard不读取评估输出；paired指标、confirmation20和非等价runtime
delta仍不可用于控制。任何恢复都必须在同一宿主由完整checkpoint hash门验证，并保留回执。
