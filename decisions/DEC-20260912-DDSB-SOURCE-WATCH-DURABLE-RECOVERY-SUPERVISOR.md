# DEC-20260912：DDSB 权威源码守望增加可恢复监督层

## 裁决

DDSB 仍为 `REPRODUCTION_INCOMPLETE`，不是科学负结果；没有权威作者实现时仍禁止猜测长训。
本次只加强既有公开来源守望的工程连续性，不改变来源、训练授权或论文实验队列。

## 已部署

- source watcher PID `22104` 保持原进程、原输出与 6 小时轮询，当前为
  `WAITING_FOR_AUTHORITATIVE_SOURCE`。
- recovery supervisor PID `26164` 绑定 control commit `1e90905`、source commit `40c4b37`、
  Python、脚本与冻结 contract；仅当 child 在 WAIT/TRANSIENT 状态真实死亡时恢复，否则
  fail closed。当前 restart count 为 0。
- combined health PID `30196` 同时监视 supervisor 与 source watcher，已连续三个轮询健康、
  零告警。
- 替代链稳定后，冗余 health PID `31124/16220` 经命令身份核对后退役；source watcher 和
  所有科学训练均未停止或重启。

自动来源接受和自动训练授权仍为 false，未读取 checkpoint、性能值或 confirmation20。

机器可读证据：
`evidence/paper_aio/DDSB_SOURCE_WATCH_DURABLE_RECOVERY_SUPERVISOR_20260912T062623.json`。
