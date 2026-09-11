# DEC-20260911：5090B matched plain e66 精确恢复闭环与告警传播修复

## 裁决

5090B matched plain 已从经验证的 e64 full-state 连续完成 e65、e66，共前进 17,106 updates。
e66 checkpoint 可在 CPU 完整加载，step、physical epoch、target steps、sidecar 和 heartbeat 相互
一致，因此 e64 停滞恢复正式升级为 `EXACT_RESUME_CLOSURE`。新 trainer PID 76358、原 supervisor
PID 534983、successor PID 523993 和 exporter PID 534982 均健康；没有丢失任何已完成 epoch，
runtime cohort 与训练协议未改变。

## 监控修复

根因不是 progress watcher 没报警，而是通用 health watcher 只传播 `BLOCKED/FAIL/FATAL`，没有
传播 `ALERT_*`。提交 `34c39698a365245c23b37fd86946e6f80c14ade` 将 `ALERT` 加入上游故障
前缀，并增加回归测试（12项通过）。5090B 已部署只读哈希一致源码，新的 watcher PID 83714
同时监控 successor、progress watcher 和 lane heartbeat；状态为 HEALTHY、零告警。旧 watcher
PID 667507 在替代项健康后退役，训练未被触碰。

## 时间边界

按 e66 的 2360 秒/epoch 估计，e200 约在 9 月 15 日 14:15 完成。此前由静默停滞产生的时间损失
无法挽回，仍建议把 5090B 至少保留到 9 月 16 日 12:00，以覆盖 e200 export 和异常余量。

本次不读取性能值、不使用 paired 控制、不改变模型或数据协议，confirmation20 继续封存。

证据：
`evidence/paper_aio/PAPER_AIO_5090B_MATCHED_PLAIN_E66_EXACT_RESUME_AND_ALERT_HEALTH_CLOSURE_20260911T224500.json`
