# 5090B matched plain 启动后的交付链复核

时间：2026-09-06 08:25:24 +08:00

## 裁决

保留当前训练、source-bound exporter、v3 import relay 与动态统一评估链，不重启或改写
任何健康进程。Git 中 matched-plain 局部状态仍指向已退役 v2 relay 的两个 PID，现仅把
控制面引用修正到真实 v3 relay PID `3414652` 和覆盖它的 active-frontier health watcher
PID `3460191`。

## 依据

- formal plain successor/supervisor/trainer/exporter PID `523993/534983/534984/534982`
  全部存活。
- exporter 已绑定 lane、训练 commit、协议 fingerprint 与720小时等待边界，状态为等待
  e200，不存在“训练完成但无人导出”的尾部断点。
- v3 relay 状态每分钟刷新，合同 SHA256
  `5a4ea22cfbd55d4da6b799e0025e209122fe4505605fd77d8775c3e466d91005`；总健康监控为
  `HEALTHY`、零告警，并显式覆盖该 relay。
- 统一评估、ST-CGR matched evaluation 与最终组合三个 successor 均存活，继续等待合法
  fixed imports 和 GPU 释放。

## 边界

这是陈旧控制面引用修正，不是运行时迁移或科学变更；不读取性能，不触碰 checkpoint，
不使用 paired 控制，不打开 confirmation20。

证据：
`evidence/paper_aio/PAPER_AIO_5090B_MATCHED_PLAIN_DELIVERY_CHAIN_REVALIDATION_20260906T082524.json`
