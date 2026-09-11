# DEC-20260911：DCLGAN source exporter 持久恢复

## 决策

保留本地 GTX1660 上连续运行的 DCLGAN 训练，不重启、不迁移、不修改协议；退役已经死亡的
source-bound exporter PID 23168 及只会持续报警的旧 health watcher PID 22088。使用提交
`3888436cb7884c4acf35336a5d2425ff7c247bbe` 的干净独立 worktree，把同一冻结 source commit、
adapter fingerprint 和 export destination 重新置于可恢复的 supervisor 下。

## 事实边界

- 失败组件仅是等待 e200 的 exporter。训练 wrapper/supervisor/trainer PID
  20832/12824/22020 始终存活，未损失已完成 epoch 或 checkpoint。
- 新 supervisor PID 14596 管理 exporter child PID 20760；首次启动计数为 1，当前状态为
  `WAITING_FOR_COMPLETE_E200`。health watcher PID 17572 同时核查训练、export supervisor、GPU
  锁和磁盘余量，当前零告警。
- child command、control commit、source commit、adapter fingerprint 和 supervisor contract
  均已哈希绑定；e200 完成后的 publish-last push/evaluation 链未改变。
- 旧进程退出的精确异常文本不可恢复，因此只把该事件裁决为工程等待器故障，不推断科学原因。

## 科学约束

本恢复未读取性能值，未使用 paired 指标，未改变数据顺序、训练更新、runtime cohort 或评估协议，
也未打开 confirmation20。DCLGAN 仍是非阻塞外部基线；本事件不是算法失败。

证据：
`evidence/paper_aio/PAPER_AIO_DCLGAN_EXPORTER_DURABLE_RECOVERY_20260911T185642.json`
