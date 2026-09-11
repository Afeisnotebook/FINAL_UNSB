# DEC-20260912：DCLGAN exporter Windows 文件锁恢复

## 裁决

本地 DCLGAN 训练持续健康，故障只发生在等待 e200 的 source-bound exporter 监督器。
旧 supervisor PID 14596 因 Windows 对 child-state JSON 的一次短暂访问拒绝退出；exporter
child PID 20760 没有退出，仍处于 `WAITING_FOR_COMPLETE_E200`。这不是训练停滞、数据损坏或
算法失败。

## 修复与部署

- 在 `paper_aio_control_supervisor._read_json` 增加与原子写入相同性质的有界重试：短暂
  `PermissionError` 最多重试十次，持续拒绝仍然抛错并 fail closed。
- 新增短暂拒绝后恢复和持续拒绝后失败的测试；相关 control-supervisor/health 测试共
  38 项通过。修复提交为 `c092b9d9a6d469a129c1d7d0c5e8e3fc48335a0b`。
- 从该提交建立干净独立 control worktree，启动新 supervisor PID 19524。它根据冻结 child
  state 原位收养 PID 20760，`restart_count=0`，没有启动第二个 exporter。
- 新 health watcher PID 27044 连续核查训练、export supervisor、lane heartbeat、GPU 锁和
  磁盘，状态为 `HEALTHY`、零告警；确认健康后才退役绑定死亡 PID 的旧 watcher 17572。
- 下游 push supervisor/child/health PID 26020/24396/20500 仍健康等待 e200 export，
  export→push→4090A import→统一评估链没有变化。

## 科学边界

DCLGAN wrapper/supervisor/trainer PID 20832/12824/22020 未收到信号、未重启、未迁移，已自然
到达 e141 / 1,205,973 updates。修复只增强等待器对 Windows 短暂文件锁的容忍，不修改训练
协议、数据顺序、runtime cohort 或评估设置；没有读取性能值，没有使用 paired 指标，也没有
打开 confirmation20。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_DCLGAN_EXPORTER_WINDOWS_LOCK_RECOVERY_20260912T031536.json`。
