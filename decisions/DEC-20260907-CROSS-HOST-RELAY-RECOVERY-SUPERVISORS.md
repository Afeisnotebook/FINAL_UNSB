# DEC-20260907：跨宿主e200导出relay增加持久恢复层

## 裁决

四条从5090A、5090B和5090C汇入4090A统一评估目录的source-bound relay此前都健康，
但只有告警监视，没有自动恢复。任何relay在数日等待中意外退出，训练本身仍会继续，却可能
让e200导出、统一评估和最终论文交付链静默停滞。

保留全部原relay和健康训练，不重启、不迁移checkpoint；在4090A新增四个独立的
hash-pinned recovery supervisor。它们分别收养ST-CGR、CUT/CycleGAN、matched plain和
Proposal relay，当前均为`MONITORING_EXISTING_RELAY`，restart count为0。退出SSH后，四个
supervisor和原relay的PPID均为1，且系统中仍恰好只有四个匹配的relay进程。

## 恢复边界

- 每个supervisor冻结控制commit、恢复脚本SHA、原relay contract SHA及原relay脚本SHA。
- 只有relay状态尚未终止且匹配进程确实不存在时才可按冻结命令恢复。
- 活PID命令不匹配或出现重复进程时fail closed；完成、失败或超时状态不会被盲目重启。
- 密码只通过对应进程环境继承，不进入contract、state、日志或命令行。
- 新健康监视器PID 401539检查四个supervisor的PID、状态新鲜度和真实磁盘余量；首次状态
  `HEALTHY`、零告警，536.52 GiB余量覆盖160 GiB剩余写入估计和20 GiB headroom。

这次变更只加固交付控制面，不改变任何训练、算法、数据顺序、runtime cohort或matched
关系。没有读取paired性能值，没有打开confirmation20。完整回执见
`evidence/paper_aio/PAPER_AIO_CROSS_HOST_RELAY_RECOVERY_SUPERVISORS_20260907T172500.json`。
