# DEC-20260906：4090A deleted runtime复核与外层恢复保护

## 结论

错误清理确实删除了AM-TNC活动进程原先使用的完整`unsb_cov`环境，PID 3446758因此仍显示
`(deleted)`；这不能也不应通过重启健康训练来“修好”。但“原路径仍无Python、未来一定无法
启动”已经不是当前事实：三个原Python入口现为只读relay，并已实际启动隔离运行时完成导入。
活动trainer和supervisor保持原PID，e29完整状态已原子保存并验签。

## 本次核验与加固

- 对活动进程映射清单中的111个二进制和动态库重新逐项计算SHA256，隔离运行时仍为
  `111/111`一致；不是只依赖旧回执。
- 隔离解释器通过Python、NumPy、Pillow、PyTorch导入，并在4090上完成CUDA张量往返。
- 从当前e29、248037-step完整状态复制出独立CPU探针，成功恢复并执行到248038步；探针只
  写入`/home/yc/recovery_snapshots/AMTNC_RESUME_PROBE_E29_20260906T134800`，未修改主lane。
- 恢复运行时、活动映射捕获和relay源已移除写权限，降低再次误清理或误安装覆盖的风险。
- 新增外层恢复guard PID 3966089。它当前只收养并监视原supervisor/trainer；只有二者都
  消失、冻结源码/协议仍一致且最新checkpoint文件哈希通过时，才从隔离解释器重启原
  supervisor命令。重复进程、无进展重启或身份漂移均fail closed。独立健康监视器
  PID 3968393已脱离交互会话持续检查guard状态，目前零告警。

## 损失边界

截至复核，没有数据、checkpoint或已经完成的训练进度丢失。若活动进程此刻故障，可从e29
完整状态恢复，最多丢失正在进行但尚未落盘的一个data epoch；健康进程继续运行比主动重启
更安全。隔离恢复不建立新的runtime cohort，也不声称与仍在运行的deleted-inode进程做GPU
下一步逐位等价。统一评估继续使用此前单独冻结的评估运行时，不能误用训练relay。

本次没有读取性能指标，没有打开confirmation20，没有改变算法、训练协议、数据顺序或
matched关系。详细机器回执见
`evidence/paper_aio/PAPER_AIO_4090A_DELETED_RUNTIME_FRESH_AUDIT_AND_OUTER_GUARD_20260906T135200.json`。
