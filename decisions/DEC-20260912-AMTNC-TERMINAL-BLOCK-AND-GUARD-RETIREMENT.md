# AM-TNC 终端阻塞与旧恢复 guard 退休裁决

第三个冻结 trainer PID `3698062` 于18:16:46第三次在同一e178→e179转换触发完全相同的非有限几何。原监督器按三次上限正确进入 `BLOCKED_AFTER_REPEATED_ENGINEERING_FAILURE`，source-bound successor也记录 `BLOCKED_SUCCESSOR_SUPERVISOR_EXIT`。当前AM-TNC实现因此终端记为 `blocked_current_implementation`，仍不等于机制证伪。

旧外层恢复guard PID `439182` 未理解新的科学终端裁决，在GPU空闲检查后的TOCTOU窗口重新启动 supervisor/trainer `3716027/3716028`。这与只读localizer首个执行 `3716115/3716116` 同时发生。发现共驻后先终止诊断child；随后退休旧guard、health和未取得任何完整epoch的第4次训练链，防止它把已冻结的三次失败预算扩展为新一轮三次甚至更多重启。

所有相关PID退出后，活动与隔离e178 checkpoint SHA仍完全相同，没有丢失完成epoch，也没有触碰其他宿主或算法。首次localizer运行只有约92秒、没有形成定位receipt，明确不可作为结果；保留其失败状态作为工程证据。

V2合同增加“旧guard及所有AM-TNC恢复PID必须缺席”的启动门，使用新的独立输出目录。只有重新验签、GPU无未知进程并取得共享锁后，才允许从只读e178副本定位故障。该诊断不修复算法、不恢复主训练、不读取paired性能。
