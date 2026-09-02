# 5090A：ST-CGR时间优先重排

日期：2026-09-03

用户明确授权在紧迫工期下暂停5090A正在运行的matched plain，立即推进
ST-CGR full-data e200，并要求不影响其余健康任务。本决策只改变5090A的执行顺序，
不改变算法、数据、seed、batch、更新量、优化器、评估协议或其他宿主的运行。

切换前，5090A plain已完成e9（76,977 updates）。其
`full_state_latest.pt` SHA256为
`9758471e2ad9a90810196fc1b66206e23b4d77eb94bf5a0cf9c63879c26362b0`，
切换后复核哈希相同。plain的触发器、监督器和训练子进程已停止，自动重启权被
撤销；e200 exporter保留等待，以便未来从e9精确恢复后继续原有交付链。

ST-CGR已有e1（8,553 updates）完整状态。本次以冻结candidate commit
`656670cdb828620cb500f7a74fd3afd4df7e375d`和protocol fingerprint
`2fbdd6f58971657134c305224ab14ae0e0ba53cf421c64f9935c68a4c0873e20`
从e1精确恢复。当前continuation、exporter、supervisor和trainer均由持久进程维护，
并由新的metric-blind health watcher监控。切换过程没有读取性能值，未打开
confirmation20，也没有修改正在运行的4090A、5090B或5090C任务。

该时间优先决策有明确科学代价：在5090A plain完成e200前，ST-CGR只能产生合法的
绝对长程轨迹，不能形成严格matched delta；5090C Proposal虽然已通过与5090A的
runtime-twin关系，也同样要等待该plain恢复并完成后才能形成最终matched delta。
因此仓库不得把ST-CGR绝对轨迹写成相对plain收益、机制成功或论文终局。plain的
暂停是`paused_by_explicit_user_time_priority`，不是失败、淘汰或机制证伪。

按ST-CGR e1实测9,609.63秒/epoch做保守线性外推，直接运行消除了约五天的plain
前置等待，e200暂估在2026-09-25附近完成；该ETA只用于容量规划，将在e2/e5
heartbeat后按无指标吞吐重估，绝不用于依据paired指标停止或修改训练。

