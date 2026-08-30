# DEC-20260830：4090终局消融按单流顺序执行

## 决定

最终赢家的 `proposal-only` 与 `observable-only` 两条真实 e200 消融在4090上按固定顺序
单流执行。`projected/full` 使用已经完成并绑定终点receipt的赢家轨迹，不重复训练。

该决定只改变任务调度，不改变算法、batch、数据、seed、初始化、训练更新或评估协议。
两条新消融仍为batch1、seed2026、共同e0、30000 updates/200 data epochs，且不得使用
中间paired指标控制训练。

## 依据

4090上的真实候选chunk显示：单流每5 data epochs约315--330秒；两条相同规模训练流
并发时每条约720--750秒。完成两条必需轨迹时，顺序单流的总墙钟短于并行双流，且
减少评估重叠与GPU争用。显存仍有空余不构成并行吞吐证据。

因此合同固定：

- `e200_execution_policy=SEQUENTIAL_SINGLE_STREAM_BY_MEASURED_WALL_CLOCK`；
- `maximum_parallel_e200_executors=1`；
- 第一条完整退出并写出终点状态后，第二条才允许启动；
- 任一条失败时fail closed，不以另一条结果完成最终交付。

GPU micro gate仍可独立执行；它不产生长期科学裁决，也不改变上述e200单流要求。

## 防漂移边界

这不是缩减消融。最终交付仍必须同时拥有proposal-only、observable-only和原赢家
projected/full三个完整e200终点receipt。该调度也不授权seed2027/2028、增大batch、
最佳checkpoint选择、路线二、退出窗口或confirmation20访问。
