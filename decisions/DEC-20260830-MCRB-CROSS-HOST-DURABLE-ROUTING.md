# DEC-20260830：MCRB跨宿主持久裁决链

## 裁决

5090上的`G1-03-STATE-FEEDBACK-MISSING`保持独立batch1、seed2026、共同e0和真实e200。
只有它形成完整e200 trajectory和source-bound terminal receipt，并通过注册的全部数值门，
才在4090从该宿主自己的共同e0启动逐字相同算法身份的e200权威复赛。5090若为完整终点
负结果，则保留为跨运行时负证据，不额外消费4090复赛。

该条件只用于完整算法轨迹之间的算力路由。中间paired PSNR、最佳checkpoint、退出窗口、
handoff和controller均不可访问。5090与4090的delta不合并；4090复赛不得续接或复制5090
checkpoint。

## 为什么保持条件复赛而不立刻双流

4090单流实测每5 data epochs约315--330秒；双流实测约720--750秒。虽然显存仍有空间，
同时运行AM-TNC和MCRB会降低总吞吐并延后两个终点。当前最短关键路径是4090单跑AM-TNC、
5090单跑MCRB；只有后者完整过门，才让空闲后的4090承担同宿主复赛。

这与紧急单seed协议一致：释放的seed2027/2028预算已投入独立MCRB机制，但不把跨宿主
运行伪装成同一数值排名，也不为一个完整终点负机制再支付重复宿主成本。

## 持久性

`operations/local_route1_mcrb_cross_host_successor.py`使用专用SSH key而非密码，并冻结：

- 控制worktree与MCRB训练worktree commit；
- manifest、环境记录和后继源码hash；
- 5090 endpoint、known-hosts与key文件hash；
- batch1、seed2026、e200/30000 updates；
- confirmation20和paired-control关闭状态。

后继独立于Codex会话运行。它等待5090完整终点和4090 AM-TNC终点，按上述条件执行或跳过
MCRB复赛，随后生成包含全部4090权威receipt的`ROUTE1_FINAL_E200_SELECTION.json`，再启动
真正赢家的proposal-only和observable-only e200消融。
