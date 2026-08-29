# DEC-20260830：路线一因果审计双 worker 裁决

同一宿主仍由一个 durable auditor 独占审计队列和最终矩阵，但 RTX 4090/5090
允许最多两个相互独立的 `(probe, data_epoch)` worker；GTX 1660默认单worker。

并发只改变墙钟调度。每个cell使用独立工作目录；atlas与variance atlas的
read/merge/replace受跨进程文件锁保护；matrix重建串行加锁；stall只观察当前cell的
落盘行；每个虚拟分支继续验证parent full-state hash前后一致。batch、horizon、
operator、observable、paired标签时机、confirmation20锁与候选生成门均不改变。

任何OOM、非一致重复行、父状态污染或完整性缺口都属于工程失败，不能产生科学结论；
已有canonical row可在单worker模式幂等恢复。
