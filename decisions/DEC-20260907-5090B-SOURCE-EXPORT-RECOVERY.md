# DEC-20260907：补齐5090B源端导出的持久恢复层

## 裁决

5090B的CycleGAN和matched plain均健康推进，分别到e178和e27；训练监督器、连续性
guard、进度监视与下游4090A relay都在工作。剩余的真实薄弱点是两条源端
`paper_aio_export_successor`：它们虽有健康告警，却没有进程恢复。导出器若在长等待期间
退出，训练不会丢失，但e200 hash-bound receipt、统一评估和论文交付可能静默停滞。

保留两个原导出器和全部训练进程，在5090B新增两个独立的fail-closed recovery
supervisor：

- CycleGAN supervisor PID 673800收养原exporter PID 9516；
- matched plain supervisor PID 673803收养原exporter PID 534982；
- 两者均为`MONITORING_EXISTING_EXPORT`，restart count为0；
- 健康监视器PID 673880检查二者PID、状态新鲜度及真实磁盘余量，首次和SSH断开后的
  复核均为`HEALTHY`、零告警。

恢复器冻结自身commit、源export contract、原控制checkout commit以及exporter依赖的
三个源码hash。只有状态非终止且不存在语义匹配进程时才恢复；命令不匹配、重复进程、
边界字段异常或终止状态一律fail closed。它不读取checkpoint和性能值。

## 科学边界

本次只加固控制面：没有中断或修改CycleGAN/matched plain，没有改变算法、数据顺序、
协议、runtime cohort或matched关系，没有打开confirmation20。实现已在commit
`ae897ba1ef78d2c9e59d80b2cbe5e7245b22b3ab`通过721项测试。部署回执见
`evidence/paper_aio/PAPER_AIO_5090B_SOURCE_EXPORT_RECOVERY_DEPLOYED_20260907T175518.json`。
