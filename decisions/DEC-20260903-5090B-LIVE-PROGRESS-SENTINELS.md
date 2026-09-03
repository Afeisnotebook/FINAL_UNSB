# 5090B：为共驻长训补齐metric-blind进度哨兵

日期：2026-09-03

5090B上的CUT与CycleGAN均持续健康运行，分别完成到e74和e45；GPU共驻利用率约
99%，训练PID、科学配置和后继队列均未改变。既有health watcher能够确认PID与
epoch心跳存在，但不能覆盖“进程仍活着而epoch不再前进”的故障，因此为两条长训
补充只读进度哨兵。哨兵不读取性能指标、不发送信号、不恢复训练，也不打开
confirmation20。

CUT的legacy supervisor同时保留bootstrap shell与trainer两个直接child，原v2哨兵
因要求恰好一个child而保持等待。该无效哨兵被单独终止，训练未受影响。v3把可选的
trainer命令片段纳入冻结合同：只有唯一child匹配时才采样其I/O和CPU；零个或多个
匹配仍fail closed。部署后v3唯一选择CUT trainer PID 5771，CycleGAN v2唯一选择
trainer PID 11846；两者30秒I/O与CPU均持续增长，SSH断开后PPID均为1，且零告警。

matched-plain与DCLGAN successor仍分别等待CUT e200和matched plain e200，未提前或
重复启动。此次变更只补工程连续性证据，不更改任务排序、算法、数据协议或任何科学
裁决。
