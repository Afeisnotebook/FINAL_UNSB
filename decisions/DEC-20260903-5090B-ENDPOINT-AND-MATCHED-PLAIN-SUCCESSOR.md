# 5090B端点识别与matched-plain后继

日期：2026-09-03

用户给出的新SSH端点经实时GPU UUID、现有进程、run目录和health watcher共同核验，
对应的正是已经登记的5090B，而不是第五张空闲GPU。该卡上的CUT与CycleGAN均健康
共驻，因而不能按“新增容量”重新分配，也没有中断或迁移任何现有训练。

5090A为立即推进ST-CGR已按用户授权在plain e9暂停，这使5090C Proposal的严格
matched plain成为当前论文证据缺口。5090B的CUT预计先于CycleGAN结束；CUT释放的
计算槽位可以由fresh-e0 plain接替，在不增加同时训练进程数的前提下恢复这条对照。
因此新增一个metric-blind持久后继：只等待CUT supervisor的`COMPLETE_E200`状态，
随后用冻结commit `e4a5eed...`执行完整preflight、resume、evaluation-repeat和
2000-update peer runtime-twin。只有网络、优化器、sampler、RNG与step core相对
5090A receipt完全一致，才授权plain e1--e200；失败则停止，不用近似结果。

该plain必须从e0在5090B重新训练，不复制或续接5090A e9 checkpoint。其source host
也诚实记录为`5090B_MATCHED_PLAIN`。它通过门禁后可为5090C Proposal新增一条显式
runtime-relation并提供严格matched control，但在relation提交前不能计算delta。
它不会自动替代ST-CGR的5090A parent：ST-CGR的跨代码candidate gate仍绑定5090A，
任何跨宿主传递关系必须另行推导和实现，不能假装具有传递性。

按当前无指标吞吐估计，CUT约在9月6日释放槽位。plain在CycleGAN剩余阶段按已观测
共驻速率、之后按5090隔离速率保守估算，约在9月12--13日完成，与Proposal e200
窗口接近。这比等待ST-CGR完成后再恢复5090A plain显著提前Proposal论文裁决，同时
不收缩Proposal、ST-CGR和AM-TNC的多算法前沿。

部署的successor和health watcher均为PPID 1，状态每分钟刷新；它们不读取性能值，
不改变训练协议、不选择checkpoint、不访问confirmation20。5090B租期尚未在仓库中
得到明确终止时间，若实际租期早于9月13日，必须在CUT结束前后按租期做一次人工容量
确认，不能让该外部条件被误写为算法或工程失败。

4090A同时部署了独立的source-bound relay，提前等待未来5090B plain的固定
e100/e125/e150/e175/e200 exports。首次用错不含`paramiko`的conda解释器，relay在
复制前fail closed；保留该回执后改用已承载现有5090B relay的系统Python，v2现为
健康等待状态。该relay只复制带哈希的checkpoint、sidecar和export receipt，不读取
metric文件。它不会自行创建runtime relation或触发论文delta。

辅助watcher随后把successor状态新鲜度窗口从600秒校正为3600秒：PID仍每分钟检查，
但同步执行2000-update gate时不会因状态文件暂时不刷新而误报。该重绑不重启successor。

## 共驻触发的墙钟复核与后继补强

05:42的无指标刷新得到CUT e45、CycleGAN e25，最近完整epoch分别为1674.17秒和
2334.91秒。按CUT剩余155个epoch计算，其预计在约72.08小时后释放一个计算槽；届时
CycleGAN预计还剩约63.86个epoch。使用已经测得的5090 plain隔离时间
2325.01--2495.93秒，以及CUT共驻/隔离比1.499--1.589推导CycleGAN隔离时间，逐一配对
计算得到：CUT结束后立即让plain与CycleGAN共驻，plain终点尚需142.95--154.01小时；
等待CycleGAN先结束再启动plain则需155.24--166.30小时。相同假设配对下，立即启动
预计节省10.72--13.85小时。因此仍以CUT完成作为触发点，而不改成等待两个外部基线。

静态估计不能代替未来真实共驻吞吐。后继因此增加一个metric-blind两epoch容量门：
完整runtime门通过后先把plain精确运行到e2，使用没有里程碑评估开销的第二个epoch和
当时CycleGAN heartbeat重新计算两种makespan。只有预计至少节省一小时才立即接续；
否则plain保持在可精确恢复的e2 full-state，等待CycleGAN释放GPU后再由同一supervisor
接管。容量门写入不可变回执，只读取data epoch和wall time，不读取paired指标，不改变
batch、优化器、数据顺序、训练目标或200-epoch终点。
