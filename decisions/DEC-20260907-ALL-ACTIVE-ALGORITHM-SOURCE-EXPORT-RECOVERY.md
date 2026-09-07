# DEC-20260907：全部活动算法的源端 e200 exporter 已纳入恢复层

AM-TNC、ST-CGR、Proposal，以及5090B上的CycleGAN和matched plain，均已有独立的
source-bound exporter。现在每个 exporter 都由一个 fail-closed recovery supervisor 收养，
并由状态新鲜度 health watcher 保护；部署后均为监控原进程、零重启、零训练变更。

5090A与5090C访问GitHub超时时，使用完整Git bundle传送同一个`ae897ba`控制commit；最初
假定存在`/usr/bin/python3`的控制进程立即退出，随后改用服务器已有的miniconda解释器。
这两次失败发生在恢复监督器启动前，没有加载checkpoint，也没有影响训练或原exporter。

这项控制只保护 e200 后的source-bound导出，不改变算法、训练协议或matched关系。完整回执
见`evidence/paper_aio/PAPER_AIO_ALL_ACTIVE_ALGORITHM_SOURCE_EXPORT_RECOVERY_20260907T182725.json`。
