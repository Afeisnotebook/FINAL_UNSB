# DEC-20260906：本地增量checkpoint relay恢复

## 裁决

4090A AM-TNC和5090A ST-CGR训练、远端增量exporter以及本地terminal-audit主/子进程均健康，但两条本地增量relay已经退出。日志表明它们不是训练故障或网络暂态，而是旧contract引用了可变的`main`工作树；被导入的base relay脚本随后发生合法代码更新，relay按设计以`control source changed`失败关闭。

两条relay已用当前提交`46f9077`的独立detached工作树重新部署，分别登记为`incremental_4090A_amtnc_v3`和`incremental_5090A_stcgr_v5`。新contract同时冻结入口脚本与base relay脚本哈希，后续`main`更新不会改变其运行来源。两条新relay连续更新状态；新的综合健康监视器已通过两轮独立检查并保持零告警，旧的告警监视器随后退出。

## 论文流程影响

这次恢复没有重启或修改任何训练，不复制尚未到达的checkpoint，不读取性能指标。两条来源目前均未到固定审计点e100，因此`available_epochs=[]`是预期等待，不是数据缺失。4090A plain的e100/e150/e200审计已经完成，terminal audit继续等待Proposal、AM-TNC和ST-CGR的固定来源。

本地DCLGAN仍通过PID 16164持有与terminal JVP相同的GPU文件锁；即使e100在DCLGAN完成前到达，terminal audit也只能等待，不能与DCLGAN在GTX1660上共驻。

机器可读回执见`evidence/paper_aio/PAPER_AIO_LOCAL_INCREMENTAL_RELAY_RECOVERY_20260906T065415.json`。
