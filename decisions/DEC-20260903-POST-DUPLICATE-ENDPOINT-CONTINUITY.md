# 重复端点后的四宿主连续性与队列裁决

日期：2026-09-03

确认`:44804`只是现有5090B入口后，重新检查了全部真实训练根、supervisor、exporter、
health watcher、后继和控制时限。没有新增物理GPU，因此不能用虚构的第五卡重排任务。

当前队列保持：

- 4090A plain e68继续独占，到e200后由PID 2182585启动AM-TNC；
- 5090A ST-CGR e7独占，continuation/supervisor/exporter/incremental exporter均存活；
- 5090B CUT e54与CycleGAN e31继续共驻，CUT后PID 61721运行fresh-e0 exact-runtime
  matched plain容量门；
- 5090C Proposal e10独占并由source-bound exporter等待e200；
- 本地GTX1660保持固定checkpoint relay、terminal audit与posthoc successor，不启动
  无证据训练。

4090A不提前共驻AM-TNC。plain只余约46.34小时，是AM-TNC及统一结果链的高扇出control；
同一训练骨干的既有4090A共驻门曾使plain epoch约慢70%。AM-TNC虽未单独做当前时刻的
共驻测速，但为了得到该测速必须实际占用并延后plain。缺少能预先保证总makespan下降的
证据时，等待e200后立即接力优于再次扰动这条最早control路径。

持久性覆盖满足当前外推：4090A health 600小时；5090A ST-CGR 600小时；5090B external
480小时且matched-plain后继720小时；5090C Proposal 480小时。各训练supervisor均只按
相同lane/commit/protocol执行exact resume，连续三次工程失败才停止并由独立health
watcher显式报警。磁盘最坏写入预算均小于真实余量。

因此本次最优动作是保持现有队列，不为占满GPU增加lane。唯一无法由仓库内supervisor
修复的风险仍是云端租期：5090C需覆盖到至少9月15日、5090A至少9月16日、5090B至少
9月14日。该风险不授权提前停止、改算法或选最佳checkpoint。

此裁决只使用heartbeat、进程、时间与容量；未读取paired性能，未打开confirmation20，
未合并非等价宿主delta。
