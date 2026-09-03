# 5090A ST-CGR-only实时复核

日期：2026-09-03 14:13:47 +08:00

用户再次明确要求取消5090A上的matched plain等待，优先直接完成ST-CGR full-data
e200。实时核查确认这项调度已经生效：plain训练停在e9，trainer、supervisor、export
waiter以及两条历史plain-resume successor均无存活PID；保存的完整状态SHA256仍为
`9758471e2ad9a90810196fc1b66206e23b4d77eb94bf5a0cf9c63879c26362b0`。历史successor
JSON中的`WAITING_FOR_STCGR_COMPLETE_E200`只是无进程持有的旧状态，不能恢复plain。

ST-CGR的continuation/supervisor/trainer/exporter/health watcher均存活，完成e10、85,530
updates，健康监控为零告警。因此本次不重启、不迁移、不修改ST-CGR，也不触碰4090A、
5090B、5090C或本地1660上的任务。5090A保持ST-CGR-only，plain e9 checkpoint仅作可追溯
保存；未来若要恢复，必须有新的用户明确决定，不能由旧successor自动触发。

本次时间优先让步改变的是结果可比较性，不改变算法训练协议。ST-CGR仍按seed=2026、
batch1、8,553张/侧和200 data epochs运行；未读取paired性能、未选择最佳checkpoint、未打开
confirmation20。5090A不再提供e200 matched plain，所以ST-CGR终点在新的合法runtime关系
审核完成前只能报告绝对轨迹，不能把它写成相对plain收益。

证据：
`evidence/paper_aio/PAPER_AIO_5090A_STCGR_ONLY_RECONFIRMATION_20260903T141347.json`。
