# 当前及后继自研lane的静默停滞覆盖

日期：2026-09-03

在不重启或修改训练的前提下，4090A plain与5090C Proposal均已增加只读live
progress sentinel。两条watcher均唯一绑定现有trainer，30秒I/O和CPU继续增长，
断开SSH后PPID为1且零告警。plain当前完成e94，Proposal完成e17。

4090A上等待plain e200的AM-TNC也已预先绑定后代watcher。它当前保持等待，不参与
调度或启动训练；合法successor产生唯一trainer后才开始诊断。由此，当前plain、
Proposal、ST-CGR、CUT、CycleGAN以及已排队的AM-TNC、matched plain、DCLGAN均有
PID/心跳健康检查和独立的live-process进度覆盖。

这些watcher不读取paired指标、不发送进程信号、不改变successor顺序，也不打开
confirmation20；它们只缩小工程上“进程活着但训练不前进”的盲区。
