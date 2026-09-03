# 5090C加入后的实时全局重排裁决

日期：2026-09-03 13:44 +08:00

## 结论

重新读取四台远端GPU和本地GTX1660的进程、heartbeat、checkpoint状态、实测epoch时间、
后继条件与运行时回执后，当前四条在飞训练保持不变：4090A运行plain并在e200后自动接
AM-TNC；5090A只运行ST-CGR；5090B共驻CUT/CycleGAN，并在CUT结束后由两epoch容量门决定
是否立即共驻fresh-e0 matched plain；5090C继续独占Proposal-only。

这不是机械沿用旧队列。5090C Proposal已经通过与5090A的2000-update exact runtime twin，
当前e13实测约4851秒/epoch，预计约09-14 01:43完成。若现在改跑AM-TNC，既不能早于
4090A在09-05释放后启动的同宿主matched AM-TNC，也会使历史最稳定的Proposal失去可用
槽位；HJCGR的预计成本超过当前租期，且跨宿主small25终点符号仍不一致；DDSB没有通过
权威源码门；DCLGAN属于外部基线而非独立自研算法替代。因此换卡或共驻都降低当前论文
价值/墙钟比。

## 两个时间目标

- 训练轨迹层面的核心基线：4090A plain约09-05 08:17、5090B CUT约09-06 05:14、
  CycleGAN约09-07 23:52完成。
- 第一条带合法同宿主control的自研full-data算法预计为4090A AM-TNC，仍在09-11/12窗口。
- 5090B fresh-e0 matched plain经动态容量门预计09-12/13完成；只有其2000-update运行时关系
  经Git审核后，才允许把它作为5090C Proposal或5090A ST-CGR的control。
- Proposal预计09-14 01:43，ST-CGR预计09-14 14:03；因此有层次的Proposal/AM-TNC/ST-CGR
  结果集合预计约09-15形成。以上只依据速度和依赖，不读取中间paired指标。

## 本次直接执行

没有打断任何健康训练，也没有为了填卡新增实验。为避免已排队的DCLGAN在训练完成后成为
不可用孤立结果，新增并部署了source-bound DCLGAN导入和固定e100/125/150/175/200统一
评估链。4090A上的relay PID 2491699、evaluator PID 2491700、health PID 2491724均为
PPID 1且第二次heartbeat健康；它们当前只等待5090B DCLGAN完整导出和未来动态first-wave
统一cohort，不占训练GPU、不读取性能值。DCLGAN本身仍按5090B matched plain之后的既定
资源顺序启动；该前驱只是GPU排程，不被写成算法比较依赖。

## 保留的多算法前沿

- Proposal-only：running，条件方差缩减主线；
- ST-CGR：running，bridge-time分层估计主线；
- AM-TNC：queued，独立Adam-metric优化几何主线；
- HJCGR：deferred，成本和跨宿主证据不足，但未证伪；
- DCLGAN：外部对手已通过本地门并持久排队；
- DDSB：`reproduction_incomplete`，未获权威源码前不猜测实现。

5090C必须保证至少运行到09-15，给e200 checkpoint和source-bound export留出安全余量；若
原十天租期早于该时间，需要在09-11前扩容。租期风险不授权修改batch、提前停止、跨宿主
续训或选择最佳checkpoint。

