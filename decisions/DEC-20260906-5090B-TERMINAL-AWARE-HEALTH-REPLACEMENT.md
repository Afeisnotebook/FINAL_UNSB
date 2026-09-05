# 5090B完成态感知健康监控替换

## 裁决

5090B原主健康监控只把`COMPLETE_E200`等少数字面量识别为终态。CUT及其source-bound
export均已完整结束后，exporter PID按设计退出，旧监控却把
`COMPLETE_SOURCE_BOUND_EXPORT_SET`误报为`ALERT_PID_DEAD`。这不是训练、导出或磁盘故障，
也不需要恢复CUT。

使用提交`e831ac7334791faa8f4ee533eae85455859a7be5`中的通用`COMPLETE_*`终态语义部署新
只读监控PID `533823`。它连续三次报告零告警并归属PID 1后，才终止旧监控PID `30355`。
新监控继续覆盖CUT终态、CycleGAN supervisor/lane/export及真实磁盘余量；没有读取
checkpoint或performance文件。

替换后CycleGAN supervisor/trainer PID仍为`11845/493306`，5090B matched-plain
successor/容量探针PID仍为`523993/524019`。当时容量门已完成e1且e2仍在运行；任何训练、
checkpoint、runtime cohort和协议均未改变。

## 后续

5090B主健康入口改用
`/root/autodl-tmp/runs/FINAL_UNSB_HEALTH_5090B_TERMINAL_E831AC7/HEALTH_WATCH_STATE.json`。
容量门仍只按工程吞吐和makespan裁决，不能读取paired结果；confirmation20继续封存。

