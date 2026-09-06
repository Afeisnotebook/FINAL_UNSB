# 五宿主健康刷新与陈旧监控退休裁决

## 裁决

保持全部科学训练与队列不变。退休 4090A 的旧活动前沿健康监控 PID 3460191，以及 5090B 仍绑定已退役 successor PID 148465 的 PID 148593/148594。权威替代分别为 4090A PID 3885710，以及 5090B PID 524155/524156。

## 原因

4090A 的 successor/supervisor JSON 是状态转换文件，并非周期 heartbeat；旧监控即使看见 PID 和 lane heartbeat 正常，也会在文件超过一天后报告 stale。替代监控继续检查两个 PID，但只对真正周期更新的 lane heartbeat、exporter 和 transport relay 应用新鲜度门。

5090B 的旧监控仍绑定容量门前已经退役的 successor，因此产生的告警没有恢复意义。正式 matched-plain successor、supervisor、trainer、exporter 以及新健康/进度监控均在运行，退休旧监控不会改变训练或恢复所有权。

## 科学边界

本次没有读取性能值、没有调整训练协议或队列、没有重启任何训练，也没有打开 confirmation20。5090B 共驻的继续/等待裁决仍完全来自预注册 metric-blind makespan 容量门。
