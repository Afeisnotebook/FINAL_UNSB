# 5090A：撤销plain自动后继，保持ST-CGR唯一训练优先级

日期：2026-09-03

用户再次明确要求：5090A不得因matched plain阻塞ST-CGR；在紧迫工期内，优先取得
ST-CGR的full-data e200轨迹，并且不影响其他宿主正在运行的任务。

实时核查表明，执行本决策前ST-CGR已经是5090A上唯一占用GPU的训练进程；plain
trainer和supervisor均不存在，plain停在e9可精确恢复状态。实际可能改变后续调度的
对象是：等待ST-CGR完成后自动恢复plain的successor、该successor的health watcher，
以及等待plain e200的source exporter。因此本次只停止这三个plain链进程，并替换了
包含plain exporter依赖的旧5090A health watcher。ST-CGR的continuation、supervisor、
trainer、完整e200 exporter和增量审计exporter均未停止、迁移或重启。

5090A现在没有plain自动恢复授权。e9 checkpoint和两类状态hash完整保留，未来如需
恢复，必须由新的明确决策重新部署；旧successor不得静默拉起。新的5090A监控只守护
ST-CGR主训练与其交付链，第二轮刷新仍为`HEALTHY`且零告警。

这项时间优先决策不修改任何科学配置，但会牺牲当前的matched-control交付时间：
ST-CGR e200完成后可先形成合法的绝对长程轨迹与target-blind审计，不能在没有合法
plain e200前写成相对收益。5090C Proposal与5090A plain的既有runtime关系仍保留，
但也不能把未完成的plain当成最终matched control。4090A、5090B、5090C和本地1060
均未查询或改动。

训练继续固定为seed=2026、batch1、8,553张/侧、200 data epochs；没有读取中间paired
指标，没有以指标控制调度，没有选择最佳checkpoint，也没有打开confirmation20。

