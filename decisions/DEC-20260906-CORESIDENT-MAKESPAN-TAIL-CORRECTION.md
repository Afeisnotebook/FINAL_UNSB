# 共驻容量门必须计算两项任务的完整尾部

## 裁决

共驻容量门的目标是比较“所有既定任务完成”的总墙钟时间。源码复核发现，当plain先于
companion完成时，旧公式直接把plain完成时刻当作总makespan，漏掉companion随后独占GPU的
剩余尾部；companion先完成的另一分支已经正确计算plain尾部。

提交`a4f4b46ca93482371a65a187295b9ce7893b44ce`补齐对称分支：先计算plain完成前companion
实际推进的epoch，再按companion独占吞吐加入其剩余时间。确定性反例从旧错误的1000秒修正
为5750秒，而等待释放方案为5500秒，因此应判等待而不是错误地判立即共驻。

## 适用边界

该修复只用于未来容量门，不热修改已冻结并正在运行的5090B gate，也不改变batch、算法、
数据顺序、checkpoint或runtime cohort。门的输入仍只有epoch吞吐和任务进度，不读取paired
性能。定向测试11项、全套705项通过；confirmation20继续封存。
