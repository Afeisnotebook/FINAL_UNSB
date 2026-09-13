# DEC-20260913：AM-TNC e200 完成与评估参数冲突恢复

AM-TNC 已在4090A上完成全量 `e200 / 1,710,600 updates`。最终full-state、sidecar和
source-bound固定epoch导出均闭合；554个张量无非有限值，e195恢复后6个Adam二阶矩继续以
float64保存。因此训练结果没有丢失，训练本身也没有在终点失败。

首次e200后评估失败发生在加载模型之前：外层评估器曾使用`--mode static_pair`，而UNSB模型
在内部再次读取进程参数，把同一个`--mode`解释成只允许`sb/FastCUT/fastcut`的模型参数。
这是一处CLI命名空间冲突，不是AM-TNC算法、checkpoint或性能结论。旧V7失败输出保留且未覆盖。

提交`7881b07`把外层参数改为`--evaluation-mode`；提交`f643036`进一步以“完成的run-state元数据
+ 完整source-bound export”共同证明旧监督器未重复写出的metric-blind边界，不修改任何历史状态，
live release仍保持原先的严格fail-closed规则。41项针对性测试、compileall和`git diff --check`
通过；4090A上嵌套解析实际得到UNSB `mode=sb`。

新的V8链现由AM评估监督器/child `246338/246356`运行，状态为
`EVALUATING_FIXED_ALGORITHM_CHECKPOINT`，GPU进程存在；终局等待器为`246339/246357`。新链自身
health `246425`和跨V7/V8总health `248074`连续为`HEALTHY`。V7中仍有效的统一评估和ST-CGR
等待器`125400/125436`、`125402/125437`保留；只退休依赖失败AM状态的旧AM、旧final和旧总health。

当前裁决：AM-TNC训练与交付链正常，算法机制未被证伪；但真实性能仍必须等待固定e100/e125/
e150/e175/e200 matched评估完成后才能判断。未读取性能值用于调度，未选择最佳checkpoint，
confirmation20继续封存。
