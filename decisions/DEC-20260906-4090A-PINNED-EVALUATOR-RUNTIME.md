# DEC-20260906：4090A专用评估运行时与DCLGAN恢复链冻结

## 结论

4090A 现在有两套职责互不混用的隔离运行时：AM-TNC 的恢复环境只续接训练；新的只读
评估环境只负责统一评估及其控制进程。由此消除了 DCLGAN evaluator 意外退出后通过
AM-TNC 训练 relay 自动恢复、形成不可见运行时混合的风险。

## 门禁与部署

评估候选保留现有动态 evaluator 的 Python 3.10.20、NumPy 2.1.3、Pillow 11.3.0 和
Torch 2.5.1 数值栈。第一次完整测试因缺少 Paramiko 而 fail closed，未接入 successor；
在隔离副本补齐与已验证环境一致的控制依赖后，冻结 DCLGAN bundle 为 679 passed、2 skipped，
dynamic delivery bundle 为 699 passed、2 skipped。候选随后设为只读，测试前后完整树哈希
均为 `4c9ca0c7d5e0e4aaa4d1026ed6229136d50e646aeccaf73401a565ce3f4e1bfe`。

新的 DCLGAN supervisor PID 3870693 从原 child state 收养 PID 3587264，没有重启 child，
restart count 为零；health watcher PID 3870938 在 SSH 断开前后连续三次报告 HEALTHY，
两者均已归属 PID 1。确认 replacement 健康后，才退役旧 supervisor 3804113 和旧 health
watcher 3804260。AM-TNC supervisor/trainer PID 3446757/3446758 未改变。

## 科学边界

本次只修改未来工程恢复路径。没有读取 paired 性能，没有生成评估结果，没有修改训练
checkpoint、队列、算法或协议，也没有打开 confirmation20。动态 first-wave、ST-CGR 和
AM-TNC 现有健康 waiter 继续保留；若它们未来退出，必须先复核只读评估树哈希和输出中
是否已有部分结果，再决定同运行时续接或从不可变 checkpoint 重建完整 evaluation cohort。
