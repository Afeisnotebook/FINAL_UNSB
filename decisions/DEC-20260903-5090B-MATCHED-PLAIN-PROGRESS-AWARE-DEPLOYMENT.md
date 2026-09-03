# 5090B未来matched plain采用progress-aware监督器

日期：2026-09-03

5090B的CUT与CycleGAN当前健康，不重启任何现有训练。等待CUT e200的未来matched
plain后继被原位替换为源码绑定的progress-aware版本：只有一次失败前出现了更晚的完整
checkpoint，历史失败串才会被重置；同一checkpoint上的重复失败仍在第三次后fail closed。
这避免多日训练把彼此独立且已恢复的工程退出错误累计为连续失败，同时没有弱化真正的
无进展故障门。

新后继继续使用相同科学checkout、manifest、peer runtime receipt、fresh e0要求、输出
目录和状态路径。DCLGAN仍读取这个状态路径，因此级联依赖没有改线。部署后退出交互式
SSH并重新核查：successor、health watcher和progress watcher均已归属PID 1；matched
plain尚未启动，系统中没有重复plain trainer。CUT、CycleGAN以及DCLGAN等待链PID均未
变化。

一次初始watcher启动因nohup重定向目录尚不存在而立即退出；创建专属目录后按同一冻结
命令重启，两个失败PID均已死亡且未接触训练或checkpoint。本事件作为可恢复工程事件写入
回执，不隐去。

本变更只增强未来工程连续性，不修改seed、batch、数据、优化器、更新量或算法，也不读取
paired性能、不选择checkpoint、不打开confirmation20。
