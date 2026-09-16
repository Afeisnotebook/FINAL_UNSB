# 第一波统一评估闭合与后续评估恢复

固定的21个第一波评估单元已经全部完成，并发布合法的统一cohort。中间指标没有参与训练、调度、早停或算法修改。

ST-CGR随后暴露的是控制命令错误，而不是算法失败：旧控制脚本把`dynamic_candidate`放入`--mode`，被UNSB模型解析器误读。旧contract、state和lock已完整归档；commit `314d48c`使用隔离的`--evaluation-mode`重新建立durable supervisor，固定e100评估已经正常运行。

DCLGAN暴露的是只读审计顺序错误，而不是模型或checkpoint损坏：统一环境在审计快照后执行确定性reseed，使训练RNG哈希发生预期变化。修复只在指标计算前保存训练RNG、计算后恢复，再执行完整状态与checkpoint重哈希；模型、优化器、scheduler、sampler和checkpoint门禁均保留。新V3评估等待共享GPU，不覆盖失败证据。

两条恢复均不修改训练结果、不读取局部性能用于调度，并继续封存confirmation20。下一门是先观察ST-CGR首个固定单元闭合，再顺序完成ST-CGR和DCLGAN全部固定评估、disposition、复杂度及最终多算法portfolio。
