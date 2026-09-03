# 统一论文评估的逐图完整性闭环

日期：2026-09-03

对未来统一评估链的代码审计发现三个延迟风险：LPIPS初始化异常会被静默转成空指标；
cohort lock只验收已写入的汇总字段而不从逐图证据重算；`training_checkpoint_read_only`
是声明而不是评估前后的文件证明。这些问题不会改变当前任何训练，但可能在长训结束后
产生不完整或内部不一致的论文表格。

commit `bc29b379722a8fd7b3c3a4facbff121b890cbd25`将尚未部署的替换评估链升级为：

- 固定并显式记录RGB PSNR、SSIM与`lpips==0.1.4` AlexNet v0.1的数值语义；
- LPIPS请求失败时fail closed，禁止把依赖/权重错误伪装为可接受的`null`；
- 从逐图记录重算每个NFE、每个replicate、逐域和六域宏指标及随机性标准差；
- 验证完整的domain×image×replicate×NFE笛卡尔积、有限数值和冻结primary NFE；
- 在同一epoch跨所有lane核对精确sample identity、evaluation-input hash与CRN identity；
- 对复制的训练checkpoint在评估前后重新计算SHA256，并把相等证明写入metric与receipt；
- future successor不接受缺少上述只读重哈希证明的陈旧评估产物。

`python -m compileall -q production research operations tests`、完整`pytest`（540项）和
`git diff --check`均通过。5090A ST-CGR、5090B CUT/CycleGAN、5090C Proposal、4090A
plain及其现有监督/导出链均未重启、迁移或修改。本提交只约束未来统一评估与结果锁；
部署replacement evaluator时必须从包含本提交和已审核runtime-relation registry的干净
checkout生成新的protocol fingerprint，不能拿旧receipt冒充新完整性闭环。

该修复不读取中间paired性能、不控制训练、不选择最佳checkpoint、不合并非等价宿主，
也不打开confirmation20。
