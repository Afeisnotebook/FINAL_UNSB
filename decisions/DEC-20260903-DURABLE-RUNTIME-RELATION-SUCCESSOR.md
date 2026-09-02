# Proposal新matched control的持久关系继任器

日期：2026-09-03

## 问题

5090A按用户时间优先授权暂停plain并直接运行full-data ST-CGR。为避免Proposal等待该
plain，5090B已经武装了CUT完成后的fresh-e0 plain继任器。但其2000-update runtime twin
即使未来通过，仍需要人工转运回执和构造关系；若Codex会话届时不在，这会造成不必要的
论文关键路径停顿。

## 决策

新增持久的`paper_aio_runtime_relation_successor.py`，部署在统一评估宿主4090A。它：

1. 使用固定SSH host key和仅存在于进程环境的密码，等待5090B的runtime receipt；
2. 要求状态为`PASS_EXACT_RUNTIME_COHORT`、2000 updates、指定host/protocol/manifest、
   `differences={}`且confirmation封存；
3. 将回执按原字节不可变发布，再与5090C Proposal的runtime receipt及其hash绑定授权
   一起验证；
4. 只生成`review-only`关系候选并退出。

继任器明确不会修改`PAPER_AIO_MATCHED_RUNTIME_RELATIONS.json`，不会授权比较、启动训练、
读取性能值或打开confirmation。关系进入registry仍需在真实回执出现后进行人工科学复核、
Git提交和测试。因此自动化缩短的是工程交接时间，不放松matched delta门。

当前5090A ST-CGR、4090A plain、5090B CUT/CycleGAN和5090C Proposal的训练进程均不受
该变更影响。
