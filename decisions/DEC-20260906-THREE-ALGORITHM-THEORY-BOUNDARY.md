# Full-data三算法数学边界冻结

日期：2026-09-06
裁决：`PRE_RESULT_THEORY_BOUNDARY_FROZEN / CONTINUE_ALL_THREE_E200`

在不读取full-data中间paired性能、不加载checkpoint和不触碰任何训练进程的前提下，重新
核对了Proposal-only、ST-CGR和AM-TNC的derivation card、冻结源码及既有公式—实现审计。
三条路线共享“顺序随机博弈中的条件stochastic-measure design”母题，但不是重复算法：

- Proposal只在原生D/E提交后，用两个条件iid G/F视图降低给定batch后的完整视图方差；
- ST-CGR在Proposal的计算预算内，把两次time抽样改为边际仍均匀的有序无放回耦合，只
  进一步删除between-time conditional-mean covariance；
- AM-TNC作用于D、E、G/F全部player，通过replica交换反对称性保留Adam度量切向分歧，
  是独立几何路线。

三者能够严格支持的共同结论仅到pre-Adam条件梯度均值。它们都不能据此宣称期望Adam
参数位移、完整训练转移或PSNR无偏。ST-CGR也不能被包装成terminal singular-drift修复：
该假设没有通过既定跨算法/跨域证据门，ST-CGR的真实父证据是time-stratum异方差。

论文写作将在e200与合法matched control之后按结果分支：Proposal与ST-CGR可成为同一
estimator family的基线成员/结构扩展，AM-TNC保持独立贡献；若某个full-data operator失败，
只关闭该实现，不用small25结果或最佳checkpoint替代。HJCGR仍是有证据的历史族成员，
当前`deferred`不等于机制证伪。

为防止白纸Codex按旧快照恢复错误队列，README、START_HERE、上下文胶囊、paper contract、
active plan和AGENTS均已更新为当前plain已完成、AM-TNC/ST-CGR/Proposal/CUT/CycleGAN/
DCLGAN在飞、5090B matched plain待CUT终点的真实结构。公式、比较表、结果依赖写作路由
及禁止主张统一冻结在`research/paper_aio/ALGORITHM_THEORY_MAP_CN.md`。

本次只读刷新时六条训练均健康：AM-TNC e18、ST-CGR e52、Proposal e58、CUT e195、
CycleGAN e127、本地DCLGAN e11；5090B matched-plain successor及wait guard仍正常等待，
guard restart为0。没有改变训练源码、科学协议、PID、checkpoint、GPU队列或confirmation锁。

验证结果：Proposal/ST-CGR定向29项、AM-TNC与runtime relation定向23项、全仓696项测试
全部通过；11个公式/源码绑定hash、JSON解析与`git diff --check`均通过。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_THREE_ALGORITHM_THEORY_BOUNDARY_20260906T025900.json`。
