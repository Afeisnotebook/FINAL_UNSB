# 投稿前相关工作与新意边界裁决

时间：2026-09-06 03:18 +08:00

## 裁决

在等待 full-data e200 的非破坏性窗口，依据论文原文、正式论文页和作者仓库，对
Proposal-only、ST-CGR、AM-TNC以及terminal endpoint叙事做了pre-result碰撞审计。

三条方法都不能把通用的两样本平均、timestep stratification、antithetic sampling或
gradient projection作为首创新意。它们可能成立的论文单位分别收紧为：

- online sequential UNSB 中以真实 post-D/E 状态为条件、只改 G/F 的完整视图估计器；
- 在上述估计器内保持 uniform marginal 的有限 bridge-time 无放回耦合；
- same-player exchangeable replicas 在 frozen pre-step Adam metric 中的反对称切向几何。

NADB与SDDBM分别公开讨论endpoint underfitting/noise mismatch和hard-endpoint singularity。
相关工作不得被换名隐去；当前三条方法也不改变bridge/endpoint law，且本项目target-blind
terminal门没有通过，因此不授权添加terminal模块或声称解决奇异性。

DDSB仍是最接近的已发表无配对restoration对手。权威源码门的状态没有因论文存在而改变：
没有可靠源码/公式到实现映射时保持`reproduction_incomplete`，不能用猜测结果补主表。

## 对运行的影响

本裁决没有读取中间paired性能，没有打开confirmation20，没有修改协议、源码、队列、
checkpoint或任何健康训练。它只在结果出现前冻结“能写什么、不能写什么”，并要求投稿前
再做一次最新版检索。

权威边界：`research/paper_aio/RELATED_WORK_NOVELTY_BOUNDARY_CN.md`

compact evidence：
`evidence/paper_aio/PAPER_AIO_RELATED_WORK_NOVELTY_BOUNDARY_20260906T031839.json`
