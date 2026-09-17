# FINAL_UNSB 当前项目状态

日期：2026-09-17

## 当前阶段

项目已经结束 full-data discovery 执行阶段，当前状态是：

`结果已归档 → 等待论文 claim review → confirmation20 继续封存`

目前没有由本仓库授权的主训练队列、服务器 successor 或自动新实验。旧文档中的 PID、
heartbeat、SSH 端点和“running”字段只记录当时如何得到结果，不能作为当前状态。

## 当前科学结论

- Proposal-only 是唯一通过预注册 full-data 长期门的自研方法：late-three 宏 PSNR
  delta `+1.723565 dB`，e200 delta `+0.839347 dB`。
- ST-CGR 与 AM-TNC 的当前实现未通过长期门；这不是其父机制族的数学证伪。
- HJCGR 是 deferred；DDSB 是 reproduction incomplete，均不能写成失败。
- CUT、DCLGAN、CycleGAN 的 e200 绝对 PSNR 高于 Proposal，所以不能声称总体 SOTA。
- 当前证据只有 seed 2026，不能声称跨 seed 稳定。
- terminal low-variance/singular-drift 假设没有通过证据门，不能添加或宣称对应修复模块。

## 当前允许做什么

1. 审阅并冻结论文主张、算法集合、披露文字和 confirmation policy；
2. 使用已归档结果写方法、实验、局限和相关工作；
3. 单独审阅服务器 independent proof，再决定是否建立 proof archive；
4. 在用户新授权后执行 confirmation20 或额外 seed，但不得反向修改算法。

## 当前不允许做什么

- 从旧 active plan、PID、successor 或服务器路径恢复训练；
- 用中间 paired 指标选择算法、退出点或 checkpoint；
- 把不同 runtime cohort 拼成 matched delta；
- 把 Proposal 写成总体 SOTA或多 seed 稳定；
- 把 ST-CGR/AM-TNC/HJCGR/DDSB 的当前处置扩大成机制证伪；
- 把未裁决 independent proof 自动并入 canonical 结果；
- 在没有新决策的情况下打开 confirmation20。

## 权威入口

按以下顺序阅读：

1. `FINAL_HANDOFF_CN.md`
2. `FINAL_HANDOFF.json`
3. 本文件
4. `CLAIM_BOUNDARIES.md`
5. `archive/paper_aio/final_v10/PAPER_ALGORITHM_PORTFOLIO_WITH_DCLGAN.json`
6. `research/paper_aio/RESEARCH_HANDOFF_CN.md`
7. `DISASTER_RECOVERY_CN.md`

文档分类规则见`configs/DOCUMENT_AUTHORITY_REGISTRY.json`。
