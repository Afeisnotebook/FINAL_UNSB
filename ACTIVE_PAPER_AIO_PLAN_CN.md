# ACTIVE：FINAL_UNSB 全量论文实验与下一阶段算法重构

状态：`FIRST_WAVE_RUNNING / THREE_ALGORITHM_PATHS_RUNNING / CONFIRMATION_LOCKED`
日期：2026-09-03

## 当前不可误读的事实

- small25路线一已经得到多算法前沿：4090上HJCGR与Proposal-only严格通过，AM-TNC为
  独立脆弱正方向；Proposal-only是唯一在4090/5090两种运行时都严格通过的方法。
- 当前阶段不是把这三个旧结果重跑一次，而是获得全量论文对照，并继续从长期因果证据
  生成更好的算法。
- terminal low-variance/singular-drift假设未通过跨算法/跨域门，所以没有加入终端修复。
- time-stratum梯度异方差及Proposal/HJCGR父状态审计支持ST-CGR；small25 e200、全量
  candidate lock、跨代码runtime和独立authorization均已通过，当前运行full-data e200。

## 当前在线任务

| 宿主 | 当前任务 | 冻结身份 | 自动后继 |
|---|---|---|---|
| 4090A | full plain e200 | 当前live commit/fp见`PROJECT_STATE.json` | plain后运行AM-TNC |
| 5090A | full ST-CGR e200 | candidate `656670c` / fp `2fbdd6f...` | e200后source-bound export；plain稍后精确恢复 |
| 5090B | full CUT + CycleGAN e200同卡 | 各自冻结外部基线协议 | CUT后先exact twin，再fresh-e0 matched plain |
| 5090C | full Proposal e200 | commit `e4a5eed` / fp `e5704e...` | source-bound export |
| 本地1660 | 合同、代码、只读审计 | 不与远端训练混用 | 准备统一评估和只读runtime关系证据 |

进度以各宿主`HEARTBEAT.json`为唯一工程事实，不在计划文档中冻结会迅速过时的epoch；
这些心跳不是科学结论。所有训练均由持久监督器执行。5090A plain停在e9完整状态且不会
自动重启；5090B的plain继任器只在CUT完成和精确runtime门通过后启动。ST-CGR与Proposal
均有独立source-bound exporter；任何relay或关系候选都不能自动改Git registry或授权结论。

## 最近硬门

1. ST-CGR已通过small25 e200和全量双阶段授权，必须自然运行到full-data e200；只按
   e150/e175/e200、e200、六域、SSIM/LPIPS、绝对轨迹和成本裁决。它虽具有独立训练协议
   指纹，但评估使用固定`68f53a8e...` CRN bundle；只有late三点逐图CRN完全一致且
   matched plain身份合法时才允许计算delta。
2. plain/CUT必须自然到e200；中间discovery值不触发早停、算法修改或资源调度。
3. DDSB维持`reproduction_incomplete`，除非出现权威作者源码；资源不会给猜测实现。
4. 4090A plain完成后执行AM-TNC；5090B的CUT/CycleGAN保持连续，CUT完成后才运行已冻结
   的Proposal matched-plain继任器。
5. 5090B的runtime receipt必须先被只读转运、验证并生成review-only关系候选；只有人工
   审阅和Git提交才能把它加入comparison registry。

## 第一波完成后的顺序

- 汇总plain、Proposal、CUT、CycleGAN固定e200主表；
- 对通过small25门的新算法安排自己的全量e200，而不是把Proposal当唯一方法；
- 有可靠源码和资源时补DDSB或第二优先级DCLGAN/NEGCUT；
- 统一4090评估容器完成e200 discovery80、5 bundles、NFE1–5、LPIPS/KID/FID；
- 在统一评估前分别为四条source lane的e100/e125/e150/e175/e200生成checkpoint export
  receipt；复制后只读复算，四lane×五epoch完整同容器门通过才形成第一波结果锁；
- 冻结算法集合与论文主张，之后只打开一次confirmation20；
- 仅对全量seed2026严格通过的plain和最终论文算法考虑seed2027/2028。

## 禁止项

- 跨4090/5090计算matched delta或搬checkpoint续训；
- 用中间paired指标控制训练、选择NFE、改公式、改time权重或提前停止；
- 增大batch/梯度累积后仍声称与batch1轨迹等价；
- 把`CANDIDATE.json`行动优先级写成唯一科学算法；
- 在没有终端漂移证据时加入BMC/终端修复；
- 打开confirmation20；
- 把DDSB“未能可靠复现”写成DDSB性能失败。
