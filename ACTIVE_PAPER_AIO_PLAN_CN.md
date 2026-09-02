# ACTIVE：FINAL_UNSB 全量论文实验与下一阶段算法重构

状态：`FIRST_WAVE_RUNNING / STCGR_SMALL25_RUNNING / CONFIRMATION_LOCKED`  
日期：2026-09-02

## 当前不可误读的事实

- small25路线一已经得到多算法前沿：4090上HJCGR与Proposal-only严格通过，AM-TNC为
  独立脆弱正方向；Proposal-only是唯一在4090/5090两种运行时都严格通过的方法。
- 当前阶段不是把这三个旧结果重跑一次，而是获得全量论文对照，并继续从长期因果证据
  生成更好的算法。
- terminal low-variance/singular-drift假设未通过跨算法/跨域门，所以没有加入终端修复。
- time-stratum梯度异方差及Proposal/HJCGR父状态审计支持ST-CGR；其固定状态门已通过，
  当前运行small25 e200，尚未获全量授权。

## 当前在线任务

| 宿主 | 当前任务 | 冻结身份 | 自动后继 |
|---|---|---|---|
| 4090A | full plain e200 | commit `31f2fb8` / paper fp `68f53a8e...` | plain完整后重新验门并跑Proposal |
| 5090B | full CUT e200 | 同一paper commit/fp，独立baseline | CUT完整后重新验门并跑CycleGAN |
| 5090A | ST-CGR small25 e200 | candidate commit `0637880` | 只在完整e200后裁决，不看中间paired指标 |
| 本地1660 | 合同、代码、只读审计 | 不与远端训练混用 | 准备候选全量冻结接口 |

进度以各宿主`HEARTBEAT.json`为唯一工程事实，不在计划文档中冻结会迅速过时的epoch；
这些心跳不是科学结论。三条训练均由持久监督器执行，plain→Proposal和CUT→CycleGAN另有fail-closed
后继等待器。ST-CGR也已部署source-bound终点receipt后继：只在完整e200轨迹出现后发布
正式receipt，状态为`WAITING_FOR_COMPLETE_E200_TRAJECTORY`；它不会自动授权全量。

## 最近硬门

1. ST-CGR完成small25 e200后，只按e150/e175/e200、e200、六域、SSIM/LPIPS、绝对
   轨迹和成本裁决；若通过，生成独立full candidate lock，不修改在飞paper协议。
   该lock还必须通过跨代码e0/2000-step、zero-intervention、resume与重复评估门，且只有
   随后的独立authorization能启动全量；任何一门失败都保持未授权。
2. plain/CUT必须自然到e200；中间discovery值不触发早停、算法修改或资源调度。
3. DDSB维持`reproduction_incomplete`，除非出现权威作者源码；资源不会给猜测实现。
4. plain完成后在同一4090执行Proposal专属resume/identity/eval门；CUT完成后在同一
   5090执行CycleGAN门。
5. 额外4090只有在用户提供实际宿主后才进入runtime-twin或新算法全量分配。

## 第一波完成后的顺序

- 汇总plain、Proposal、CUT、CycleGAN固定e200主表；
- 对通过small25门的新算法安排自己的全量e200，而不是把Proposal当唯一方法；
- 有可靠源码和资源时补DDSB或第二优先级DCLGAN/NEGCUT；
- 统一4090评估容器完成e200 discovery80、5 bundles、NFE1–5、LPIPS/KID/FID；
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
