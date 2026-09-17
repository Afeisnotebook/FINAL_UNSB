# FINAL_UNSB 当前论文主张边界

状态：`POST_DISCOVERY / CLAIM_REVIEW_NOT_FROZEN / CONFIRMATION20_SEALED`

## 目前可以陈述

- 在冻结的full-data、seed2026、batch1、e200协议内，Proposal-only相对合法matched plain
  取得late-three `+1.723565 dB`、e200 `+0.839347 dB`宏PSNR收益，并通过预注册护栏。
- Proposal作用于post-D/E的G/F条件iid双视图估计，当前证据与降低player-selective条件
  梯度方差一致。
- ST-CGR与AM-TNC当前实现未通过同一长期门，负结果必须报告。
- CUT、DCLGAN、CycleGAN的固定e200绝对指标高于Proposal。
- 当前实验说明短程正收益不能自动外推为长期收益，算法对象、时间耦合和优化几何需要分别
  裁决。

## 目前不能陈述

- Proposal是总体SOTA、在全部基线上最佳或已具备跨seed稳定性；
- ST-CGR/AM-TNC/HJCGR/DDSB的父机制已被证伪；
- 方差缩减必然改善图像质量、收敛性或终端桥分布；
- terminal low-variance/singular drift已被本项目确认或修复；
- DCLGAN与UNSB方法之间存在matched delta；
- confirmation20支持当前结论；它尚未打开；
- 最佳checkpoint、跨非等价runtime delta或图像bootstrap可以替代冻结e200和训练seed证据。

## 必须披露

- 只有seed2026；分辨率128；六域custom unpaired split；
- legacy cohort的一sample stream phase offset；
- AM-TNC epoch178/194数值恢复及其method-only runtime relation边界；
- Proposal额外随机视图带来的计算成本；
- DDSB reproduction incomplete、HJCGR deferred；
- 数据集许可限制与不可公开再分发的图像资产。

## 下一门

只有提交式claim review和claim-freeze receipt才能把上述开发结论转成最终论文措辞，并决定
是否一次性打开confirmation20。当前文件不构成confirmation授权。
