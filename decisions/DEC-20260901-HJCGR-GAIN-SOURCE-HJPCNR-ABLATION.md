# DEC-20260901：HJCGR收益来源的一视图HJ条件重采样对照

状态：`FROZEN_FOR_SOURCE_BOUND_GATE_AND_E200`

## 已完成父证据

4090上的`G3-02-HJ-CONDITIONAL-GF-RESAMPLING`（HJCGR）已经从共同e0完成
seed2026、small25、batch1、真实200 data epochs：晚三点宏PSNR delta为
`+0.820751 dB`，e200为`+0.873408 dB`，三个晚期点均达到至少4/6域正，
SSIM/LPIPS、最差域、回撤和plain-collapse护栏全部通过。训练与验证commit均为
`36090652782c225f41ae1065fd67cd69fae00eeb`，confirmation20封存，paired指标未进入
公式、训练或控制。

HJCGR相对continuous HJ父轨迹的晚三点/e200增量分别为`+0.171420`和
`+0.714066 dB`。这证明组合在完整非线性训练轨迹上改善了HJ父场，但该差值同时包含：

1. D/E提交后重新抽取G/F随机视图，解除跨玩家随机复用；
2. 把两个fresh HJ G/F梯度在Adam前取均值，降低给定batch后的条件方差。

原生UNSB场已有PCNR一视图对照，但不能直接替代HJ场内的因子分解。因此4090空闲槽位
只新增一个预注册的来源绑定对照，而不是扩展算法网格。

## 冻结对照

新身份：`ABL-G3-02-HJCGR-SINGLE-VIEW`（HJ-PCNR）。

在当前状态和官方unpaired batch上：

1. 使用原生单视图提交D和E；
2. 在已经实现的post-D/E状态与continuous HJ controller上重新抽取一个fresh随机视图；
3. 只用该视图的HJ structure-projected PatchNCE joint G/F梯度提交一次Adam更新。

公式为

\[
\widehat g_{\mathrm{HJ,1}}(S,q)=g_{\mathrm{HJ}}(S,q;\xi_1),
\qquad
\mathbb E[\widehat g_{\mathrm{HJ,1}}\mid S,q,b]
=\mathbb E[g_{\mathrm{HJ}}\mid S,q,b].
\]

它保持单视图条件协方差；HJCGR则使用

\[
\widehat g_{\mathrm{HJ,2}}=(g_{\mathrm{HJ}}(\xi_1)+
g_{\mathrm{HJ}}(\xi_2))/2,
\qquad
\operatorname{Cov}(\widehat g_{\mathrm{HJ,2}}\mid S,q,b)=
\tfrac12\operatorname{Cov}(g_{\mathrm{HJ}}\mid S,q,b).
\]

因此`HJ-PCNR - HJ`隔离同一HJ目标下的post-D/E resampling，
`HJCGR - HJ-PCNR`隔离相同事件顺序下增加二视图均值。两者都是共同e0完整轨迹差，
不能解释成单条轨迹内部的可加因果贡献。

## 防漂移与算力裁决

- 这不是退出阈值、窗口、退火、强度、replica-count网格或路线二handoff；
- HJCGR完整e200 paired结果只授权该事后机制消融，不参与HJ-PCNR公式或训练控制；
- HJ-PCNR仍必须通过zero identity、resume、e20/e100/e200跨状态、400-update工程门，
  并从4090共同e0完成固定e200；中间paired结果不能早停；
- 该对照不进入最终算法冠军排名，也不删除HJCGR、Proposal-only或AM-TNC；它进入
  `mechanism_gain_source_decomposition`，用于说明收益来源；
- 4090当前无其他训练进程，本任务不挤占5090上的HJCGR跨运行时复核；
- batch1、seed2026、small25、同宿主plain、confirmation20封存和单seed声明边界不变。

若HJ-PCNR接近HJCGR，则收益主要可以由HJ场内的重采样耦合改变解释；若HJ-PCNR接近HJ
或失败而HJCGR持续通过，则证据指向二视图条件方差缩减；若处于两者之间，则两部分均有
贡献。无论结果如何，只裁决该来源分解，不把一个对照升级成新的超参搜索空间。
