# DEC-20260901：HJCGR收益来源的一视图HJ条件重采样对照

状态：`COMPLETE_E200 / CURRENT_IMPLEMENTATION_CLOSED / GAIN_SOURCE_RESOLVED`

## 完整e200裁决

4090上的`ABL-G3-02-HJCGR-SINGLE-VIEW`已从共同e0完成seed2026、small25、
batch1、30000 updates/真实200 data epochs。source-bound终端收据状态为
`ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT`，训练与验证commit均为
`83504c58678875e1d91203c1f3c3892a95b44eaa`；算法fingerprint为
`b19bafa514cbbf72dbc9d76805887528f8445870edc5a2d8294473fd46a7db37`。
confirmation20保持封存，paired指标只在完整轨迹冻结后用于标签，未进入公式、训练、
路由或早停。

结果不通过长期门：晚三点宏PSNR delta为`-1.282081 dB`，e200为
`-0.502174 dB`，三个晚期点分别只有2/6、2/6、1/6域正，晚期平均最差域为
`-2.876955 dB`；晚期SSIM为`-0.033007`，LPIPS为`+0.043368`，两项护栏也均失败。
这只关闭HJ-PCNR当前一视图算子，不关闭HJ场、条件重采样问题或无偏方差缩减机制。

同宿主完整轨迹的因子差为：

- `HJ-PCNR - HJ`：晚三点`-1.931412 dB`，e200 `-0.661516 dB`；
- `HJCGR - HJ-PCNR`：晚三点`+2.102832 dB`，e200 `+1.375581 dB`；
- `HJCGR - HJ`：晚三点`+0.171420 dB`，e200 `+0.714066 dB`。

因此，在已测试的continuous HJ场中，仅把G/F随机视图移到D/E提交之后重新抽取并不能
解释收益，反而形成长期负轨迹；把两个条件独立的fresh HJ G/F梯度在一次Adam提交之前
求均值，才把该算子变成严格长期正结果。当前最有力的收益来源解释是条件方差缩减，而
不是事件重排或重采样本身。上述数值是三条非线性完整轨迹之差，不声称可分解为单条
样本路径上的可加因果贡献，也不证明继续增加replica数量一定继续获益。

2026-09-01补充：首轮门禁正确发现disabled复合算子仍序列化dormant HJ诊断状态，
所以full-state hash不等于plain。活动公式和训练路径未改变；修复只在关闭算子时删除
该方法私有状态。旧implementation与失败日志已归档，修复commit
`83504c58678875e1d91203c1f3c3892a95b44eaa`通过373项测试，重跑门禁的zero identity、
resume、跨状态反事实、target-blind observable、400-update micro run及HJ/PCNR事件计数
全部通过。正式e200仍从共同e0重训，未复用门禁状态。

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
- 该对照首先进入`mechanism_gain_source_decomposition`说明收益来源；若它自身通过与
  其他算法完全相同的e200数值和护栏，则也进入同宿主算法总榜和可行算法集合，因为
  一个更低计算量的通过方法不能被单冠军交付逻辑删掉。它仍标记为基于已完成HJCGR
  结果提出的posthoc development control，不是confirmation或多seed证据；
- 4090当前无其他训练进程，本任务不挤占5090上的HJCGR跨运行时复核；
- batch1、seed2026、small25、同宿主plain、confirmation20封存和单seed声明边界不变。

若HJ-PCNR接近HJCGR，则收益主要可以由HJ场内的重采样耦合改变解释；若HJ-PCNR接近HJ
或失败而HJCGR持续通过，则证据指向二视图条件方差缩减；若处于两者之间，则两部分均有
贡献。无论结果如何，只裁决该来源分解，不把一个对照升级成新的超参搜索空间。
