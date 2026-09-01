# DEC-20260901：路线一多算法终局裁决

状态：`ACCEPTED_SINGLE_SEED_SMALL25_E200 / MULTIPLE_VIABLE_ALGORITHMS`

## 结论

路线一没有收缩成“唯一冠军”。4090同宿主、共同e0、seed2026、batch1、small25、真实
200 data epochs裁决出两条严格可行算法：

1. `G3-02-HJ-CONDITIONAL-GF-RESAMPLING`（HJCGR）：late-three
   `+0.820751 dB`，e200 `+0.873408 dB`；
2. `ABL-G1-02B-PCRSMG-PROPOSAL-ONLY`：late-three `+0.541507 dB`，
   e200 `+0.451092 dB`。

`G2-01-ADAM-METRIC-TANGENTIAL-CONSENSUS`（AM-TNC）在4090为正但LPIPS护栏脆弱：
late-three `+0.105302 dB`，e200 `+0.383135 dB`。它作为独立几何机制保留，不与前两条
共享估计器的算法混为同一名字。

因此最终`ALGORITHM_SET.json`的科学结论是`MULTIPLE_VIABLE_ALGORITHMS`；
`ACTION_PRIORITY.json`把HJCGR排在第一，只表示下一份4090算力的默认入口。Proposal-only
仍是严格可行算法，AM-TNC仍是证据充分的脆弱正方向，二者没有因不是第一名而被删除。

## 为什么这是一个相关算法族，而不是名字堆叠

Proposal-only与HJCGR共享同一个数学算子：先提交原生单视图D/E，在已实现的post-D/E
状态、同一官方unpaired batch和父控制器状态下，抽取两个条件iid的G/F随机视图，并在
一次Adam提交前求均值：

\[
\widehat g^{GF}=\frac{g^{GF}(\xi_1)+g^{GF}(\xi_2)}{2},\qquad
\mathbb E[\widehat g^{GF}\mid\mathcal F]=\mathbb E[g^{GF}\mid\mathcal F],
\quad
\operatorname{Cov}(\widehat g^{GF}\mid\mathcal F)=\frac12\operatorname{Cov}(g^{GF}\mid\mathcal F).
\]

Proposal-only把它用于原生UNSB场；HJCGR把它用于continuous HJ的结构投影PatchNCE场。
共享定理只约束固定父状态下的pre-Adam估计器，不声明有限步Adam位移或完整随机路径与
父算法相同，也不降低A/B identity、domain或其他跨batch采样方差。

HNEK迁移`HPCGR`没有通过：相对HNEK父轨迹late-three/e200分别为
`-0.752090/-0.677788 dB`。这不是坏消息被隐藏，而是给出了重要适用边界：相同估计器
不会在任意父场自动有效；HJ的latent/time/bridge方差轴与该算子直接对齐，HNEK的主要
缺陷更多落在当前共享batch算子没有处理的跨batch轴。

## 收益来源已经被因子对照收紧

HJ场内的单个fresh post-D/E G/F view（HJ-PCNR）完整e200为late-three
`-1.282081 dB`、e200 `-0.502174 dB`，而双视图HJCGR严格通过。共同e0完整轨迹差为：

- `HJ-PCNR - HJ`：`-1.931412/-0.661516 dB`；
- `HJCGR - HJ-PCNR`：`+2.102832/+1.375581 dB`；
- `HJCGR - HJ`：`+0.171420/+0.714066 dB`。

所以当前支持的来源不是D/E后重新采样或事件重排本身，而是选择性G/F条件方差缩减。
compute-only观察分支与plain e200 dynamics精确一致；全player双视图在终点回到非正，说明
收益也不是“方差降得越多越好”。这些都是不同完整非线性轨迹之差，不解释为单路径内的
可加因果贡献。

## 5090独立运行时证据

5090仍使用seed2026，因此不是第二个seed，不能与4090 delta平均。但它给出了很有价值的
运行时敏感性边界：

- Proposal-only在两台宿主都严格通过；5090 late-three/e200为
  `+0.845316/+0.573796 dB`。它是目前唯一跨运行时均严格通过的算法；
- AM-TNC在5090严格通过（`+0.678000/+0.233197 dB`），在4090为脆弱正信号；
- HJCGR在5090晚三点仍为`+0.612437 dB`，但e200为`-0.094231 dB`，所以只在4090
  严格通过。其5090轨迹有很强正窗口和改善的晚期SSIM/LPIPS，不是无信号，但不能声明
  跨运行时终点稳定。

这正是最终保留算法集合而不是单冠军的原因：HJCGR拥有4090上最高同宿主收益，
Proposal-only拥有最强跨运行时可移植性和更低计算成本，AM-TNC提供独立的优化几何证据。

## 算力与进一步优化裁决

已有算力没有再用于`m=3/4`视图网格。对`m`个条件iid视图，协方差按`1/m`下降而计算量
近似线性增加；HPCGR和全player复制已经证明长期PSNR不随这种缩放单调。下一步数学优化
应提高单位额外计算的方差下降，例如可证明零均值的廉价control variate或保持官方测度的
分层估计。现有Gaussian反号构造已在固定状态审计中劣于compute-matched iid pair，不进入
长训。新构造仍须先有无偏/自消隐推导和target-blind证据，不能由paired收益拟合副本数、
窗口或阈值。

## 证据与边界

- fail-closed Goal审计证明8条算法的e200/30000-update/六域轨迹、474条反转证据、
  140条采样方差证据、13项hypothesis ledger和全部来源哈希均完整；
- `confirmation20`仍封存，paired指标未进入公式、训练控制、退出或checkpoint选择；
- 当前是small25单seed开发结论，不保证10000张全量数据、200 epochs、多seed或
  confirmation上的最终效果；
- `CANDIDATE.json`的唯一身份只是兼容行动接口，不覆盖本决策的多算法科学结论；
- 终点紧凑证据见
  `evidence/remote_route1_offload/RELATED_MULTI_ALGORITHM_TERMINAL_20260901.json`。

## 后续默认顺序

本Goal不自动启动全量训练。若进入10000张/200-epoch阶段，应把Proposal-only与HJCGR
作为两条独立主线与matched plain并行；AM-TNC作为第三条有条件保留线。由于Proposal-only
是唯一跨运行时严格通过且成本更低的方法，它应拥有与HJCGR同等的科学保留权，即使
4090小视图行动排名把HJCGR列为第一。任何全量结果仍须同宿主、共同e0、固定更新量报告，
不得以本轮small25排名预先剪掉另一条严格算法。
