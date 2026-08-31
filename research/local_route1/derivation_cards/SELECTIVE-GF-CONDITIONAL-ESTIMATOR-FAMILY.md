# Selective post-D/E conditional G/F estimator family

状态：`FROZEN_MATHEMATICAL_SCOPE_PENDING_RELATED_E200_ADJUDICATION`

## 1. 被修改的UNSB对象

把一次UNSB更新写成顺序随机博弈：D先提交，E随后提交，最后G/F联合提交。令

\[
S_k^{DE}=\Phi_E\!\left(\Phi_D(S_k,b_k,\xi_k^{DE}),b_k,\xi_k^{DE}\right)
\]

是原生单视图D/E提交后实际实现的对手状态；`b_k`是本次官方batch1无配对A/B样本。
本算法族不平均D/E，不改data sampler，也不把输出拉向plain。它只在
`S_k^{DE}`已经实现后，重新抽取两个G/F随机视图。

令条件sigma-field

\[
\mathcal F_k^{DE,b,q}=\sigma(S_k^{DE},\text{optimizer state},b_k,q_k),
\]

其中`q_k`只在HJ父目标中存在。给定该sigma-field，抽取
\(\xi_{k,1}^{GF},\xi_{k,2}^{GF}\)为父目标原生随机测度下的条件独立同分布视图，并定义

\[
\widehat g_k^{GF}
=\frac{1}{2}\left[
g_{P,GF}(S_k^{DE},b_k,q_k;\xi_{k,1}^{GF})+
g_{P,GF}(S_k^{DE},b_k,q_k;\xi_{k,2}^{GF})
\right].
\]

`P`分别代表原生UNSB、HNEK physical-horizon父场或HJ结构投影PatchNCE父目标。
G/F的Adam优化器只在均值形成后执行一次。

## 2. 条件无偏与协方差

若两个视图在\(\mathcal F_k^{DE,b,q}\)下独立同分布且具有有限二阶矩，则线性性给出

\[
\mathbb E[\widehat g_k^{GF}\mid\mathcal F_k^{DE,b,q}]
=\mathbb E[g_{P,GF}\mid\mathcal F_k^{DE,b,q}],
\]

而条件独立性给出

\[
\operatorname{Cov}(\widehat g_k^{GF}\mid\mathcal F_k^{DE,b,q})
=\frac{1}{2}\operatorname{Cov}(g_{P,GF}\mid\mathcal F_k^{DE,b,q}).
\]

因此算法不增加一个新的期望G/F向量场；它改变的是给定实际D/E状态与batch后的估计器
协方差及随后的有限步随机耦合。这个结论只在pre-Adam梯度层成立。由于Adam是非线性的，
不推出期望有限步Adam位移等于父算法，也不推出随机训练路径相同。

## 3. 为什么必须选择性保留D/E随机性

若D/E也采用双视图均值，即使每个player的估计器分别无偏，随机对手状态
\(S_k^{DE}\)的有限步分布仍会改变。G/F随后面对的是另一套内生训练分布。因此“每个player
都降方差”并不等价于“更接近同一UNSB训练过程”。

同宿主共同e0、seed2026、真实e200因子对照为：

| 算子 | late-three宏PSNR delta | e200 delta | 裁决 |
|---|---:|---:|---|
| PCNR：原生D/E + 一个fresh G/F view | -0.532685 | -0.030690 | resampling alone未通过 |
| Proposal：原生D/E + 两个fresh G/F view均值 | +0.541507 | +0.451092 | strict pass |
| Full PC-RSMG：D/E/G/F均双视图 | +0.620959 | -0.001379 | 终点未通过 |

Proposal相对PCNR的完整非线性轨迹差为late-three `+1.074192 dB`、e200
`+0.481782 dB`。这支持“选择性G/F双视图均值”是当前已测试原生场算子中严格通过的关键
组件，但不把两条轨迹之差解释为单路径内可加因果量，也不证明所有可能方法都必须使用两个
视图。

## 4. 三个相关族成员并非同一失败模式

1. **Proposal-only**：原生UNSB父场。PCNR、compute-only与all-player消融直接限定其
   收益来源。
2. **HJCGR**：HJ的`latent_time_bridge_rng`轴在4090因果矩阵中22/22行由方差主导，
   与同batch双视图算子直接对齐。两个replica从同一HJ状态开始；整数物理计数只推进一次，
   浮点诊断增量取均值。
3. **HPCGR**：HNEK同一随机轴18行中0行由方差主导，而跨batch轴18行中9行由方差
   主导。由于HPCGR共享batch，它不是对该主导轴的直接修复，而是检验已验证的选择性G/F
   估计器能否迁移到独立有效的HNEK父场。相对HNEK的完整e200增量决定迁移是否成立。

共享无偏定理只说明三个实现属于同一估计器家族，不预先保证三个父目标都得到PSNR收益，
也不把家族成员资格当作通过裁决。

## 5. 没有被减小的随机量

两个G/F视图共享同一官方A/B batch。因此被减小的是给定batch后的latent、bridge time、
bridge noise、PatchNCE feature sampling等条件协方差。以下方差不在定理的减半结论中：

- A/B样本identity；
- domain draw；
- 官方sampler产生的其他跨batch数据方差。

若未来要处理跨batch方差，必须另行构造保持官方无配对测度的stratified或importance-
weighted估计器；不能把当前结果直接外推。

## 6. 训练与结论边界

- 不读取paired PSNR、SSIM、LPIPS或plain输出；
- 不使用退出阈值、固定handoff、最佳checkpoint或paired controller；
- confirmation20封存；
- 当前裁决为small25、seed2026、200 data epochs，不声明跨seed稳定；
- 5090是cross-runtime复现，不是第二个seed；
- compute-only精确plain控制排除“仅额外观察计算或墙钟变化”解释，但算法仍有额外计算
  成本，不能声称native compute-budget等价；
- AM-TNC保留为独立的Adam度量切向机制，不属于本条件均值家族。

## 7. 最终可证伪问题

终局不再问“哪个名称必须成为唯一冠军”，而分别问：

1. Proposal的原生场严格正结果能否在独立runtime保持方向性证据；
2. HJCGR是否把直接对齐的HJ随机轴转化为相对HJ的长期增量；
3. HPCGR的跨父场迁移是否相对HNEK为正；
4. AM-TNC是否作为独立几何路线保持正向且通过感知护栏；
5. 最终有几条方法满足同宿主共同e0的严格e200门槛。

`ACTION_PRIORITY`只安排下一份算力，`ALGORITHM_SET`才是科学交付；允许两到三条分别通过的
算法同时保留。
