# DEC-20260831：相关多算法族与非排他性交付

状态：`ACCEPTED_AND_RUNNING`

## 覆盖的旧字段

本决策覆盖此前把终局科学结论收缩为“唯一冠军算法”的解释，但不覆盖：

- seed2026、small25、batch1、共同e0、真实200 data epochs；
- paired target不得进入公式或训练控制；
- confirmation20封存；
- 跨宿主delta不合并、cross-runtime不冒充cross-seed；
- `CANDIDATE.json`作为兼容行动接口。

## 科学判断

当前证据不要求真实贡献必须收缩成一个名字。严格的终局对象可以是两到三条彼此关联、
动机呼应且分别完成e200裁决的算法。单一action priority只回答“下一步先运行谁”，不回答
“其余机制是否应从论文或后续研究中删除”。

已有4090证据给出三个可以继续构造而非盲目调参的事实：

1. PC-RSMG proposal-only在原生UNSB目标上取得严格长程正结果；
2. HNEK是正的physical-horizon bridge父机制，且在plain/自身状态的审计中持续；
3. HJ是正的结构投影PatchNCE父机制，同时22/22 independent-batch与22/22
   latent/time行均由方差主导；next-batch cosine 22/22为负，因此不授权阈值控制器。

由此冻结一个相关算法族。共同算子是在原生D/E提交后，对当前父目标抽取两个条件独立的
G/F视图并取均值：

\[
\hat g_{2}(S)=\frac{1}{2}\left(g(S;\xi_1)+g(S;\xi_2)\right),
\qquad \xi_1\perp\xi_2\mid S.
\]

对每个父状态与父目标分别有：

\[
\mathbb E[\hat g_2\mid S]=\mathbb E[g\mid S],\qquad
\operatorname{Cov}(\hat g_2\mid S)=\frac12\operatorname{Cov}(g\mid S).
\]

因此三个成员共享估计器动机，但不是同一算法的超参网格：

- `ABL-G1-02B-PCRSMG-PROPOSAL-ONLY`：原生UNSB场；
- `G3-01B-PHYSICAL-HORIZON-CONDITIONAL-GF-RESAMPLING`（HPCGR）：
  HNEK physical-horizon bridge game；
- `G3-02-HJ-CONDITIONAL-GF-RESAMPLING`（HJCGR）：
  HJ structure-projected PatchNCE目标。

`G2-01-ADAM-METRIC-TANGENTIAL-CONSENSUS`（AM-TNC）保留为独立机制成员。它不是上述
相关族的重复，也不因其LPIPS护栏脆弱而被事后删除。

## 计算量归因边界

已完成的`observable_only`消融执行额外随机视图计算，但仍只提交原生梯度。其e200
dynamics-state hash与plain精确一致，late-three和e200 matched delta均为0。相比之下，
proposal-only提交G/F双视图均值并取得长期正结果。因此观察计算或墙钟副作用不能解释收益；
收益必须经过提交估计器的协方差/有限步耦合变化产生。但该算法仍需要额外采样计算，不能据此
声称与原生UNSB算力预算等价，论文阶段仍需报告计算成本。

同一e200消融还给出player-scope边界：仅在post-D/E状态复制G/F时，late-three为
`+0.541507 dB`且e200为`+0.451092 dB`；D/E/G/F全复制虽把late-three提高到
`+0.620959 dB`，e200却回到`-0.001379 dB`。因此收益不是“降方差越多越好”，而是
保留原生D/E对抗随机性、选择性降低joint G/F条件方差。这一结论解释相关族为何统一采用
G/F-only算子。

上述无偏结论的数学对象是固定已实现父状态下、进入Adam之前的随机梯度估计器。Adam只在
均值之后执行一次；不声称期望有限步Adam位移或随机样本路径与单视图父算法相同。

进一步的同宿主因子分解把“重新采样”和“方差缩减”区分开来。只保留一次fresh post-D/E
G/F view的PCNR，late-three为`-0.532685 dB`、e200为`-0.030690 dB`；相同player顺序下
改为两个fresh G/F view的pre-Adam均值后，proposal-only分别为`+0.541507 dB`和
`+0.451092 dB`。两条共同e0完整轨迹之差为late-three `+1.074192 dB`、e200
`+0.481782 dB`。因此，在已测试的原生场算子中，解除D/E到G/F的随机复用本身不足以严格
通过；关键证据指向选择性G/F双视图均值。该轨迹差不是单路径内的可加因果量，也不证明
所有可能算法都必须使用两视图。

方差定理还必须按正确的条件域阅读。两次G/F view共享同一官方unpaired batch；条件
sigma-field包含已实现post-D/E模型/优化器状态、该A/B batch及父控制器状态。被减半的是
给定batch后的latent、bridge time、bridge noise和PatchNCE feature sampling等条件
协方差，不是A/B identity、domain或其他跨batch数据采样方差。这个边界不改变正在运行的
算法，只限制论文能够声明的数学对象。

## 为什么当前不机械增加DT变体

DT证据已经驱动过moving covariance-rate barrier、Adam/Euclidean约束几何、
residual-feasible数值修复和rollout-velocity路线。这些当前实现均已完成或形成长程负证据。
算力空闲不构成再次盲目采样同一机制空间的科学授权。该结论是
`closed_current_operator_space`，不是DT父思想的永久证伪；若出现新的数学对象或新的
target-blind因果证据，可另开推导，不用本轮已有结果拟合窗口、强度或退出阈值。

## 资源与裁决

- 4090并行完成HPCGR与HJCGR，使相关族三成员和AM-TNC可以在同一宿主、同一plain下排序；
- 5090继续完成proposal-only、AM-TNC与排队HJCGR，作为独立runtime证据；
- 同卡有限并发上限由吞吐和状态隔离决定，不用剩余显存机械堆满任务；
- 所有候选必须跑到e200，中间paired结果不得早停；
- 最终发布`ALGORITHM_SET.json`，分别列出strict、positive-but-fragile和
  closed-current实现；`ACTION_PRIORITY.json`仅提供下一步默认入口；
- 同时发布与行动优先级严格一致的`CANDIDATE.json`和两个证据排序递补的
  `ALTERNATES.json`。它们是可执行接口，不会把`ALGORITHM_SET.json`收缩为科学唯一冠军；
- 每个算法成员必须绑定复杂度、风险、冻结executor合同和seed2026 e200复现命令。

## 非声明

- 单seed开发结果不声明多seed稳定；
- 跨runtime一致或分歧都不等于cross-seed；
- 相关族成员共享定理不代表它们必然同时取得PSNR收益；
- 若最终只有一条严格通过，仍保留其余完整轨迹和数学失败边界；
- 若有多条严格通过，不人为选一个“科学唯一冠军”。
