# FINAL_UNSB 相关工作与新意边界

状态：`PRE-RESULT PRIMARY-SOURCE COLLISION AUDIT`

冻结时间：2026-09-06

适用对象：Proposal-only、ST-CGR、AM-TNC，以及尚未由证据批准的 terminal 路线。

这是一份投稿前的主张边界，不是“检索不到相同标题即证明新颖”的法律或穷尽性结论。
它只使用论文主页、论文原文和作者仓库等一手来源；full-data 性能仍未知，本文不读取任何
中间 paired 指标，也不改变正在运行的训练。

## 1. 必须面对的最近邻工作

| 工作 | 已发表/公开的核心对象 | 与本项目的真实交集 | 必须保留的边界 |
|---|---|---|---|
| [UNSB, ICLR 2024](https://openreview.net/forum?id=uQBW7ELXfO) / [作者代码](https://github.com/cyclomon/UNSB) | 用顺序 adversarial sub-problems 学习无配对 neural SB；作者也报告过 NFE 过多时的 over-translation | 本项目的 canonical 基座和顺序 `D -> E -> G/F` 博弈 | 所有方法都必须写成 UNSB 上的训练估计器/优化几何扩展，不能把原生 bridge 或多步 refinement 当作新贡献 |
| [Improved DDPM, ICML 2021](https://proceedings.mlr.press/v139/nichol21a.html) / [作者代码](https://github.com/openai/improved-diffusion) | 用 loss-second-moment importance sampler 降低 VLB 训练的 timestep gradient noise | 证明“time 不同导致梯度噪声，改变采样能降方差”早已有先例 | ST-CGR 不能宣称首次发现 time 异方差或首次用采样降 diffusion 梯度方差；它的可检验差异是**不改变 uniform time 边际**的二视图无放回耦合 |
| [Adaptive Non-uniform Timestep Sampling](https://arxiv.org/abs/2411.09998) | 在线学习非均匀 timestep sampler，明确研究跨 time 的梯度方差与子问题耦合 | 与 ST-CGR 共享 time-stratum 训练难度这一动机 | 该工作改变单次 time 分布并学习 sampler；ST-CGR 固定每个 replica 的 uniform marginal，不得写成 adaptive time weighting |
| [CARV, 2026 preprint](https://arxiv.org/abs/2605.21489) | 对 frozen diffusion teacher 的下游 Monte Carlo 梯度做 amortized resampling、importance sampling 和 stratified inverse-CDF；同时报告降方差未必改善 FID | 是 Proposal/ST-CGR 在“复用条件状态、增加噪声视图、无偏分层”层面的最近通用碰撞 | 不能把 iid averaging、复用上游计算、unbiased stratification 或“方差下降不保证质量”本身当新意；差异必须落在在线 UNSB 顺序博弈、post-D/E player boundary、完整 G/F view 与 e200 长期因果验证 |
| [Differentiable Antithetic Sampling](https://arxiv.org/abs/1810.02555) | 用边际正确的负相关 Monte Carlo 样本降低随机梯度方差 | 覆盖了“相关但仍无偏的两样本”这一通用原则 | ST-CGR 的新意不能建立在一般 antithetic/without-replacement 原理上；必须明确有限 bridge-time 与 UNSB player-conditional 实现 |
| [PCGrad, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) 与 [CAGrad, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html) | 对不同任务目标的冲突梯度做 Euclidean 投影或在平均梯度邻域求 conflict-averse 方向 | 与 AM-TNC 共享“直接修改梯度方向”的表面形式 | AM-TNC 的两个向量是**同一 player、同一状态、同一 batch 的 exchangeable replicas**，不是两个任务；它删除 frozen Adam metric 下 disagreement 对 consensus 的径向分量，不能称为首次 gradient surgery |
| [Stochastic gradient manipulation convergence audit, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f91bd64a3620aad8e70a27ad9cb3ca57-Abstract-Conference.html) | 说明用瞬时 stochastic gradients 决定组合权重的多目标方法可能不收敛 | 是 AM-TNC 不得从单步几何越界到全程收敛主张的直接警示 | exchange symmetry 只证明 pre-Adam 条件均值；不能据此声称 Adam displacement、完整 Markov kernel 或长期收益无偏 |
| [DDSB, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/039c30e9af8039fbd1b58da9d04f38e9-Abstract-Conference.html) | 直接面向无配对 restoration，以 degradation-aware OT 与 dynamic transport consistency 减少迭代误差和早期细节损失 | 是论文主任务最接近的已发表外部对手 | 必须在 related work 和主实验中正常处理；当前无权威公开实现，故保持 `reproduction_incomplete`，不得猜测复现或虚构数字 |
| [NADB, 2026 preprint](https://arxiv.org/abs/2605.28962) | 研究 paired diffusion bridge 的 target-endpoint underfitting；改变 interpolant/target，并用 posterior-mean network 做方向对齐 | 与用户提出的“低方差末段导致 variance/direction drift”高度相关 | 不得隐去；但它用 paired target、改变 bridge law/回归目标，Proposal/ST-CGR/AM-TNC 均不做这些，因此不能声称在不改桥的情况下等价实现 NADB |
| [SDDBM, 2026 preprint](https://arxiv.org/abs/2608.08594) | 将 hard terminal condition 改为非退化 Gaussian terminal marginal，以避免 terminal-boundary singularity | 覆盖更严格的“硬端点导致 drift ill-conditioning”理论叙事 | 本项目三条在飞方法不改变 endpoint law，不能声称解决该类奇异性；若未来采用 soft endpoint，必须作为新路线显式引用、重新推导和重新门禁 |
| [DBIM, ICLR 2025](https://openreview.net/forum?id=eghAocvqBk) 与 [Consistency Diffusion Bridge, NeurIPS 2024](https://openreview.net/forum?id=FFJFGx78OK) | 在 paired denoising diffusion bridge 上改变采样过程或蒸馏 consistency function，重点是少 NFE 推理 | 与本项目的 bridge/time 术语接近，但作用阶段不同 | 本项目当前贡献是训练时 stochastic measure/geometry，不是 fast sampler；NFE=1--5 只作固定评估，不能宣称推理加速贡献 |

投稿前还应追踪两个2026年8月公开的概念近邻：
[denoising score-matching loss floor 的 conditional-variance/Fisher 分解](https://arxiv.org/abs/2608.23916)
与 [Bridge Graphical Models 的 Markovization gap](https://arxiv.org/abs/2608.19144)。它们使“bridge
中存在条件方差/压缩缺口”不能作为宽泛首创主张；本项目若成功，贡献必须由具体 UNSB
operator、严格状态边界和长期受控实验来承担。

## 2. 三条方法可以写什么

### Proposal-only

若 e200 与合法 matched plain 通过，可以主张：在在线、无配对、顺序 adversarial UNSB
博弈中，先让 D/E 形成真实 `post-D/E` 状态，再仅对 G/F 抽取两个条件 iid 完整视图并在
pre-Adam 合并，是一个 player-selective estimator；它与“所有 player 都扩大 batch/平均”
具有不同的有限步状态分布。论文价值来自该状态边界、历史反事实和长期结果的共同证据。

不能主张首次 Monte Carlo averaging、首次 diffusion variance reduction、等计算量、完整
Adam update 无偏或跨 seed 稳定。

### ST-CGR

若 Proposal 与 ST-CGR 都有合法 e200 对照，可以主张：在 Proposal 的同一 `post-D/E`
operator 上，把两个 T=5 的 time index 从 iid-with-replacement 改为 ordered uniform
without-replacement；每个 replica 的边际不变，并相对 Proposal 精确删除
`V_mu/[2(T-1)]` 的 between-time conditional-mean covariance。经验贡献是这一个可隔离
增量是否在真实200 data epochs转化为长期收益。

不能主张首次 timestep stratification、首次无偏分层、adaptive time sampling、terminal
singularity 修复，或仅凭协方差半正定下降推导 PSNR 改善。

### AM-TNC

若同宿主 e200 对照通过，可以主张：对 D/E/GF 各 player 的 exchangeable stochastic
replicas，在提交前冻结的 Adam 对角度量内，把 disagreement 分解为 consensus 的径向和
切向部分；交换两个 replica 后修正项反对称，从而保持 pre-Adam 条件均值，同时区别于
纯平均掉全部 disagreement。

不能主张首次 gradient projection、multi-task conflict resolution、真实 Adam step 无偏、
一般收敛保证，或把一次随机 tangent 保留解释为确定的“好方向”。

## 3. terminal 现象的处理原则

NADB 与 SDDBM 说明 endpoint underfitting/terminal singularity 是必须公开讨论的既有方向，
不能为了让方法显得独立而换名隐藏。当前 target-blind terminal audit 没有达到预注册的
跨算法、跨域先行门，因此正确处理是：

1. 不给 Proposal、ST-CGR 或 AM-TNC 追加 terminal module；
2. 不把三者描述为 NADB/soft-endpoint 的隐式版本；
3. 若最终某方法在 time-index 4 或 NFE5 更稳，只能先报告为 estimator-level empirical
   association，除非新的 target-blind 因果审计证明它确实先于并解释性能变化；
4. 若未来证据支持 terminal-law 修改，必须另立 derivation card，显式说明与 NADB/SDDBM
   的差异，并从 e0 重新运行，不能把它偷偷融合进当前方法。

这并不削弱当前算法：它把论文问题收紧为“在不改变 unpaired UNSB bridge law 与 endpoint
law 的条件下，顺序 player-conditional stochastic estimator/geometry 能否改善长期优化”。

## 4. 投稿主张模板与失败路由

- 所有三条都通过：写成一个共同问题下的两个 estimator 层级（Proposal、ST-CGR）和一条
  独立 stochastic-geometry 路线（AM-TNC），不强行合并为单冠军。
- 只有 Proposal/ST-CGR 通过：核心是在线顺序博弈中的 player-selective conditional
  measure design；必须把 CARV、importance/non-uniform sampling 与 generic antithetic
  sampling 列为近邻并说明状态边界差异。
- 只有 AM-TNC 通过：核心是 same-player replica geometry；必须与 PCGrad/CAGrad 和随机
  gradient manipulation 的收敛限制正面对照。
- 三者都失败：不能用公式新意替代结果；保留负结果、长期因果图谱和 terminal 审计，禁止
  改用最佳 checkpoint 或隐藏既有 terminal work。

## 5. 提交前强制复核

在论文方法与摘要冻结前，必须再做一次最新版检索，并逐条核对：

- “first / novel / unbiased / stable / solves”每个词是否有对应定理和实验门；
- CARV之后是否出现直接面向在线 diffusion/SB training 的同状态分层估计器；
- AM-TNC是否已有完全相同的 exchange-antisymmetric Adam-metric operator；
- NADB/SDDBM是否已有正式会议版本或新的 unpaired 变体；
- DDSB是否发布权威源码，若有则重新打开复现门，而不是继续沿用“无源码”。

本文件只能防止已知越界，不能替代投稿前的新颖性检索或审稿判断。
