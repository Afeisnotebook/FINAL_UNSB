# FINAL_UNSB 全量论文算法数学关系图

状态：`PRE-RESULT THEORY BOUNDARY FROZEN`
适用范围：full-data、seed 2026、batch 1、200 data epochs 的 Proposal-only、ST-CGR 与
AM-TNC
结果入口：`PROJECT_STATE.json` 与 `configs/PAPER_DELIVERY_COMPLETION_MATRIX.json`

这份文件只冻结公式、作用对象和可声明边界，不读取或预判任何 full-data 中间性能。
它不改变训练协议，也不把三条在飞算法合并成一个未经实验的组合方法。

## 1. 共同随机博弈表示

一次原生 UNSB 更新按 `D -> E -> joint G/F` 顺序提交。令 $b_k$ 为一次官方 batch-1
无配对样本，$S_k^{DE}$ 为 D、E 提交后的实际网络与优化器状态。在这个边界上，把一次
G/F 随机视图写成

\[
g_{I,\omega}=g_{GF}(S_k^{DE},b_k;I,\omega),
\]

其中 $I\in\{0,\ldots,T-1\}$ 是均匀 bridge-time index，$\omega$ 汇总 endpoint latent、
rollout noise、PatchNCE latent 与 patch sampling 等
给定 batch 后的原生随机量。定义

\[
\mu_i=\mathbb E_{\omega}[g_{i,\omega}],\qquad
\bar\mu=\frac1T\sum_i\mu_i,
\]

\[
\bar\Sigma=\frac1T\sum_i\operatorname{Cov}_{\omega}(g_{i,\omega}),\qquad
V_\mu=\frac1T\sum_i(\mu_i-\bar\mu)(\mu_i-\bar\mu)^\top.
\]

下述“无偏”全部只指：给定各 player 实际优化状态与官方 batch 后，提交给原生 Adam 的
`pre-Adam` 梯度估计器保持父目标的条件均值。它不等于期望 Adam 参数位移不变，也不等于
完整顺序博弈的有限步转移、RNG 轨迹或训练分布不变。

这里还隐含以下最小概率条件：每个所用梯度具有有限二阶矩；标记为 iid 的两个视图在给定
状态与 batch 后条件独立同分布；标记为 exchangeable 的两个 replica 具有交换不变的联合
分布。所有结论均针对冻结的源码采样顺序，而不是任意实现了“两次 forward”的近似版本。

## 2. Proposal-only：选择性降低 post-D/E G/F 条件方差

Proposal 保留原生单视图 D/E，待 $S_k^{DE}$ 已形成后，独立抽取两个完整 G/F 视图：

\[
\widehat g_P=\frac12(g_{I_1,\omega_1}+g_{I_2,\omega_2}),
\quad (I_1,\omega_1),(I_2,\omega_2)\stackrel{iid}{\sim}\mathcal P_{GF}.
\]

因此

\[
\mathbb E[\widehat g_P\mid S_k^{DE},b_k]=\bar\mu,
\qquad
\operatorname{Cov}(\widehat g_P\mid S_k^{DE},b_k)
=\frac12(\bar\Sigma+V_\mu).
\]

它减少的是给定同一 A/B identity 后的随机协方差，不包括跨 batch identity 或域抽样方差。
它对 G/F 只执行一次 Adam step，但需要一个原生 D/E 视图和两个新的 G/F 视图。收益若成立，
必须同时报告额外网络计算量；不能写成等 FLOP 优势。

Proposal 的核心选择性不是“所有 player 都降方差”。历史 full PC-RSMG 同时改变 D/E 噪声，
从而改变 G/F 所面对的内生 $S_k^{DE}$ 分布；它与 Proposal 是不同有限步随机博弈。

## 3. ST-CGR：在 Proposal 之上去除重复 time stratum

ST-CGR 保持 Proposal 的 post-D/E 两视图和一次 G/F Adam commit，但把两个 time index 的
联合分布从 iid-with-replacement 改成均匀有序无放回：

\[
\Pr(I_1=i,I_2=j)=\frac1{T(T-1)},\qquad i\ne j.
\]

两个坐标的边际仍严格为 (1/T)，所有非 time 随机量仍按视图独立抽取，所以

\[
\mathbb E[\widehat g_S\mid S_k^{DE},b_k]=\bar\mu,
\]

\[
\operatorname{Cov}(\widehat g_S)
=\frac12\bar\Sigma+\frac{T-2}{2(T-1)}V_\mu
=\operatorname{Cov}(\widehat g_P)-\frac1{2(T-1)}V_\mu.
\]

相对 Proposal，它只保证删除 between-time conditional-mean covariance 的一个半正定部分；
当 $V_\mu=0$ 时，两者协方差相同。ST-CGR 不改变 time 边际、不做 importance weighting，
也不增加相对 Proposal 的网络视图数。

这一协方差式要求：给定两个被抽中的 time index 后，两份非 time 随机量条件独立，且第
$i$ 个 stratum 的噪声协方差确为上文定义的 $\operatorname{Cov}_\omega(g_{i,\omega})$。
当前冻结实现逐视图独立抽取这些随机量并通过门禁；若未来共享 latent、bridge noise 或
PatchNCE sampling，上式就不能直接沿用。对当前 $T=5$，公式具体化为

\[
\operatorname{Cov}(\widehat g_S)=\frac12\bar\Sigma+\frac38V_\mu,
\qquad
\operatorname{Cov}(\widehat g_P)-\operatorname{Cov}(\widehat g_S)=\frac18V_\mu\succeq0.
\]

ST-CGR 的证据来源是已审核的 time-stratum 梯度异方差。终端低方差/奇异值漂移没有通过
预注册的跨算法、跨域门，因此不得把 ST-CGR 写成“终端奇异漂移修复”，也不得借这个叙事
加入一个未验证的 terminal module。

## 4. AM-TNC：保留 Adam 度量下的切向 replica 分歧

AM-TNC 在 D、E、G/F 各 player 的实际状态上抽取两个条件可交换的梯度 $g_1,g_2$。令

\[
m=\frac{g_1+g_2}{2},\qquad d=\frac{g_1-g_2}{2},
\]

并冻结提交前 Adam 二阶矩给出的正对角算子

\[
A=\operatorname{diag}\left((\sqrt{v_{prev}}+\epsilon)^{-1}\right).
\]

若 \(\|Am\|>0\)，定义

\[
c=\frac{\langle Am,Ad\rangle}{\|Am\|^2},\qquad
\widehat g_A=m+d-cm.
\]

于是

\[
A(\widehat g_A-m)=Ad-\operatorname{Proj}_{Am}(Ad),
\]

即只删除 replica disagreement 在共识方向上的径向部分，保留其 Adam 度量切向部分。
交换两个 iid replica 时 $m$ 不变、$d,c$ 同时变号，因而

\[
\widehat g_A(g_1,g_2)+\widehat g_A(g_2,g_1)=2m.
\]

在交换性条件下，AM-TNC 的 pre-Adam estimator 仍具有原生条件均值。若两个梯度逐位相同，
实现直接返回第一份梯度；配置为单 replica 时则逐位 dispatch 到 plain。

零共识分支也必须单独说明：由于 $A$ 为正对角算子，$\|Am\|=0$ 蕴含 $m=0$。冻结实现此时
提交有序第一份梯度 $g_1=d$；交换 replica 后提交 $g_2=-d$，所以交换对的平均仍为
$0=m$。无偏证明要求 $A$ 只由提交前 optimizer state 决定、对 replica 顺序可测且不变，
以及两份梯度可积。当前源码在计算两份梯度前冻结 `exp_avg_sq` 与 epsilon，满足该条件。

上述交换恒等式在实数算术中精确成立。除“两个 replica 逐位相同”这一显式 identity 分支外，
通用浮点交换测试只应声称在预注册数值容差内成立，不能写成任意硬件上逐位相同。

AM-TNC 不是 Proposal/ST-CGR 的“第三个方差平均版本”：它对 D/E/G/F 全部 player 生效，
使用两套 pre-opponent 视图和 D/E 提交后的新 G/F 视图，并有意保留随机切向分量。这里的
$A$ 是冻结的 pre-step 几何，不包含当前梯度对 Adam 新二阶矩的非线性影响；因此不能声称
期望 Adam 位移无偏，也不能把它叫作对真实 Adam step 的精确正交投影。

## 5. 三条 full-data 算法的非重复关系

| 方法 | 修改 player | 两视图耦合 | 直接处理的对象 | 相对父估计器的严格性质 | 每 update 随机 forward 视图 |
|---|---|---|---|---|---:|
| Proposal-only | G/F only | 完整视图 iid | 给定 post-D/E 状态和 batch 的全部 G/F 视图方差 | 条件均值保持；协方差减半 | 3 |
| ST-CGR | G/F only | time 无放回，其他随机量独立 | Proposal 中 between-time 条件均值方差 | 条件均值保持；相对 Proposal 减少 $V_\mu/[2(T-1)]$ | 3 |
| AM-TNC | D、E、G/F | 完整视图 iid + 交换反对称算子 | Adam 度量下 replica disagreement 的径向/切向几何 | 交换平均回到两视图共识；保留切向分歧 | 4 |

这里的 3 个 Proposal/ST-CGR 视图是“1 个 D/E 原生视图 + 2 个 post-D/E G/F 视图”；
AM-TNC 的 4 个视图是“2 个 D/E 视图 + D/E commit 后 2 个 G/F 视图”。它们保持 optimizer
step 数和 batch identity 暴露量，但不保持相同 FLOP。

三条路线可以共享“顺序随机博弈中的条件 stochastic-measure design”这一研究母题，
但在 full-data 终点到达前不能预先写成一个统一成功算法：

- Proposal 是 player-selective conditional averaging；
- ST-CGR 是在该 estimator 上增加的 time-coupling 结构；
- AM-TNC 是独立的 all-player stochastic geometry 路线，不是 ST-CGR 的组件消融；
- HJCGR 是重要 small25 父证据与相关族成员，当前未运行 full-data，`deferred` 不等于
  mechanism falsified。

## 6. 论文主张分级门

### 现在已经可以主张

- 公式到冻结源码、执行顺序、恢复状态和 target 不可访问边界已经审核一致；
- Proposal、ST-CGR、AM-TNC 各自具有上述 pre-Adam 条件均值性质；
- ST-CGR 相对 iid Proposal 的协方差差是半正定项；
- terminal singular-drift 假设目前没有达到算法生成门。

### 只有完整 e200 与合法 matched control 后才能主张

- 任一方法在 full-data 上获得 sustained PSNR/SSIM/LPIPS 收益；
- Proposal 与 ST-CGR 谁更优，以及 time stratification 是否提供额外经验收益；
- AM-TNC 的切向保留是否比全平均更适合长期训练；
- 算法集合包含一条、两条还是三条 full-data 可行方法。

Proposal/ST-CGR 的跨宿主 matched 关系必须等未来 5090B plain 完成严格 runtime twin、
Git registry 审核和统一 CRN 评估；AM-TNC 只使用其 4090A 同宿主 plain。没有合法关系时，
只能报告绝对轨迹，不能用跨宿主 delta 代替。

### 即使 e200 为正也不能自动主张

- 多 seed 稳定性；
- 期望 Adam 参数位移或完整训练 Markov kernel 无偏；
- 与 plain 等计算预算；
- 终端奇异漂移已被解决；
- 中间最佳 checkpoint 代表最终算法；
- 一个行动优先项是唯一科学赢家。

## 7. 结果依赖的诚实论文路由

- Proposal、ST-CGR 均通过：主线可写成 player-selective conditional averaging，并把
  stratified time coupling 作为有额外完整证据的结构化扩展；是否声明 ST-CGR 增益仍需
  合法共同 control 与逐图 CRN 比较。
- Proposal 通过、ST-CGR 未通过：保留 Proposal；结论是 time 去重复在当前 full-data
  operator 中没有转化为长期收益，不否定条件双视图机制。
- Proposal 未通过、ST-CGR 通过：time coupling 是必要经验组件；不得把 Proposal 的
  small25 结果外推成 full-data 成功。
- AM-TNC 通过：作为独立优化几何贡献报告，不把它与 Proposal/ST-CGR 的收益相加。
- 三条均未通过：论文只能报告当前 operator 的长期负结果与因果审计，不能通过最佳
  checkpoint、退出窗口或 confirmation 调参制造成功。

## 8. 权威证据

- Proposal derivation：`research/local_route1/derivation_cards/G1-02B-PLAYER-CONDITIONAL-RSMG.json`
- ST-CGR derivation：`research/local_route1/derivation_cards/G4-01-STRATIFIED-TIME-CONDITIONAL-GF.json`
- AM-TNC derivation：`research/local_route1/derivation_cards/G2-01-ADAM-METRIC-TANGENTIAL-CONSENSUS.json`
- Proposal formula/source audit：`evidence/paper_aio/PROPOSAL_FULL_DATA_FORMULA_IMPLEMENTATION_AUDIT_20260903T0754.json`
- ST-CGR formula/source audit：`evidence/paper_aio/STCGR_FULL_DATA_FORMULA_IMPLEMENTATION_AUDIT_20260903T074759.json`
- ST-CGR independent operator audit：`evidence/paper_aio/STCGR_INDEPENDENT_OPERATOR_SEMANTIC_AUDIT_20260903T102300.json`
- AM-TNC formula/source audit：`evidence/remote_route1_offload/AMTNC_FORMULA_IMPLEMENTATION_AUDIT_20260830.json`

实时训练进度、runtime relation 与最终 disposition 不在本文件硬编码，以免静态数学文档
覆盖后来完整证据；它们始终由 `PROJECT_STATE.json`、完成矩阵和 source-bound receipts 决定。
