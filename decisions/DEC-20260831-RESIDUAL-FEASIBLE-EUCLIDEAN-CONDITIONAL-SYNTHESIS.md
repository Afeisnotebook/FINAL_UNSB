# DEC-20260831：保留残差可行 Euclidean 条件合成兄弟项

状态：`CONDITIONAL_PREIMPLEMENTATION_NOT_LONG_RUN_AUTHORIZATION`

## 为什么新增这一项

用户明确要求额外算力不要把证据前沿过早压缩成唯一算法。RF-AMMCRB 与 RF-MCRB
不是同一算子的强度超参数：前者在 Adam 二阶矩诱导的对角度量中求最近可行位移，后者
在 Euclidean 参数几何中求最近可行位移。两者都修复了旧实现的固定绝对余量事故，且
RF-MCRB是原始MCRB推导所声称的直接数学算子。因此，若RF-MCRB自身在4090同宿主
matched e200严格通过，只推进Adam合成会无证据地删掉一个独立优化几何。

新身份为
`G3-03-CONDITIONAL-SAMPLING-RESIDUAL-FEASIBLE-EUCLIDEAN-BARRIER`。它保留与G3-02
完全相同的条件采样父项和启动门，只把屏障父项替换为
`F2-02-RESIDUAL-FEASIBLE-EUCLIDEAN-COVARIANCE-BARRIER`。

## 数学边界

在D/E提交后，由严格条件采样父项得到G/F梯度并实现原生Adam位移 `d_R`。在同一随机
测度上计算moving covariance defect的切向量 `a`。若 `<a,d_R> <= 0`，逐位提交
`d_R`；否则解

\[
\min_d \frac12\|d-d_R\|_2^2\quad\text{s.t.}\quad\langle a,d\rangle\le0.
\]

解析系数在float64中计算，实际parameter dtype中的位移必须重新测量可行性；只允许用
`residual / <a,a>` 和相对dtype ULP作最多八次fail-closed细化。不存在固定绝对余量、
强度、epoch窗口、paired阈值、plain输出锚定或最佳checkpoint选择。

这与G3-02的差别只有约束问题的度量 `P=I` 对 `P=H^{-1}`，但该差别会改变投影方向，
属于两个可证伪的优化几何，不是超参搜索。两项及其父算法在科学交付中分别保留。

## 启动硬门

G3-03只有同时满足以下条件才可从共同e0运行真实e200：

1. 4090上的严格条件采样父项完成e200并通过持续收益、域覆盖和感知护栏；
2. 4090上的RF-MCRB完成e200并严格通过同一门；
3. 两个receipt绑定相同e0、protocol、manifest和plain e200权威；
4. e20/e100/e200、1/8/32-step的target-blind组件校正cosine全部不低于`-0.2`；
5. disabled identity、full-state resume、父状态隔离、represented feasibility、
   optimizer顺序和sampling provenance全部通过。

任一门失败只记录`SYNTHESIS_INAPPLICABLE`，不得降低门槛或改用中间paired结果。
G3-03可与已完成冻结门的G3-02在4090上并行，但必须等G3-02先完成共享ledger冻结，
避免并发写入算法谱系。

## 多候选与停止边界

- G3-02、G3-03及其独立父项都保留完整轨迹；canonical `CANDIDATE.json`只表示后续
  行动优先级。
- 不因为小幅e200分差提前只留一个度量，也不在父项不严格时用空闲GPU强行启动。
- 本决策不授权第三种度量、强度网格、多seed、全量数据、confirmation20、route2、
  exit/handoff或跨宿主delta。
- 若G3-03完整e200失败，只关闭这一条件采样—Euclidean屏障组合，不自动证伪两个父机制。

