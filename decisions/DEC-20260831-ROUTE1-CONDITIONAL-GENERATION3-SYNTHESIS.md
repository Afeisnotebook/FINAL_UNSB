# DEC-20260831：路线一条件式Generation-3双机制合成门

状态：`SUPERSEDED_DO_NOT_RUN`。该设计使用的旧 AM-MCRB 固定绝对余量已被
`DEC-20260831-RESIDUAL-FEASIBLE-CONDITIONAL-SYNTHESIS.md`确认不符合注册的尺度无关
KKT 算子。源码仅保留历史谱系；任何后续终点结果都不得重新授权 G3-01。

替代身份为 `G3-02-CONDITIONAL-SAMPLING-RESIDUAL-FEASIBLE-ADAM-BARRIER`，它需要两个
同宿主、source-bound、完整 e200 且严格通过的父 receipt，并使用 residual-feasible
parameter-dtype 投影。

## 目的（历史）

canonical主候选只是当前规范入口，不是算法发现的提前停止点。为合理利用PCNR与
AM-MCRB完成后的5090算力，同时防止重新退化成任意点子或超参搜索，在读取两条终点
paired结果前冻结以下唯一合成入口。

## 进入条件

只有全部满足时才生成一个Generation-3候选：

1. 两个父算子都已从共同e0完成small25、batch1、seed2026、真实e200；
2. 两者分别通过完整持续收益门，而不只是late-three或单一endpoint为正；
3. 两者解决独立状态因素：一个是player-conditional采样/方差机制，另一个是
   moving covariance-rate/Adam度量安全机制；
4. 不组合PCNR与PC-RSMG proposal-only，因为它们属于同一采样机制族；
5. 在冻结e20/e100/e200状态上的target-blind 1/8/32-step兼容审计中，两组件校正
   cosine不得低于`-0.2`，且parent full-state hash保持不变；
6. 合成必须仍能写成一个一致的约束更新，不含窗口、退火、paired阈值、plain输出或
   checkpoint选择。

采样父项按以下固定规则选择：若PCNR严格通过，使用计算更低且保留原生单样本方差的
PCNR；否则只有已经完成且严格通过的PC-RSMG proposal-only可进入。约束父项必须是
AM-MCRB严格通过；MCRB当前失败实现、AM-TNC以及旧DT/HJ/HNEK不进入该合成。

## 条件公式

采样组件先在D/E实际提交后的状态生成G/F估计并更新Adam矩：

\[
g_k^R=\operatorname{PCNR}(S_k)\quad\text{或}\quad
g_k^R=\tfrac12(g_{k,1}+g_{k,2}),
\qquad d_k^R=\operatorname{AdamStep}(g_k^R).
\]

若采样父项为两视图PC-RSMG proposal，则约束缺陷必须在产生上述梯度的同一对G/F
bridge view上计算，并以共同latent作交换对称平均：

\[
C_k=\tfrac12(C_{k,1}+C_{k,2}),\qquad
a_k=\nabla C_k=\tfrac12(a_{k,1}+a_{k,2}).
\]

PCNR只有一个G/F view，因此直接在该view上计算`a_k`。不得另抽一个任意view来约束
已经形成的采样位移。

随后用AM-MCRB当前moving covariance tangent `a_k`和由该实际Adam状态得到的
`P_k=H_k^{-1}`求唯一最近可行位移：

\[
d_k^*=\begin{cases}
d_k^R,&\langle a_k,d_k^R\rangle\le0,\\
d_k^R-\dfrac{\langle a_k,d_k^R\rangle}
{\langle a_k,P_ka_k\rangle}P_ka_k,&\text{otherwise}.
\end{cases}
\]

采样项保持其条件原生均值；约束项在native-like位移已安全或tangent为零时严格自消隐。
整体不声称无偏，因为active barrier有意改变有限步位移，但其偏差由target-blind半空间
和唯一Adam-metric最近点完全定义。

## 执行和停止规则

- 最多生成一个合成候选、最多两个组件、没有可调强度。
- 400--800 updates只验证有限性、identity、resume和事件顺序，不能作科学淘汰。
- 通过兼容门后从同宿主共同e0运行seed2026、small25、batch1、e200；不拼接父checkpoint。
- 任何一个父项未严格通过，或兼容cosine失败，则形成`SYNTHESIS_INAPPLICABLE`证据，
  不用“接近通过”放松门槛。
- 当前PCNR/AM-MCRB训练、4090复跑与赢家自身消融不因本决定停止或改变。
- 条件实现与持久后继已冻结于`95cefe5`；后继只等待完整terminal adjudication，父门不
  满足时写出`SYNTHESIS_INAPPLICABLE`并退出，不会为了占用空闲GPU放松条件。
- confirmation20、seed2027/2028、全量数据、路线二和跨宿主delta仍不属于本门。
