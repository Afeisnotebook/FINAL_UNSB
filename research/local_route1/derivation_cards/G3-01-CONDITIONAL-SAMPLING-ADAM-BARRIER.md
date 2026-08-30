# G3-01：条件采样—Adam度量协方差屏障

状态：`CONDITIONAL_PREIMPLEMENTATION_CARD`。只有
`DEC-20260831-ROUTE1-CONDITIONAL-GENERATION3-SYNTHESIS.md`的父项终点门和兼容门都通过后，
才允许生成可执行JSON卡、冻结候选并开始e200；本卡和源码本身不构成长训授权。

## 父证据与修复对象

- 采样父项固定为严格通过的PCNR；若PCNR未严格通过，则只允许使用已经严格通过的
  PC-RSMG proposal-only。二者修复的是D/E提交后G/F估计仍复用旧player view的条件耦合，
  不得彼此叠加。
- 约束父项固定为严格通过的AM-MCRB。它修复的是moving covariance-rate安全半空间在
  Euclidean参数度量下与实际Adam位移不一致的问题。
- 两个父项必须分别有共同e0、同宿主、seed2026、small25、batch1、e200终点receipt；
  当前PCNR与AM-MCRB尚在运行，因此父证据尚未冻结。

## 算子

采样组件在D/E实际提交后形成G/F估计

\[
g_k^R=g_k^{\mathrm{PCNR}}
\quad\text{或}\quad
g_k^R=\frac12(g_{k,1}+g_{k,2}),
\qquad d_k^R=\operatorname{AdamStep}(g_k^R).
\]

若使用两视图父项，AM-MCRB缺陷也在同一对G/F bridge view上用共同latent求值，并取
交换对称的平均

\[
C_k=\tfrac12(C_{k,1}+C_{k,2}),\qquad
a_k=\nabla_\theta C_k=\tfrac12(a_{k,1}+a_{k,2}).
\]

这避免用第二个任意随机视图约束第一个估计，也避免把两个采样父项重复叠加。最后在
Adam后更新的二阶矩所定义的对角度量中求唯一最近可行位移

\[
d_k^*=\begin{cases}
d_k^R,&\langle a_k,d_k^R\rangle\le 0,\\
d_k^R-\dfrac{\langle a_k,d_k^R\rangle}
{\langle a_k,H_k^{-1}a_k\rangle}H_k^{-1}a_k,&\text{otherwise}.
\end{cases}
\]

F只提交一次，EMA teacher在最终G位移后只更新一次。

## 性质与边界

- `pcammcrb_enable=false`必须逐位回到plain，且不创建方法状态。
- 条件采样保持各自已证明的条件原生均值；active barrier有意改变有限步位移，因此整体
  不声称无偏。
- native-like位移已满足半空间或tangent为零时，屏障严格自消隐；没有强度、退火、窗口、
  paired指标、plain输出或最佳checkpoint输入。
- paired target对训练和兼容门均不可访问；e20/e100/e200的1/8/32-step兼容门只读取
  full state、unpaired batch、RNG、梯度、Adam状态和moving teacher。
- 两组件相对plain的校正cosine必须在所有注册状态/步长不低于`-0.2`；任一父项资格失败、
  parent hash改变或组合位移违反屏障，均判`SYNTHESIS_INAPPLICABLE`。

## 明确判死反例与成本

- 若完整e200后AM-MCRB不严格通过，或不存在严格通过的采样父项，本方法不生成。
- 若兼容门任何一行cosine低于`-0.2`，不放松阈值、不换状态、不调强度。
- 通过门后若matched e200持续收益门失败，关闭该两组件算子；不拆成窗口或handoff。
- PCNR父项约增加一次训练forward及MCRB的`m=4`方向探测；PC-RSMG proposal父项约增加
  一个G/F训练view，并对两个view各做`m=4`方向探测。新增恢复状态为sampling provenance、
  moving teacher、屏障计数和最后一次KKT诊断。

