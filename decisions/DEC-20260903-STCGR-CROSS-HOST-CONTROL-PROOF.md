# ST-CGR跨宿主matched control的显式两段证明

日期：2026-09-03

## 背景

5090A plain按用户时间优先授权停在e9，并撤销自动恢复链。ST-CGR继续在5090A运行，
而5090B已有fresh-e0 plain后继。直接把5090B plain称为ST-CGR的matched control并不
严谨：ST-CGR使用独立candidate commit和protocol fingerprint，既有运行时关系只证明
Proposal/5090C与plain/5090A等价，不能靠宿主型号或口头“传递性”替代证据。

## 决策

增加一种独立的cross-host/cross-code candidate-control relation。它只在以下两段证明
同时成立时生成review-only关系候选：

1. candidate runtime gate证明ST-CGR代码在5090A上与parent plain具有相同e0 scientific
   core、相同2000-step native transition，以及zero-intervention逐位identity；
2. 5090B fresh plain的runtime-twin receipt证明其e0 core、2000-step core、parent protocol、
   manifest和归一化运行环境与第一段中的parent plain完全相同。

因此使用的是显式相等链

\[
(e_0,U_{2000},\mathcal E)_{C\rightarrow P_A}
=
(e_0,U_{2000},\mathcal E)_{P_A}
=
(e_0,U_{2000},\mathcal E)_{P_B},
\]

而不是把“同为5090”当作等价。候选authorization、portable authority和已验签metadata
import必须逐哈希绑定candidate gate；任一差异均fail closed。

## 边界

- successor只等待已有Proposal runtime-relation successor发布的已验证plain receipt；
  不再建立第二条SSH信任链，也不读取密码。
- 产物只是一份review-only relation candidate，不修改Git中的关系registry，不授权比较，
  不计算delta，不启动或改变训练。
- 即使runtime relation通过，仍必须等5090B plain完整e200并由后续人工Git审查把关系写入
  registry，ST-CGR的matched评估才可开始。
- 5090B gate失败、租期中断或e200未完成均不能被写成ST-CGR机制失败。
- confirmation20继续封存；paired指标不参与关系建立。

