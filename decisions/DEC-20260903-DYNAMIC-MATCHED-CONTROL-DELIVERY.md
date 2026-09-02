# 取消5090A plain后的动态matched-control交付

日期：2026-09-03

## 实时裁决

新增5090C后的训练分配保持不变：4090A继续plain并在e200后自动接AM-TNC；5090A只跑
ST-CGR；5090B继续CUT与CycleGAN共驻并在CUT e200后从fresh e0启动plain；5090C继续
Proposal。四条正在训练的lane和所有导出/健康监督均保持连续。

这一分配同时优化两个时间目标：Proposal不再排在4090A plain之后，最稳算法结果提前；
ST-CGR和AM-TNC分别保留采样估计与优化几何两条独立前沿。此时给5090C增加第二条训练会
直接拖慢Proposal关键路径，而没有另一条已完成全部门禁、且能在租期内闭环的算法，因此不做
为了占满GPU的共驻。

## 发现的交付依赖缺陷

已部署的unified evaluator和final-delivery successor冻结于取消5090A plain之前，均把
`plain=5090A`写入不可变contract。5090A plain现已按用户指令停在e9，且自动恢复链已撤销，
所以这两个旧waiter虽然健康，却不再是可完成的交付路径。

代码中的final-delivery contract升级为v2：

- first-wave每条lane的来源由部署时的`--lane-source lane=host`显式冻结；
- ST-CGR来源由`--stcgr-source-host`冻结；
- 最终portfolio不再写死matched plain，而是从e150/e175/e200三点的已审核运行时关系推导；
- Proposal只接受same-source或标准exact cross-host关系；ST-CGR只接受cross-code candidate
  gate；host、CRN与三点关系任一不一致都会fail closed。

实现提交为`1d424c0fe0f0f913cf3a34808e054fd8e84d7ab0`。

## 安全切换顺序

1. 等5090B fresh-e0 plain产生exact runtime receipt；
2. 现有两个metric-blind successor分别产生Proposal和ST-CGR的review-only关系候选；
3. 由Codex审查候选并把精确关系作为独立Git提交写入registry；
4. 从该已审核commit部署新的unified evaluator，来源固定为
   `plain=5090B_MATCHED_PLAIN, proposal=5090C, cut=5090B, cyclegan=5090B`；
5. 新successor通过两次heartbeat后，才退役旧的5090A-plain evaluator/final waiter。

这一顺序确保任何时刻都有持久交付链，同时不会把尚未产生的关系当作既成事实。旧waiter不
读取性能值、不占训练GPU，也不会影响正在运行的实验。

## 科学边界

- 5090C与5090A已有2000-step exact relation，但5090A没有e200 plain，因此当前不能计算
  Proposal matched delta。
- 5090B将来能否同时控制Proposal和ST-CGR，取决于两份独立关系候选和Git审核，不靠GPU型号
  或口头传递性。
- confirmation20继续封存；不根据中间paired结果调度、早停或改算法；主表仍使用e200，
  sustained仍使用e150/e175/e200。
- HJCGR保持deferred，DDSB保持reproduction incomplete；都不写成机制证伪。
