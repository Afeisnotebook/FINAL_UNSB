# 先从这里开始

## 当前状态（2026-08-29）

这是一个重新打开的本地路线一算法发现项目。此前的“四张4090、四条冻结 lane”
执行计划已经暂停，不是当前任务。

- 当前硬件：本地 GTX 1660 6GB。
- 当前目标：从 clean UNSB、DT/HJ/HNEK 及后续机制的历史证据中主动构造一个
  能在真实 200 data epochs 上维持收益的新算法。
- 当前不是：只复现 HJ、验证四个旧算法、寻找退出窗口或准备4090服务器。
- HJ 的角色：第一项时间尺度正对照，因为它历史上 e100 为负、e125--e200
  延迟转正；它不是唯一候选，也不是预定论文算法。

## 为什么此前的“路线一完成”需要重审

此前 small25 的 2400 updates 只有 16 data epochs；full100 的 12000 updates
只有 20 data epochs。历史 HJ 的关键正收益却在 e125 后出现。因此旧门禁只能证明
“当前实现曾在前 16--20 epochs 反转”，不能证明其父机制在 e200 不可能有效。

同类时间尺度问题不只影响 HJ：

- HNEK 有历史 e200 正结果，clean 只延长到约 e20 且多次变号；
- DT 在 clean 短程有正窗口后反转，尚未被当前 deterministic 基座做等价 e200
  裁决；
- PCOA 最长 e16；LBST/PTQ/DCUM/AEB 多数只到 e8 左右；它们的当前实现失败，
  但不能因短门禁自动判死父机制；
- TA_MINIMAL 的直接恢复实现已有 matched e200 负结果，可保留为真正的长程负对照。

## 当前执行顺序

1. 完成 DT/HJ/HNEK 以及 clean canonical 的语义谱系审计。
2. 共享同一 small25 e0，建立 plain、continuous HJ、HNEK、DT 的 e200
   长程锚点轨迹。HJ先跑只是为了校准时间量尺；不以 HJ 结果结束研究。
3. 在 e80/100/125/150/175/200 上建立 target-blind 长程因果图谱，定位每个
   校正何时仍有效、何时改变符号、何时只是方差或博弈状态问题。
4. 从最清楚的失败机理构造新的数学算法；旧名字可以完全消失。
5. 新候选完成真正的 e200 matched local trajectory 后，才讨论是否值得外部算力。

完整边界见 `LOCAL_ROUTE1_RESEARCH_CONTRACT_CN.md`。旧 `configs/FOUR_LANES.json`
只保留为暂停方案的历史记录，不得执行。
