# 先从这里开始

## 当前状态（2026-08-30）

这是一个重新打开的多宿主路线一算法发现项目。此前的“四张4090、四条冻结 lane”
执行计划已经暂停；后来授权的一张4090和一张5090只执行host-matched路线一，不是
恢复旧计划。

- 当前硬件：本地 GTX 1660 6GB、受控 RTX 4090 24GB、受控 RTX 5090 32GB。
- 当前目标：从 clean UNSB、DT/HJ/HNEK 及后续机制的历史证据中主动构造一个
  能在真实 200 data epochs 上维持收益的新算法。
- 当前不是：只复现 HJ、验证四个旧算法、寻找退出窗口、恢复旧四卡四lane或跨主机拼接结果。
- HJ 的角色：第一项时间尺度正对照，因为它历史上 e100 为负、e125--e200
  延迟转正；它不是唯一候选，也不是预定论文算法。

当前真实阶段以`PROJECT_STATE.json`为准：4090的plain/HJ/HNEK/DT和474条长期反转、
140条采样方差因果图谱已经冻结。图谱生成了BVCP与PC-RSMG两个新算法；BVCP冻结当前
实现已完成e200且为长期负，PC-RSMG正在从共同e0运行真实e200。当前工作不是重新跑
lineage或锚点，也不是用BVCP失败结束父机制搜索。

## 为什么此前的“路线一完成”需要重审

此前 small25 的 2400 updates 只有 16 data epochs；full100 的 12000 updates
只有 20 data epochs。历史 HJ 的关键正收益却在 e125 后出现。因此旧门禁只能证明
“当前实现曾在前 16--20 epochs 反转”，不能证明其父机制在 e200 不可能有效。

同类时间尺度问题不只影响 HJ：

- HNEK曾只有历史 e200 正结果、clean短轨迹多次变号；现在4090/5090已各自完成
  host-matched e200，本地轨迹也在接近e200；
- DT现在已在4090 deterministic基座完成e200：晚三点均值为正但终点、域覆盖和护栏
  不合格，因此是`positive_probe_not_candidate`，不是最终算法；
- PCOA 最长 e16；LBST/PTQ/DCUM/AEB 多数只到 e8 左右；它们的当前实现失败，
  但不能因短门禁自动判死父机制；
- TA_MINIMAL 的直接恢复实现已有 matched e200 负结果，可保留为真正的长程负对照。

## 当前执行顺序

1. 已完成DT/HJ/HNEK与clean canonical的语义谱系和200-data-epoch锚点审计。
2. 已在4090冻结target-blind长期因果图谱，并从最清楚的rollout移动与采样方差机理
   构造两项新算法；旧名字没有候选保护席位。
3. 当前让PC-RSMG在权威4090和独立5090完成真正的e200 matched trajectory；中间
   paired分数只作固定描述，不控制代码、调度或退出。
4. e200后由source-bound后继执行排名。正向赢家冻结后做新seed与三项消融；若两项
   全负，先做target-blind终点缺陷和梯度子空间审计，再只允许一次证据授权的数学修订。
5. 任何宿主都不得读取confirmation20、运行全量数据、搜索handoff/退出窗口或把方法
   分数减去另一宿主的plain。

完整边界见 `LOCAL_ROUTE1_RESEARCH_CONTRACT_CN.md`。旧 `configs/FOUR_LANES.json`
只保留为暂停方案的历史记录，不得执行。
