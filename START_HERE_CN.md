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
- 紧急成本策略：完整seed2026/e200足以冻结开发候选；seed2027/2028延期，释放算力
  用于赢家机制消融、因果数学修订和新的独立方向。任何文件都不得把这写成已证明
  跨seed稳定。

当前真实阶段以`PROJECT_STATE.json`为准：4090的plain/HJ/HNEK/DT和474条长期反转、
140条采样方差因果图谱已经冻结。BVCP与PC-RSMG均已完成e200：BVCP为长期负；
PC-RSMG晚三点仍为正但e200为-0.00138 dB，因此当前实现未通过严格终点门。终点
target-blind审计据此授权唯一二代修订AM-TNC；它正在4090从共同e0运行e200。紧急
单seed协议释放的另一份算力用于独立机制MCRB，它正在5090运行e200。两台宿主的
数值不合并；只有MCRB在5090为正时，才会在4090完成同plain身份的权威复赛。
4090侧已用与5090逐字相同的`7fa9081`身份提前通过MCRB宿主门禁，但尚未启动长训；
因此正结果可立即复赛，负结果不会额外消耗一条4090 e200。

截至`2026-08-30 23:46 +08:00`，AM-TNC到e119，MCRB已完成e125并继续e130；MCRB的真实e100
描述性delta为`+2.158 dB`，但已验证它不会触发早停、4090复赛、排名或冻结，科学硬门
仍是完整e200。最终赢家的projected/full使用其既有完整receipt，不重复训练；4090按
实测总墙钟依次运行proposal-only和observable-only两条e200消融，seed2027/2028仍延期。
最终`CANDIDATE.json`还会绑定真实executor contract的相对路径、hash、训练commit和算法
指纹，不再根据候选名称猜文件名；缺失、歧义或身份不符时拒绝交付。
MCRB的e125描述性delta已反转为`-0.770 dB`：候选绝对值较e100下降约`0.360 dB`，
同宿主plain则恢复约`2.567 dB`。这说明e100优势包含plain低谷效应，但e125仍不是终点，
不授权退出、改算法或机制判死。

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
3. 当前让AM-TNC在权威4090、MCRB在独立5090完成真正的e200；中间paired分数只作
   固定描述，不控制代码、调度或退出。PC-RSMG的冗余5090复现已在e88安全暂停，资源
   让给独立机制。
4. e200后先决定MCRB是否需要4090同宿主复赛，再由source-bound总排名冻结单seed
   开发赢家。projected/full使用赢家已经从共同e0完成的e200 receipt；proposal-only
   和observable-only也从共同e0依次完成真实e200；seed2027/2028不自动启动。
5. 任何宿主都不得读取confirmation20、运行全量数据、搜索handoff/退出窗口或把方法
   分数减去另一宿主的plain。

上述第4步由4090上的持久跨宿主后继执行，不依赖Codex会话存活：它只读取5090完整
e200 terminal receipt；完整正结果才启动4090 MCRB复赛，完整负结果则跳过。它同时等待
AM-TNC e200，最终统一冻结4090同宿主总排名后才启动赢家消融。协议见
`decisions/DEC-20260830-MCRB-CROSS-HOST-DURABLE-ROUTING.md`和
`decisions/DEC-20260830-WINNER-ABLATION-SINGLE-STREAM.md`。

完整边界见 `LOCAL_ROUTE1_RESEARCH_CONTRACT_CN.md`。旧 `configs/FOUR_LANES.json`
只保留为暂停方案的历史记录，不得执行。
