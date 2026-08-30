# 先从这里开始

## 当前状态（2026-08-31）

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

当前真实阶段以`PROJECT_STATE.json`为准：4090的plain/HJ/HNEK/DT、474条长期反转和
140条采样方差因果图谱已经冻结；BVCP、PC-RSMG、AM-TNC均已在4090完成matched e200，
独立MCRB也已在5090完成matched e200。MCRB终点为`-0.730 dB`、3/6域正、最差域
`-2.175 dB`，因此当前实现判负并按合同跳过4090复赛，跨宿主delta从未合并。

AM-TNC给出了目前最清楚的正而脆弱信号：晚三点PSNR均值`+0.105 dB`，e200为
`+0.383 dB`、4/6域正、最差域`-0.239 dB`，终点SSIM和LPIPS也优于plain；但晚三点
平均LPIPS仍差`0.00628`，所以没有通过完整严格门。按预注册的“晚三点PSNR均值优先”
排序，PC-RSMG以`+0.621 dB`排第一，尽管其e200仅`-0.00138 dB`；AM-TNC排第二，BVCP
排第三。因为没有方法通过全部数值门，PC-RSMG只被冻结为seed2026当前最优fallback，
不是已证实的稳定赢家；AM-TNC是正终点递补。

截至`2026-08-31 01:17 +08:00`，4090持久后继已启动PC-RSMG的机制消融。projected/full
直接复用原完整e200 receipt；proposal-only和observable-only先并行完成短GPU门，随后
严格单流、依次从共同e0各跑真实e200。完整结果可以证明收益究竟来自双随机proposal、
observable/投影还是二者组合；在两条消融结束前不会生成最终`CANDIDATE.json`。
seed2027/2028继续延期，confirmation20、全量数据、路线二、退出阈值和handoff仍关闭。

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
