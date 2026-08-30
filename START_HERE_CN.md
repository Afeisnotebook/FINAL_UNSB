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

截至`2026-08-31 01:58 +08:00`，PC-RSMG的proposal-only和observable-only均已通过
完整短GPU门，包括zero-intervention身份、精确resume、跨状态父hash隔离、target-blind
以及400-update有限性。projected/full直接复用原完整e200 receipt；proposal-only现已从
共同e0进入真实e200，结束后才由同一持久后继启动observable-only。完整结果
可以证明收益究竟来自双随机proposal、observable/投影还是二者组合；在两条消融结束前
不会生成最终`CANDIDATE.json`。compact证据见
`evidence/remote_route1_offload/PCRSMG_WINNER_ABLATION_GATES_AND_PROPOSAL_E200_START_20260831.json`。
seed2027/2028继续延期，confirmation20、全量数据、路线二、退出阈值和handoff仍关闭。

“唯一候选”是最终交付的主排名要求，不是提前终止算法发现的规则。5090的空闲算力现已
用于一个严格受限的可信候选前沿：PCNR把D/E提交后的G/F随机视图改为条件原生重采样，
AM-MCRB把MCRB的欧氏投影改为Adam二阶矩度量下的最小安全位移。两者都来自长期近失配
证据而非超参网格，已通过完整GPU门，并从同一5090 e0/plain、batch1、seed2026并行跑
真实e200。代码冻结于`c874e37`，持久训练调度器为`9c8c987`；独立终点裁决器
`47bbb85`只在两个完整receipt出现后进行5090宿主内排名，并且只有严格通过门的第一名
才能生成一条4090复跑请求。4090上的`827183a`后继已经休眠等待：它先等待PC-RSMG
两项消融结束，再读取完整远端决定；若有赢家则重新执行4090 GPU门并从共同e0复跑唯一
一条，否则零复跑退出。compact证据见
`evidence/remote_route1_offload/FRONTIER_GATES_AND_E200_START_20260831.json`。4090与5090
先各自在宿主内matched裁决，绝不直接合并小数级跨宿主delta。

最终交付也不再依赖人工接续：`dc3f982`后继等待上述终点链，先原样归档旧的
`final/CANDIDATE.json`等四个文件，再以4090同宿主裁决写出一个明确canonical主候选、两个
证据备选和完整5090宿主内前沿。主候选只是最终优先级，不是搜索阶段的提前剪枝。若PCNR
或AM-MCRB在4090复跑后胜出，必须先完成该算法自身的proposal-only、observable-only和
projected/full e200证据；不能借用PC-RSMG消融。若没有4090复跑，旧主候选保持不变。
详见`decisions/DEC-20260831-FRONTIER-PRESERVATION-AND-WINNER-ABLATION-BINDING.md`。

终交付本身也已升级到`68dcea0`：它不再只输出主候选摘要，而是强制保留主候选及三项
机制消融的逐域candidate/plain绝对轨迹和delta、4090/5090宿主分离完整前沿、公式与
源码/合同身份，并在最终报告中分别说明科学结论、工程失败、代理失真和未测试假设。
任一终点证据或hash缺失都会拒绝交付。见
`decisions/DEC-20260831-FRONTIER-FINAL-EVIDENCE-COMPLETENESS.md`。

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
3. AM-TNC与MCRB均已完成e200；PC-RSMG是当前fallback但还在完成proposal-only和
   observable-only真实e200机制消融。中间paired分数只作固定描述，不控制代码、调度或退出。
4. 5090同时推进PCNR与AM-MCRB两个证据驱动的新候选到e200。两者不是旧算法网格，
   也不因任何中间checkpoint提前淘汰；完整结果先在5090宿主内与matched plain裁决。
5. 所有轨迹完成后保留一个明确主候选和两个有完整证据的备选；只有5090候选严格通过时
   才考虑4090同宿主复跑，seed2027/2028不自动启动。
6. 任何宿主都不得读取confirmation20、运行全量数据、搜索handoff/退出窗口或把方法
   分数减去另一宿主的plain。

上述长任务均由独立持久后继执行，不依赖Codex会话存活。4090后继固定顺序完成PC-RSMG
的两项机制消融；5090后继并行完成PCNR和AM-MCRB，并在单条失败时保留另一条继续运行。
最终只读取完整e200 terminal receipt作科学裁决。协议见
`decisions/DEC-20260830-WINNER-ABLATION-SINGLE-STREAM.md`和
`decisions/DEC-20260831-ROUTE1-FRONTIER-EXPANSION.md`。

完整边界见 `LOCAL_ROUTE1_RESEARCH_CONTRACT_CN.md`。旧 `configs/FOUR_LANES.json`
只保留为暂停方案的历史记录，不得执行。
