# 先从这里开始

## 当前状态（2026-08-31）

### 12:00 两种残差可行合成都进入条件前沿

用户再次确认：额外5090/4090算力应推进多数有独立证据的算法，唯一`CANDIDATE.json`
不是科学剪枝原则。当前两条5090父修复流约为RF-AMMCRB e71、RF-MCRB e77，双流GPU
约98%，仍未读取中间paired分数。除两条父算法及各自来源绑定消融外，现在预实现并持久
武装了第二个Generation-3合成：
`G3-03-CONDITIONAL-SAMPLING-RESIDUAL-FEASIBLE-EUCLIDEAN-BARRIER`。

G3-02与G3-03不是强度网格。它们固定相同的严格条件采样父项，分别在Adam二阶矩度量和
Euclidean参数几何中求represented-residual最近可行位移；只有对应4090父项自身完整
e200严格通过，且e20/e100/e200、1/8/32-step的target-blind组件兼容门通过时才会长训。
两项可并行，但G3-03等待G3-02先完成共享ledger冻结。实现与4090持久后继提交为
`733645e`，决策见
`decisions/DEC-20260831-RESIDUAL-FEASIBLE-EUCLIDEAN-CONDITIONAL-SYNTHESIS.md`。

实现时同时发现旧G3-02复合类在disabled路径用未进入MRO的修复mixin执行`super()`，会让
zero-intervention门在真正GPU gate处失败。它尚未冻结算法、未启动训练、没有算力或科学
轨迹损失；旧等待器已留档并由`733645e`的修复等待器替换。4090当前两个新后继都只等待
完整repair portfolio，stderr为0。部署证据见
`evidence/remote_route1_offload/RESIDUAL_EUCLIDEAN_SYNTHESIS_4090_SUCCESSOR_ARMED_20260831.json`。

### 11:25 多候选原则与跨宿主持久接力

用户的最新提醒被确认为科学边界而非临时资源偏好：最终唯一`CANDIDATE.json`只决定下一份
算力和论文主线先给谁，不得把尚未完成的可信前沿提前剪成一个算法。5090仍先把
RF-AMMCRB、RF-MCRB跑完共同e0、batch1、seed2026、真实e200；两者完整后，所有属于
strict、near或有独立因果证据的alternate分别保留full和来源绑定消融。已有小幅长期差距
说明排序尚未拉开，不能作为只留第一名的依据；但空闲算力也不能被随机点子或超参网格填满，
新长程流仍须先有target-blind证据、推导卡和工程门。

11:25时两条5090修复流分别到e61/e66，GPU约98%占用且顶层stderr为空。5090完整e200
权威组合到达后，本地持久relay会验证完整性并原字节原子传到4090；4090可从自己的共同
e0并行复跑最多两条算法，不搬checkpoint、不跨宿主合并delta。relay实现提交为`e8356ff`，
compact证据见
`evidence/remote_route1_offload/REPAIRED_PORTFOLIO_DURABLE_RELAY_ARMED_20260831.json`。

### 10:50 修复后双机制合成已预实现，但未获长训授权

5090上的RF-AMMCRB与RF-MCRB已通过e40固定里程碑并进入e45；两个batch1流合计约
6.7GB显存、GPU持续高负载，当前无fatal或非空顶层stderr。中途paired分数没有被读取或
用于路由。

旧`G3-01`仍调用已确认存在固定绝对余量语义事故的AM-MCRB，现已在main上硬性标为
`SUPERSEDED_DO_NOT_RUN`。新身份
`G3-02-CONDITIONAL-SAMPLING-RESIDUAL-FEASIBLE-ADAM-BARRIER`已预实现：只允许一个
同宿主严格通过的条件采样父项与严格通过的RF-AMMCRB组合，并使用实际parameter-dtype
residual与relative-ULP refinement。它还必须通过e20/e100/e200、1/8/32-step的
target-blind组件兼容门；当前只是计算就绪，不构成长训授权，也不替代任何独立父算法。
实现提交为`12468d1`，决策见
`decisions/DEC-20260831-RESIDUAL-FEASIBLE-CONDITIONAL-SYNTHESIS.md`。
持久后继已在4090以`tmux:FINAL_UNSB_RESIDUAL_SYNTHESIS_b164df8`部署，当前只等待修复
算法同机复赛完成；RF-AMMCRB缺席或不严格通过时会留下不适用记录而非启动。部署证据见
`evidence/remote_route1_offload/RESIDUAL_SYNTHESIS_4090_SUCCESSOR_ARMED_20260831.json`。

另一个独立的无偏方向也已先做target-blind固定状态筛查：固定time/patch并把全部Gaussian
latent/bridge noise反号后平均G/F梯度。在plain e20/e100/e200各8 pairs中，反号梯度
协方差均为正，pair-mean方差分别是compute-matched iid pair的1.340/1.029/1.558倍。因此
当前Gaussian反号involution不占用e200槽位；这不影响已经严格通过的iid PC-RSMG
proposal-only，也不证伪其他无偏控制变量。见
`decisions/DEC-20260831-ANTITHETIC-GAUSSIAN-GRADIENT-AUDIT.md`。

### 10:30 双算法跨宿主复验链已武装

“唯一候选”现在只表示下一步行动优先级，不表示只保留一个算法。5090继续同时运行
RF-AMMCRB与RF-MCRB；完整e200后，所有strict/near修复算法（最多两条）会被导出为
源码、公式、训练commit、receipt与trajectory哈希绑定的便携组合。4090的两个原始训练
commit工作树和双流持久执行器已预置，组合到达后从4090共同e0分别重训真实e200；不搬运
checkpoint，不跨主机混算delta。5090上strict、near或有独立证据的alternate仍各自保留
proposal-only/observable-only长程流，因此4090复验队列不会替代5090多前沿搜索。

跨宿主组合、导出守护和4090双流执行器分别冻结在`c1021d8`、`7a66246`和`2c21ea2`，
部署证据见
`evidence/remote_route1_offload/REPAIRED_PORTFOLIO_CROSS_HOST_ARMED_20260831.json`。

### 09:53 多算法前沿不再只给冠军做后续

5090上的RF-AMMCRB与RF-MCRB约到e15/e16，运行正常。新的持久后继已经武装：完整
e200后，两条修复中凡属于strict、near或仍有独立正证据的alternate，最多两条都分别
运行自己的proposal-only和observable-only真实e200；不会因为几百分之一dB的排名差异
只留下第一名。每个父算法内部串行、父算法之间最多两流并行，仍固定batch1、seed2026、
共同e0。唯一canonical候选只代表下一步行动优先级。实现提交为`52c0ad3`，证据见
`evidence/remote_route1_offload/REPAIRED_MULTI_PARENT_FOLLOWUP_ARMED_20260831.json`。
其后的`0b69fff`持久终点后继会统一比较所有full与完成的proposal-only；若某个消融本身
更好，它可以进入行动主项，observable-only则固定排除在候选排名之外。

### 09:27 两条修复算法已启动长期训练

旧AM-MCRB已完成e200，但它的固定绝对余量实现只保留为诊断：late-three
`-0.666038 dB`、e200 `-1.467358 dB`，不能进入科学排名。ledger状态切换与旧普通receipt
生成之间暴露的竞态没有损失任何训练；现在旧MCRB/AM-MCRB都使用完整验证、永久不可排名的
专用诊断receipt。

RF-AMMCRB和RF-MCRB已经分别通过5090 GPU门，从共同e0开始两个batch1、seed2026、
真实e200长程任务，目前约e2/e3。持久裁决器会在PCNR与两条修复线全部完成后保留完整
三算法前沿，再给出一个行动优先级和两个备选。实现与竞态修复提交为`c98f967`。

### 08:53 数值语义事故与第二波修复

完整源码不变量复核发现，G1-03 MCRB和F1-02 AM-MCRB都向解析投影系数加入了与真实
更新尺度无关的固定float32绝对余量。在小切向量的确定性反例中，这会让校正/原生更新
比达到`9.5e9`，违反两张推导卡声称的“唯一最近可行位移”和尺度不变性。这是
`engineering_invalid`，不是paired结果驱动的调参，也不能据旧轨迹判死父机制。

- PCNR已经自然完成5090同宿主e200：late-three为`+0.034821 dB`，但e200为
  `-0.300890 dB`、2/6域正，因此当前算子长期不通过；其完整轨迹仍作为独立采样机制证据。
- 冻结旧AM-MCRB继续原样跑到e200作诊断，当前约e184；所有旧屏障G3、4090复跑和终交付
  自动等待器均已暂停，避免错误公式扩散。
- 新建两个新identity并从共同e0重训：RF-AMMCRB恢复Adam度量最近点，RF-MCRB恢复欧氏
  最近点；两者都只用“解析系数 + 实际表示残差 + 相对dtype ULP”，不含强度、窗口、
  paired阈值或绝对余量。
- 两条修复线分别占5090第二波的两个e200名额；它们比较“数值修复本身”与“数值修复+
  Adam坐标”。这是可信多候选前沿，不是超参网格。最终一个canonical候选仍只是行动优先级。

代码与事故冻结在`1d9f9cb`和`613a10a`，持久screen已武装，完整边界见
`decisions/DEC-20260831-AMMCRB-NUMERICAL-SEMANTIC-REPAIR.md`。

这里的“最终唯一候选”只表示下一步算力行动优先级，不会把搜索阶段提前收缩为一个算法。
修复后5090会同时保留一个主优先级、两个备选和完整同宿主前沿；细则与自动裁决见
`decisions/DEC-20260831-REPAIRED-FRONTIER-PRESERVATION.md`。

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

AM-TNC给出了正而脆弱的长期信号：晚三点PSNR均值`+0.105 dB`，e200为
`+0.383 dB`、4/6域正、最差域`-0.239 dB`，终点SSIM和LPIPS也优于plain；但晚三点
平均LPIPS仍差`0.00628`，所以没有通过完整严格门。PC-RSMG full曾按旧的
“晚三点PSNR均值优先”规则以`+0.621 dB`排在AM-TNC与BVCP之前，但它的e200仅
`-0.00138 dB`。该排名只描述三条full轨迹，已被下述来源绑定消融和严格资格优先裁决
更新，不能再把PC-RSMG full称为当前canonical fallback。

PC-RSMG的proposal-only、observable-only和projected/full现已全部完成真实e200。
proposal-only的late-three为`+0.541507 dB`、e200为`+0.451092 dB`，SSIM/LPIPS
护栏通过；observable-only与plain的完整下一步动力学和e200指标精确一致；full的
late-three为`+0.620959 dB`但e200为`-0.001379 dB`。旧裁决曾直接按late-three排序，
错误地让严格失败的full压过严格通过的proposal-only；现已修正为“先过持续收益门，
再在同资格层排序”。因此当前canonical开发主候选是
`ABL-G1-02B-PCRSMG-PROPOSAL-ONLY`，不是full。完整训练没有重跑或改写，事故与裁决
证据见`evidence/remote_route1_offload/PCRSMG_WINNER_ABLATION_E200_AND_RECOVERY_20260831.json`。
seed2027/2028继续延期，confirmation20、全量数据、路线二、退出阈值和handoff仍关闭。

“唯一候选”是最终交付的主排名要求，不是提前终止算法发现的规则。5090的空闲算力现已
用于一个严格受限的可信候选前沿：PCNR把D/E提交后的G/F随机视图改为条件原生重采样，
AM-MCRB把MCRB的欧氏投影改为Adam二阶矩度量下的最小安全位移。两者都来自长期近失配
证据而非超参网格，已通过完整GPU门，并从同一5090 e0/plain、batch1、seed2026并行跑
真实e200。代码冻结于`c874e37`，持久训练调度器为`9c8c987`；独立终点裁决器
`47bbb85`只在两个完整receipt出现后进行5090宿主内排名，并且只有严格通过门的第一名
才能生成一条4090复跑请求。4090上的持久后继现正等待完整远端决定；若有赢家则重新
执行4090 GPU门并从共同e0复跑唯一一条，否则零复跑退出。compact证据见
`evidence/remote_route1_offload/FRONTIER_GATES_AND_E200_START_20260831.json`。4090与5090
先各自在宿主内matched裁决，绝不直接合并小数级跨宿主delta。

完整e200后也不会机械地只留下第一名：严格通过者、具有单一target-blind可修复缺陷的
近边界方法和已有完整证据的正向递补会保留为可信前沿。当前两条5090流结束后，最多再
并行两条由终点证据授权的合成、来源绑定消融或一次最小数学修订；不会用随机新点子填卡。
详见`decisions/DEC-20260831-EVIDENCE-QUALIFIED-MULTI-CANDIDATE-ADVANCEMENT.md`。

最终交付也不再依赖人工接续：持久后继等待上述终点链，先原样归档旧的
`final/CANDIDATE.json`等四个文件，再以4090同宿主裁决写出一个明确canonical主候选、两个
证据备选和完整5090宿主内前沿。主候选只是最终优先级，不是搜索阶段的提前剪枝。若PCNR
或AM-MCRB在4090复跑后胜出，必须先完成该算法自身的proposal-only、observable-only和
projected/full e200证据；不能借用PC-RSMG消融。若没有4090复跑，旧主候选保持不变。
详见`decisions/DEC-20260831-FRONTIER-PRESERVATION-AND-WINNER-ABLATION-BINDING.md`。

终交付本身已继续加固至`405197d`：它不再只输出主候选摘要，而是强制保留主候选及三项
机制消融的逐域candidate/plain绝对轨迹和delta、4090/5090宿主分离完整前沿、公式与
源码/合同身份，并在最终报告中分别说明科学结论、工程失败、代理失真和未测试假设。
任一终点证据或hash缺失都会拒绝交付。见
`decisions/DEC-20260831-FRONTIER-FINAL-EVIDENCE-COMPLETENESS.md`。

本地1660补充因果审计也已结束：HJ/HNEK共174条反转记录和52条采样方差记录，完整matrix
再次显示状态/epoch依赖和采样方差问题，但没有通过跨方法target-blind控制信号门。本地
proxy未校准，因此没有在本地追加DT或候选；它是代理失真证据，4090的474/140矩阵仍是
canonical因果权威。

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
3. AM-TNC与MCRB均已完成e200；PC-RSMG三项机制消融也已完成，proposal-only是当前
   严格通过的seed2026开发主候选。中间paired分数从未控制代码、调度或退出。
4. 5090同时推进PCNR与AM-MCRB两个证据驱动的新候选到e200。两者不是旧算法网格，
   也不因任何中间checkpoint提前淘汰；完整结果先在5090宿主内与matched plain裁决。
5. 所有轨迹完成后保留一个明确主候选和两个有完整证据的备选；只有5090候选严格通过时
   才考虑4090同宿主复跑。若两个独立机制严格通过，最多生成一个预注册的两组件合成；
   seed2027/2028不自动启动。
6. 任何宿主都不得读取confirmation20、运行全量数据、搜索handoff/退出窗口或把方法
   分数减去另一宿主的plain。

上述长任务均由独立持久后继执行，不依赖Codex会话存活。4090后继固定顺序完成PC-RSMG
的两项机制消融；5090后继并行完成PCNR和AM-MCRB，并在单条失败时保留另一条继续运行。
最终只读取完整e200 terminal receipt作科学裁决。协议见
`decisions/DEC-20260830-WINNER-ABLATION-SINGLE-STREAM.md`和
`decisions/DEC-20260831-ROUTE1-FRONTIER-EXPANSION.md`。

完整边界见 `LOCAL_ROUTE1_RESEARCH_CONTRACT_CN.md`。旧 `configs/FOUR_LANES.json`
只保留为暂停方案的历史记录，不得执行。
