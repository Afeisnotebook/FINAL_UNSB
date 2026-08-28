# ACTIVE：本地路线一长期算法发现计划

状态：`PLANNING_LOCKED / COMPUTE_NOT_STARTED`  
日期：2026-08-29  
当前硬件：GTX 1660 6GB  

## 0. 北极星目标

从 deterministic clean UNSB 出发，利用 DT、HJ、HNEK 及后续机制的历史正窗口、
反转与负结果，**主动发现并构造**一个能在真实 200 data epochs 上保持 matched
收益的新算法。

本计划的完成标准不是：

- 复现HJ；
- 给旧算法排一次名；
- 找到退出窗口或handoff；
- 证明某个预先写好的点子有效；
- 为4090准备运行包。

HJ只是第一项延迟收益时间正对照；HNEK是第二项历史e200正锚点；DT是方向/协方差
锚点。三者都是用来获得因果证据的探针，不是受保护的最终候选。

## 1. 最高优先级防漂移门

任何新动作开始前，必须逐项回答：

1. 它是否直接帮助我们发现、推导或裁决一个长期算法？
2. 它是路线一算法构造，还是悄悄变成路线二退出/handoff？
3. 时间是否已经换算为data epochs，而不是只写updates？
4. 如果在运行旧算法，它作为哪一种因果探针，而不是为什么“值得再验证一次”？
5. paired target是否只作事后标签，绝不进入训练、调度或控制？
6. 当前结论关闭的是实现、协议还是父机制？证据是否足以支持该词？

任一问题回答不清楚，动作不得启动。目标变更只有用户明确同意才生效，不能由最新
实验、最新总结或算力便利性自动改写。

### 固定裁决词汇

- `engineering_invalid`：工程身份或确定性门禁失败，不能作科学解释；
- `short_horizon_negative_current_implementation`：命名实现在不足e200的协议中负；
- `long_horizon_negative_current_implementation`：命名实现完成校准后的e200仍负；
- `mechanism_falsified`：数学不变量被反证，或父机制在适用状态上有充分长期反事实；
- `positive_probe_not_candidate`：提供有价值信号，但尚未形成论文算法；
- `route1_candidate`：有derivation card且进入matched e200裁决；
- `route2_only`：退出、handoff或状态交接证据，不得混入路线一排行榜。

旧文档中的笼统 `CLOSED_NEGATIVE` 必须重标为上述具体状态。

## 2. Phase A：证据与语义谱系重建

### A1. 时间尺度重标

把 DT/HJ/HNEK、SEARCH-001/002/003/004/005 以及 LBST/PTQ/DCUM/AEB/PCOA 的所有
关键运行换算成：训练身份数、batch、updates、data epochs、是否continuous、是否
handoff、是否matched plain、是否独立评估。

产出：`evidence/LONG_HORIZON_EVIDENCE_INDEX.jsonl`。

### A2. 多方法lineage审计

分别比较历史正结果实现与clean实现：

- 数学对象和公式；
- forward是否恒等、backward具体改了哪一项；
- 参数、激活语义与坐标；
- optimizer顺序、参数集合、AMP/精度；
- batch composition、sampler和latent RNG；
- 数据视图、seed、分辨率和评估器。

DT、HJ、HNEK都必须完成，不得只审HJ。后续机制至少记录它们真正作用的UNSB对象。

产出：

- `evidence/lineage/DT_LINEAGE.json`
- `evidence/lineage/HJ_LINEAGE.json`
- `evidence/lineage/HNEK_LINEAGE.json`
- `evidence/lineage/MECHANISM_OBJECT_MAP.json`

### A3. Phase A出口

只有在每个长期锚点都有一个语义明确、可恢复、zero-intervention正确的clean实现后，
才能冻结训练协议。若历史与clean不等价，先分离语义差异，不能把数值差直接解释成
算法失效。

## 3. Phase B：校准后的长期锚点图谱

### B1. 共同协议

- small25：25张/域，共150张；
- official unpaired训练语义；
- batch1，seed2026，共同e0；
- 200 data epochs = 30000 updates/lane；
- milestone：e1/5/10/20/40/60/80/100/125/150/175/200；
- discovery70只作固定评估；confirmation20封存；
- e200前不因中间PSNR为负科学早停；只允许工程hard stop。

### B2. 执行顺序

1. `P0_PLAIN_LONG`：共享plain长期轨迹；
2. `P1_HJ_CONTINUOUS_LONG`：按lineage冻结的权威continuous HJ，不handoff；
3. `P2_HNEK_LONG`：按lineage冻结的bridge-coordinate锚点；
4. `P3_DT_LONG`：按lineage冻结的direction/covariance锚点。

为避免HJ成为唯一校准器，HJ和HNEK共同承担历史正对照。若二者都不能在语义等价
条件下呈现任何晚期正信号，先暂停新算法筛查，诊断small25/batch1 proxy，而不是
宣布两个机制死亡。DT仍是算法证据，但不在失真的proxy上盲目消耗长程算力。

### B3. 记录内容

除PSNR/SSIM/LPIPS和逐域绝对轨迹外，每个milestone保存完整状态，并记录：

- 方法校正与原生梯度、下一独立batch梯度的cosine；
- generator block、bridge time、domain级符号和幅值；
- GAN/SB/NCE方向导数与冲突；
- endpoint dispersion、bridge KDD、rollout velocity；
- D/G/E平衡与Adam moment-gradient夹角；
- time/domain/latent方差；
- 每种方法自己的内部缺陷量。

这些量全部target-blind；paired指标只在轨迹完成后给未来收益标标签。

## 4. Phase C：长期反转因果图谱

根据完整轨迹定位每种探针的关键转折，不预设只有“早正晚负”。在转折前、附近、后
分别从plain状态和方法状态计算：

\[
u_0(S^P),\;u_i(S^P),\;u_0(S^i),\;u_i(S^i)
\]

并运行不污染主状态的1/8/32-step和必要的200-step虚拟分支。重点区分：

- 校正本身在当前状态已变成有害；
- 校正仍局部有效，但改变了长期博弈/优化状态；
- 均值方向无问题，方差主导结果；
- clean实现与历史数学对象并不等价；
- proxy没有保留历史机制。

产出：`evidence/LONG_REVERSAL_ATLAS.jsonl` 与
`evidence/LONG_CAUSAL_MATRIX.json`。

## 5. Phase D：证据驱动的新算法生成

### D1. 生成原则

从长期因果图谱中选择证据最清楚的失败机理。第一代最多生成三个相互不同的最小
构造，不为DT/HJ/HNEK保留席位。允许出现完全新方法。

每个候选必须先有 `DERIVATION_CARD`：

- 父探针和长期证据；
- 修复的UNSB数学对象；
- 新update/operator/measure的推导；
- identity、self-null、自稳定或无偏条件；
- 是否改变目标、梯度估计、bridge coordinate或endpoint law；
- paired target不可访问证明；
- 预期有效状态及明确判死反例；
- 计算、显存和恢复状态成本。

### D2. 晋级漏斗

1. 数学不变量和zero-intervention测试；
2. 早/中/晚已有checkpoint上的反事实；
3. 400--800 updates工程micro run，只排除实现灾难，不作长期科学淘汰；
4. 只有跨状态反事实成立的候选才运行small25 e200。

不得用短程PSNR网格挑候选，也不得把固定退火包装成自适应算法。

## 6. Phase E：新算法真实长期裁决

晋级候选从共同e0完成matched small25 e200。主排序为：

1. e150/e175/e200宏PSNR delta均值；
2. e200 delta；
3. 正域数和最差域；
4. 候选绝对轨迹及峰值回撤；
5. SSIM/LPIPS护栏和计算成本。

不挑最佳checkpoint。若候选降低了目标缺陷但e200仍失败，允许根据新的因果证据修订
一次；每个机制最多两代。失败后不能偷偷转向handoff或退出阈值。

若第一名在seed2026晚期为正，再在本地运行seed2027；符号不一致才考虑seed2028。
本计划不自动进入4090。

## 7. 诚实停止与交付

路线一只有在以下工作完成后才可收口：

- proxy经至少一个历史长期正对照校准，或明确证明proxy失真；
- 多方法长期因果图谱完成；
- 至少一个证据派生的新算法完成e200裁决；
- 允许的一次因果修订完成或明确不适用。

必须交付：完整证据索引、lineage、长期atlas、候选推导、逐域轨迹、唯一当前候选、
两个备选/失败方向以及不能获得正收益时的具体原因。

若proxy失真，结论是“本地代理无法判断目标机制”，不是“路线一死亡”。若proxy
有效但新算法均失败，才可诚实写“在已审计机制空间内未找到持续候选”，且不能用
路线二结果冒充路线一成功。

## 8. 本地资源预估

现有实测约2400步/24分钟。普通30000步约5小时/lane；HJ及诊断会更慢。Phase B
串行预计约24--36小时GPU墙钟，Phase C约4--8小时，单个新候选e200约6--10小时。
整个发现循环预计2--4天本地连续工作，具体以首条长期轨迹实测修正。
