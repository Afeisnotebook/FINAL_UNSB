# ACTIVE：本地路线一长期算法发现计划

状态：`FINAL_474_140_CAUSAL_MATRIX_FROZEN / BVCP_AND_RSMG_PASS_GATES / MATCHED_E200_RUNNING`
日期：2026-08-30
当前硬件：GTX 1660 6GB、RTX 4090 24GB、RTX 5090 32GB

## 2026-08-30 多宿主算力补充

用户已明确授权 `192.168.0.30` 的RTX 4090作为受控算力副本。该授权不恢复旧四lane
计划，也不改变本地canonical：远端方法必须与远端plain从同一shared e0完整matched，
不得拿remote方法减local plain、不得跨硬件续接同一轨迹。远端先通过代码/数据内容/
e0/zero-intervention/resume/守护门，才可并行运行plain/HJ/HNEK的e200副本和后续
host-matched审计。confirmation20、全量数据、路线二和退出阈值仍关闭。详细边界见
`decisions/DEC-20260830-ROUTE1-REMOTE-OFFLOAD.md`。

远端已完成570个允许身份、1140个文件的内容hash核对（confirmation触碰为0），
shared e0文件/科学状态hash复制一致，CPU/GPU门全部通过。独立tmux守护器已从e0启动
remote plain；训练commit仍为`0da2a37`。compact证据见
`evidence/remote_route1_offload/REMOTE4090_PREFLIGHT_AND_START.json`。

用户随后又明确授权一台RTX 5090，并授权主控在不改变科学目标的前提下决定batch与
并发。当前决定是：科学lane继续固定batch1；通过同宿主、同e0、隔离output root的
独立进程并行提高总吞吐。batch增大会改变unpaired B采样、PatchNCE negatives、Adam
轨迹及HJ/DT校正对象，不能与现有长期证据直接比较。并行度按实时算力和科学依赖动态
分配：一张卡最多两个相互隔离的batch1训练或审计进程，接近饱和后不再开启第三个。
并行结果只能显式验完整文件树和同机plain hash后导入canonical root，绝不覆盖已有lane。
详细边界见
`decisions/DEC-20260830-ROUTE1-INDEPENDENT-PROBE-CONCURRENCY.md`。

## 当前执行事实（2026-08-30 14:45）

- 4090权威因果图谱已经冻结：`474`条长期反转记录、`140`条采样方差记录；最终
  `LONG_CAUSAL_MATRIX.json`状态为`COMPLETE_CAUSAL_AUDIT`。matrix SHA256为
  `dc54569a...e0d3`，atlas SHA256为`965faf9a...d2e1`。该阶段不再是运行中。
- 证据驱动的Generation-1只冻结两个具有充分构造权限的算法，不为填满名额强造第三个：
  BVCP（Bridge-Velocity Chord Projection）修复rollout分布移动速度；RSMG
  （Replicated Stochastic-Measure Gradient）以条件无偏的双随机视图梯度平均降低
  原生随机场方差。DT/HJ/HNEK仍是父证据探针，不是候选名单。
- 两个算法均已通过数学不变量、zero-intervention逐位身份、active resume、e20/e100/e200
  跨状态分支、父状态隔离和400-update工程门。门禁冻结证据见
  `evidence/remote_route1_offload/GENERATION1_GATE_FREEZE_20260830.json`。
- 4090正在从共同e0并行运行BVCP与RSMG的权威small25/e200轨迹；5090正在运行相同算法的
  独立跨运行时复核。两处均固定batch1、seed2026、30000 updates、最多5 data epochs一个
  可恢复chunk，中期paired指标不控制训练。5090与4090运行时不等价，因此5090只作稳定性
  证据，绝不与4090 plain混算或并入4090排名。
- 当前早期指标仅作健康检查：4090 BVCP e1/e5/e10约为
  `+0.241/+0.167/+0.307 dB`，RSMG e1/e5约为`+0.367/+0.259 dB`。这些结果不能晋级、
  不能早停；科学裁决仍固定为e150/e175/e200。
- 本地1660继续原HNEK锚点，当前约e80；4090、5090两张卡的独立batch1进程均达到约
  98--99%利用率。显存空闲不是吞吐证据，增大batch会改变科学协议，故只使用独立进程
  并行而不改变batch。
- 下一硬门：两候选完整e200。若某候选通过晚三点、e200、域覆盖、最差域、SSIM/LPIPS、
  绝对轨迹和plain-collapse门，则立即冻结公式后运行seed2027；若当前实现失败，只有在
  target-blind缺陷量确实下降但长期收益仍反转时，才允许一次因果修订。不得转成窗口、
  handoff、退火或paired控制。

## 历史执行快照（2026-08-30 08:04，以下进度数字已由上节取代）

- plain 的 e100 正式指标已从冻结 milestone 两次独立回填，完整payload哈希一致，
  且评估前后科学状态哈希不变。
- 本地plain已正式完成并接受e200（30000 updates），PSNR `18.095297`、SSIM
  `0.583635`、LPIPS `0.316534`；完整文件hash和重算scientific-state hash均与sidecar
  一致。current-user计划任务`FINAL_UNSB_ROUTE1_EXECUTOR`已自动从共同e0启动HJ；当前
  已到e103（15450 updates），继续按最多5 data epochs分块执行。
- 通用milestone验收器已在本地e175和4090 plain e200上实际重算checkpoint文件hash、
  scientific-state hash、420张discovery70 CRN、六域计数和LPIPS，结果与已有正式证据
  一致；后续每个关键e200都使用同一验收边界。
- 同一plain进程此前也在e60附近无traceback消失，说明失败属于进程托管而非e100
  科学路径。隔离LPIPS评估已正常完成420张图像。
- 4090同宿主plain、continuous HJ与HNEK均已完成并接受e200。HJ晚三点宏PSNR
  delta均值为`+0.649331 dB`，HNEK为`+0.806229 dB`，两者均通过注册的域覆盖规则，
  因而small25 proxy已正式`CALIBRATED`。一次HNEK交接竞态已通过隔离、哈希验收、
  可恢复quarantine和显式导入解决，没有混合full state。
- 5090已完成数据内容、shared e0、CPU/GPU、resume和zero-intervention门；plain与隔离
  HNEK均已验收e200并导入canonical root，HJ当前到e75。HNEK晚三点宏delta均值虽为
  `+0.608222 dB`，但只有一个晚期点达到4/6域正，单独不足以校准该宿主；DT继续锁定到
  HJ e200，不能为了提高利用率绕过科学依赖。
- 5090已利用剩余算力启动HNEK固定点的部分因果审计。e20首两条记录已验证audit/training
  身份、父状态隔离和target blindness。冻结同机plain/HNEK轨迹随后通过正式选择函数确定
  完整HNEK子图为e10/e20/e40/e100/e150/e175/e200；这只提前计算最终队列必含的固定/动态
  审计点，不启动DT、不冻结矩阵。HJ heartbeat保持正常，双流GPU利用率约95--97%。
- 4090 DT已完成并验收e200。其e40/60/80/100/125/150/175/200宏PSNR delta依次为
  `-0.4910/+1.2333/+0.1148/+3.2237/-0.6441/+0.6784/+0.5011/-0.4711 dB`；晚三点均值
  `+0.236124 dB`，但e200仅2/6域正、最差域`-1.5765 dB`，SSIM/LPIPS也退化。因此DT是
  `positive_probe_not_candidate`：它证明e21--e45的有限支持校正能改变很长的原生尾部，
  但当前实现没有形成顺畅的终点收益保持。中间paired指标从未控制训练。
- 冻结executor worktree固定于`0da2a37`。恢复后训练仍使用原协议指纹
  `b0786b...9b2`；主开发worktree负责补齐审计和候选执行器，不得污染锚点身份。
- 4090持久审计器固定于audit commit `729826f`、source fingerprint `41434187...5e1a`。
  四锚点终态哈希验收后冻结了25个审计点：HJ 8个、HNEK 9个、DT 8个；无缺失和重放。
  当前两个隔离worker并行审计DT e60/e100，已形成70条atlas row和12条variance row，已检查
  的记录均证明parent full-state hash前后相同、paired controller访问为false、
  confirmation20未打开；实测双流GPU利用率
  约95%，不再增加第三个进程。审计覆盖连续介入、短pulse后的原生流传播以及8重复
  batch/latent-time校正方差；pulse只作因果诊断，不是路线二策略。
- paired discovery70只在双分支冻结后作为未来标签；只有跨方法准确率、相关性、域
  一致性和反转领先性通过合同的target-blind量才可驱动自适应构造。若没有跨方法共享
  信号，分析器现在按原计划保留满足同样数学门槛的探针专属信号；候选card必须把该信号
  同时绑定到对应探针和父失败机理，不能借给其他算法。若两类信号都没有，才只允许从
  可证明无偏的估计器/重参数化路线派生，不允许拟合退出阈值。实现和门禁见`42a19c6`。
- 远端4090 plain e200 checkpoint与官方metric已实物落盘并二次核验；远端剩余空间
  约418 GB。远端方法仍只允许与同一远端plain比较。5090数据盘剩余约230 GB。
- 在看到长期atlas之前，矩阵后处理已补齐rollout-speed与bridge-time conditioning
  路由；分析commit、源码和atlas/variance/queue输入hash将单独记录，不改原分支行。
- 候选执行链已区分算法定义指纹与host/e0证据执行指纹，并要求源码绑定的可执行gate
  hook实际通过数学不变量、zero identity、resume、跨状态和400--800 update工程门。
- 新候选derivation card还必须绑定长期矩阵、历史证据索引、UNSB对象图和SEARCH-005
  等价性边界，声明与旧实现的实质差别、四类数学改动、适用状态、恢复成本及三项消融；
  因果矩阵完成后会初始化不可静默覆盖的`HYPOTHESIS_LEDGER.json`。候选executor同时
  绑定产生plain e200的GPU/PyTorch/CUDA环境及baseline哈希，跨宿主delta会在启动前失败。
- 新算法公式仍未预写。必须等25点atlas和不可变`LONG_CAUSAL_MATRIX`完成后，才依据跨探针
  证据生成最多三个Generation-1构造；历史已失败的future-batch selector、HJ cosine
  matching、同batch Adam projection、DT sensitivity preconditioner等不得换名重跑。
- Generation-1排序已从原先仅按“支持探针数”修正为固定字典序：跨探针支持、已观测
  前兆领先性、六域一致性、冻结的200-update反事实收益、最小数学复杂度、最小计算/恢复
  成本。复杂度和成本在矩阵阶段仅是构造家族下界，最终必须由derivation card替换为实际
  算法开销；这不等于提前写好算法。200 updates在small25等于1.333 data epochs，也是唯一
  实际带discovery70 post-branch标签的短分支；1/8/32只作局部几何诊断。实现见`56f2740`。
- 一次因果修订不再依赖手工改ledger。`641559a`新增追加式G2授权：只有G1完成e200且
  当前实现长期为负、target-blind缺陷量数值上确实下降但收益仍反转、并形成新的因果
  失败理由时才能创建；同一父机制只能一次，G2不能再生G3，窗口、handoff、网格和paired
  控制全部硬拒绝。G2 card必须绑定父候选、修订请求和缺陷证据哈希。
- 原始audit一直记录block几何及每个replicate的domain/time，但旧screen没有将其转成候选
  可用前兆。`8e934f2`补入最差非零block、最差域、最差bridge-time方向一致性和block方差
  护栏；精确零校正不会因为零方差被误判成安全信号，域/时间量至少需要两个group。这只
  改最终后处理，不改不可变原始行、不重启worker，也没有增加paired阈值。
- `cff1d49`继续补齐计划要求的UNSB内部状态：endpoint dispersion、bridge KDD、generator
  梯度尺度、Adam moment夹角、D/G和E/G平衡。只有这些target-blind量先恶化且冻结的
  200-step反事实为负时才形成父失败机理；endpoint构造硬禁止改变endpoint law，game构造
  仍须通过SEARCH-005等价性门，不能把失败的AEB、PCOA或NPOOA换名重跑。
- 当前分析器已在4090独立目录用真实raw快照烟测：86/432 reversal rows、17/128 variance
  rows时严格返回`PARTIAL_CAUSAL_AUDIT`和退出码7，ranked mechanism为0，screen保持blocked，
  stderr为空。它没有触碰primary atlas，证明新增后处理在真实schema上兼容且仍fail-closed。
- 机制证据边界进一步固定：零校正只作identity、不参与排名；单状态负方向或过大幅值不会
  被全程均值掩盖；plain/self两种状态都受害时只能做当前状态rate/curvature数学重写；
  “共享信号”要求每个探针单独过0.65，不能用平均值掩盖失败探针。当前完整测试为100项。
- DT矩阵后处理已在主分支`b2e4480`区分有限support内的registered机制证据与support外的
  forced-active机制诊断，并允许同一DT机制跨两种记录形成连续时间证据。e20的registered
  路径受schedule/warmup限制，故机制有效性使用forced-active；e40使用真实registered。
  audit commit
  `729826f`继续只生成不可变原始行；最终矩阵必须由修复后的主分支重算，不能直接采用
  审计分支运行期间的partial matrix。

持久恢复门已完整通过：缺失milestone两次一致回填、5-data-epoch幂等chunk、外部
计划任务托管、退出审计、15分钟stall检测、人为终止后的精确恢复，以及首个正式
生产分块验收均有compact evidence。冻结训练继续运行，主worktree并行实现真实
Phase C；两者通过独立git/audit identity隔离。

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
4. proxy校准；
5. 仅在`CALIBRATED`宿主运行`P3_DT_LONG`：按lineage冻结的direction/covariance锚点。

这里的编号是科学依赖而不是强制串行调度。HJ与HNEK都从同一shared e0独立开始，
可在隔离目录并行；plain必须在任何相对比较前完成并验收，DT仍受proxy硬门限制。

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

若第一名在seed2026晚期为正，冻结算法后再运行host-matched seed2027；符号不一致才
考虑seed2028。4090/5090现已是用户明确授权的路线一执行节点，但这不授权全量数据、
confirmation20或跨宿主delta；这些仍需候选冻结后另行裁决。

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

## 8. 多宿主资源与调度

所有科学裁决仍以data epochs而非墙钟。单进程batch1没有吃满4090/5090，因此每张
高端卡当前运行两个相互独立的lane；实测利用率已接近饱和，不再增加第三个进程。
checkpoint、指标和audit只在同宿主内匹配。Phase C可在不同宿主形成独立因果副本。
同一宿主仍由单一durable auditor拥有queue和最终矩阵；在跨进程atlas锁、独立工作目录、
per-job stall计数和parent-state隔离通过后，4090/5090最多运行两个独立审计worker，
1660默认单worker。候选派生后最多两个算法可分配到不同宿主并行完成各自matched e200，
不能把不同宿主的method与plain混算。
