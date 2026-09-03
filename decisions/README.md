# Decision order

Read decisions numerically. A later accepted decision may supersede an earlier
one only when it names the superseded fields explicitly. Proposal files do not
authorize compute.

- `DEC-0001-FOUR-LANE-FREEZE.md`: scientific portfolio freeze.
- `RUN_AUTHORIZATION.example.json`: schema/example only; not authorization.
- `CONFIRMATION_UNLOCK.example.json`: schema/example only; confirmation remains sealed.
- `DEC-0002-LOCAL-PREFLIGHT-ACCEPTED.md`: local engineering gate result and HJ RNG correction.

## 当前路线一覆盖顺序（2026-08-31）

旧的四lane冻结已被后续用户授权明确暂停。接手者在上述历史文件后应按以下顺序读取：

1. `DEC-20260829-LOCAL-ROUTE1-REOPEN.md`：重新打开长期算法发现，旧短门不再充当e200结论。
2. `DEC-20260830-ROUTE1-REMOTE-OFFLOAD.md`与
   `DEC-20260830-ROUTE1-REMOTE5090.md`：授权4090/5090仅作host-matched路线一加速。
3. `DEC-20260830-ROUTE1-SINGLE-SEED-EMERGENCY.md`：当前以seed2026完整e200作开发选择，
   seed2027/2028延期但不得声称稳定。
4. `DEC-20260831-ROUTE1-FRONTIER-EXPANSION.md`：PCNR与AM-MCRB组成两条证据驱动前沿，
   不是超参网格。
5. `DEC-20260831-FRONTIER-PRESERVATION-AND-WINNER-ABLATION-BINDING.md`与
   `DEC-20260831-FRONTIER-FINAL-EVIDENCE-COMPLETENESS.md`：唯一主候选只是交付入口，
   两个备选、完整前沿和主候选自身三项e200消融都必须保留。
6. `DEC-20260831-WINNER-ABLATION-SCIENTIFIC-GATE-FIRST.md`：严格通过者优先于数值更高
   但长期门失败的fallback；当前PC-RSMG proposal-only因此优先于full。
7. `DEC-20260831-ROUTE1-CONDITIONAL-GENERATION3-SYNTHESIS.md`已因旧AM-MCRB固定绝对
   余量事故标为`SUPERSEDED_DO_NOT_RUN`；替代文件
   `DEC-20260831-RESIDUAL-FEASIBLE-CONDITIONAL-SYNTHESIS.md`要求两个同宿主严格父项、
   residual-feasible屏障与target-blind兼容门，才允许一个最多两组件的Generation-3合成。
8. `DEC-20260831-EVIDENCE-QUALIFIED-MULTI-CANDIDATE-ADVANCEMENT.md`：canonical主候选
   只是行动接口；严格通过和因果可修复近边界机制继续组成受限可信前沿。
9. `DEC-20260831-5090-MATCHED-PLAIN-LPIPS-RECOVERY.md`：晚期plain LPIPS缺失属于工程
   证据缺口，必须从冻结checkpoint双重确定性恢复并通过父状态隔离，不能机械记为算法失败。
10. `DEC-20260831-ANTITHETIC-GAUSSIAN-GRADIENT-AUDIT.md`：梯度级Gaussian反号虽保持
    期望无偏，但e20/e100/e200相对iid两视图均增加方差，因此当前involution不启动e200；
    该结论不外推到所有无偏variance estimator。
11. `DEC-20260831-RESIDUAL-FEASIBLE-EUCLIDEAN-CONDITIONAL-SYNTHESIS.md`：在对应父项
    同宿主严格通过的前提下，保留条件采样与Euclidean residual-feasible屏障的G3-03；
    它与G3-02的Adam度量是两个独立约束几何，不是强度网格，也不预先删除任一父算法。
12. `DEC-20260831-EVIDENCE-BACKED-ALTERNATE-CROSS-RUNTIME-REPLAY.md`：完整e200后仍有
    late或terminal正信号的修复alternate也可进入最多两条4090复跑；closed仍排除，且
    G3父项严格门不降低。
13. `DEC-20260831-COMPLETE-MULTI-CANDIDATE-FRONTIER-DELIVERY.md`：唯一
    `CANDIDATE.json`只表示行动优先级；完整4090/5090宿主分离算法前沿、历史探针、
    因果图谱、消融和逐域绝对/相对轨迹必须同时交付。
14. `DEC-20260831-TERMINAL-GOAL-COMPLETION-AUDIT.md`：终交付返回本地后还须通过独立
    fail-closed完成审计；机器通过不能代替最终人工科学复核、Git裁决和推送。
15. `DEC-20260831-PCNR-ALTERNATE-4090-REPLAY-ROUTING.md`：旧修复portfolio将PCNR
    alternate误过滤为空是编排事故而非科学淘汰；PCNR须按完整5090权威绑定，从4090
    共同e0完成真实e200复赛后再重算完整多候选前沿。
16. `DEC-20260831-5090-CROSS-RUNTIME-MULTI-CANDIDATE-PORTFOLIO.md`：空闲5090并行
    复赛4090严格第一的PC-RSMG proposal-only与机制独立的AM-TNC；两者保持batch1、共同
    e0和宿主分离delta，closed当前算子不因算力空闲而机械重跑。
17. `DEC-20260831-RELATED-MULTI-ALGORITHM-FAMILY-DELIVERY.md`：终局允许多个严格可行
    算法；原生/HNEK/HJ父对象分别组合相同的条件无偏双视图估计器，AM-TNC作为独立机制
    保留。4090有限并发HPCGR与HJCGR，最终`ALGORITHM_SET`而非单一冠军拥有科学解释权。
18. `DEC-20260901-HJCGR-GAIN-SOURCE-HJPCNR-ABLATION.md`：HJ场内的一视图完整e200
    对照失败，而二视图HJCGR严格通过；当前收益来源因此收紧为post-D/E选择性G/F条件
    方差缩减，不再归因于重新采样或事件重排本身。该结论只关闭HJ-PCNR当前实现，并明确
    禁止把后续优化退化为replica-count或paired阈值网格。
19. `DEC-20260901-ROUTE1-RELATED-MULTI-ALGORITHM-TERMINAL.md`：终局保留HJCGR与
    Proposal-only两条4090严格算法、AM-TNC独立脆弱正机制及全部负边界；HJCGR只是行动
    优先级，Proposal-only是唯一4090/5090均严格通过的跨运行时方法。结论仍限于small25、
    seed2026、e200，不声明多seed或全量数据稳定。

同日文件若涉及不同对象并非互相覆盖；只有明确写出被替代字段的后决策才覆盖前决策。

## 2026-09-02 全量论文阶段覆盖

20. `DEC-20260902-PAPER-AIO-ACTIVATION.md`：用户明确授权全量paper runner和继续算法
    重构；只覆盖旧small25-only算力限制，不覆盖confirmation、paired控制、跨宿主delta等
    安全边界。
21. `DEC-20260902-PAPER-AIO-DDSB-SOURCE-GATE.md`：DDSB无权威公开实现，状态为
    `reproduction_incomplete`，不能用猜测版冒充论文对手。
22. `DEC-20260902-PAPER-AIO-NEW5090-CUT-ASSIGNMENT.md`：新5090通过门后运行CUT。
23. `DEC-20260902-PAPER-AIO-DURABLE-SUCCESSION.md`：plain→Proposal与CUT→CycleGAN只在
    前序固定e200完成、后继全部工程门通过后自动衔接。
24. `DEC-20260902-PAPER-AIO-CANDIDATE-CROSS-CODE-GATE.md`：新算法不能复用在飞plain的
    旧代码授权；small25正向终点收据、同宿主跨代码e0/2000步、zero identity、resume与
    重复评估全部通过后仍只形成证据lock，必须再签当前commit授权才能启动全量。
25. `DEC-20260902-PAPER-AIO-COMMON-CRN-DYNAMIC-ADJUDICATION.md`：训练源码指纹和CRN
    bundle身份分离；新候选保留独立训练指纹，但固定复用首波`68f53a8e...`随机bundle，
    逐图CRN完全匹配后才计算同宿主delta，并报告e200五bundle随机性标准差。
26. `DEC-20260902-PAPER-AIO-UNIFIED-EVALUATION-COHORT.md`：四条首波lane的五个固定
    checkpoint必须先做source-bound export，再在单一4090评估环境只读复算；缺少20个
    receipt中的任意一个时，`PAPER_RESULTS`不得宣称第一波完成。
27. `DEC-20260902-PAPER-AIO-DURABLE-SOURCE-EXPORT.md`：plain、Proposal、CUT、CycleGAN
    各自拥有独立GPU-free export继任器；只在本lane密封e200后签收五个固定checkpoint，
    不读取性能、不复制权重，也不干扰原训练继任链。
28. `DEC-20260902-5090C-GLOBAL-PORTFOLIO-RESCHEDULE.md`：5090C与5090A通过严格runtime
    twin后直接运行Proposal，4090A在plain后改跑独立几何路线AM-TNC；ST-CGR继续等待
    5090A matched plain。该重排并行得到至少三条自研全量轨迹，HJCGR仅因成本与跨runtime
    风险延期，未被写成机制证伪。
29. `DEC-20260902-PAPER-MATCHED-RUNTIME-RELATION.md`：统一评估环境只消除评估器差异，
    不能单独证明跨宿主训练matched。Proposal/5090C与plain/5090A的delta必须同时通过
    CRN逐图身份和预训练2000步runtime relation；缺少任一项即fail closed。
30. `DEC-20260902-PAPER-ISOLATED-THROUGHPUT-RECALIBRATION.md`：首个脱离共驻的5090A
    plain完整epoch把matched control与ST-CGR关键路径显著提前；5090C Proposal仍保留
    独立卡位并等待首个完整epoch校准。该决策只覆盖旧完成日期，不覆盖算法、训练协议或
    科学裁决。
31. `DEC-20260902-5090C-PROPOSAL-E1-THROUGHPUT.md`：5090C Proposal首个8,553-update
    完整epoch以4,696.60秒通过，确认约9月13日完成以及至少续租1天、稳妥续租2天；
    5090A第二个隔离plain epoch同时确认禁止共驻的关键路径判断。该证据不读取paired性能。
32. `DEC-20260902-PAPER-DURABLE-HEALTH-WATCH.md`：四宿主新增独立、PPID 1、每分钟刷新
    的metric-blind健康监控，覆盖训练heartbeat、successor、exporter、relay和真实磁盘需求；
    只产生可恢复告警，不自动修改训练。5090B控制仓对象缺失事故由完整bundle恢复并留痕。
33. `DEC-20260902-DURABLE-UNIFIED-EVALUATION-SUCCESSOR.md`：4090A新增持久化首波统一
    评估继任器，在固定checkpoint全部验签且AM-TNC释放GPU后自动完成单容器评估、cohort
    锁定与事后裁决。控制状态不含性能值，不改变训练，也不等于候选集或confirmation冻结。
34. `DEC-20260902-PAPER-DURABLE-MATCHED-ALGORITHM-EVALUATION.md`：AM-TNC不能误用首波
    5090A plain，故单独绑定4090A plain；ST-CGR只绑定5090A plain并等待首波cohort锁定。
    两条e200后评估/裁决链均已持久化，固定五个epoch且不能以指标控制训练或选择checkpoint。
35. `DEC-20260902-PAPER-FINAL-DISCOVERY-DELIVERY.md`：全部固定e200结果到齐后，4090A
    自动完成六条lane的统一复杂度测量，并在保持5090A/4090A两个合法plain关系的前提下
    生成多算法full-data discovery portfolio；该产物仍需人工论文审查，不能自动打开confirmation。
36. `DEC-20260903-POST-FREEZE-DISTRIBUTION-EVALUATION.md`：KID/FID接口已实现但不部署；
    只有算法、基线、e200结果和论文主张形成Git内显式冻结收据后，才可在discovery80上
    运行macro KID与补充pooled KID/FID，confirmation20仍不得打开。
37. `DEC-20260903-5090A-STCGR-TIME-PRIORITY.md`：按用户明确的紧迫工期授权，5090A
    plain在可精确恢复的e9暂停并撤销自动重启，ST-CGR从完整e1状态立即恢复长训。
    在plain未来补完前只承认绝对轨迹，不宣称ST-CGR或Proposal的matched收益。
38. `DEC-20260903-5090B-ENDPOINT-AND-MATCHED-PLAIN-SUCCESSOR.md`：新提供的SSH端点
    实际是正在运行CUT/CycleGAN的既有5090B，并非新增空闲卡。部署metric-blind后继，
    CUT e200后只有通过与5090A的2000-update精确runtime门才从e0启动plain，以尽早
    恢复Proposal的matched control；该关系不会被偷换成ST-CGR跨宿主delta。
39. `DEC-20260903-MULTI-CONTROL-RUNTIME-RELATION.md`：统一评估器兼容同一方法的多条
    精确plain关系，并按实际method/plain宿主唯一匹配；新增由三份原始gate receipt
    物化关系候选的fail-closed接口，但在5090B真实receipt出现前不修改当前registry。
40. `DEC-20260903-DURABLE-RUNTIME-RELATION-SUCCESSOR.md`：4090A持久等待5090B未来的
    2000-update exact runtime receipt，验明Proposal授权和三份原始回执后只生成
    review-only关系候选；不会自动改registry、启动任务或授权跨宿主delta。
41. `DEC-20260903-STCGR-TO-PLAIN-DURABLE-RESUME.md`：5090A新增metric-blind接力，
    只在ST-CGR完整e200后复验暂停于e9的plain checkpoint、授权、代码和协议身份，再从
    原宿主原状态恢复plain；防止长程任务完成后再次等待人工上线。
42. `DEC-20260903-INCREMENTAL-TERMINAL-AUDIT-PIPELINE.md`：现有source exporter只有
    e200后才发布五个checkpoint，导致本可并行的e100/e150 target-blind审计被串行推迟。
    新增固定e100/e150/e200增量source-bound export/relay；每个部分导入仍需checkpoint、
    sidecar、source receipt和relay-set哈希闭环，不改变训练、主表或matched-delta门。
43. `DEC-20260903-DYNAMIC-MATCHED-CONTROL-DELIVERY.md`：5090A plain取消后，旧统一评估
    与最终交付waiter虽健康但无法完成；新的部署contract显式冻结lane来源，并只从三个
    晚期点的已审核runtime relation推导matched plain。待5090B关系入Git后安全替换旧链。
44. `DEC-20260903-RUNTIME-RELATION-REGISTRY-REVIEW.md`：增加候选到Git registry之间的
    metric-blind确定性审核器，验证关系类型、两端宿主、2000-step、proof chain、全部哈希和
    唯一性；只输出review proposal，不自动修改Git或授权delta。
45. `DEC-20260903-TERMINAL-ROLLOUT-JACOBIAN-AUDIT.md`：纠正终端谱审计把单步netG
    Jacobian误作完整rollout Jacobian的缺口；新增lane-blind CRN方向的NFE5幂迭代、
    endpoint运输方向定义和严格输出验收，并在本地1060真实模型上通过容量smoke。
46. `DEC-20260903-TERMINAL-PATHOLOGY-POSTHOC-ADJUDICATION.md`：在任何paired结果可读前
    验签四条固定probe的12个target-blind审计，以e100→e150诊断预测e150→e200变化；
    固定阈值、同一统一评估器和跨方法/域支持共同决定是否只授权后续数学推导，绝不自动
    启动终端模块或控制在飞训练。
47. `DEC-20260903-DURABLE-RELATION-REGISTRY-REVIEW.md`：Proposal与ST-CGR关系候选生成后
    的机械Git审核不再依赖人工上线；持久successor等待两份精确完成状态并自动生成review
    proposal，但继续禁止修改tracked registry、授权delta或启动评估，保留Codex显式准入。
48. `DEC-20260903-PAPER-INITIALIZATION-EXPOSURE-AUDIT.md`：当前冻结e0在DDI后保存sampler，
    使UNSB/CUT的训练暴露错开一个stream位置；现有同cohort比较继续有效且不重启，论文
    必须披露有界偏差。未来fresh e0从DDI前保存sampler并形成不可与legacy混用的新cohort；
    同时把CycleGAN准确标为官方损失、受控共享骨干，而非逐字官方实现。
49. `DEC-20260903-LIVE-CRITICAL-PATH-AND-LEASE-HORIZON.md`：五节点训练、监督、导出和
    交付链全部健康，不做无故重启或改协议；按实测吞吐明确5090C Proposal与5090A
    ST-CGR的e200日期，并把云端租期识别为supervisor无法自行修复的唯一当前外部风险。
50. `DEC-20260903-DYNAMIC-DELIVERY-HASH-CLOSURE.md`：修复取消5090A plain后future
    replacement delivery只看完成状态、未把全部结果/disposition重新绑定到路径与SHA的
    延迟故障窗口；v3交付在profiling前后均复验固定产物，并重新核验跨宿主2000-step core。
51. `DEC-20260903-UNIFIED-EVALUATION-METRIC-INTEGRITY.md`：统一评估改为从逐图证据重算
    NFE/replicate/逐域/宏指标，跨lane精确核对sample与CRN身份，并以评估前后checkpoint
    SHA256证明只读；LPIPS 0.1.4 Alex v0.1不可用时fail closed，不再静默生成空结果。
52. `DEC-20260903-NEW-ENDPOINT-IDENTITY-IS-5090B.md`：用户提供的44804端点经hostname、
    machine-id、仓库/manifest、run inode、PID和heartbeat共同证明是现有5090B，不是新增
    物理GPU；保持CUT/CycleGAN及matched-plain后继不变，禁止误开第三lane或重复登记算力。
53. `DEC-20260903-POST-DUPLICATE-ENDPOINT-CONTINUITY.md`：在真实四宿主上复核全部训练、
    后继、exporter、health watcher、磁盘与480–720小时控制窗口；当前队列覆盖e200且0告警，
    4090A不以新容量探针推迟只剩约46小时的高扇出plain control。
54. `DEC-20260903-PHYSICAL-GPU-IDENTITY-GATE.md`：新增GPU UUID主键的宿主身份注册表与
    fail-closed接入门；三个AutoDL容器machine-id相同，不能单独作物理身份。`:44804`远端
    实跑返回已登记5090B并拒绝5090D长训身份，现有训练进程保持不变。
55. `DEC-20260903-STCGR-INDEPENDENT-OPERATOR-AUDIT.md`：在full-data长训早期独立复核
    ST-CGR公式、MRO执行、随机量来源、恢复证明和e7 live state。确认条件pre-Adam G/F
    梯度无偏与相对iid双视图的PSD方差下降，但不扩大为Adam、完整game或PSNR保证；冻结
    源码无需修改，5090A不重启并继续e200。
56. `DEC-20260903-DDSB-DURABLE-SOURCE-WATCH.md`：DDSB仍无权威作者实现，继续严格标记
    `reproduction_incomplete`；新增冻结commit、每六小时轮询的来源守望器及独立健康监控。
    任何官方页或GitHub命中只触发人工源码/公式审核，不自动授权或启动训练。
57. `DEC-20260903-GLOBAL-RESCHEDULE-AND-EXTERNAL-SOURCE-GATE.md`：基于五条实时heartbeat和
    实测吞吐保留5090C Proposal、5090A ST-CGR、4090A plain→AM-TNC及5090B外部基线→
    matched plain；同时锁定DCLGAN/NEGCUT作者源码。DCLGAN先进入full-state adapter工程，
    NEGCUT因明确工程与许可门暂缓但未证伪；没有打断或新增GPU长训。
58. `DEC-20260903-DCLGAN-SOURCE-BOUND-ADAPTER-LOCAL1000-GATE.md`：DCLGAN官方源码绑定适配器
    已冻结，覆盖完整训练状态、严格恢复、只读重复评估和confirmation锁；短GPU门通过后，
    在本地1660启动完整1000/500正式门。该门不读取性能、不授权远端训练，目标GPU仍须重跑。
59. `DEC-20260903-DCLGAN-PORTABLE-GATE-AND-DURABLE-QUEUE.md`：修复Git换行策略导致的
    raw-byte源码锁跨平台失效，以LF归一化hash在Windows/Linux得到同一fingerprint；新版
    本地正式门全部通过，5090B已部署target-gate→e200及导出的持久链并等待matched plain。
    旧门只因source-lock表示被取代，不构成DCLGAN机制失败或训练失败。
60. `DEC-20260903-5090C-LIVE-GLOBAL-RESCHEDULE.md`：按e79/e10/e62/e37/e13实时吞吐重新
    求解四卡排程，确认5090C继续独占Proposal而非换跑或共驻；保留4090A AM-TNC与5090A
    ST-CGR两条独立数学前沿，并部署DCLGAN的持久导入、统一评估和健康链，避免长训结束后
    形成不可交付结果。
61. `DEC-20260903-DCLGAN-COMMON-RUNTIME-LOAD-SMOKE.md`：把本地正式1000-update
    DCLGAN checkpoint在4090A统一Linux/PyTorch环境中真实恢复并执行两次target-blind CPU
    前向；checkpoint、完整科学状态和输出均逐字节稳定，提前排除长训结束后的跨运行时加载
    故障。该门不产生性能结果、不占训练GPU，也不改变5090B资源顺序。
62. `DEC-20260903-5090A-STCGR-ONLY-RECONFIRMATION.md`：再次实时确认5090A只运行ST-CGR；
    plain停在e9且trainer、export waiter及两条历史resume successor均无存活PID。保留完整
    checkpoint但禁止旧状态文件自动恢复，并明确没有e200 matched control时只报告绝对轨迹。
63. `DEC-20260903-CODEX-GOAL-HEARTBEAT-CONTINUITY.md`：五节点与本地控制链实时健康；
    44804仍是已登记5090B。发现旧Codex heartbeat已不存在后，新建并启用当前论文Goal
    heartbeat `final-unsb-goal`，持续监管多算法e200、runtime关系、统一评估和Git终局。
64. `DEC-20260903-LEASE-CRITICAL-PATH-REFRESH.md`：进程健康但云租期仍是外部关键路径；
    5090C十天租期可能比Proposal e200早约30小时结束，要求9月11日前确认覆盖至9月15日；
    5090B的9月14日边界只覆盖control，DCLGAN需先预留至9月22日再按目标门实测收敛。
65. `DEC-20260903-FUTURE-DYNAMIC-CONTROL-INTEGRATION.md`：用当前tracked registry的真实
    单relation形态闭环演练未来5090B control接入；证明Proposal可保留旧关系并增加新关系，
    ST-CGR cross-code关系也能通过晚三点最终交付门。该测试不写production registry、不读
    性能且不预授权delta，真实5090B e200回执与人工review仍是必要条件。
