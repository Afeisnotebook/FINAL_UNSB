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
