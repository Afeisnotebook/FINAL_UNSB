# 论文证据冻结采用两阶段人工/Codex审批门

日期：2026-09-03

审计确认，KID/FID实现会要求一个已提交Git的freeze receipt，但此前仓库没有生成或审批
该receipt的入口，且校验只检查`source_portfolio_sha256`是64位字符串，没有复验真实
portfolio文件。这会在e200后留下两个风险：靠手写JSON跳过论文主张审阅，或者冻结收据
指向已经变化的结果组合。

新接口分为两段。`freeze-draft`只从已完成的portfolio提取全部算法/基线lane和用户明确
给出的claim文本，状态固定为`PENDING_EXPLICIT_HUMAN_CODEX_REVIEW`，不能自批准。
`freeze-materialize`只接受仓库内已经提交的独立review decision；该决定必须逐字绑定
portfolio路径/哈希、全量distribution lanes、claim集合以及human/Codex双重审阅标志。
物化后的freeze receipt仍须再次提交Git，KID/FID才会接受。

最终distribution门同时复验三层不可变身份：真实外部portfolio的SHA256、仓库内review
decision的文件/Git blob/commit、仓库内freeze receipt的文件/Git blob/commit。算法、
baseline、e200结果和论文claim冻结后，`confirmation_authorized`仍为false；本接口不会
打开confirmation20，也不能根据KID/FID回改方法。

该改动只补全未来论文交付控制面，不部署运行、不读取当前中间性能、不修改任何训练。
