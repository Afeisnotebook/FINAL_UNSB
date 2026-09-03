# DDSB权威来源持久守望与fail-closed边界

日期：2026-09-03

DDSB是当前论文协议要求关注的直接当代对手，但论文正文、官方补充材料和作者机构页面
仍不能唯一确定完整实现。截至本次复核，NeurIPS官方页面没有代码链接，Westlake作者实验室
页面没有与该论文绑定的repository链接；GitHub精确标题搜索的唯一结果仍是无关聚合，未发现
可信作者实现。因此DDSB继续标记为`REPRODUCTION_INCOMPLETE`，不是性能失败。

为避免长训期间反复人工搜索或作者源码公开后无人发现，新增
`operations/paper_aio_ddsb_source_watch.py`。它每六小时只检查冻结的两个权威页面和两条
GitHub repository query，并保存响应SHA256和候选元数据。权威页面附近若出现repository
链接，状态只能变为`AUTHORITATIVE_SOURCE_CANDIDATE_REVIEW_REQUIRED`；搜索命中只能变为
`UNVERIFIED_SOURCE_CANDIDATE_REVIEW_REQUIRED`。两种情况都不会修改paper protocol、不会
放行DDSB lane、不会启动训练，仍需Codex人工核验作者身份、公式覆盖、源码hash、更新顺序、
stop-gradient和full-state语义，并通过Git decision准入。

守望器从detached commit `cd74b2d`运行，避免主分支后续变化改变其行为。独立health watcher
检查PID、state freshness和磁盘余量；如果source watcher因发现候选而正常退出，health状态会
把PID终止显式暴露，而不是静默停止。首次在线轮询得到
`WAITING_FOR_AUTHORITATIVE_SOURCE`，全部网络请求成功、无候选。该部署不读取checkpoint、
paired指标或confirmation20，也不触碰任何训练宿主。

这项工程只把“等待权威来源”变成可持续观察的前驱，不改变DDSB的科学状态。若租期内始终
没有来源，论文应如实写为复现未完成，并使用已经合法运行的CUT与受控CycleGAN结果；不得
用本地猜测实现填表。

