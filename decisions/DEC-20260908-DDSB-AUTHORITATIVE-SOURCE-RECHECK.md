# DDSB权威源码门复核：继续fail-closed

日期：2026-09-08

DDSB仍是论文时间点上重要的直接对手，因此在长训健康等待期间再次复核其权威来源。
NeurIPS官方页面仍只提供论文和补充材料，没有Code或GitHub链接；Westlake作者实验室的
论文条目仍只指向论文文件，没有与该工作绑定的repository。四条GitHub repository搜索只
命中一个无关的NeurIPS论文聚合仓库，没有作者实现、官方checkpoint或可核验实现记录。

2026-09-03的完整论文与补充材料审计已经证明，公开文档没有唯一确定U-Net图、MINE实现、
判别器、优化器、更新顺序、stop-gradient边界、损失归一化和All-in-One协议适配。此次没有
出现能够关闭这些歧义的新材料，因此DDSB继续标记为`REPRODUCTION_INCOMPLETE`，而不是
算法性能失败；不得为了补论文表格而启动猜测实现。

本地持久源码守望器PID 20388和健康监控PID 10780仍真实存活，状态为
`WAITING_FOR_AUTHORITATIVE_SOURCE`、零告警。它们继续每六小时检查冻结来源，但任何候选
只会触发人工来源与公式审查，不会自动授权训练。

本裁决不改变任何GPU队列，也不缩小当前算法前沿。DDSB一旦发布权威源码即可重新打开门禁；
在此之前，论文外部比较由已经合法运行的CUT、CycleGAN和DCLGAN承担，并如实披露DDSB
复现未完成。
