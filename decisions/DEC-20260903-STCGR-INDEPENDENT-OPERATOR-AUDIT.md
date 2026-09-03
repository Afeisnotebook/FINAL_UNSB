# ST-CGR独立operator语义审计

日期：2026-09-03

5090A上的full-data ST-CGR仍处于早期长训，因此在不读取paired指标、不加载训练GPU、
不修改在飞checkout的前提下，对冻结公式、实际MRO dispatch、完整G/F累计顺序、随机量
来源、full-state恢复证据和当前e7 checkpoint再次进行独立审计。

审计确认，当前实现先用一份原生视图各提交一次D与E，再在固定post-D/E状态和同一官方
无配对batch上生成两份G/F视图。第一份时间索引仍由原生`Uniform{0,...,T-1}`采样；第二份
从其余`T-1`个索引均匀采样。两个G/F loss各以`1/2`反传，G与F优化器各只提交一次。
当前协议使用InstanceNorm、关闭dropout和flip；rollout噪声、endpoint latent、最终latent、
PatchNCE latent与patch permutation均由每份视图/损失自己的后续RNG调用产生，没有共享的
可写随机模块状态。共享的官方A/B batch和post-D/E参数状态是条件估计器的设计边界，不是
随机量泄漏。

为避免只测试纯采样函数，新增了完整mixin-MRO的可微标量博弈测试。该测试真实执行
`StratifiedTimeConditionalGFMixin -> PCRSMGAblationMixin`路径，验证D/E只读原生视图、
两份G/F视图位于D/E提交之后、时间不重复、非时间随机量重新抽取、两个梯度算术平均，
以及D/E/G/F各只step一次；关闭ST-CGR时直接dispatch到原生optimizer且不写方法状态。
冻结候选源码没有改变。

当前e7 full state在独立CPU进程中只读加载。checkpoint加载前后SHA256一致；59,871个
ordered pair全部位于非对角线，pair matrix的row/column sum分别精确等于保存的第一/第二
边际计数，Proposal update、G/F bundle和ST-CGR bundle计数也都等于59,871。经验频数只作
描述性诊断，未用于继续、停止或修改训练。

数学结论必须保持限定：无偏性针对给定post-D/E状态与batch的**pre-Adam联合G/F梯度**；
有限总体方差下降相对于Proposal的iid双视图时间耦合。它不声称Adam是线性的，不声称
完整sequential game转移与plain相同，也不保证PSNR改善。最终科学裁决仍必须等待e200、
合法matched plain和冻结统一评估。

因此裁决为`PASS_NO_CHANGE_CONTINUE_E200`：不改算法、不重启5090A、不移动checkpoint。
ST-CGR继续作为多算法前沿的一条独立时间分层估计器，而不是唯一算法。confirmation20
继续封存。

