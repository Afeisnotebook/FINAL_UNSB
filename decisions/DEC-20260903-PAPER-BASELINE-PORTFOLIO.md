# 论文基线组合：主表、分域上下文与 paired ceiling 分层

全量长训已经开始，但此前的交付矩阵只明确了正在运行的 Input、CUT、CycleGAN 与
UNSB 家族，没有把“六域 All-in-One 直接比较”“单退化上下文”以及“paired ceiling”
严格分开。若在结果出现后再划分，容易发生按结果挑表、把不适用域补零或把 paired
方法描述成同协议对手。现在在不读取任何性能值的前提下冻结这三层边界。

论文核心受控表保留 Input、CycleGAN、CUT、plain UNSB、Proposal-only、ST-CGR 和
AM-TNC。DCLGAN 是已经完成作者源码绑定与完整恢复门的非阻塞扩展；它按既有 5090B
队列运行，不阻塞较早的核心结果。DDSB 仍是重要的当代直接对手，但当前没有通过作者
源码—公式—恢复锁，因此只能如实标为 `reproduction_incomplete`，不能猜实现或生成
数值。NEGCUT 的暂缓是工程和许可状态，不是机制证伪。

DehazeSB 的作者仓库与固定 commit 已确认，但它是 dehazing 专用方法。只有后续完成
source-bound 适配并限定到适用域时才能作为分域上下文；不得把缺失的其余域拼成六域
宏平均。其他单退化 Schrödinger-bridge 工作在身份和源码未锁定前只保留文献搜索槽位。

RestoreVAR 与 PromptIR 的作者仓库已确认，但二者属于 paired/pretrained All-in-One
restoration ceiling。若时间允许，只能在算法冻结后对兼容重叠域做单独 ceiling 推理，
清楚披露训练监督与数据差异；不能与无配对方法计算 matched delta，也不能占用当前
核心全量算法长训资源。

本裁决不新增训练、不改变四张远端卡的队列，也不读取中间 paired 指标。它只提前冻结
最终论文表格的语义，确保缺源码、缺域或监督条件不同不会在结果出来后被掩盖。

权威配置：`configs/PAPER_BASELINE_PORTFOLIO.json`。
