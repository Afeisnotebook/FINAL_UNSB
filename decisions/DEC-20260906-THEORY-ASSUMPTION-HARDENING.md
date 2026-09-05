# 三条算法定理假设与数值主张加固

## 裁决

Proposal、ST-CGR 与 AM-TNC 的现有推导没有发现会要求中断或重启 full-data 训练的数学错误。
但论文级表述必须显式区分概率定理、冻结实现与浮点数值事实。因此，在读取任何 full-data
性能前，为共同无偏/协方差结论补充有限二阶矩、条件独立和交换性假设，并把 AM-TNC 的
零共识分支与浮点边界写入权威理论图。

ST-CGR 的半正定协方差改进只适用于冻结实现中“time 无放回、其他随机量逐视图独立”的
联合测度。对当前 `T=5`，相对 Proposal 删除的项为 `V_mu/8`；未来若共享 latent、bridge
noise 或 PatchNCE sampling，不能继续直接引用该式。

AM-TNC 的无偏结论只到 pre-Adam 梯度估计器。它要求度量由提交前 optimizer state 冻结、
不随 replica 顺序改变。零共识时源码提交有序第一 replica，交换后提交第二 replica，交换
平均仍回到零共识。一般交换恒等式是实数算术结论；除相同 replica 的显式 identity 分支外，
跨硬件实现只声称通过冻结数值容差，不声称逐位相同。

## 运行影响

本裁决没有改训练源码、超参数、随机过程、队列或任何健康进程；没有读取性能值，也没有
打开 confirmation20。它只收紧结果出来后允许写入论文的数学主张。

权威证据：
`evidence/paper_aio/PAPER_AIO_THEORY_ASSUMPTION_HARDENING_20260906T035100.json`。
