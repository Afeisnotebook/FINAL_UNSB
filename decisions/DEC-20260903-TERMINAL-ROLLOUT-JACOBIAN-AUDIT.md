# 终端谱审计：完整rollout Jacobian而非单步netG代理

日期：2026-09-03

预注册的终端低方差/谱漂移审计要求用JVP幂迭代估计从每个桥状态到最终输出的
rollout最大奇异值。复核现有实现后发现，旧字段
`local_jacobian_top_singular_proxy`只测量当前一次`netG(X_t,t,z)`，而后续
桥状态仍通过此前endpoint更新，因此该局部量不能替代完整rollout Jacobian。
旧JVP初始方向还来自checkpoint RNG，不保证不同lane、epoch使用相同探针方向。

本次只修正离线审计器，不修改训练、checkpoint或任何算法：

1. 新增从`X_t`到固定NFE=5最终endpoint的可微数值rollout；梯度包含endpoint
   介导的后续桥状态转移。
2. 保留原有单步Jacobian，同时增加
   `rollout_jacobian_top_singular_proxy`，二者不再混称。
3. JVP和有限差分都从同一lane-blind CRN bundle的对应bridge-noise张量起步，
   因而跨lane与跨epoch可比。
4. endpoint方向一致性改为`endpoint - current bridge state`的运输方向，避免
   原始图像均值主导余弦。
5. successor验收现在必须看到六个域、五个time index、32个随机bundle、四个
   gradient replicates、完整rollout字段以及不变的model/RNG hash；旧的局部-only
   输出不能被误签收。

在本地GTX 1660上，用既有step-1 plain checkpoint完成了每域一张、2个rollout
replicate、1个gradient replicate的工程smoke。30个桥位置都生成了完整rollout
JVP字段，model和RNG前后hash一致；GPU峰值观测约5,924 MiB/6,144 MiB，未出现
OOM、NaN或状态污染。该smoke只验证实现和容量，不用于任何性能结论或算法选择。

旧的持久terminal-audit successor冻结于旧source hash，必须在本提交后以新commit
重新部署；在新successor通过至少两次健康轮询前，旧进程保持不动。

