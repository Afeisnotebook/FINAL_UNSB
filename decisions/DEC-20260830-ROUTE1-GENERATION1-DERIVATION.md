# DEC-20260830：Generation-1 只冻结两张有充分构造权限的新算法卡

## 裁决

最终 `474` 条长期反转证据与 `140` 条采样方差证据冻结后，Generation-1
只冻结以下两个新构造：

1. `G1-01-ROLLOUT-DISTRIBUTION-SPEED`：Bridge-Velocity Chord Projection
   （BVCP）。它改变的是构造训练桥状态的 no-grad rollout transition，最终可微
   endpoint、原生损失和推理均不改。
2. `G1-02-SAMPLING-VARIANCE`：Replicated Stochastic-Measure Gradient
   （RSMG）。它在同一官方 batch-1 无配对样本上串行计算两份独立原生随机梯度并
   在 Adam 之前平均；数据暴露量、PatchNCE batch 语义和期望目标不改。

两张卡分别绑定 causal matrix SHA256
`dc54569ac474706cbe001c061f94836f90b7a1baf0ba9be944e5ebbf4f87e0d3`
和 reversal atlas SHA256
`965faf9a4eaf7279aed4caddc4379b5da3892d3024652210238e90d3fad3d2e1`。

## 为什么不为了凑满三个而冻结 DT 状态反馈公式

自动 derivation queue 的第三条路线是 `state_feedback_missing`。它只有 DT 的
method-specific 信号通过门槛：最直接的 `dt_covariance_mismatch_descent_margin`
只有四个记录。最自然的实现是对某个 DT defect gradient 做一侧 Adam-metric
projection，但 SEARCH-005 的 `G2-HNEK-PHRSUP` 已经测试过同一约束模板，只是被约束
对象不同。当前矩阵虽允许“不同对象”的新测试，却没有足够证据说明这个 operator
优先于前两条路线。

因此第三条保持 `DERIVATION_REQUIRED`，不把名额本身当作科学目标。若 BVCP 或 RSMG
的跨状态工程门揭示一个新的、数值上明确的 DT defect，仍可在 Generation-1 上限内
形成第三张卡；否则它不会被换名重跑。

## 防漂移与算力边界

- 旧 DT/HJ/HNEK 继续只作为父证据，不作为候选。
- 不使用 paired PSNR/SSIM/LPIPS 生成权重、阈值或分支。
- 不使用固定介入窗口、退出点、handoff 或最佳 checkpoint。
- 4090/5090 只并行运行彼此独立的 batch-1 gate、反事实和候选轨迹；不会通过增大
  scientific batch size 改变无配对采样、PatchNCE negatives、博弈或 Adam 轨迹。
- 最多两个候选进入从共同 e0 开始的 small25 e200；400--800 updates 只排工程故障。

