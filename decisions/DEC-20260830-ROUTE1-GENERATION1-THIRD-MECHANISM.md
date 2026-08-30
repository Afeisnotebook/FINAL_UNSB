# DEC-20260830：用释放的多 seed 预算补足第三个独立数学机制

## 裁决

紧急协议不再自动运行 seed2027/2028。释放的预算不用于扩大 batch、旧算法调参或
寻找退出窗口，而用于实现 `G1-03-STATE-FEEDBACK-MISSING`：Moving Covariance Rate
Barrier（MCRB）。它占用原 derivation queue 已存在的第三个 Generation-1 名额，
不是对 BVCP 或 PC-RSMG 的组合。

## 为什么现在有构造权限

冻结 causal matrix 中没有任何跨方法共享控制信号，但 DT 有四个方法专属信号通过
预注册门。其中 `dt_covariance_mismatch_descent_margin` 有 4 条记录、未来符号准确率
1.0、Spearman 0.60、反转前兆领先率 1.0、平均 5/6 域一致。它只授权 DT 谱系的
state-feedback 构造，不能被写成普适控制器。

MCRB 不再拟合冻结 teacher 的绝对协方差值。它以候选自身的一 data-epoch EMA 为移动
参考，在相同当前无配对桥状态和 common-random-number latent 上测量方向协方差 gap；
原生 Adam 先生成真实位移，只有该位移会一阶放大 gap 时，才取到安全半空间的最近
Euclidean 位移。安全时完全保留 native Adam；稳定时 EMA 追上当前 G，算子自消隐。

## 与旧实现的边界

- 不是 DT frozen-teacher additive loss：没有绝对早期目标或 lambda。
- 不是 CNDRP/BCNRP：不按协方差敏感度缩放原生 gradient。
- 不是 HJ ACMP/FBCMP：不投影或 gate 辅助 HJ correction。
- 不是 HNEK PHRSUP：被约束的是移动 latent direction-covariance gap，且投影对象是
  PyTorch Adam 已产生的真实参数位移，不是 frozen-metric raw gradient。
- 不是 LBST：EMA 网络从不生成训练 rollout 或 inference endpoint。

## 不变边界

- seed2026、small25、batch1、共同 e0、真实 e200；
- 无 paired 控制、无 confirmation20、无 fixed window、无 handoff、无 best checkpoint；
- 400 updates 只作工程门，不能作科学判死；
- 先通过等价性、数学、zero-intervention、resume、RNG 和跨状态门，才允许长训；
- 单 seed 结果只叫 development signal，不声称跨 seed 稳定。
