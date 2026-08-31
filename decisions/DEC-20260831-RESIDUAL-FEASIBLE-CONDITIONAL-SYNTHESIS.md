# DEC-20260831：条件采样与残差可行 Adam 屏障的双机制合成门

状态：`PREIMPLEMENTED_NOT_AUTHORIZED_FOR_LONG_RUN`

## 原因

旧 `G3-01` 把条件采样和带固定绝对数值余量的 AM-MCRB 组合。后续 scale audit 已证明，
该余量会在小 tangent/native displacement 状态下产生无界比例的校正。因此旧 G3 只保留为
历史预注册设计，不得运行，也不得把它的结果或门禁转授给新合成。

新身份 `G3-02-CONDITIONAL-SAMPLING-RESIDUAL-FEASIBLE-ADAM-BARRIER` 只允许组合：

- 一个完成共同 e0、small25、batch1、seed2026、真实 e200 且严格通过的条件采样父项：
  PCNR full 或 PC-RSMG proposal-only；
- 同一宿主、同一 baseline authority 下完成真实 e200 且严格通过的 RF-AMMCRB full。

两个父项解决不同对象：前者修复 D/E 提交后 G/F 随机场的条件构造，后者把实际 Adam
位移约束在 moving covariance-rate 的 target-blind 安全半空间。任一父项未严格通过，
或不属于同一宿主 authority，均只写 `SYNTHESIS_INAPPLICABLE`，不启动训练。

## 数学算子

在 D/E 已提交的状态形成父采样器的 G/F 梯度 `g_R`，并由原生 Adam 得到实际位移
`d_R`。在产生该位移的同一 G/F 随机视图上计算 current/EMA direction-covariance
defect 的 tangent `a`。若父项是两视图 PC-RSMG proposal，则 `a` 是同一两视图、共同
latent 探针的交换对称均值。

令 `P=H^{-1}` 为 post-native-step Adam 对角逆度量：

```text
if <a,d_R> <= 0 or <a,P a> is numerically zero:
    d* = d_R
else:
    lambda0 = <a,d_R> / <a,P a>                 # float64
    d(lambda0) = represent_param_dtype(d_R-lambda0 P a)
    while <a,d(lambda)> > 0:
        lambda = next_dtype_ulp(lambda + <a,d(lambda)>/<a,P a>)
        d(lambda) = represent_param_dtype(d_R-lambda P a)
    d* = d(lambda)
```

最多八次 residual-only 数值 refinement；失败时 fail closed。没有固定绝对 margin、
强度、退火、epoch 窗口、paired 阈值、plain 输出或 checkpoint 选择。

## 启动门

1. 两个父 receipt 各自严格通过且 baseline authority 完全相同；
2. 旧 AM-MCRB source/fingerprint 不得出现在新 implementation；
3. e20/e100/e200、1/8/32-step target-blind 分支上，两组件相对 plain 的校正 cosine
   均不低于 `-0.2`；
4. 每行实际 parameter-dtype 位移满足 `<a,d*><=0`，parent hash 前后不变；
5. disabled identity、full-state resume、RNG/sampler 恢复和 400-update finite gate 通过。

门通过只授权从共同 e0 的一条 e200 合成轨迹，不授权调参或减少现有独立算法前沿。

## 与多算法前沿的关系

该合成是独立候选，不替代 RF-AMMCRB、RF-MCRB、PCNR 或 PC-RSMG proposal-only。
最终唯一 `CANDIDATE.json` 仍只表示下一步行动优先级；所有完整 e200 父项与合成结果均
保留在科学排名、消融和失败机理记录中。5090/4090 的数值只在各自宿主内排序，不合并
delta。confirmation20 继续封存。

