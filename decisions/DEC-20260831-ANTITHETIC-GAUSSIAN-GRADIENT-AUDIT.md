# DEC-20260831：Gaussian 反号梯度估计器不进入 e200

状态：`CLOSED_CURRENT_ANTITHETIC_INVOLUTION_BEFORE_LONG_RUN`

## 检查的问题

AEB把`G(x,z)`和`G(x,-z)`先做输出平均，改变了有限步endpoint law。这里检查一个不同的、
严格无偏的构造：在D/E已提交的固定状态，用相同batch、time index、PatchNCE patch和
非Gaussian随机量，把forward与G-loss中的全部Gaussian latent/bridge noise作`xi -> -xi`
变换，然后只平均两次G/F梯度。Gaussian对称性保证两个视图各自具有原生边缘分布，因而
梯度平均在条件期望上等于plain UNSB；推理和endpoint law不变。

## Target-blind结果

在本地plain e20/e100/e200固定状态各运行8 pairs，并同时构造两次独立原生采样的梯度
平均作为compute-matched对照。parent checkpoint、post-D/E网络/优化器状态和每对的RNG
消耗均精确保持；没有读取paired图像或质量指标。

| epoch | antithetic / native variance | iid-pair / native variance | antithetic / iid-pair | native-antithetic trace covariance |
|---:|---:|---:|---:|---:|
| 20 | 0.958985 | 0.715672 | 1.339979 | +47.354292 |
| 100 | 0.978793 | 0.950871 | 1.029365 | +402.727465 |
| 200 | 0.936517 | 0.601105 | 1.557993 | +162.823036 |

三个状态的反号梯度协方差都为正，且反号pair mean方差都高于两次独立采样。UNSB的非线性
bridge/GAN/SB/NCE梯度并不随Gaussian输入近似奇对称；形式上的`z/-z`对称没有转化成有效
的负相关control variate。

## 裁决边界

- 不冻结、不启动该Gaussian反号梯度算法的e200，节省一个长期槽位；
- 这只关闭“固定time/patch、全部Gaussian取反”的当前involution，不证伪所有无偏
  stratified/control-variate估计器；
- 已严格通过的PC-RSMG proposal-only使用条件iid两视图，仍是不同且有长期证据的算法；
- 不因本结果改变RF-AMMCRB、RF-MCRB、它们的消融或G3-02条件合成安排。

