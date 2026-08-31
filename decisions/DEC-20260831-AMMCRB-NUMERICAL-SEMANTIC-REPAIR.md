# DEC-20260831：AM-MCRB数值语义修复与多候选前沿

## 裁决

`F1-02-ADAM-METRIC-MOVING-COVARIANCE-BARRIER`继续运行到真实e200，以保留冻结实现的
完整诊断轨迹；但它不再被解释为推导卡中“唯一Adam度量最近可行投影”的有效检验，也
不得自动进入Generation-3合成。

原因与paired分数无关。源码审计发现它向解析投影系数的分子加入固定绝对float32余量。
当原生更新或缺陷切向量很小时，该余量会主导解析项，使校正相对原生更新无界增长，破坏
最近点和尺度不变性。确定性标量反例中，原生更新从`1e-2`缩小到`1e-8`时，旧实现的
校正/原生比从约`9.5e3`增大到约`9.5e9`。

## 修复身份

新身份为`F2-01-RESIDUAL-FEASIBLE-ADAM-METRIC-BARRIER`。它保持原数学问题、移动协方差
缺陷、Adam度量、EMA、target-blind输入和安全identity不变，只修复实数公式到参数dtype
的表示：

1. 在float64中计算解析系数，不加入固定余量；
2. 形成参数dtype中实际提交的位移并重新测量`<a,d>`；
3. 仅当表示残差仍为正时，加入`残差/分母`并推进一个相对参数dtype ULP；
4. 有界重检仍不可行则fail closed，不接受不安全更新。

这不是窗口、退火、强度调参或根据PSNR作出的修订。它不消耗父机制的一次数学修订额度，
但必须用新fingerprint从共同e0重训到e200。

## 资源与多候选原则

最终仍输出一个canonical主候选，以便复现和资源决策；研究阶段不因此只保留一条算法。
5090第二波最多保留两条有独立证据的e200轨迹：

- 修复后的RF-AMMCRB占一条，因为这是已注册公式的必要语义恢复；
- 第二条固定为`F2-02-RESIDUAL-FEASIBLE-EUCLIDEAN-COVARIANCE-BARRIER`。后续源码审计确认
  G1-03 MCRB使用同一个固定绝对余量；其e150曾达到`+0.894228 dB`且6/6域正，这使得从
  共同e0恢复其本来声称的欧氏最近点算子，比运行一个理论上必然等于plain的
  observable-only控制更有信息量。

两条修复线不是重复网格：RF-MCRB保持欧氏几何，RF-AMMCRB使用Adam几何。它们在同一
数值语义下的长期比较会区分“去掉尺度灾难已经足够”与“还需要优化器坐标适配”。PCNR
完整e200作为第三个、已在第一波支付算力的独立采样机制保留，不占这两个修复名额。

旧AM-MCRB完成、RF-AMMCRB完成之前，所有使用旧屏障的G3、自动4090复跑和最终交付链
保持暂停。PCNR完整轨迹独立有效，不因本事故失效。

## 不变边界

- 只使用seed2026作紧急发现；不声称跨seed稳定。
- small25、batch1、真实200 data epochs、共同e0和同宿主matched plain不变。
- confirmation20、paired训练控制、退出窗口、handoff、最佳checkpoint和全量训练继续封存。
- 中间paired指标不能启动、停止或改变RF-AMMCRB。
