# G3-03：条件采样—残差可行 Euclidean 协方差屏障

状态：`CONDITIONAL_PREIMPLEMENTATION_CARD`。

完整公式、父项资格、identity、target不可访问性和启动边界由
`decisions/DEC-20260831-RESIDUAL-FEASIBLE-EUCLIDEAN-CONDITIONAL-SYNTHESIS.md`
冻结。只有同宿主条件采样父项与RF-MCRB各自完成真实e200并严格通过，且固定的
e20/e100/e200组件兼容门通过后，才生成可执行JSON卡并获得长训资格。

它与G3-02不是强度网格：两者分别求Euclidean和Adam度量下的最近可行位移。两项都
使用represented residual和relative-ULP refinement，不允许复用旧固定绝对余量算子。

