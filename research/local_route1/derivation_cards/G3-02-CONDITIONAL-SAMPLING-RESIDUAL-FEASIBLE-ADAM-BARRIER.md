# G3-02：条件采样—残差可行 Adam 度量协方差屏障

状态：`CONDITIONAL_PREIMPLEMENTATION_CARD`。

完整推导、父项资格、公式、identity、target 不可访问性、反例和启动边界由
`decisions/DEC-20260831-RESIDUAL-FEASIBLE-CONDITIONAL-SYNTHESIS.md`冻结。只有同宿主
条件采样父项与 RF-AMMCRB 父项各自完成真实 e200 且严格通过后，才生成可执行 JSON 卡；
本文件和源码本身不构成长训授权。

旧 `G3-01` 使用数值语义失真的 AM-MCRB，不得执行。G3-02 使用实际 represented
residual 检验和 relative-ULP refinement 的 RF-AMMCRB，新旧算法身份、源码 hash、
receipt 和结果不可互换。

