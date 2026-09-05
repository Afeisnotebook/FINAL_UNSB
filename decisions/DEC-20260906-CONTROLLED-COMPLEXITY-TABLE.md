# 论文复杂度表的受控训练成本闭环

## 裁决

Proposal、ST-CGR 和 AM-TNC 的每次 optimizer update 包含不同数量和耦合方式的随机视图，
只报告参数量或推理延迟不足以支持论文中的收益—成本判断。另一方面，各 lane 来自不同宿主
且部分曾共驻，直接拿训练期间的 epoch wall time 做算法倍率也不具可比性。

最终复杂度表现在只使用冻结后、同一 evaluator runtime、同一 profiler 对各 e200 checkpoint
执行的完整训练 step 计时。它强制导出 median、p90、CUDA peak allocated memory，以及相对
同一环境 plain median 的倍率。推理侧固定报告 UNSB family 的 NFE5、外部 translation 方法
的 NFE1，同时保留完整 NFE JSON 与环境记录。plain profile、p90、峰值显存或参考 NFE 任一
缺失都会 fail closed。

FLOPs 仍不声明：当前自定义 stochastic bridge 与 lazy PatchNCE 尚无覆盖完整 operator 的
审计计数器。该限制比输出一个不完整 FLOP 数字更可解释。

本修改不读取性能、不改变任何在飞训练、不改变 checkpoint 或算法配置，也不授权
confirmation20。

实现提交：`bebfa05b6a66e4c2caf18a2a13fba57364a96ce0`。

证据：`evidence/paper_aio/PAPER_AIO_CONTROLLED_COMPLEXITY_TABLE_20260906T041200.json`。
