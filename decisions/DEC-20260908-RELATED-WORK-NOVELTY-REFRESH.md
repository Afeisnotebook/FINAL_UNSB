# DEC-20260908：相关工作与新颖性边界增量复核

## 裁决

在不读取训练中 paired 指标、不改变算法和 GPU 队列的前提下，按当前日期复核一手论文页、
论文原文、作者项目页与作者仓库。原边界仍成立，但必须新增四个约束：SBF 已覆盖在线无配对
SB 的宽泛叙事；IBCD 已覆盖 bridge consistency 与单步双向无配对翻译；E-Bridge 已覆盖
低能量 restoration bridge、entropy-regularized 起点与少步 consistency solver；通用不放回
梯度估计早已覆盖“不放回且无偏”本身。

因此 Proposal、ST-CGR、AM-TNC 的潜在贡献继续限定在固定 UNSB 顺序博弈内部的
player-conditional state boundary、有限 bridge-time coupling 和 same-player Adam-metric
geometry，不能扩张成上述通用首创主张。

## 实验和复现影响

- DDSB 仍未找到足以关闭公式—源码—实现歧义的作者实现，保持
  `reproduction_incomplete`，不是机制失败；
- IBCD 项目页仍写明 code coming soon，不能临时伪造成已复现基线；
- E-Bridge 虽有作者代码，但 foundation-model、任务和推理协议不等价于当前六域无配对
  All-in-One 主协议，只作为相关工作/ceiling 语境；
- 本轮没有改变冻结主表、健康训练、confirmation20 或任何 successor；
- 在最终摘要和 claim freeze 前仍需进行一次当日一手来源复核。

原八分支写作合同对相关工作文档做了 fail-closed 哈希绑定。增量文档形成后，完整回归按预期
拒绝旧哈希；因此本裁决同时将合同重新绑定到新文档 SHA256
`0e6ed53cf7a44caead5e4b06bd70e6ffacf5665436a3dd185a6ebb72bed93585`，合同新 SHA256 为
`16af7c33f766b2b8161d378cbb70eafb4b5ac26b66acaa186e542871336a984e`。八种算法结果分支、
门禁和历史证据均未改写。

机器可读回执：
`evidence/paper_aio/PAPER_AIO_RELATED_WORK_NOVELTY_REFRESH_20260908T033000.json`。
