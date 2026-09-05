# 论文冻结必须绑定精确算法理论证据

## 裁决

Proposal-only、ST-CGR 与 AM-TNC 的数学材料已经足够，不再为了文件形式重复发明一张
Proposal 专属 derivation card。Proposal 的父卡明确给出 `proposal_only` 消融，族推导给出
post-D/E 条件双视图估计器，full-data formula audit 又把它绑定到冻结源码和运行状态；三者
共同构成完整且更诚实的谱系。

真正的缺口在结果冻结端：旧 two-stage freeze 绑定最终 portfolio 和文字 claim，却没有强制
证明 Codex/人工审阅的是哪一版推导和公式—源码映射。现在新增
`configs/PAPER_ALGORITHM_THEORY_BUNDLE.json`，逐文件冻结共同理论图、三条 derivation 谱系和
formula/source audits。机器 review draft 会写入 bundle、object 和 artifact-set hash；人工/Codex
decision 必须逐位复述同一引用；freeze receipt 物化及其后的 distribution、manuscript table、
confirmation 路径会重新哈希全部文件，任何漂移都 fail closed。

冻结入口还会核对最终 portfolio 的方法集合、algorithm ID 与 lane ID 必须逐项对应 bundle
中的 Proposal-only、ST-CGR、AM-TNC。这样不能用一套正确的理论哈希为另一套结果或一个
悄悄缩减的算法集合背书。

对称性复核还发现 AM-TNC 原先只引用 route1 阶段审计。只读比对证明旧审计提交与当前
full-data 提交中的 operator、model registry 和 derivation card 是相同 Git blob；4090A
固定 e20 checkpoint 的双 bundle 计数、五事件 schedule、RNG 与双 sampler 状态也一致。
因此新增 full-data 现场审计并由 bundle 直接绑定它，旧审计作为其哈希父证据保留。

## 科学边界

这不是对算法、超参或训练队列的修改，也不读取 full-data 性能。共同定理仍只到 pre-Adam
条件梯度均值；不声明期望 Adam 位移、完整 Markov kernel、等 FLOP 优势或 terminal singular
drift 修复。最终 full-data 收益、多方法集合及论文主张仍必须等待固定 e200 与合法 matched
control 后裁决。

在审计时，AM-TNC e22、ST-CGR e55、Proposal e61、CycleGAN e133、本地 DCLGAN e15 均保持
健康；5090B matched plain 的两 epoch metric-blind 容量门仍在 e2 运行，未触碰任何训练进程。
