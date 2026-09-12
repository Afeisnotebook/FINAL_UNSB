# DEC-20260912: DCLGAN e150 哈希与离机恢复闭合

## 决策

本地 GTX1660 上的 DCLGAN 已完成固定里程碑 `e150`（step `1,282,950`）。checkpoint、sidecar、heartbeat 与 supervisor receipt 在训练继续运行的情况下完成源端复核，并复制到4090A隔离恢复目录；逐文件 SHA256 一致，远端文件为只读。因此 `e150` 取代 `e125`，成为 DCLGAN 最新受保护的固定恢复点。

本次复制仅用于恢复保护，不构成统一评估的合法 import，也不产生任何跨运行时指标关系。论文评估仍必须等待既有 e200 source-bound exporter、验签、push 和 publish-last import 链。

## 影响与后继

- wrapper、supervisor、trainer、exporter 和 push 进程均未修改或重启，训练继续 e200。
- 未读取中间 paired 指标，没有依据 e150 结果调度、早停或选择模型。
- 下一固定里程碑为 e175；主表仍固定使用 e200，sustained 仍使用 e150/e175/e200。
- 若本地机器在 e200 前发生不可恢复损失，可以从4090A的只读 e150 full-state恢复；任何真实恢复必须单独记录 runtime provenance。

证据：`evidence/paper_aio/PAPER_AIO_DCLGAN_E150_HASH_OFFHOST_CLOSURE_20260912T140025.json`。
