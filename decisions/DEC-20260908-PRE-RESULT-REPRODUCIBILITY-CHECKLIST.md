# DEC-20260908：冻结论文复现与 Artifact 清单

## 裁决

在 full-data 结果产生前冻结一份英文 reproducibility and artifact checklist。它不是实验结果，
不选择算法，也不改变训练队列；其作用是提前规定论文数值、运行身份、恢复事故和公开产物必须
如何被审计，避免结果出现后按赢家重写复现标准。

清单由 `configs/PAPER_REPRODUCIBILITY_CHECKLIST_CONTRACT.json` 绑定，并进一步绑定可执行协议、
三算法理论包、外部基线组合、runtime relation registry、结果分支合同和英文 Methods 草稿。
任何 prose 与这些权威对象冲突时，以被哈希绑定的可执行对象为准。

## 强制范围

- 每条 lane 必须公开 commit、protocol/lane fingerprint、manifest、e0、runtime cohort、完整命令、
  checkpoint/sidecar/metric/export 哈希及共驻情况。
- 主结果固定 e200，长期行为固定 e150/e175/e200；不允许最佳 checkpoint 或跨非法 runtime delta。
- 第一波只有 seed 2026，必须明确为 single-seed evidence，不能声称多 seed 稳定。
- DDSB 在权威复现门未通过前继续标记 `REPRODUCTION_INCOMPLETE`，不能生成猜测结果。
- 4090A deleted-inode 清理事故作为工程 provenance 披露，不把它写成算法收益或失效证据，也不
  声称未经证明的 CUDA 下一步逐位等价。
- confirmation20 只有在方法、claims、表格与图形共同冻结后才能按单次合同开启。
- 凭据、SSH 密码和私钥不得进入 Git artifact。

## 当前影响

这项工作不占用训练 GPU、不读取中间 paired 指标、不修改任何 PID 或队列。完成时 4090A
AM-TNC、5090B CycleGAN/matched plain 和本地 DCLGAN 均保持健康运行。目标测试覆盖哈希绑定、
协议常量、科学边界和无经验收益表述。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_PRE_RESULT_REPRODUCIBILITY_CHECKLIST_20260908T081500.json`。
