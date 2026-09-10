# DEC-20260910：ST-CGR e125 固定里程碑与异机保护闭环

## 裁决

5090A 上的 ST-CGR（`G4-01-STRATIFIED-TIME-CONDITIONAL-GF`）在 supervisor/trainer PID 336345/863376 不变、outer guard 零重启的条件下，按冻结协议自然完成 e125（1,069,125 updates）。固定 checkpoint、sidecar 与指标产物均完成 SHA256 闭环，并复制到 `E:/UNSB_Expl/recovery_backups/5090A_STCGR_20260910_E125` 后设为只读；指标文件仅复制和哈希，没有读取数值。

## 后继

- 保持现有训练连续运行，不迁移、不重启、不改算法协议。
- e150 由既有 source-bound incremental exporter 处理；e125 不产生算法裁决。
- 合法 matched delta 仍等待已准入 runtime relation 下的 5090B matched plain e200。
- 不使用中间 paired 指标控制训练或调度，不选择最佳 checkpoint，confirmation20 继续封存。

机器可读证据：`evidence/paper_aio/PAPER_AIO_STCGR_E125_HASH_AND_OFFHOST_CLOSURE_20260910T144500.json`。
