# 5090B runtime relation 准入与 gate-5 工程恢复

## 裁决

5090B matched plain 的五项门禁没有暴露科学失败。前四项均已通过；第5项被 e4a5eed 中已知且随后由 909a2c3 修复的回执字段混用阻塞：重复评估已经逐位一致，但旧实现把 evaluation-bundle fingerprint 写入 training-protocol fingerprint 字段。

本次只迁移这一已知元数据字段。原始失败状态、重复评估回执和失败授权均按字节归档；runtime twin `c1471a79...0b82` 未重跑、未改变。修复后的重复评估回执为 `2249627f...f7c7`，e4a5eed 原生授权代码重新签发 PASS 回执 `e43cb66d...db16`。随后由 commit `f8c38c1` 的恢复 successor 从门禁之后进入预注册的两 epoch、metric-blind 共驻容量门。

当前 successor PID `523993`，容量探针训练子进程 PID `524019`，健康与进度 watcher 分别为 `524155` 和 `524156`。CycleGAN 及其他健康训练未被中断或修改。容量门仍只能依据 epoch wall time 计算 makespan；它不得读取 paired 结果。

## Runtime relation

4090A 上自动生成的两个 review-only candidate 及统一 review receipt 均通过：

- Proposal `5090C -> 5090B_MATCHED_PLAIN`：`PASS_EXACT_RUNTIME_RELATION`；
- ST-CGR `5090A -> 5090B_MATCHED_PLAIN`：候选到父 runtime 与父 runtime 到 plain 的两段证明均通过。

review 生成的 proposed registry SHA256 为 `e728492b...f891`。本提交把与该文件字节完全一致的内容纳入 `configs/PAPER_AIO_MATCHED_RUNTIME_RELATIONS.json`，同时保留原有 Proposal `5090C -> 5090A` 关系。由此准许未来在各自 e200 固定评估完成后做严格 matched comparison，但当前 plain 尚未完成，所以不存在也不得声称任何实际 matched delta。

## 后续

1. 等待两 epoch 容量门完成，核对唯一进程、完整状态和 `CORESIDENT_MAKESPAN_CAPACITY_GATE.json`。
2. 若裁决为立即共驻，则确认唯一 plain supervisor/trainer；若裁决为等待 CycleGAN 释放，则保留 e2 full state，不新增训练。
3. 关系准入后部署 dynamic unified/final-delivery successor，替换仍绑定旧 5090A plain 的 legacy completion path。
4. confirmation20 保持封存，所有中间性能值继续不得参与调度。

完整机器证据见 `evidence/paper_aio/PAPER_AIO_RUNTIME_RELATION_ADMISSION_AND_GATE5_RECOVERY_20260906T053848.json`。
