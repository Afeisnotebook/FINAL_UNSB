# DEC-20260912: AM-TNC e175 离机恢复点完成

## 决策

4090A 上仍在运行的 AM-TNC 进程保持不动。`e175` 固定 checkpoint、sidecar、lane heartbeat 与精确授权文件已经复制到本机独立目录，源端与目的端 SHA256 一致，目的文件只读。因此，`e175`（step 1,496,775）取代 `e170`，成为训练状态的最新离机恢复权威。

原 `/home/yc/unsb_cov` 路径仍不存在，当前进程的解释器仍属于 deleted inode；这项事实没有被掩盖，也没有通过覆盖旧路径来“修复”。若健康进程意外退出，只允许恢复监督器在启动前重新核验隔离 runtime 的 Python、111 个映射文件和最新 full-state，并从该隔离 runtime 精确恢复。

## 影响

- 普通进程退出的最坏进度损失已从旧的 e170 后退缩小为 e175 之后最多一个未完成 epoch。
- 即使 4090A 整机或本地存储同时损失，e175 full-state 与隔离 runtime 的离机归档仍可用于恢复。
- 健康训练、sampler、RNG、优化器和方法状态未被重启、迁移或修改。
- 实际发生恢复后第一步 CUDA 更新能否与当前 deleted-inode 进程逐位一致，仍不可在不中断健康训练的前提下证明；发生恢复时必须如实标注该 provenance 边界。

## 后继

训练按冻结协议继续到 e200。e200 到达后使用既有 source-bound exporter、统一评估和 terminal audit 链路闭环；不重复复制 e175，也不读取 paired 指标来控制训练或调度。confirmation20 继续封存。

证据：`evidence/paper_aio/PAPER_AIO_AMTNC_E175_OFFHOST_PROTECTION_COMPLETE_20260912T130435.json`。
