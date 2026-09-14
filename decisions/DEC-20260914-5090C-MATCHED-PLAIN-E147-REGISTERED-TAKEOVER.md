# 5090C 接管迁移后的 matched plain：从 e147 严格分段续训

## 决策

5090C 不重复运行已经完成并完成 source-bound 导出的 Proposal-only。它接管迁移自已关机 5090B 的逻辑 `5090B_MATCHED_PLAIN` 轨迹，从磁盘中最新、完整且逐哈希可信的 e147 full state 精确续训到 e200。

这不是把不同运行时的结果直接相减，也不是把新宿主伪装成原物理 GPU。最终报告必须把该轨迹标为 `SEGMENTED_EXACT_RUNTIME_COHORT`，保留原逻辑 lane 名称及每个物理段，并在 e200 后重新审核 runtime relation，之后才允许生成 matched delta。

## 为什么这是当前最合理的恢复

- Proposal-only 已经完整到 e200；重跑它不会增加论文证据，却会继续阻塞 Proposal、ST-CGR 以及统一评估所需的合法 matched control。
- 迁移盘上的 e147 比 Git 中旧的 e134 状态更新，并且 full state、sidecar 与 scientific-state 三组哈希一致；没有理由丢弃 13 个已经完成的 data epochs。
- 当前 5090C 是已登记宿主，但仍重新完成了物理身份门、2000-update exact runtime twin 与两个相互独立的 8-step exact-resume probe。两个 probe 的 checkpoint/scientific hash 逐位一致，因而允许工程上的分段精确续训。
- 5090A ST-CGR、本地 DCLGAN 和 4090A 交付链均健康；它们未被停止、迁移或重启。把空闲的 5090C 用于 matched plain 是对现有科学关键路径的直接缩短。

## 已执行

1. 在 5090C 上把 e147 源状态复制到只读恢复目录，并逐哈希确认；又复制到本地 `E:/UNSB_Expl/recovery_backups/5090C_MATCHED_PLAIN_20260914_E147` 形成异机备份。
2. 在隔离 control worktree `50f58d6` 执行身份门、exact runtime twin 和双 resume probe，全部通过。
3. 使用原科学 checkout、原 run 目录、原 commit、原 protocol、原 sampler/RNG/full-state，从 e147 启动 supervisor 和 trainer；外层 guard、progress watcher、source exporter、incremental exporter、两类恢复监督器和聚合 health watcher均已部署。
4. 本地与 4090A 的 relay 全部重绑到 36525 端点；旧 43172/44804 的本地定时任务及只读支持进程已退休，防止重复拉取或误恢复。训练进程没有被这些控制面操作影响。
5. 已完成的 Proposal checkpoint、导出集和两条异机导入链保持不变，没有覆盖，也没有启动 Proposal 重跑。

## 当前验收边界

5090C 的基本健康验证已经通过：trainer 持续消耗 CPU/GPU，guard 零重启，progress 与聚合 health 零告警，磁盘最坏写入估算低于真实剩余空间，并保留 `USER_CAPACITY_OVERRIDE`。第一个迁移后完整 epoch 是 e148；在其 full state 与 scientific hash 发布前，只宣告“主训练健康运行”，不把分段连续性写成已最终闭合。

后续顺序固定为：e148 闭合 → e150 增量异机导出闭合 → e200 source-bound 导出 → runtime relation 最终审核 → 4090A 统一固定点评估与论文组合裁决。全过程不读取中间 paired 性能控制训练或调度，confirmation20 继续封存。

## 风险与回滚

- 若 e148 前 trainer/supervisor 消失，只允许 frozen guard 从 e147 完整状态恢复；不得启动第二条主训练。
- 若 e148 已闭合后发生工程故障，从最新完整 epoch exact-resume；不跨宿主迁移，也不退回旧 e134。
- 5090C 预计 e200 在 2026-09-16 09:00–21:00 完成，取决于迁移后完整 epoch 实测吞吐。e148 完成后必须据真实 wall time 收窄区间。
- 只有 e200 source export、异机导入及 runtime relation review 全部通过后，matched delta 才可用；此前 Proposal、ST-CGR 的算法裁决保持 pending。

权威证据：`evidence/paper_aio/PAPER_AIO_5090C_MATCHED_PLAIN_E147_REGISTERED_TAKEOVER_20260914T235729.json`。
