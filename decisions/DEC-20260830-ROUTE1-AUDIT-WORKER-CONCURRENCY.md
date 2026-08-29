# DEC-20260830：路线一因果审计双 worker 裁决

## 裁决

同一宿主仍由一个 durable auditor 独占审计队列和最终矩阵，但 RTX 4090/5090
允许最多两个相互独立的 `(probe, data_epoch)` worker。GTX 1660 默认保持单 worker。
这只改变已授权审计单元的墙钟调度，不改变 batch、更新算子、horizon、observable、
paired 标签时机或候选生成门。

## 并发前必须成立的隔离

- 每个审计单元使用独立 `audit/work/<probe>_e<epoch>`，不共享 options 或临时模型目录；
- `LONG_REVERSAL_ATLAS.jsonl` 和 `SAMPLING_VARIANCE_ATLAS.jsonl` 的
  read/merge/replace 使用跨进程文件锁，canonical 排序和重复行一致性检查保持不变；
- `LONG_CAUSAL_MATRIX.json` 的重建同样串行加锁，最终完整性仍由冻结 queue 的 expected
  row keys 判定；
- stall 计时只观察当前 `(probe, epoch)` 的已落盘行数，不能由另一个 worker 的进展
  掩盖卡死任务；
- 每个虚拟分支继续验证 parent full-state hash 前后一致；confirmation20 和 paired
  controller 继续不可访问。

## 资源策略

4090 在 DT e200 和终点验收后以 `maximum_parallel_jobs=2` 启动审计。若第一批双
worker 出现 OOM、非确定性、父状态污染、非一致重复行或总吞吐不增，durable
supervisor 按工程失败停下，不把不完整 atlas 解释为科学证据，并以同一已落盘行恢复
到单 worker。5090只有在本机四条锚点均验收后采用相同策略。

正式候选训练仍为 `batch_size=1`，本裁决不授权预写候选、超参网格、退出窗口、
跨宿主 delta、全量数据或 confirmation20。
