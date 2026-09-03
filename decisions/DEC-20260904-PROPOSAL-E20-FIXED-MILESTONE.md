# Proposal全量e20固定里程碑验收

## 决定

接受5090C Proposal的e20完整checkpoint与固定评估产物，并保持当前训练和调度不变。
e20只是预注册轨迹点，不用于选择checkpoint、修改算法、提前停止或重排算力。

## 依据

- e20对应`171060`次更新，checkpoint实算SHA256与sidecar均为
  `213ac61bab087be0fe366b0e609b4eee03084b6d742025f70bbc4a98af951365`。
- sidecar锁定seed 2026、batch 1、官方image-proportional unpaired sampling、manifest、
  training commit及protocol fingerprint。
- 固定评估文件已生成；本次只记录文件大小和SHA256，未解析或读取任何性能值。
- Proposal随后已推进到e21，supervisor、trainer、export、health和progress watcher全部存活，
  零告警。
- 其余四个执行节点全部推进或健康等待；5090B两条continuity guard仍为零重启。

完整回执见
`evidence/paper_aio/PAPER_AIO_BIHOURLY_HEALTH_AND_PROPOSAL_E20_20260904T003000.json`。

## 边界

e20不进入主表裁决；主表仍使用e200，sustained仍使用e150/e175/e200。confirmation20继续
封存，不合并非等价runtime delta，也不因该里程碑新增或关闭算法。
