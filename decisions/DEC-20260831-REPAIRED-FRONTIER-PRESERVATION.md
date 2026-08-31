# DEC-20260831：修复前沿不提前收缩为单算法

状态：`ACCEPTED_AND_IMPLEMENTED`

## 决策

`CANDIDATE.json`中的唯一主候选只表示“下一份算力先给谁”的行动优先级，不表示算法发现
阶段只能留下一个方法。小幅的e200收益差距也不构成提前关闭独立机制的证据。

5090第二波完成后，同宿主科学排名固定包含：

1. `F1-01-PLAYER-CONDITIONAL-NATIVE-RESAMPLING`（PCNR）；
2. `F2-01-RESIDUAL-FEASIBLE-ADAM-METRIC-BARRIER`（RF-AMMCRB）；
3. `F2-02-RESIDUAL-FEASIBLE-EUCLIDEAN-COVARIANCE-BARRIER`（RF-MCRB）。

旧`G1-03 MCRB`和旧`F1-02 AM-MCRB`均完整保留为冻结实现诊断，但固定绝对余量事故使
它们不再有资格代表推导卡的最近点算法，不能混进科学排名或合成。

裁决同时输出：

- 一个`action_priority_candidate_id`；
- 两个`priority_alternate_candidate_ids`；
- 全部同宿主排名、分类检查、逐轨迹哈希和计算成本；
- 旧实现诊断及其语义事故绑定；
- 所有strict/near机制的4090复跑优先队列。

主候选、备选和完整前沿是三种不同层次：主候选解决“下一步先做什么”，备选防止单次
复跑失败后回到旧算法，完整前沿用于继续数学修订、消融和论文机制解释。

## 5090后续使用原则

当前两个batch1槽已经分别保留给RF-AMMCRB与RF-MCRB，因此不塞入第三条并发训练。
两条完整e200后，5090不会默认闲置或只做多seed；它优先执行以下证据门通过的工作：

1. 严格通过算法自身的proposal-only、observable-only、projected/full消融；
2. 对“缺陷量下降但长期收益反转”的机制进行唯一一次target-blind数学修订；
3. 两个standalone长期为正且数学兼容时，执行最多两组件的Generation-3合成。

这些工作均需新identity、共同e0、真实e200和完整推导卡。不得因paired得分差异设置窗口、
改变公式或调参。seed2027/2028继续让位于更有信息量的机制探索，直到用户另行要求。

## 工程实现

- `research/local_route1/repaired_frontier_adjudication.py`负责语义事故感知的同宿主排名；
- `operations/local_route1_repaired_frontier_successor.py`只在五条轨迹均有source-bound
  terminal receipt后自动裁决；
- `research/local_route1/repaired_frontier_followups.py`不只消费第一名：两条数值修复中
  凡属于strict、near或仍有独立正证据的alternate，最多两条都冻结自身proposal-only与
  observable-only；closed当前算子不会因空闲算力被机械重跑；
- `operations/local_route1_repaired_followup_successor.py`在5090上为每个合格父算法保留
  一个独立流，父算法之间最多两流并行，父算法内部按proposal-only→observable-only串行，
  每项仍从共同e0完成batch1/seed2026/e200；
- `algorithm_discovery_collapsed_to_single_candidate`被固定为`false`；
- `canonical_candidate_is_action_priority_only`被固定为`true`；
- paired指标只在完整e200后参与排序，不进入训练、公式或运行调度。

## 不改变的边界

- 北极星仍是长期算法发现，不是旧算法验证；
- batch1、small25、seed2026、200 data epochs不变；
- 不打开confirmation20；
- 不合并4090与5090的delta；
- 不用单个中间checkpoint提前淘汰；
- 不把当前实现失败升级为父机制被证伪。

上述多父后继已在提交`52c0ad3`以独立screen武装；compact记录见
`evidence/remote_route1_offload/REPAIRED_MULTI_PARENT_FOLLOWUP_ARMED_20260831.json`。
