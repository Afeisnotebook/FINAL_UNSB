# DEC-20260831：完整多候选前沿，而非单算法科学剪枝

## 决策

路线一最终仍写出一个`CANDIDATE.json`，但它只表示下一阶段的默认行动优先级，不表示
其余算法被科学淘汰。完整e200链结束后必须额外写出`RESEARCH_FRONTIER.json`：保存
4090同宿主全部可排名轨迹、5090宿主分离的full/proposal-only/observable-only证据、
每条算法的前沿处置以及完整来源身份。

只要一个分支具有独立的长期机制证据，并且属于严格通过、因果上可修复的近边界或
证据型递补，它就继续留在研究前沿。单seed下几个十分之一dB以内的名次差只决定下一步
算力顺序，不能升级成父机制证伪。只有当前实现完成e200且既不满足正向证据、又没有
target-blind可修复缺陷时，才标记为`closed_current_implementation_on_current_protocol`；
这个词仍不等于`mechanism_falsified`。

## 当前算力使用

- 5090继续并行完成RF-AMMCRB与RF-MCRB的共同e0、batch1、seed2026、真实e200；随后为
  所有符合证据门的父算法完成各自proposal-only和observable-only流。
- 4090最多复跑两个来源绑定的修复算法，并分别保留Adam几何与欧氏几何的条件合成。
- 这不是超参网格：两条修复线对应不同约束几何，两个合成分支只有在各自严格父项和
  target-blind兼容门通过时才运行。
- seed2027/2028继续延期。额外算力先用于扩大独立算法/消融证据，而不是重复seed。

## 最终接口

- `CANDIDATE.json`：默认下一步行动主项；
- `ALTERNATES.json`：兼容既有合同的前两项递补；
- `RESEARCH_FRONTIER.json`：全部值得保留、修订或扩尺度验证的算法前沿；
- `RESULTS.json`：4090与5090完整宿主分离证据，不合并跨宿主delta。

最终交付守护器固定在`0f267ae2cf7cc99de9c7ae0decf888289e03da65`，只在4090完整
同宿主裁决和5090完整可携带机理证据同时到达后原子发布。中间paired指标仍不得参与
训练、路由、退出或checkpoint选择，confirmation20继续封存。

