# DEC-20260831：最终主项必须重新经过完整4090前沿裁决

状态：`ACCEPTED_AND_DURABLE_SUCCESSOR_ARMED`

## 问题

已有`final/CANDIDATE.json`只冻结了PC-RSMG三角色消融时的前沿。它不认识后续两条
residual-feasible修复算法、两种Generation-3几何或它们的新terminal receipt。若仍调用旧
finalizer，新算法即使真实e200更好也无法成为最终主项，这会把算法搜索错误地收缩成
旧候选验证。

## 新裁决边界

- 保留当前PC-RSMG proposal-only主项与旧完整候选，但不保护它们的名次。
- 等待4090两条修复复赛和两个G3终点结果；G3父门不通过时也必须写明确
  `inapplicable`结果，不把未运行写成负结果。
- 只排名source-bound、完整e200、共同4090 plain/e0的receipt；strict资格优先，再按冻结
  late-three/e200/域护栏/成本key排序。
- 唯一canonical id仍只是行动优先级；全部完整行、来源角色和结论边界均保留。
- 不合并跨宿主delta，不使用中间paired结果，不开启seed2027/2028或confirmation20。

实现为`research/local_route1/complete_frontier.py`及其4090持久后继，提交`757eb2d`。
