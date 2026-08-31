# DEC-20260831：evidence-backed alternate 可占用4090修复复核席位

状态：`ACCEPTED_RESOURCE_POLICY_EXTENSION_WITHOUT_SCIENTIFIC_GATE_RELAXATION`

## 覆盖内容

此前便携4090复跑队列只接受完整e200后的`strict_sustained`和
`causally_repairable_near_boundary_pending_target_blind_audit`。用户明确指出，当前多个
200-epoch结果差距很小本身说明收益保持已比旧50-epoch反转更接近目标；额外算力应推进
多数仍有证据的算法，而不是把唯一主项误作科学剪枝。

因此，RF-AMMCRB与RF-MCRB中完整e200后属于`evidence_backed_alternate`者也可以进入
4090的两个修复复跑席位。该分类要求late-three或e200至少一项为正；late-three和e200
都非正的`closed_current_operator`仍不得复跑。最多仍为两条，且只允许这两个已经冻结的
residual-feasible修复身份。

## 不变边界

- 复跑仍从4090共同e0重新训练到真实e200，不迁移5090 checkpoint；
- 每个算法只与4090 plain同宿主比较，不合并跨宿主delta；
- 5090的完整e200 terminal receipt、源码、公式、训练commit和trajectory hash仍是便携
  权威，任何中间paired结果都不参与；
- 这只是跨运行时复核资源分配，不修改公式、不授权数学修订、不降低最终持续收益排名门；
- G3-02/G3-03的两个父项仍必须在4090本机`strict_sustained`，alternate身份不能启动合成；
- batch1、seed2026、confirmation20封存、无route2/exit/handoff、无full-data保持不变。

这一扩展让4090的空闲资源用于回答“可能的长期信号是否受运行时影响”，同时避免把晚期和
终点都负的当前算子仅因算力充足而机械重跑。

