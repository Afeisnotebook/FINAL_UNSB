# 终端低方差/谱漂移的事后因果裁决

日期：2026-09-03

已有终端审计已经修正为完整的`X_t → NFE5 endpoint` rollout Jacobian，但“看到谱异常”
本身不能证明它导致后续性能下降，也不能据此把一个新模块接入正在训练的算法。为避免用
paired结果倒推阈值，本项目在任何全量结果到达前冻结如下lead-lag裁决。

固定probe为4090A plain、5090C Proposal-only、4090A AM-TNC和5090A ST-CGR；固定epoch
为e100/e150/e200。裁决器必须先验签完整12个target-blind审计，以及每个审计的checkpoint、
源码、父状态和RNG不变证明。缺少任意一格时，paired metric binding连内容都不得解析。

诊断窗口固定为e100→e150，未来标签固定为e150→e200。paired标签只使用三者共同包含的
discovery70、replicate 0、NFE 5；e200额外10张和额外rollout不参与lead-lag标签。阈值在
结果到达前固定为：terminal increment trace与effective rank均降至不高于0.90倍，或末段
full-rollout Jacobian与finite perturbation gain均升至不低于1.10倍；未来下降定义为不高于
−0.05 dB。同一机理必须同时覆盖至少两个probe和三个域，不能把两种各自证据不足的病理
并集合并成“通过”。

paired结果必须来自单一统一评估runtime，且manifest、CRN bundle、lane/source host、
checkpoint、metric receipt和training protocol逐项闭环。该裁决不计算跨宿主matched delta；
它只标注每条lane自身的未来绝对变化，所以不会把非等价runtime误写成算法收益。

即使病理通过，输出也只授权编写一个derivation card和判死反例，不自动实现、启动或堆叠
终端修复模块。若未通过，则明确保持“不要加入终端模块”。整个successor不读取中间性能、
不控制训练、不选择最佳checkpoint，也不打开confirmation20。

为避免12格审计完成后再次依赖人工上线，持久successor只有在完整target-blind release
成立后，才获取本地共享GPU锁并在同一个冻结评估runtime内顺序生成12格标准统一指标，
随后自动写出绝对路径binding并裁决。生成中的paired值不会进入successor状态的调度逻辑；
若进程中断，已有metric必须通过原收据和哈希验证后才能继续。该自动化缩短工程关键路径，
不改变上述科学信息边界。
