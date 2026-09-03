# Paper e0 初始化暴露与外部基线语义审核

日期：2026-09-03

本次只读审核发现，当前冻结训练checkout在建立e0时先用首个stream sample执行
data-dependent initialization（DDI），随后才保存sampler state。因此当前已在运行的
UNSB family从stream位置2开始第一次optimizer update；CUT的primary stream同样如此；
CycleGAN的DDI不取数据，因而没有该偏移。

远端e0和完整状态直接验证了这一点：4090A plain、5090A parent/ST-CGR、5090C Proposal
的primary/secondary e0均为epoch1/cursor1；5090B CUT的primary为epoch1/cursor1、
secondary未使用；5090B CycleGAN两条stream均为初始状态。epoch-boundary full state和
scheduler state也与这一解释一致，既有exact-resume及重复评估门继续通过。

这不会改变一次data epoch等于8,553次optimizer update或e200等于1,710,600次update；
但当前冻结轨迹的每个8,553-update窗口由当前permutation的余下8,552项和下一
permutation的首项组成。对每条受影响stream，完整e200相对严格未偏移序列只替换一个
端点sample，比例为`1 / 1,710,600 = 5.846×10^-7`。这是一项有界的暴露索引偏差，
不是算法转移、恢复或matched-runtime失败。

裁决如下：

- 不重启、不迁移、不修改任何当前健康训练；用户要求的5090A ST-CGR-only调度保持不变。
- 当前UNSB family的合法matched comparison必须继续限定在共享同一legacy偏移和精确
  runtime证明的cohort内。偏移不会替代缺失的e200 plain control。
- 当前CUT和CycleGAN仍可报告固定e200绝对结果，但外部基线表脚注必须披露CUT的一项
  stream偏移以及CycleGAN无偏移；不得构造二者相对plain的matched delta。
- commit `f973c68ed71d5e2ad481f90a4012b4a4978127f0`已把未来fresh e0改为保存DDI前
  sampler state，并保留DDI后的模型和全局RNG，使首个optimizer update可确定性重放
  初始化所用batch。新策略指纹为
  `4cb394ea4b41cb546c38448173e96ee76be72f88eb602f4247bd200288cf9564`。
- 新策略只用于未来新output root；它与当前legacy-offset cohort不是同一runtime cohort，
  未经新的exact twin不得混合或计算matched delta。已经武装的5090B matched-plain
  successor必须继续使用其冻结旧checkout，才能服务当前Proposal/ST-CGR关系证明。

外部源码审核另确认：CUT的损失、更新顺序和关键默认值与固定上游commit
`b3ac297708dfb6f7589d04662277e53c0d579c27`一致；当前CycleGAN的损失、image pool、
G→D更新顺序和`100 constant + 100 linear decay`日程与官方实现一致，但生成器/判别器
使用本项目共享的CUT/UNSB antialiased、Xavier初始化骨干，而非当前官方CycleGAN默认的
normal初始化和非antialiased骨干。因此论文中必须写作
`CycleGAN (official-loss, controlled shared backbone)`，不能写成逐字官方复现。

审核没有读取paired performance，没有据此控制训练，没有打开confirmation20。
