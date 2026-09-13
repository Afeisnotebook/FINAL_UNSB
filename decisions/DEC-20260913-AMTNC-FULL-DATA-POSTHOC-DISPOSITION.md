# DEC-20260913：AM-TNC全量长期事后裁决

在AM-TNC训练完整到e200、十个plain/AM固定epoch只读评估单元全部完成之后，首次读取其
paired discovery结果。合法同宿主、审计过的method-only recovery关系和所有晚期CRN门均通过；
因此下面是科学结果，不是CLI故障或checkpoint损坏造成的假失败。

AM-TNC在e100相对plain为`+0.066 dB`且5/6域PSNR为正，但SSIM与LPIPS已轻微退化。此后
PSNR delta在e125/e150/e175/e200依次为`-0.419/-0.134/-2.117/-4.072 dB`；e175和e200均为
0/6域正。晚三点均值为`-2.108 dB`，e200同时退化`SSIM -0.1334`、`LPIPS +0.1285`。
AM自身从e150到e200下降`5.564 dB`，plain同期下降`1.626 dB`，所以不能把结果解释成仅由
plain终点坍塌或评估噪声制造。

裁决为：冻结AM-TNC当前实现不具备全量200 data epochs长期收益，标记
`closed_current_implementation`，不追加seed或把e100选作最佳checkpoint。该结果强烈否定当前
持续operator，但不等同于证伪整个Adam-metric/tangential geometry思想；后者只能在未来由新的
长期因果证据和新推导重新开启，不能为填卡立即重跑。

Proposal、ST-CGR、DCLGAN和5090B matched plain的健康轨迹与队列不变。指标只在完整固定评估
后事后读取，未用于训练、调度或早停；confirmation20继续封存。
