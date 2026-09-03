# 五节点实时关键路径与租期边界

日期：2026-09-03 09:05 +08:00

对4090A、5090A、5090B、5090C和本地GTX1660执行metric-blind实时巡检。四条在飞远端
训练全部连续，trainer/supervisor/exporter/health watcher均存活，supervisor连续失败为0，
健康监控告警为0。未发现允许自动恢复的工程故障，因此不重启、不迁移、不增加伴随训练，
也不改变任何冻结科学协议。

实测最新进度和单epoch外推为：4090A plain e65，约2026-09-05 08:42完成；5090B CUT
e52，约09-06 05:54完成；5090B CycleGAN e30，约09-07 23:18完成；5090C Proposal
e9，约09-13 19:49完成；5090A ST-CGR e6，约09-15 04:25完成。外推只用于资源和
租期规划，不读取或使用paired performance。

5090C被用户登记为预计租用10天，但仓库没有权威的云端到期时间。当前Proposal尚需约
250.7小时训练，固定e200后的source export还需要额外余量。因此如果十天从其约
2026-09-02上线时开始计算，现有租期很可能比训练闭环短约1至2天。该风险不能通过增大
batch、修改算法、提前停止或跨宿主续训规避；建议把5090C至少保障到09-15，给e200、
checkpoint密封与source-bound export留下约24小时余量。

5090A ST-CGR的当前外推更晚，且本地没有权威租期记录；必须保障该宿主至少运行到
09-16。5090B后续还承担CUT完成后的exact twin和fresh-e0 matched plain，当前队列预计
延伸到09-12/09-13，故必须保障到09-14。4090A为本地长期资源，不新增租期假设。

当前持久接力覆盖：4090A plain→AM-TNC；5090B CUT→exact runtime gate→matched plain；
四条source-bound export；Proposal和ST-CGR到未来5090B control的review-only关系候选；
统一评估、算法posthoc裁决与最终discovery交付。所有timeout均覆盖当前外推窗口，磁盘
自由空间也高于各自保守写入需求。租期本身是唯一不能由进程内supervisor修复的外部风险。

本次巡检没有读取paired指标、没有打开confirmation20、没有更改任何远端进程或训练状态。
