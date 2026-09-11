# DEC-20260912：terminal audit 保持本地，不建立低收益跨机分支

## 关键路径结论

当前 terminal audit 已完成 plain 的 e100/e150/e200 三个单元，剩余 9 个。按真实 GPU smoke
约 777 秒/单元估算，剩余约 1.94 小时。DCLGAN 预计约 68.66 小时后释放本地 GTX1660；最晚
输入 ST-CGR 预计约 70.17 小时后完成，因此审计预计约 72.11 小时后完成。当前总关键路径
5090B matched plain 预计约 83.08 小时后完成，terminal audit 仍有约 10.97 小时余量。

将 AM-TNC 三个单元提前放到 4090A 理论上最多节省约 0.65 小时，却需要新增跨机 audit receipt
重绑定或新的结果权威，会扩大论文解释与恢复面。因此不为了表面并行建立新分支：本地审计链
保持健康等待 DCLGAN 释放，4090A 在 AM-TNC e200 后优先承担固定 import 与统一评估。

只有后续实测 ETA 显示 terminal audit 会比 matched plain 至少晚 2 小时，并且已有经过测试的
hash-preserving receipt 合并门，才重新审议迁移。此次没有停止、重启或修改任何训练。
