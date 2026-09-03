# confirmation20采用一次可恢复逻辑session

日期：2026-09-03

论文结果交付审计发现，仅有“confirmation20保持封存”的拒绝门不足以完成最终流程：仓库
没有在算法、基线、e200结果和论文主张冻结之后安全打开一次、跨故障恢复并最终封闭该
split的实现。同时DCLGAN缺少冻结后的KID/FID入口，会使包含该基线的最终portfolio无法
完成分布指标cohort。

本次补齐四层控制。第一，所有冻结lane的discovery80 KID/FID必须在同一个评估runtime
完成并形成hash绑定的distribution cohort，DCLGAN通过其锁定的官方源码adapter接入同一
接口。第二，机器只能生成不具授权力的confirmation review draft；另一个已经提交Git的
human/Codex decision必须逐项确认freeze、distribution cohort、lane集合、claim集合和
唯一session identity。第三，授权文件本身仍须提交Git后，才能原子地claim一个逻辑
session；故障只能恢复相同session，不能生成第二次开封。第四，全部冻结lane必须使用
固定e200 checkpoint、同一评估runtime和同一120张confirmation identity完成，才能写出
最终completion receipt；完成后session不能再次恢复。

confirmation只报告冻结后的PSNR/SSIM/LPIPS，UNSB固定NFE=5和五个预定rollout，外部
方法固定NFE=1，Input固定NFE=0。它不选择checkpoint、不修改算法或claim、不参与训练或
调度。所有checkpoint、模型状态和RNG在评估前后均复验。当前只提交控制接口和测试：不
创建review approval、不materialize authorization、不claim session，因此
`confirmation20_opened=false`仍然成立。
