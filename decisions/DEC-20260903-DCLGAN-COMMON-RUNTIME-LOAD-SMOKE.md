# DCLGAN统一评估运行时checkpoint加载门

日期：2026-09-03 14:05 +08:00

DCLGAN的Windows/GTX1660正式1000-update工程门已经证明适配器的恢复和重复评估正确，
但未来论文评估将在4090A的Linux、PyTorch 2.5.1+cu121统一容器中完成。只做Python import
不能证明350MB完整checkpoint能跨平台、跨PyTorch小版本恢复；若在5090B完整e200后才发现
这一问题，会造成数日级交付延迟。

因此把正式门的只读checkpoint复制到4090A独立工程目录，在CPU上通过冻结的e45973a
适配器与DCLGAN作者commit f7a7b8e执行真实模型构建、完整状态恢复和一张discovery输入的
两次target-blind前向。该检查没有读取target图像、没有计算PSNR/SSIM/LPIPS、没有占用GPU，
也没有改变4090A plain训练或任何队列。

结果为`PASS_CROSS_RUNTIME_CHECKPOINT_LOAD_AND_TARGET_BLIND_INFERENCE`：原checkpoint SHA256
保持`92a975...4363e`；两次输出逐字节hash均为`d10b26...dfb4`；恢复前后完整科学状态hash
均为`ea5340...4d10`。这证明未来DCLGAN统一评估的模型加载、状态身份和确定性前向在目标
Linux环境中可执行。它不是性能结果，也不授权DCLGAN提前于5090B matched plain占用GPU。

远端回执：
`/home/yc/runs/FINAL_UNSB_DCLGAN_CROSS_RUNTIME_SMOKE_B6BF3A4/DCLGAN_CROSS_RUNTIME_LOAD_SMOKE.json`
（SHA256 `8bbcb9e80db4252e363776008c684a5433b9b9e8a79de858bda68b8fcac5e972`）。

