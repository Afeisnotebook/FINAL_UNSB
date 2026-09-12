# AM-TNC e178 重复数值阻塞裁决

AM-TNC 从同一个 e178 完整状态恢复后，在约 46 分 48 秒内第二次于 e178→e179 转换复现完全相同的 `AM-TNC replica geometry is nonfinite`。两次失败均发生在同一 checkpoint、同一协议和通过 111/111 文件验签的隔离运行时上。因此它不再按一次性基础设施中断处理，当前实现正式记为 `current_implementation_numerically_blocked`。

这不是机制证伪。e178 之前的轨迹与固定 e175 证据仍然有效，但当前冻结实现尚未产生 e200 结果，不能拿 e178 代替主表终点，也不能用修剪、跳过 update、换 checkpoint 或临时调参把长训“做完”。

对受保护 e178 checkpoint 的只读数值审计显示：G/F/D/E 参数和全部 Adam moment 都仍为有限数，最后一个已完成 update 的三组 replica geometry 也都有限，所以不是已落盘 checkpoint 损坏。与此同时，Adam 二阶矩的范围极端（最小约 `7e-43`，E 的最大值约 `3.49e26`），AM-TNC 又显式使用 `1/(sqrt(v)+eps)` 加权 replica gradient。当前 line 92 只能确定“加权后的共识/分歧几何在 e179 内成为非有限”，尚不能在不重放故障 batch 的情况下区分上游梯度先变成非有限，还是有限梯度乘极端度量后溢出。最终冻结尝试释放 GPU 后，应在受保护副本上做只读 player/parameter 定位；这项诊断不能热改主轨迹。

原监督器已按既有上限自动启动第三个、也是预算内最后一个 child PID `3698062`。这个进程直接使用已验签的隔离 Python；不进行手工中断、重启或协议修改。如果它再次在相同转换失败，允许冻结监督器按合同进入阻塞态，禁止再次自动或人工重启。若它意外推进到新的完整 epoch，则先验签新 checkpoint，再单独解释重复失败后的可重复性边界。

e178 full-state 已在 4090A 与本地只读备份中保持相同 SHA256，已完成 epoch 损失为 0；每次失败最多重做一个未落盘 epoch。其他健康算法、matched plain、外部基线和终端审计均继续，路线不会因 AM-TNC 当前实现受阻而收缩为单一算法。

用户独立证明实验保持原样；它已获授权且自然完成，没有被停止，也没有证据支持把 AM-TNC 数值故障归因给它。

全程未读取 paired 性能、未打开 confirmation20、未改变算法或训练协议。证据：`evidence/paper_aio/PAPER_AIO_AMTNC_E178_REPEATED_NUMERICAL_BLOCK_20260912T173722.json`。
