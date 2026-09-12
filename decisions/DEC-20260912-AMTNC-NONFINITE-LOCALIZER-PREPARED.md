# AM-TNC 非有限几何定位器准备裁决

在不触碰正在运行的第三个冻结 child 前提下，新增独立故障重放工具 `operations/paper_aio_amtnc_nonfinite_localizer.py`。它从受保护的 e178 full-state、e0、sampler 和 RNG 重放，逐次标记 D、E、GF player，并把首个异常参数区分为：原始梯度非有限、Adam scale 非有限、replica 均值/差分溢出、度量乘法溢出或 float32 几何乘积溢出。

工具只写独立 diagnostic output，运行前后重算源 checkpoint SHA；不保存科学 checkpoint、不读取 paired target/指标、不改变训练协议。当前仅完成代码和纯张量测试，尚未部署或运行。必须等待现有最后冻结尝试释放 4090A 后，才可从 e178 只读副本执行，避免影响主轨迹。

这项工作只定位当前实现的数值故障，不修复算法，也不提前决定 AM-TNC 机制的论文裁决。
