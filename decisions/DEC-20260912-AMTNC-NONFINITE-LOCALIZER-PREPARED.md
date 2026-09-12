# AM-TNC 非有限几何定位器准备裁决

在不触碰正在运行的第三个冻结 child 前提下，新增独立故障重放工具 `operations/paper_aio_amtnc_nonfinite_localizer.py`。它从受保护的 e178 full-state、e0、sampler 和 RNG 重放，逐次标记 D、E、GF player，并把首个异常参数区分为：原始梯度非有限、Adam scale 非有限、replica 均值/差分溢出、度量乘法溢出或 float32 几何乘积溢出。

工具只写独立 diagnostic output，运行前后重算源 checkpoint SHA；不保存科学 checkpoint、不读取 paired target/指标、不改变训练协议。代码与纯张量测试完成后，已把 e178、e0、manifest、localizer 和共享GPU锁runner复制到4090A独立只读目录并逐文件闭合SHA；import/CLI门通过。工具仍未运行，必须等待现有最后冻结尝试释放 4090A 后，才可执行，避免影响主轨迹。

这项工作只定位当前实现的数值故障，不修复算法，也不提前决定 AM-TNC 机制的论文裁决。

冻结命令与启动前提见 `configs/AMTNC_E178_NONFINITE_LOCALIZER_CONTRACT.json`，部署证据见 `evidence/paper_aio/PAPER_AIO_AMTNC_E178_NONFINITE_LOCALIZER_DEPLOYED_20260912T175934.json`。
