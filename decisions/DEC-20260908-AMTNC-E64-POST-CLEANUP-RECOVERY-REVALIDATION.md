# DEC-20260908：4090A 清理事故后的 AM-TNC e64 恢复再验证

## 裁决

错误清理确实删除了原完整 `unsb_cov` Conda 环境，运行中 trainer PID 3446758 仍显示
`(deleted)`。这不是一个应该通过原地重建或重启来“消除标签”的问题。当前正确设计是：
旧路径下只保留不可写的 Python relay，权威训练恢复运行时位于隔离目录，且另有本机只读
归档。健康的 supervisor 3446757 和 trainer 3446758 保持连续。

本次现场复核纠正了“按原路径启动一定失败”的判断。完整 Conda 环境的确不存在，但三个
原 Python 入口是 326 字节的只读 relay；从该入口新建解释器已成功落到隔离运行时，并导入
Python 3.10.20、Torch 2.5.1+cu121、NumPy 2.2.6 和 Pillow 12.2.0。relay 不应被当作已
恢复的环境，也不应被覆盖，但它能够承接仍存活旧 supervisor 的冻结重启命令。

## 当前恢复证据

- AM-TNC 已由原进程完成 e64 / 547,392 updates；latest full-state SHA256 为
  `dbafbeef...0c731`，科学状态 SHA256 为 `6422f810...91a79`。
- 以该 e64 完整状态复制到隔离输出，在隐藏 GPU 且 `gpu_id=-1` 的条件下真实执行一更新，
  从 547,392 到 547,393，stderr 为零。主轨迹 checkpoint 哈希前后未变。
- 隔离运行时 Python 与活动 deleted inode 的 SHA256 相同；111 个活动映射依赖再次逐项
  复哈希，结果 0 缺失、0 不匹配、0 可写。
- 外层 fail-closed guard PID 439182 和健康监视器 PID 439447 均为 PPID 1、零重启、零告警。
  只有原 supervisor 与 trainer 同时消失，且代码、协议、checkpoint、Python 与全部依赖重新
  验签通过时，guard 才能启动恢复；任一不一致都会停止而不是猜测恢复。
- 5.49 GB 隔离运行时归档在4090A与本机各有一份只读副本，两端重新计算 SHA256 均为
  `6396a1f2...b2a52`。
- e200 source-bound exporter、其恢复监督器和独立的统一评估监督链均在线、零重启、零告警。

第一次新的CPU探针错误地同时隐藏CUDA并传入 `gpu_id=0`，在模型构造前按预期失败；随后
仅修正探针设备参数为 `gpu_id=-1` 即通过。该失败没有触碰主轨迹，也不是checkpoint或隔离
运行时不可恢复的证据。

## 损失与剩余风险

目前没有已完成 epoch、优化器、sampler、RNG 或方法状态损失。若活动进程未来异常退出，
可从最近完整 epoch 恢复，最多损失当时尚未完成的一个 data epoch。不能声称恢复后的下一个
GPU update 与仍存活进程逐位一致；能够严格证明的是运行时身份一致、完整状态可加载并能继续
执行更新。因此继续保留原进程到 e200 仍是最低风险方案。

不得把旧路径扩建回一个“看似完整”的环境，不得覆盖relay，也不得为了消除 `(deleted)` 标签
主动重启。后继任务必须继续分别使用已冻结的训练恢复运行时和统一评估运行时。完整机器可读
回执见 `evidence/paper_aio/PAPER_AIO_AMTNC_E64_POST_CLEANUP_RECOVERY_REVALIDATION_20260908T003100.json`。
