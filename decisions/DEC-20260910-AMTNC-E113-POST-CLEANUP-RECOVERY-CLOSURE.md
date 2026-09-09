# DEC-20260910：AM-TNC e113 清理事故恢复闭环复核

## 裁决

4090A 的原 `unsb_cov` 完整 Conda 环境仍然不存在，运行中的 AM-TNC supervisor/trainer
也仍指向 deleted inode；因此不能把事故改写成“旧环境已恢复”。当前原 prefix 下的三个
Python 文件只是只读 fail-closed relay，不是完整环境，也不得覆盖重建。

在不发送信号、不重启、不迁移健康进程的前提下，本次从当前 e113 / 966,489 updates 的
full-state 重新执行了两层恢复门：隔离 runtime 的 111 项映射逐项哈希全部通过；随后在
`CUDA_VISIBLE_DEVICES` 为空、`gpu=-1` 的独立目录中加载全部训练状态并真实执行一项 CPU
update，得到 966,490 updates，以 `ENGINEERING_PAUSE` 正常退出，stderr 为 0。主 lane
checkpoint 哈希和 PID 3446757/3446758 均未改变，guard PID 439182 仍为零重启。

## 损失边界和后继

- 已完成的 113 个 data epochs、模型、优化器、scheduler、方法状态、RNG 和双 sampler
  没有丢失。
- e113 full-state 与恢复回执已复制到本机只读目录；隔离 runtime 仍有独立的本机只读归档。
- 活进程若退出，训练 guard 只通过隔离 runtime 的绝对 Python 启动，并在启动前重新哈希
  runtime 全树、代码/协议和最新 full-state；旧 prefix 不是恢复依赖。
- 因此当前最坏计算损失限制为正在进行、尚未落盘的一个 data epoch；后续导出和统一评估
  使用另一套已固定的评估 runtime，不受原训练环境缺失阻塞。

仍无法在不中断健康进程的条件下证明“新启动 CUDA 的下一步”与 deleted-inode 进程逐位
一致；如果未来实际触发恢复，必须在论文工程 provenance 中披露这一边界。当前正确动作仍是
保留健康训练连续运行，不原地补环境、不覆盖 relay、不做试探性 GPU 重启。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_AMTNC_E113_POST_CLEANUP_RECOVERY_CLOSURE_20260910T004500.json`。
