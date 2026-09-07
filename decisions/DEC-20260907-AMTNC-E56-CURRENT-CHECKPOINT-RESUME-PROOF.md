# DEC-20260907：AM-TNC当前e56恢复能力闭环

## 裁决

4090A上的错误清理造成了真实但已被隔离的风险：原完整`unsb_cov`环境仍不存在，活动
trainer PID 3446758继续使用deleted inode。不能重启它，也不能把旧路径下的三个326字节
入口误称为重建环境；它们只是只读relay。

截至本次复核，错误清理没有造成已完成训练、科学状态或checkpoint损失。supervisor
PID 3446757和trainer PID 3446758保持连续运行，AM-TNC已经原进程完成e56、478968 steps。

## 新增的当前状态证明

- 活动deleted-inode解释器与隔离解释器SHA256一致。
- 对活动映射捕获清单重新哈希，111/111依赖字节一致，无缺失、无不匹配。
- e56 checkpoint文件哈希与sidecar一致；隔离运行时在CPU加载完整网络、优化器、调度器、
  方法状态、四类RNG和两个sampler。
- 从e56 checkpoint复制到隔离目录后执行一个CPU update，478968精确推进到478969；主lane
  checkpoint前后SHA保持不变，探针没有使用4090或修改主轨迹。
- 原路径relay实际通过PyTorch、CUDA、4090和paper runner导入；这只证明入口可用，不代表
  原环境恢复。
- 外层guard PID 3966089和健康监视器PID 3968393健康、零重启、零告警。guard冻结命令直接
  使用隔离解释器，仅在原supervisor和trainer同时消失且哈希门通过时接管。

## 损失边界

现在可避免已完成e56以前的损失；若活动进程故障，最多重算尚未原子保存的当前epoch。新
进程不声称与deleted-inode进程在GPU下一步逐位一致，因此最安全的操作仍是保持当前训练，
不为消除`(deleted)`标签主动重启，也不覆盖旧路径。

本次没有读取paired性能值、没有改变训练协议、没有打开confirmation20。完整回执见
`evidence/paper_aio/PAPER_AIO_AMTNC_E56_CURRENT_CHECKPOINT_RESUME_PROOF_20260907T164249.json`。
