# DEC-20260907：AM-TNC deleted-runtime风险复核到e55

## 裁决

4090A的错误清理确实删除了AM-TNC活动进程原来使用的完整`unsb_cov`环境，PID 3446758
显示`(deleted)`是真实风险信号，不能被忽略；但它没有破坏当前科学状态、checkpoint或已完成
进度。健康trainer PID 3446758和supervisor PID 3446757保持连续运行，本次没有为消除标签而
重启、迁移或覆盖它们。

“原路径创建新进程当前必然失败”已不再是当前事实。原来的三个Python入口并不是重建环境，
而是SHA固定的只读relay；本次从原路径实际启动新进程，成功进入隔离运行时并完成PyTorch、
CUDA和paper runner导入。必须继续区分“原完整环境仍不存在”和“原入口已经具备受控恢复
能力”。

## 本次复核

- 活动deleted-inode Python与隔离解释器SHA256完全一致。
- 对活动映射捕获清单重新逐项哈希，111个依赖全部存在且111/111字节一致。
- 最新e55、470415-step完整状态的文件哈希与sidecar一致；隔离解释器在CPU完整加载G/F/D/E、
  optimizers、schedulers、AM-TNC状态、Python/NumPy/CPU/CUDA RNG及两个sampler。
- 已有独立e29恢复探针证明隔离运行时可从248037恢复并执行到248038；本次没有重复占用GPU，
  也没有写入主lane。
- 外层guard PID 3966089和健康监视器PID 3968393均健康、零重启、零告警。冻结后继命令直接
  使用隔离Python；只有原supervisor和trainer都消失，且源码、协议与checkpoint哈希门通过时
  才会接管，重复进程会fail closed。

## 损失边界

截至e55没有已完成计算损失。即使当前活动进程随后故障，也可从最新完整epoch恢复，最多丢失
尚未原子保存的当前epoch。新的恢复进程不声称与deleted-inode活动进程在GPU下一步逐位一致，
因此最安全的处理仍是保留健康进程，不为了修复显示状态主动重启。

本次没有读取paired性能值、没有改变算法或训练协议、没有打开confirmation20。完整回执见
`evidence/paper_aio/PAPER_AIO_AMTNC_E55_DELETED_RUNTIME_RECOVERY_REAUDIT_20260907T153800.json`。
