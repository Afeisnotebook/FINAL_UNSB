# DEC-20260908：4090A 清理事故后的 AM-TNC e68 风险再审计

## 裁决

错误清理造成的事实不变：原 `unsb_cov` 完整 Conda 环境已经不存在，存活的 AM-TNC
supervisor/trainer 仍从 deleted inode 运行。不能通过覆盖旧目录消除这一标记，也不能为了验证
恢复而中断健康训练。

但“原路径不是完整环境”不等于“没有恢复路径”。当前训练外层 guard PID 439182 直接绑定
只读隔离运行时，而不是依赖旧前缀；它会在真正恢复前重新验签 Python、manifest、全部111项
映射依赖、代码、协议和最新完整 checkpoint。任一不一致均 fail closed。

## 当前核验和处理

- 原 supervisor/trainer PID 3446757/3446758 未变化，已自然完成 e68 / 581,604 updates；GPU
  正常工作，guard及health均为零重启、零告警。
- 重新逐字节验签隔离运行时的111项、4,198,514,872 bytes依赖：0缺失、0不匹配、0可写；
  新解释器探针解析到该隔离运行时并得到冻结版本集合。
- 用隔离解释器在CPU读取e68 full state，确认G/F/D/E、四个optimizer、四个scheduler、全部
  Python/NumPy/CPU/CUDA RNG、两个sampler和AM-TNC方法状态均存在。此前e65真实一步恢复
  555,945→555,946仍是执行级恢复证据。
- 新增本机只读e68快照
  `E:/UNSB_Expl/recovery_backups/4090A_AMTNC_20260908_E68`；checkpoint、sidecar和heartbeat
  均与远端SHA256一致。精确隔离运行时归档此前已完成异机备份。
- AM-TNC source exporter的恢复监督器、四条统一评估监督器及各health进程全部在线、零重启。
  评估走独立的只读统一评估runtime，不会错误地把训练relay当成通用环境。

## 影响与剩余边界

目前没有丢失任何完整epoch、checkpoint、算法状态或后续论文交付能力；若存活进程现在异常
退出，最多重算当前尚未原子保存的一个data epoch。主训练继续原PID到e200，不原地重建旧
环境，不迁移checkpoint，不修改协议。

不能诚实声称未来CUDA恢复的下一update与deleted-inode进程逐位相同，因为验证该反事实会
要求中断健康训练。可严格声称的是：活动依赖已精确捕获并只读验签、完整状态可加载、真实
一步可执行、训练guard直接使用隔离runtime、导出与评估后继不被已删除环境阻断。

机器可读回执：
`evidence/paper_aio/PAPER_AIO_AMTNC_E68_POST_CLEANUP_RISK_REAUDIT_20260908T035300.json`。
