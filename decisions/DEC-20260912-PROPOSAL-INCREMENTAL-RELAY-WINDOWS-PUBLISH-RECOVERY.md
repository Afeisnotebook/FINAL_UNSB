# DEC-20260912：Proposal 增量relay Windows发布锁自动恢复

## 裁决

5090C Proposal训练没有中断。发生故障的是本地只读增量relay：旧child在将临时
`INCREMENTAL_IMPORT_LANE.json`原子替换为发布文件时遇到Windows `WinError 5`共享锁，冻结
supervisor PID `5112`随后自动启动child PID `17936`，累计restart count为2。

新child现为`PARTIAL_VERIFIED_INCREMENTAL_AUDIT_IMPORT`，e100/e150的export receipt、checkpoint
与sidecar共六个文件均重新计算SHA256并与发布回执完全一致；combined health PID `9960`为
`HEALTHY`且alert count为0。Proposal supervisor/trainer PID `9729/766090`继续运行在e165，训练、
采样器、checkpoint和科学协议均未改变。

## 后续处理

canonical writer自提交`b85b692`起已对Windows原子替换的瞬时`PermissionError`执行有界重试，
canonical reader又由提交`f2a601a`加入同类读取重试与same-bytes解析/哈希。当前冻结relay使用更旧
的控制版本，但已健康恢复且尚有18次restart预算；不为采用修复而热替换或重启它。未来正式
部署或真实恢复必须使用同时含writer与reader hardening的canonical版本。

两小时`final-unsb-goal`监控已原位同步到replacement child、restart count和提交`b724bd5`；
频率及仅失败通知策略保持不变，prompt未持久化凭据。

这是一次已自动闭环的工程恢复，不是算法、训练或结果事件；不得据此调整训练、读取paired指标
或打开confirmation20。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_PROPOSAL_INCREMENTAL_RELAY_WINDOWS_PUBLISH_RECOVERY_20260912T084800.json`。
