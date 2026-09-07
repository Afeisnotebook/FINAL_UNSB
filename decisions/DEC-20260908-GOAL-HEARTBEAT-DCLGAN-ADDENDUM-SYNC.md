# DEC-20260908：Goal heartbeat 纳入 DCLGAN 最终 addendum

## 裁决

DCLGAN 非阻塞 addendum 已经实际部署，旧 heartbeat 仍只覆盖其训练、push 与统一评估，未记录
新增 supervisor/child/health 以及合法的等待、GPU lock 和完成后读取性能语义。若不更新，后续
低频唤醒可能把首次 child launch 计数误判为恢复故障，或遗漏 addendum 静默退出。

本次通过 Codex automation API 更新既有 `final-unsb-goal`，保留两小时 heartbeat、当前线程、
ACTIVE 状态和 `failed_runs_only` 通知策略。新合同明确：

- supervisor 638006、child 638015、health 638474 都必须纳入真实句柄检查；
- `WAITING_FOR_CORE_PORTFOLIO_AND_DCLGAN_FIXED_RESULT` 是正常状态，restart_count=1 只是首次
  child 启动；等待期不得占用共享 GPU；
- 两个依赖完成后读取固定结果与测量复杂度属于合法 post-completion 汇总，但仍不能控制训练、
  调度或 confirmation；
- 只有句柄真实消失、状态超时或 fail-closed 才按冻结 commit/command 恢复。

自动化持久化文件已只读复核，新增字段、三个 PID、最新 DCLGAN e54、增强 portfolio 输出、
ACTIVE 和失败才通知均存在；prompt 未保存任何 SSH 凭据。没有触碰训练或队列。

机器可读回执：
`evidence/paper_aio/PAPER_AIO_GOAL_HEARTBEAT_DCLGAN_ADDENDUM_SYNC_20260908T030908.json`。
