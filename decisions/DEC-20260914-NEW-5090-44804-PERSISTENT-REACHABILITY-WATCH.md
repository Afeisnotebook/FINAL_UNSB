# DEC-20260914：为 44804 部署持久、fail-closed 的可达性守望层

## 决定

`connect.weste.seetacloud.com:44804` 在多轮独立探测中持续拒绝 TCP 连接。它仍不属于可用算力 DAG，不能登记 GPU 身份或预分配实验。为避免后续完全依赖人工或 Codex 唤醒，部署一分钟轮询的持久守望层，并由冻结 commit 的 control supervisor 负责异常重启。

## 安全边界

守望层只建立 TCP 连接，不持有凭据、不进行 SSH 认证、不读取 GPU、不读取性能指标，也不能启动训练。端点首次可达时只产生 `REACHABLE_REVIEW_REQUIRED` 并退出；之后仍必须执行 GPU UUID identity gate、源码/runtime/resume/evaluation/容量门以及实时任务 DAG 裁决。

因此，该机制缩短的是“端点恢复到被发现”的工程延迟，不会把网络端口误当成新 GPU，也不会提前选择 HJCGR、NEGCUT 或任何新算法。

## 持久性

- 实现 commit：`967a785b488167582dcb689ce022f30037d606da`
- supervisor PID：`27236`
- 初始 child PID：`5592`
- 动态状态：`E:/UNSB_Expl/paper_eval_node/endpoint_reachability_44804_967a785/watch/ENDPOINT_REACHABILITY_STATE.json`
- Goal heartbeat 已加入该状态和恢复规则；现有健康训练与交付链未被修改。

端点仍不可达不是科学失败，也不是整个 Goal 的阻塞结论。其他 full-data 长训、matched control、外部基线、统一评估和终端因果审计继续推进。
