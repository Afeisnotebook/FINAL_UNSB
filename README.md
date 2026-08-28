# FINAL_UNSB

这是 UNSB 项目的最后一轮、四卡、全量六域执行仓库。它同时服务于两个角色：

1. 一台全新 Codex 项目中的**主控中心**；
2. 四台只拥有本仓库 clone 与六域数据的 RTX 4090 **执行服务器**。

仓库不搬运三个月的全部历史。它只包含：已经接受的 deterministic UNSB
canonical、HJ/HNEK 两个仍存活的候选、一个全量规模新增的 macro-marginal
对照、经过裁决的最小历史证据、完整状态恢复与统一评估协议。

## 接手顺序

无论人还是 Codex，必须依次阅读：

1. [`START_HERE_CN.md`](START_HERE_CN.md)
2. [`AGENTS.md`](AGENTS.md)
3. [`PROJECT_CONTRACT.json`](PROJECT_CONTRACT.json)
4. [`PROJECT_STATE.json`](PROJECT_STATE.json)
5. [`CONTEXT_CAPSULE_CN.md`](CONTEXT_CAPSULE_CN.md)
6. [`configs/FOUR_LANES.json`](configs/FOUR_LANES.json)
7. [`DATA_CONTRACT.json`](DATA_CONTRACT.json)

## 当前阶段

本地 GTX 1660 6GB 工程门禁已通过，当前是 `READY_FOR_SERVER_PREFLIGHT`。
本地已经验证真实9153身份数据、四lane共同e0、真实优化器更新、HJ实际介入、
full-state exact resume、评估重放和confirmation锁。验收证据见
[`evidence/local_preflight/LOCAL_PREFLIGHT_REPORT.md`](evidence/local_preflight/LOCAL_PREFLIGHT_REPORT.md)。

现在仍**没有长训授权**。下一步是在四台4090上用相同commit完成
[`server_tasks/00_COMMON_PREFLIGHT_CN.md`](server_tasks/00_COMMON_PREFLIGHT_CN.md)，
四机身份一致后再由主控生成并提交 `RUN_AUTHORIZATION.json`。

本轮目标是从四条固定 lane 中得到一个当前最优、值得继续做 seed 验证的候选。
它保证产出裁决，不保证科学事实一定给出正收益。

## 唯一执行入口

- 训练：`python -m production.train_lane`
- 评估：`python -m production.evaluate_lane`
- 排名：`python -m production.rank_lanes`

旧UNSB的网络/数据代码保留在 `src/` 作为嵌入式库；旧 `train.py/test.py` 已移除，
因为它们不满足本项目的完整状态和访问门。服务器安装、运行、评估和回传脚本见
`scripts/`，四张卡各自的任务说明见 `server_tasks/`。
