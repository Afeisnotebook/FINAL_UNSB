# FINAL_UNSB

这是 UNSB 项目的可审计 clean canonical 与研究主控仓库。2026-08-29 起，项目已
暂停四卡执行计划，重新进入本地路线一长期算法发现阶段。

当前目标不是验证固定四条lane，而是从DT/HJ/HNEK及后续机制的长期证据中主动
构造一个能在真实200 data epochs保持收益的新算法。HJ是第一项时间量尺正对照，
不是唯一研究方向。

仓库不搬运三个月的全部历史。它保留 deterministic UNSB canonical、最小历史
证据、完整状态恢复、统一评估协议，以及当前本地路线一的持久化研究契约。

## 接手顺序

无论人还是 Codex，必须依次阅读：

1. [`START_HERE_CN.md`](START_HERE_CN.md)
2. [`AGENTS.md`](AGENTS.md)
3. [`PROJECT_CONTRACT.json`](PROJECT_CONTRACT.json)
4. [`PROJECT_STATE.json`](PROJECT_STATE.json)
5. [`CONTEXT_CAPSULE_CN.md`](CONTEXT_CAPSULE_CN.md)
6. [`LOCAL_ROUTE1_RESEARCH_CONTRACT_CN.md`](LOCAL_ROUTE1_RESEARCH_CONTRACT_CN.md)
7. [`ACTIVE_LOCAL_ROUTE1_PLAN_CN.md`](ACTIVE_LOCAL_ROUTE1_PLAN_CN.md)
8. [`configs/LOCAL_ROUTE1_PROBES.json`](configs/LOCAL_ROUTE1_PROBES.json)
9. [`DATA_CONTRACT.json`](DATA_CONTRACT.json)

## 当前阶段

本地 GTX 1660 6GB 工程门禁已通过，当前是
`LOCAL_ROUTE1_LONG_HORIZON_REOPENED`。本地已经验证真实数据身份、共同e0、真实
优化器更新、full-state exact resume、评估重放和confirmation锁。验收证据见
[`evidence/local_preflight/LOCAL_PREFLIGHT_REPORT.md`](evidence/local_preflight/LOCAL_PREFLIGHT_REPORT.md)。

旧SEARCH-005 small25 2400 updates只有16 data epochs，旧full100 12000 updates只有
20 data epochs，不能充当目标所需的e200长期裁决。下一步是完成多方法语义谱系
审计，并建立plain/continuous-HJ/HNEK/DT的small25 e200长期锚点图谱。

随后必须从长期因果证据生成至少一个新数学算法并完成matched e200本地裁决。
4090服务器、固定四lane和HJ有限handoff均不属于当前任务。

## 当前本地路线一入口

独立研究runner位于 `research/local_route1`，不调用已暂停的
`production.train_lane`。默认使用冻结manifest、本机small25视图、seed 2026和
`E:\UNSB_Expl\runs\FINAL_UNSB_LOCAL_ROUTE1_E200` 作为Git外运行目录。

```powershell
python -m research.local_route1.run --stage lineage
python -m research.local_route1.run --stage gate
python -m research.local_route1.run --stage anchors --lane plain --resume
python -m research.local_route1.run --stage anchors --lane hj --resume
python -m research.local_route1.run --stage anchors --lane hnek --resume
python -m research.local_route1.run --stage evaluate
python -m research.local_route1.run --stage anchors --lane dt --resume
python -m research.local_route1.run --stage audit
python -m research.local_route1.run --stage derive
```

顺序门会阻止HJ早于plain、HNEK早于HJ，以及在proxy未校准时启动DT。训练中间
PSNR不会触发早停；`--engineering-stop-after-epoch`只用于门禁/调试，不能形成
科学裁决。`candidate`阶段必须先存在完整因果图谱、derivation card和已登记实现，
不能仅凭候选名称启动长训。

## 历史执行入口（当前暂停）

- 训练：`python -m production.train_lane`
- 评估：`python -m production.evaluate_lane`
- 排名：`python -m production.rank_lanes`

旧UNSB的网络/数据代码保留在 `src/` 作为嵌入式库；旧 `train.py/test.py` 已移除。
服务器安装、运行、评估和回传脚本保留在 `scripts/` 与 `server_tasks/` 作为
provenance；没有新的明确决策不得执行。
