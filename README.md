# FINAL_UNSB

这是 UNSB 项目的可审计 clean canonical 与论文研究主控仓库。当前已进入 full-data
All-in-One 无配对论文阶段：每侧8553张、batch1、seed2026、真实200 data epochs。

北极星不是验证固定lane或只找一个冠军，而是同时取得外部论文基线、严格matched plain，
以及多条由长期因果证据产生的完整算法轨迹。DT/HJ/HNEK继续是算法发现证据，不是必须保留
原形的候选；confirmation20仍封存，paired指标不得控制训练，非等价runtime不得合并delta。

仓库不搬运三个月的全部历史。它保留 deterministic UNSB canonical、最小历史证据、
完整状态恢复、统一评估协议和持久监督链。实时事实以`PROJECT_STATE.json`、
`configs/FULL_DATA_METHOD_PORTFOLIO.json`和`configs/PAPER_DELIVERY_COMPLETION_MATRIX.json`
为准。

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

## 当前阶段（2026-09-06）

4090A的full plain已经完成并封存，现在运行AM-TNC；5090A运行ST-CGR；5090C运行
Proposal-only；5090B同时运行CUT和CycleGAN，并在CUT e200后由已冻结继任器执行exact
runtime与容量门，再建立fresh-e0 matched plain；本地GTX1660独占运行DCLGAN。所有健康
训练均有full-state、heartbeat、监督器、export/relay和统一评估后继，不依赖当前对话存活。

论文结论仍未冻结：Proposal与ST-CGR必须等待合法的5090B matched plain关系，AM-TNC只
使用4090A同宿主plain；所有方法主表使用e200，sustained固定为e150/e175/e200。DDSB因
权威源码/公式不足保持`reproduction_incomplete`，不能用猜测实现补表。

Proposal、ST-CGR和AM-TNC的共同数学母题、严格非重复关系及可写/不可写主张见
[`research/paper_aio/ALGORITHM_THEORY_MAP_CN.md`](research/paper_aio/ALGORITHM_THEORY_MAP_CN.md)。
它们与通用Monte Carlo方差缩减、timestep sampling、gradient surgery以及最新bridge
endpoint工作的重叠和投稿边界见
[`research/paper_aio/RELATED_WORK_NOVELTY_BOUNDARY_CN.md`](research/paper_aio/RELATED_WORK_NOVELTY_BOUNDARY_CN.md)。
small25的多算法终局仍是算法来源证据，不是full-data结果替代品；单seed成本策略也不等于
跨seed稳定性证明。

## small25路线一入口（当前用于证据与审计）

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

顺序门表达科学依赖；独立HJ/HNEK可在隔离run root并行，但plain比较和proxy/DT门仍须
验收。训练中间PSNR不会触发早停；`--engineering-stop-after-epoch`只用于最多5 data
epochs的可恢复分块，不能形成科学裁决。`candidate`阶段必须先存在完整因果图谱、
derivation card、已登记实现和可执行门禁，不能仅凭候选名称启动长训。

## 历史执行入口（当前暂停）

- 训练：`python -m production.train_lane`
- 评估：`python -m production.evaluate_lane`
- 排名：`python -m production.rank_lanes`

旧UNSB的网络/数据代码保留在 `src/` 作为嵌入式库；旧 `train.py/test.py` 已移除。
服务器安装、运行、评估和回传脚本保留在 `scripts/` 与 `server_tasks/` 作为
provenance；当前仅执行后来明确授权且记录于`decisions/DEC-20260830-ROUTE1-REMOTE-OFFLOAD.md`
和`decisions/DEC-20260830-ROUTE1-REMOTE5090.md`的路线一任务。
