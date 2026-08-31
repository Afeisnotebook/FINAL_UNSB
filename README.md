# FINAL_UNSB

这是 UNSB 项目的可审计 clean canonical 与研究主控仓库。项目已暂停旧四条固定lane
计划，进入以本地1660、权威同运行时4090和独立复核5090共同执行的路线一长期算法发现阶段。

当前目标不是验证固定四条lane，而是从DT/HJ/HNEK及后续机制的长期证据中主动
构造至少一个能在真实200 data epochs保持收益的新算法，并保留所有证据合格、数学上
相互关联或独立成立的可行算法。HJ是第一项时间量尺正对照，不是唯一研究方向。

仓库不搬运三个月的全部历史。它保留 deterministic UNSB canonical、最小历史
证据、完整状态恢复、统一评估协议，以及当前路线一的持久化研究契约。服务器授权只用于
host-matched batch-1加速；跨宿主delta、全量数据、confirmation20和路线二仍关闭。

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

当前是`RELATED_MULTI_ALGORITHM_E200_FRONTIER_RUNNING`。本地和两台服务器均已验证
数据身份、共同e0、真实优化器更新、full-state exact resume、评估重放和confirmation锁。
4090的plain/HJ/HNEK/DT与最终474条反转、140条采样方差因果证据已经冻结；它们用于
生成新算法，而不是旧算法排名。当前科学终点不再被压缩成一个冠军：兼容
`CANDIDATE.json`只给出下一步行动优先级，`ALGORITHM_SET.json`拥有多算法科学解释权。

旧SEARCH-005 small25 2400 updates只有16 data epochs，旧full100 12000 updates只有
20 data epochs，不能充当目标所需的e200长期裁决。长期因果图谱已经生成并完成多项新
算法的真实e200：BVCP、PC-RSMG、AM-TNC和MCRB的当前实现均已有长期结论边界。
PC-RSMG full虽有`+0.621 dB` late-three均值，但e200为`-0.00138 dB`；其
proposal-only在同协议下达到late-three `+0.542 dB`、e200 `+0.451 dB`，并通过
SSIM/LPIPS等完整门，因此按“严格资格优先、资格层内再排序”成为当前seed2026开发
主候选。observable-only与plain的完整动力学及e200指标精确一致，收益来源被定位到
条件原生双视图G/F估计，而不是观察开销。固定四lane、HJ有限handoff、退出阈值和
paired控制仍不属于当前任务。

搜索没有因出现当前主候选而停止。当前相关算法族共享post-D/E条件独立双视图G/F均值，
但分别作用于三个不同父对象：原生UNSB场（Proposal-only）、HNEK physical-horizon
bridge game（HPCGR）和HJ structure-projected PatchNCE目标（HJCGR）。AM-TNC作为独立
Adam度量切向机制保留。4090并行运行HPCGR/HJCGR，5090并行完成Proposal-only/AM-TNC，
随后在释放的资源槽执行HJCGR跨运行时复核；所有分支都从共同e0跑到真实e200。

最终交付会分别计算Proposal-only相对plain、HPCGR相对HNEK、HJCGR相对HJ的完整轨迹
增量，从而区分父目标收益和共享估计器的匹配组合增量；这些差值不会被表述为单轨迹内的
线性因果贡献。具体边界见
`decisions/DEC-20260831-RELATED-MULTI-ALGORITHM-FAMILY-DELIVERY.md`和
`decisions/DEC-20260831-HJCGR-PORTABLE-CONSTRUCTION-AUTHORITY.md`。

紧急成本协议以完整seed2026/e200作为开发候选冻结依据，seed2027/2028延期；节省的
算力用于赢家机制消融、全负后的唯一证据驱动数学修订和额外独立方向。该协议不声称
跨seed稳定，详情见`decisions/DEC-20260830-ROUTE1-SINGLE-SEED-EMERGENCY.md`。

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
