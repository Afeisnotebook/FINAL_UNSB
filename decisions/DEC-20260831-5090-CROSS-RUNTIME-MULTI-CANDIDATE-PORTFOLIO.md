# DEC-20260831：空闲5090推进两个独立的4090长期候选

状态：`ACCEPTED_EVIDENCE_QUALIFIED_PARALLEL_REPLAY`

## 决策

额外5090不用于多seed、超参网格或已经关闭的算子，而用于把4090完整e200前沿中最有
信息量的两个算法，从5090本机共同e0重新训练到真实e200：

1. `ABL-G1-02B-PCRSMG-PROPOSAL-ONLY`：4090严格通过、当前动作优先级第一；
2. `G2-01-ADAM-METRIC-TANGENTIAL-CONSENSUS`：与条件重采样不同的优化器几何机制，
   late-three和e200 PSNR为正，仅晚期LPIPS护栏轻微失败。

两条在单张5090上以两个独立batch1进程并行。并行只提高设备利用率，不合并batch、不改变
每条lane的30000 updates、data epochs、采样器或损失。若显存或确定性工程门失败，后继
失败关闭并报告，不自动缩短协议或更换配置。

## 排除项及原因

- PC-RSMG full与proposal-only同族，且完整三角色消融已经证明proposal-only更强；不重复
  full以免把算力花在非独立信息上。
- AM-TNC保留的是独立Adam-metric机制，不把这次复赛解释为允许第三次同族数学修补。
- RF-AMMCRB、RF-MCRB和rollout-speed当前算子均在完整e200显著失败，不因卡空闲而重跑。
- PCNR已经在5090完成e200，当前有价值的下一步是4090共同e0复赛，不在5090重复自身。

## 源码与证据绑定

- 4090完整前沿、terminal receipt、trajectory、derivation card、implementation和对应
  hypothesis-ledger freeze record全部嵌入便携权威；
- 目标5090源码worktree必须精确等于各自训练commit且保持clean；
- proposal-only依赖的4090 PC-RSMG parent receipt作为只读provenance导入，只满足card
  的父项哈希绑定，不冒充5090父算法结果；
- 两个目标候选均从5090共同e0重新开始，不迁移任何checkpoint或optimizer state；
- 最终只与5090 plain比较，4090与5090 delta始终分开。

## 交付语义

这项复赛把结果组织为“多候选可信前沿、单一动作优先级”。`CANDIDATE.json`仍可指定一个
下一步主项，但不会删除PCNR、AM-TNC、PC-RSMG及其完整宿主分离证据。复赛若失败，关闭
的是该算法在该目标运行时下的当前结果，不自动证伪父机制。

## 不变边界

- seed固定2026；不声称多seed稳定；
- batch1、small25、真实200 data epochs、共同e0和CRN评估不变；
- paired结果只在完整e200后参与排名和资源分配；
- 不开启confirmation20、全量数据、路线二、handoff或退出窗口；
- 不修改任何候选公式或超参数。
