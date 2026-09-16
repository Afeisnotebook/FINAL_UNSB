# FINAL_UNSB 结果归档与后续调研入口

## 一句话状态

完整 full-data discovery 阶段已经结束。Proposal-only 是当前唯一通过预注册长期门的自研算法；ST-CGR 与 AM-TNC 只判定为当前实现/协议失败，不是机制族证伪。confirmation20 仍然封存。

## 先从哪里看

1. 最终完整结果：`archive/paper_aio/final_v10/PAPER_ALGORITHM_PORTFOLIO_WITH_DCLGAN.json`
2. 归档完整性与哈希：`archive/paper_aio/final_v10/ARCHIVE_MANIFEST.json`
3. 服务器与大文件位置：`archive/paper_aio/final_v10/SERVER_ASSET_INVENTORY.json`
4. 最终裁决：`decisions/DEC-20260916-FINAL-DISCOVERY-PORTFOLIO-V10.md`
5. 精简证据：`evidence/paper_aio/PAPER_AIO_FINAL_DISCOVERY_PORTFOLIO_V10_20260916T125639.json`

## 想研究 Proposal 的数学与实现

- 数学总图：`research/paper_aio/ALGORITHM_THEORY_MAP_CN.md`
- 方法公式与假设：`research/paper_aio/METHODS_PROTOCOL_DRAFT_EN.md`
- 冻结理论边界：`configs/PAPER_ALGORITHM_THEORY_BUNDLE.json`
- 核心实现：`src/models/route1/pcrsmg.py`
- Proposal-only 消融实现：`src/models/route1/pcrsmg_ablation.py`
- 模型注册：`src/models/route1_pcrsmg_ablation_model.py`
- 全量冻结配置：`configs/PAPER_AIO_UNPAIRED_V1.json`
- Proposal 的长期轨迹和逐域结果：最终 portfolio 中的 `methods.proposal.result`

最值得继续回答的问题不是“再找一个退出阈值”，而是：为什么 post-D/E、player-selective、条件 iid 的 G/F 双视图平均能在完整 200 epochs 保持收益，而时间分层与 Adam-metric 几何没有保持。ST-CGR 和 AM-TNC disposition 已归档，正好是反事实材料。

## 想研究负结果和下一代算法

- ST-CGR：`archive/paper_aio/final_v10/dispositions/STCGR.json`
- AM-TNC：`archive/paper_aio/final_v10/dispositions/AMTNC.json`
- 终端谱问题裁决：`decisions/DEC-20260915-TERMINAL-PATHOLOGY-NOT-CONFIRMED.md`
- HJ/Proposal 合成代码：`src/models/route1_hjcgr_model.py`
- HJ 条件重采样代码：`src/models/route1_hjpcnr_model.py`

后续若重启算法搜索，应先利用这三条完整轨迹建立“条件方差降低、时间耦合、优化几何”之间的因果区别，不能退回窗口调参或最佳 checkpoint 选择。

## 想做论文相关工作与定位

- 新颖性边界：`research/paper_aio/RELATED_WORK_NOVELTY_BOUNDARY_CN.md`
- 引用台账：`configs/PAPER_REFERENCE_LEDGER.json`
- 核心参考文献：`research/paper_aio/PAPER_REFERENCES_CORE_PRE_RESULT.bib`
- 结果分支写作框架：`research/paper_aio/MANUSCRIPT_RESULT_BRANCHING_OUTLINE_CN.md`
- 已知局限：`research/paper_aio/LIMITATIONS_PRE_RESULT_EN.md`
- 复现检查表：`research/paper_aio/REPRODUCIBILITY_CHECKLIST_EN.md`

论文不能声称总体 SOTA：CUT、DCLGAN 和 CycleGAN 的固定 e200 绝对指标均高于 Proposal。当前可辩护的主线是 UNSB 内部长期条件梯度方差控制与持续相对收益。

## 哪些东西没有进 Git

Git 已保存最终 portfolio、DCLGAN addendum、first-wave 结果、两个失败算法的 disposition、DCLGAN 结果、七条复杂度回执和三条自研算法 source-export receipt。

以下内容刻意不进 Git：

- checkpoint、完整日志和逐图 raw metrics；
- 5090C matched plain 的原始运行资产，本轮按用户要求不处理；
- 4090A 与 5090A 上的独立证明实验。

关键 checkpoint 并非服务器单点：Proposal、ST-CGR、AM-TNC、plain 和 matched plain 的固定 checkpoint 已存在本地 `E:/UNSB_Expl/paper_eval_node/checkpoint_imports/sources`；DCLGAN 原始运行位于 `E:/UNSB_Expl/runs/FINAL_UNSB_DCLGAN_LOCAL1000_E45973A`。

目前仍真正“只在服务器、尚未进入canonical归档”的，是独立证明工作：

- 4090A：`/home/yc/independent_audit_20260916`，约 1.6G，已有若干完成结果；
- 5090A：`/root/autodl-tmp/independent_audit_20260915`，约 6.8G，其中 low01-linear e40→e60 证明实验在归档时仍运行至 e57。

它们没有被删除、停止或并入论文结果。完成后应单独审阅，再决定是否建立新的 proof archive，避免未经裁决的结果污染当前主结论。
