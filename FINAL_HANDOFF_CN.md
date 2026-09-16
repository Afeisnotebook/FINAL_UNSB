# FINAL_UNSB 最终交接入口

状态：`DISCOVERY_ARCHIVED / CLAIM_REVIEW_PENDING / CONFIRMATION20_SEALED`

日期：2026-09-16

> 这是离开当前 Codex 上下文后的第一权威入口。仓库里更早的“ACTIVE”“正在训练”
> “下一继任器”文字均保留作历史 provenance，不再具有调度效力。不要根据旧 PID、旧 SSH
> 端点、旧 heartbeat 或旧队列重启任何任务。

## 1. 已经得到的结论

完整 full-data discovery 已结束。共同协议为每侧 8,553 张、batch size 1、seed 2026、
200 data epochs（1,710,600 updates）、固定 e200 主表、discovery80；没有选择最佳 checkpoint，
confirmation20 从未打开。

### 自研算法

| 方法 | 长期裁决 | e150/e175/e200 宏 PSNR delta 均值 | e200 delta | e200 绝对 PSNR |
|---|---:|---:|---:|---:|
| Proposal-only | **通过预注册 full-data 门** | **+1.723565 dB** | **+0.839347 dB** | 20.514432 |
| ST-CGR | 当前实现/协议失败 | -0.692040 dB | -2.744077 dB | 16.931007 |
| AM-TNC | 当前实现/协议失败 | -2.107622 dB | -4.072023 dB | 14.069762 |

matched plain e200 为 PSNR 19.675085、SSIM 0.642470、LPIPS 0.236559；Proposal e200
为 SSIM 0.663835、LPIPS 0.221469。ST-CGR 与 AM-TNC 的失败只关闭当前实现，不构成
time-stratification、state-coupled resampling 或 Adam-metric 几何机制族的证伪。

### 外部基线

| 方法 | e200 绝对 PSNR | 解释边界 |
|---|---:|---|
| CUT | 23.728935 | 外部协议基线，不与 UNSB matched delta 混用 |
| DCLGAN | 23.328284 | standalone fixed-protocol；没有 matched delta |
| CycleGAN | 21.655429 | 外部协议基线 |
| Input | 14.738111 | 输入下界 |

因此论文不能声称总体 SOTA。当前可辩护主线是：在固定 UNSB 内部协议中，
Proposal-only 的 player-selective、post-D/E 条件 iid 双视图估计在 200 epochs 保持了
相对 plain 的长期收益；这个结论目前只有 seed 2026，不能写成多 seed 稳定性。

## 2. 没有得到、也不能偷写的结论

- HJCGR 仍是 `deferred`，DDSB 仍是 `reproduction_incomplete`；两者都不是机制证伪。
- “末步低方差导致奇异漂移”没有通过跨算法、跨域证据门，裁决为
  `TERMINAL_PATHOLOGY_NOT_CONFIRMED_DO_NOT_ADD_MODULE`。
- 没有使用 paired PSNR 控制训练、退出、算法选择或调度。
- 没有证明跨 seed 稳定，也没有打开 confirmation20。
- 不允许把不同 runtime cohort 的数值拼成 matched delta。
- 不能把旧的短窗口正收益、最佳 checkpoint 或服务器独立证明直接并入主表。

## 3. 最重要的归档入口

1. 完整可携带结果：
   `archive/paper_aio/final_v10/PAPER_ALGORITHM_PORTFOLIO_WITH_DCLGAN.json`
2. 哈希清单：`archive/paper_aio/final_v10/ARCHIVE_MANIFEST.json`
3. 服务器与大文件地图：`archive/paper_aio/final_v10/SERVER_ASSET_INVENTORY.json`
4. 研究阅读导航：`research/paper_aio/RESEARCH_HANDOFF_CN.md`
5. 最终裁决：`decisions/DEC-20260916-FINAL-DISCOVERY-PORTFOLIO-V10.md`
6. 执行入口关闭决定：`decisions/DEC-20260916-FINAL-HANDOFF-EXECUTION-CLOSURE.md`
7. 机器可读交接：`FINAL_HANDOFF.json`

归档中的 portfolio SHA256 为：

- 无 DCLGAN：`f70ca229f2abae7026a5d60f6f359164dfd14e7da6e58e2ceeae70718ed1e36c`
- 含 DCLGAN：`c1a8229c446c65a94fcf266320467208e11d7842a1c440faf4070db1884e919e`

## 4. 代码、数学与失败证据从哪里读

Proposal 的数学和代码：

- `research/paper_aio/ALGORITHM_THEORY_MAP_CN.md`
- `research/paper_aio/METHODS_PROTOCOL_DRAFT_EN.md`
- `configs/PAPER_ALGORITHM_THEORY_BUNDLE.json`
- `src/models/route1/pcrsmg.py`
- `src/models/route1/pcrsmg_ablation.py`
- `src/models/route1_pcrsmg_ablation_model.py`

负结果和下一代算法：

- `archive/paper_aio/final_v10/dispositions/STCGR.json`
- `archive/paper_aio/final_v10/dispositions/AMTNC.json`
- `decisions/DEC-20260915-TERMINAL-PATHOLOGY-NOT-CONFIRMED.md`
- `research/paper_aio/RELATED_WORK_NOVELTY_BOUNDARY_CN.md`

三个月历史不是删除了，而是降级为证据。`START_HERE_CN.md`、`CONTEXT_CAPSULE_CN.md`、
`ACTIVE_PAPER_AIO_PLAN_CN.md` 后面的长文用于追溯推理，不能覆盖本文件的当前状态。

## 5. 大文件与服务器资产

Git 已保存 compact 结果、逐方法 disposition、复杂度、source-export receipt 和哈希；
checkpoint 与完整日志不进 Git。

本地已有关键 checkpoint 镜像：

- `E:/UNSB_Expl/paper_eval_node/checkpoint_imports/sources`（当前约 4.23 GB；固定里程碑齐全）
- `E:/UNSB_Expl/runs/FINAL_UNSB_DCLGAN_LOCAL1000_E45973A`（约 5.63 GB）

前者包含 plain、AM-TNC、ST-CGR、Proposal 和 matched plain 的固定里程碑。因此 canonical
结论不存在“只在租赁服务器保存一份”的单点风险。

仍未进入 canonical Git 的只有独立证明工作：

- 4090A：`/home/yc/independent_audit_20260916`
- 5090A：`/root/autodl-tmp/independent_audit_20260915`

它们必须完成后单独做科学裁决和 proof archive；未经裁决不能修改当前主结论。5090C 上
迁移来的 matched-plain 原始资产按用户要求没有在本次归档中搬动或删除，其关键固定
checkpoint 已在本地镜像，比较 provenance 已在最终 portfolio 中。

## 6. 后续唯一正确的顺序

1. 先做人工/Codex claim review，冻结论文主张、披露、算法集合和 confirmation policy。
2. 需要时先单独审阅上述 independent proof，不自动合并。
3. 只有显式授权后，才一次性打开 confirmation20；不能再据此改算法。
4. 若预算允许，seed 2027/2028 只优先复现 plain 与 Proposal；不得反向改 seed 2026 算法。
5. 写论文时同时报告 CUT/DCLGAN/CycleGAN 的绝对结果、Proposal 的 UNSB 内部相对收益，
   并明确单 seed、128 分辨率、runtime cohort 和 DCLGAN standalone 的边界。

除非用户重新明确授权新的科学问题，否则不要恢复旧训练队列、寻找退出阈值、重跑
ST-CGR/AM-TNC 当前版本、打开旧 5090B successor、选择最佳 checkpoint，或为了占 GPU
而新增实验。

## 7. 接手者最短阅读顺序

1. 本文件；
2. `FINAL_HANDOFF.json`；
3. `archive/paper_aio/final_v10/ARCHIVE_MANIFEST.json`；
4. `archive/paper_aio/final_v10/PAPER_ALGORITHM_PORTFOLIO_WITH_DCLGAN.json`；
5. `research/paper_aio/RESEARCH_HANDOFF_CN.md`；
6. 只有需要追溯时再读历史合同、计划与 decisions。

仓库中包含本交接文件的实际 Git commit 以 `git log -1` 为准；本文件不内嵌自身 commit，
以避免自引用哈希不可能稳定的问题。
