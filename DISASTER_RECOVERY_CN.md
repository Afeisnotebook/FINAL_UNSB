# FINAL_UNSB Git-only 灾难恢复说明

状态：`GIT_ONLY_RESEARCH_CONTINUITY_COMPLETE / BINARY_RECOVERY_EXTERNAL`

## 先说结论

如果本地电脑被格式化，只要 GitHub 仓库仍在，仅凭一次全新 clone，可以完整恢复：

- 项目的研究目标、三个月探索脉络和最终科学裁决；
- Proposal/ST-CGR/AM-TNC 的数学定义、实现、谱系和失败边界；
- full-data 协议、数据划分身份、逐文件数据 SHA256 清单；
- 最终论文数字、逐方法轨迹摘要、复杂度、运行时关系和披露项；
- 外部基线结果、论文相关工作入口和下一步正确讨论顺序；
- 训练、评估、恢复、导出和审计代码。

因此，在一台全新机器上与完全没有旧上下文的 Codex 继续讨论论文，只靠 Git 是足够的。
使用`prompts/GIT_ONLY_PAPER_DISCUSSION_BOOTSTRAP_CN.md`作为第一条提示即可。

但普通 Git **不能**恢复以下二进制资产：

- 六域图像像素；它们受数据许可限制，不能公开再分发；
- 约 4.23 GB 的关键模型 checkpoint 和约 5.63 GB 的 DCLGAN 完整运行目录；
- 完整训练日志、所有逐图原始输出；
- 尚未裁决的服务器 independent proof 工作目录。

所以“恢复研究和论文讨论”已经达到 Git-only；“拿回训练好的权重、从某个 epoch 继续训练、
无需重新取得数据就复跑”仍需要 Git 之外的私有冷备份。不要把这两个层级混为一谈。

## 全新环境恢复步骤

```bash
git clone https://github.com/Afeisnotebook/FINAL_UNSB.git
cd FINAL_UNSB
python tools/verify_git_only_recovery.py
```

验证通过后按顺序阅读：

1. `FINAL_HANDOFF_CN.md`
2. `FINAL_HANDOFF.json`
3. `archive/paper_aio/final_v10/ARCHIVE_MANIFEST.json`
4. `archive/paper_aio/final_v10/PAPER_ALGORITHM_PORTFOLIO_WITH_DCLGAN.json`
5. `research/paper_aio/RESEARCH_HANDOFF_CN.md`

若需要运行代码，在 Ubuntu/Python 3.11 环境执行：

```bash
bash scripts/bootstrap_server.sh "$PWD"
.venv/bin/python -m pytest -q
```

`scripts/bootstrap_server.sh`固定 PyTorch 2.8.0、torchvision 0.23.0、cu128以及
`environment/requirements.txt`中的依赖。不同 GPU/驱动栈不自动拥有 byte-identical runtime
资格，必须重新做相应门禁。

## 数据恢复

Git 中的`manifests/FULL_DATA_MANIFEST.csv`包含 9,153 个物理 identity 的相对路径、大小与
逐文件 SHA256，但不包含图像本身。重新合法取得六域数据后，将其放到共同`DATA_ROOT`，
使用 manifest/preflight 校验。只有所有文件哈希与
`02c01df580b882763fb0ff28dbdeac4b3729deb8bb772005f26f3e7bc2e36744`
对应的 canonical manifest 一致，才能声称恢复了同一数据集。

数据来源、许可和禁止公开再分发的原因见：

- `configs/PAPER_DATASET_PROVENANCE_CONTRACT.json`
- `research/paper_aio/DATASET_PROVENANCE_AND_REUSE_NOTICE_EN.md`
- `DATA_CONTRACT.json`

## 二进制冷备份缺口

机器可读缺口和六个关键 e200 full-state 的 SHA256 记录在
`configs/GIT_ONLY_RECOVERY_MANIFEST.json`。最低限度的私有冷备份应包含：

1. plain、matched plain、Proposal、ST-CGR、AM-TNC、DCLGAN 的 e200 full state与sidecar；
2. Proposal/plain 的 e150/e175/e200（论文 sustained 复核）；
3. 六域原始数据或可合法重新下载的来源凭据；
4. 独立证明若经裁决后值得保留，则建立单独 proof archive。

推荐至少保留两份不在同一物理地点的私有副本，并以本仓库记录的 SHA256 验收。由于数据
许可禁止公开再分发，本仓库不会擅自把图像或大型 checkpoint 推到普通 GitHub Git 历史。

## Git-only 能与不能

| 目标 | 仅 Git 是否足够 |
|---|---|
| 在全新 Codex 中继续论文讨论 | 是 |
| 恢复算法、代码、协议、最终数字和结论边界 | 是 |
| 审核最终结果来源与哈希 | 是 |
| 在重新合法取得相同数据后从头复现 | 是，但需重新做 runtime 门禁 |
| 直接加载训练好的模型权重 | 否，需要私有二进制备份 |
| 从现有 checkpoint 无损续训 | 否，需要 full-state checkpoint |
| 恢复未裁决 independent proof | 否，除非另行归档 |

任何未来接手者若只需要理解、写作、审稿或设计下一轮实验，不应因为 checkpoint 不在 Git
而重新跑旧任务；先使用 compact canonical evidence。
