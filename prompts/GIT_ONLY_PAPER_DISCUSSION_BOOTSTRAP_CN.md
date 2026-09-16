# 给全新 Codex 的 FINAL_UNSB 启动提示

请把下面整段作为新任务的第一条消息：

> 这是 FINAL_UNSB 论文工作的全新上下文。不要恢复旧训练、旧 PID、旧 successor 或旧服务器
> 队列。请先完整阅读仓库根目录的 `FINAL_HANDOFF_CN.md`、`FINAL_HANDOFF.json`、
> `DISASTER_RECOVERY_CN.md`，再读
> `archive/paper_aio/final_v10/ARCHIVE_MANIFEST.json`、
> `archive/paper_aio/final_v10/PAPER_ALGORITHM_PORTFOLIO_WITH_DCLGAN.json`和
> `research/paper_aio/RESEARCH_HANDOFF_CN.md`。以这些文件为当前权威；更早的 ACTIVE 文档只作
> provenance。先运行 `python tools/verify_git_only_recovery.py`，然后用中文向我说明：已经证明
> 的结果、不能声称的内容、Proposal/ST-CGR/AM-TNC 的关系、外部基线位置、尚未归档的二进制
> 和 independent proof、confirmation20 状态，以及下一项需要我决定的事情。不要启动实验，
> 除非我随后明确授权。

如果任务是写论文，再继续读取：

- `research/paper_aio/RELATED_WORK_NOVELTY_BOUNDARY_CN.md`
- `research/paper_aio/METHODS_PROTOCOL_DRAFT_EN.md`
- `research/paper_aio/MANUSCRIPT_RESULT_BRANCHING_OUTLINE_CN.md`
- `research/paper_aio/LIMITATIONS_PRE_RESULT_EN.md`
- `configs/PAPER_REFERENCE_LEDGER.json`

如果任务是重新开始算法研究，再读取 ST-CGR/AM-TNC disposition 和 small25 历史证据；
不得退回退出窗口搜索或最佳 checkpoint 选择。
