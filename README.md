# FINAL_UNSB

FINAL_UNSB 是 UNSB 六域 All-in-One 无配对恢复项目的可审计代码、协议与结果仓库。

当前阶段不是“正在训练”，而是：

`full-data discovery 已归档 / 等待论文 claim review / confirmation20 仍封存`

## 新环境从这里开始

无论人还是没有旧对话的 Codex，按以下顺序阅读：

1. [`FINAL_HANDOFF_CN.md`](FINAL_HANDOFF_CN.md)
2. [`FINAL_HANDOFF.json`](FINAL_HANDOFF.json)
3. [`docs/CURRENT_PROJECT_STATE_CN.md`](docs/CURRENT_PROJECT_STATE_CN.md)
4. [`CLAIM_BOUNDARIES.md`](CLAIM_BOUNDARIES.md)
5. [`archive/paper_aio/final_v10/PAPER_ALGORITHM_PORTFOLIO_WITH_DCLGAN.json`](archive/paper_aio/final_v10/PAPER_ALGORITHM_PORTFOLIO_WITH_DCLGAN.json)
6. [`research/paper_aio/RESEARCH_HANDOFF_CN.md`](research/paper_aio/RESEARCH_HANDOFF_CN.md)
7. [`DISASTER_RECOVERY_CN.md`](DISASTER_RECOVERY_CN.md)

机器可读的文档权威顺序在
[`configs/DOCUMENT_AUTHORITY_REGISTRY.json`](configs/DOCUMENT_AUTHORITY_REGISTRY.json)。
不要按文件名中的`ACTIVE`、旧 PID 或旧 heartbeat 判断当前状态。

## 最终 discovery 结论

- Proposal-only 是唯一通过预注册 full-data 长期门的自研算法：late-three 宏 PSNR
  delta `+1.723565 dB`，e200 delta `+0.839347 dB`。
- ST-CGR 与 AM-TNC 关闭当前实现，不判死其父机制。
- HJCGR 为 deferred；DDSB 为 reproduction incomplete。
- CUT、DCLGAN、CycleGAN 的固定 e200 绝对 PSNR 均高于 Proposal，不能声称总体 SOTA。
- 当前只有 seed 2026，不能声称跨 seed 稳定。
- terminal low-variance/singular-drift 假设未确认，不能宣称或加入对应修复模块。

协议为每侧8,553张、batch1、seed2026、200 data epochs（1,710,600 updates）、固定e200
主表、e150/e175/e200 sustained；不选最佳 checkpoint，不用 paired 指标控制训练。

## 当前下一门

先完成论文 claim review，冻结主张、披露、算法集合与 confirmation policy。只有新的明确
授权才能打开 confirmation20、增加 seed 或重新启动科学计算。旧服务器队列和 successor
全部没有自动恢复权。

## Git-only 恢复

```bash
python tools/verify_git_only_recovery.py
```

Git 已足以在新机器恢复研究状态、代码、协议、最终数字和论文讨论上下文；数据像素、模型
checkpoint和完整日志仍需要私有二进制冷备份。详见`DISASTER_RECOVERY_CN.md`。

## 历史材料

历史计划、合同、evidence和decision保留用于审计，不是当前调度入口。分类和重构前快照见
[`docs/HISTORICAL_PROVENANCE_INDEX_CN.md`](docs/HISTORICAL_PROVENANCE_INDEX_CN.md)。

## 验证

```bash
python tools/verify_git_only_recovery.py
python tools/validate_contracts.py
python -m pytest -q
```
