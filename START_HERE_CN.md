# 先从这里开始

本文件不再保存会过期的服务器进度。当前权威入口是：

1. `FINAL_HANDOFF_CN.md`
2. `FINAL_HANDOFF.json`
3. `docs/CURRENT_PROJECT_STATE_CN.md`
4. `CLAIM_BOUNDARIES.md`
5. `configs/DOCUMENT_AUTHORITY_REGISTRY.json`

## 一句话状态

完整 full-data discovery 已结束并归档；Proposal-only通过长期门，ST-CGR和AM-TNC当前
实现未通过，HJCGR deferred，DDSB reproduction incomplete；confirmation20仍封存。

## 新 Codex 的第一项工作

```bash
python tools/verify_git_only_recovery.py
python tools/validate_contracts.py
```

然后复述已经证明的结论、不可写的主张、二进制恢复缺口和下一决策门。不要启动实验、
恢复旧队列或读取 confirmation20。

## 重要防误读

- `ACTIVE_LOCAL_ROUTE1_PLAN_CN.md`是small25历史完成记录，不是当前计划；
- `ACTIVE_PAPER_AIO_PLAN_CN.md`是已关闭的执行计划墓碑；
- frozen protocol中的`ACTIVE_*`是历史schema token，不是任务授权；
- `PROJECT_STATE.json`、method portfolio和delivery matrix中的旧PID/heartbeat只作provenance；
- 当前结果只看final handoff与`archive/paper_aio/final_v10/`；
- independent proof未经单独裁决，不改变canonical结论。

完整历史可在 Git commit `17cd2edeb5263d7e3b46cf1f9ed24d9ab1854488`查看，索引见
`docs/HISTORICAL_PROVENANCE_INDEX_CN.md`。
