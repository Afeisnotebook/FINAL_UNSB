# FINAL_UNSB 当前工作流

## 默认工作流：论文与审阅

1. 运行`python tools/verify_git_only_recovery.py`；
2. 阅读final handoff、当前状态和claim boundaries；
3. 从final V10 portfolio引用数字，不从旧heartbeat或中间checkpoint取数；
4. 完成claim review并冻结披露、算法集合和confirmation policy；
5. 只有显式授权后才打开confirmation20或启动额外seed；
6. 所有新结论形成decision与compact evidence后提交Git。

## 如果重新启动科学计算

必须创建新的授权，记录科学问题、protocol fingerprint、数据/seed、runtime cohort、停止门和
confirmation边界。旧`RUN_AUTHORIZATION`、PID、server task或successor不得复用。

Git只保存代码、合同、manifest、哈希、compact metrics和decision；数据、checkpoint及完整
日志放在Git外，并用source-bound receipt和私有冷备份保护。

## 历史工作流

旧四机执行与return-branch说明属于provenance。查看
`docs/HISTORICAL_PROVENANCE_INDEX_CN.md`，不要从历史文本启动任务。
