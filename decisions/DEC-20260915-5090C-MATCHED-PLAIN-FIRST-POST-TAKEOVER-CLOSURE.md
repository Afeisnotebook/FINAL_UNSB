# 5090C matched plain 首个迁移后完整状态闭合

5090C上的逻辑`5090B_MATCHED_PLAIN`已从e147连续运行到e149。最新e149 full-state文件SHA256与sidecar记录完全一致，scientific-state hash已发布；因此“首个迁移后完整epoch”门最迟在e149得到严格闭合。训练supervisor、trainer、guard和health链保持原PID，guard零重启、health零告警。

迁移后的两轮完整data epoch合计耗时4530.33秒，e149单epoch耗时2241.09秒。由这一真实吞吐重估，e200训练预计在2026-09-16 08:00–12:00完成，之后仍需source-bound export、异机导入和最终segmented-runtime relation审核。

下一固定门是e150增量source export及本地异机导入。未读取paired性能，未改变训练、算法、数据、RNG、sampler或checkpoint，confirmation20继续封存。

权威证据：`evidence/paper_aio/PAPER_AIO_5090C_MATCHED_PLAIN_E149_FIRST_POST_TAKEOVER_CLOSURE_20260915T010906.json`。
