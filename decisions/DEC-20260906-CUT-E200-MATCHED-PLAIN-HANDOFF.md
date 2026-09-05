# CUT e200 与 5090B matched-plain 工程交接

Date: 2026-09-06  
Scope: fixed-e200 completion, source-bound export and metric-blind control handoff

CUT 已在 5090B 完成固定的 200 data epochs，即 1,710,600 次更新。e200
checkpoint 的实算 SHA256 为 `81c213df...6b032`，与 sidecar 和 source-bound
export receipt 一致；scientific-state hash 也在 heartbeat、sidecar 与 export 中一致。
监督器以 return code 0 写入 `COMPLETE_E200`，外层连续性 guard 零重启退出。CUT trainer、
supervisor 和 exporter 的终止均是完成后的预期终态，不得按普通 PID 消失报警重新启动。

既有 matched-plain successor PID 148465 已自行从等待态转入
`RUNNING_EXACT_ENGINEERING_GATES`，当前执行五项门中的第二项 resume gate。wait-only guard
PID 441694 已按合同写入 `HANDOFF_STARTED_RECOVERY_RELINQUISHED` 并正常退出，不再属于
可恢复对象。CycleGAN 继续原轨迹运行到最新完整 e130，未被 CUT 终态或 handoff 中断。

这仍不是 runtime-equivalence 或 matched-delta 许可。5090B 必须先完成全部精确工程门、
产生自己的 2,000-update runtime receipt，再执行预注册的两 data-epoch metric-blind
共驻容量门。只有该结果形成且 Proposal/plain、ST-CGR/plain 的 relation candidate 被
显式审阅并提交 Git 后，跨宿主 matched delta 才合法。当前没有启动长 plain、没有读取
paired 性能、没有打开 confirmation20，也没有更改任何健康训练。

权威路径、PID 与完整哈希见
`evidence/paper_aio/PAPER_AIO_CUT_E200_AND_MATCHED_PLAIN_HANDOFF_20260906T044850.json`。
