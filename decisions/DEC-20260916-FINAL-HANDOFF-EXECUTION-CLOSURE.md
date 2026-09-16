# DEC-20260916：最终交接与旧执行入口关闭

## 决定

full-data discovery 执行阶段已经完成。`FINAL_HANDOFF_CN.md`与`FINAL_HANDOFF.json`成为
离开当前 Codex 上下文后的首要入口；旧的 active plan、在线宿主快照、PID、successor、
SSH 端点和 heartbeat 只保留作历史 provenance，不再授权恢复或新增训练。

## 原因

最终 V10 portfolio、DCLGAN addendum、算法 disposition、复杂度、source-export receipt、
关键运行时关系与资产地图均已进入 commit `df20b8b`之后的 portable archive。继续把
2026-09-12 的“正在运行”描述放在根入口，会使新的人或无上下文 Codex 错误重启已完成
任务，或把独立证明误并入 canonical 结果。

## 科学边界

- Proposal-only 是当前唯一通过预注册 full-data 长期门的自研方法。
- ST-CGR/AM-TNC 只关闭当前实现；HJCGR deferred；DDSB reproduction incomplete。
- confirmation20 仍封存；没有跨 seed 稳定性结论。
- 服务器上的 independent proof 必须单独裁决，不能自动导入。
- 本决定不删除 checkpoint、日志、服务器目录，也不停止或修改任何独立实验。

## 下一门

显式的论文 claim review 与 confirmation policy freeze。除非用户重新授权新的科学问题，
不得从历史文档恢复训练或以中间 paired 指标选择算法/窗口/checkpoint。
