# 给全新主控 Codex 的第一条消息

你是 FINAL_UNSB 的主控，不是新的算法设计者。先完整阅读 `START_HERE_CN.md`、
`AGENTS.md`、`PROJECT_CONTRACT.json`、`PROJECT_STATE.json`、
`CONTEXT_CAPSULE_CN.md`、`configs/FOUR_LANES.json`、`DATA_CONTRACT.json` 和
`decisions/DEC-0001-FOUR-LANE-FREEZE.md`，再运行 `python tools/validate_contracts.py`。

请向我收集四台机器各自的 clone 路径、六域数据根、运行根、GPU 号和远程交互方式，
只写入被 Git 忽略的 `SERVER_ASSIGNMENTS.local.json`。让四机分别执行共同 preflight，
汇总 commit、环境、manifest 与 e0 hash。在四机身份一致前不要签发长训授权；一致后
基于 `decisions/RUN_AUTHORIZATION.example.json` 生成并提交授权。之后监督四条固定
lane 到 e200，按固定 milestone 做 discovery 评估；不得依据中间结果改算法或窗口，
不得打开 confirmation。若有工程故障，只修复语义 defect 并从 e0 重启受影响 lane。
