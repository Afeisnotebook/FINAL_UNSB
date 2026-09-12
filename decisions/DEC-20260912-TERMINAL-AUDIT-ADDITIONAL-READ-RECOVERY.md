# Terminal audit 追加读取恢复裁决

本地 terminal-audit 冻结读取器在既有两次 Windows 原子发布瞬时锁之后，又出现两次同类 `PermissionError`。控制 supervisor `5180` 已自动恢复出 child `18588`；状态仍为 `WAITING_FOR_PREFLIGHT_IMPORTS_OR_GPU`，三个已完成的 plain 审计单元没有丢失，combined health 为 `HEALTHY` 且零告警。

不热替换或重启当前健康冻结 child。提交 `f2a601a` 中已经测试的 PermissionError-only 有界重试仍作为正式恢复实现：只有 supervisor `5180` 真正退出或耗尽其有界预算后，才重新核验冻结命令和现有状态并采用该实现恢复。该策略既不改变审计量，也避免仅为采用代码修复而扰动健康等待链。

本事件是控制面读取竞态，不是训练、算法或科学失败；没有读取性能值，没有打开 confirmation20，也没有影响 DCLGAN 的 GTX1660 独占训练。
