# DEC-20260912: 本地 terminal-audit 读取恢复连续性

## 决策

保留当前健康的 terminal-audit supervisor `5180` 与动态 child `30948`，不为采用新代码而热替换。旧冻结读取器在 Windows 原子发布窗口内发生的两次 `PermissionError` 已由既有监督器自动恢复；当前累计恢复计数为 `3/20`，child 自 09:09 起持续处于 `WAITING_FOR_PREFLIGHT_IMPORTS_OR_GPU`，combined health 为 `HEALTHY` 且零告警。

这不是训练中断。terminal-audit 当前只等待完整 imports 和本地 GPU 释放，没有占用 GTX1660，也没有改变任何 checkpoint、sampler、RNG、算法或训练进程。

## 后继边界

- 当前 child 继续等待，不进行手工重启。
- PID `30948` 取代旧 PID `31284`，成为当前动态 child；以后不得因旧 PID 消失而误报。
- 若 supervisor `5180` 真正死亡，再在重新核验冻结命令和现有状态后，使用 `f2a601a` 中已测试的 PermissionError 有界重试读取器恢复。
- persistent read error 仍然 fail closed；不忽略哈希、不接受 torn JSON。
- terminal JVP 继续等待 DCLGAN 释放 GTX1660，不与训练共驻。

证据：`evidence/paper_aio/PAPER_AIO_LOCAL_TERMINAL_AUDIT_READ_RECOVERY_CONTINUITY_20260912T132032.json`。
