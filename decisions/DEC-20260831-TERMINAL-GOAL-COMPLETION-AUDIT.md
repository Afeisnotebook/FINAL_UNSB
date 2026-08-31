# DEC-20260831：终点 Goal 完成审计必须独立于候选排名

## 决策

路线一终交付在远端生成并逐字节返回本地后，必须再经过一项 fail-closed 的 Goal 完成
审计。候选排名或五个文件“存在”本身都不能证明研究目标完成；审计必须验证它们确实
保留完整多候选研究前沿、长期因果证据、数学身份、逐域绝对/相对轨迹和复现边界。

唯一 `CANDIDATE.json` 继续只表示下一阶段行动优先级。它不能覆盖或删除
`RESEARCH_FRONTIER.json` 中的共同领先、机制型递补、因果可修订方向或失败负对照；
单 seed 下很小的收益差也不能自动升级为父机制证伪。

## 自动审计范围

`280d685` 新增的审计器在本地结果 relay 完成后自动核验：

- pointer、relay 和全部终交付文件 SHA256 一致；
- 选中算法来自共同 e0、seed2026、batch1、small25 的真实 e200/30000 updates，且未选
  最佳 checkpoint；
- 候选具有非空的 UNSB 数学对象、公式、identity/self-null/无偏性质之一、target 不可
  访问说明、成本、风险、复现命令与延期 seed 声明；
- e150/e175/e200 六域 candidate/plain/delta 的 PSNR、SSIM、LPIPS 均完整；
- 选中机制具有 full/proposal-only/observable-only 三角色证据，或合法两组件合成的四项
  来源证据；
- 恰好两个兼容接口递补，同时完整 4090/5090 宿主分离前沿仍存在；
- 每条候选的 receipt、trajectory、derivation card、implementation 和逐域轨迹均被绑定；
- DT/HJ/HNEK e200 探针、proxy 校准、474 条 reversal、140 条 sampling variance 和完整
  hypothesis ledger 均在终交付中；
- 没有跨宿主 delta 合并、paired controller 或 confirmation20 访问；
- 最终报告明确分开科学结论、工程失败、代理失真和未测试假设。

审计通过仍不自动完成 Goal。它只证明终点产物满足机器可验证边界；主开发 worktree 还
必须完成人工科学复核、compact adjudication、完整测试、提交、推送及远端一致性检查，
之后才允许把 Goal 标为 complete。

## 持久执行

冻结 worktree 为 `E:\UNSB_Expl\FINAL_UNSB_GOAL_AUDIT_280d685`，本地独立进程 PID 5332
等待精确终交付 relay，不依赖当前 Codex 会话。合同 SHA256 为
`ad59f5cec1b5ed7ec6ee05148f89aafe30ee8052c358373b029a8ddd78aff250`；当前状态为
`WAITING_FOR_EXACT_TERMINAL_DELIVERY_RELAY`。合同不保存凭据，不传 checkpoint，不读取
中间 paired 指标。

部署证据见
`evidence/remote_route1_offload/TERMINAL_GOAL_COMPLETION_AUDIT_SUCCESSOR_ARMED_20260831.json`。
