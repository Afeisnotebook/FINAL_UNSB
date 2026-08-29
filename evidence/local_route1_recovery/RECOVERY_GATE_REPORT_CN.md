# 本地路线一持久执行恢复门

日期：2026-08-30
状态：`PASS_DURABLE_RECOVERY_GATE / ANCHORS_RUNNING`

## 正式生产分块验收

- current-user计划任务 `FINAL_UNSB_ROUTE1_EXECUTOR` 已注册并独立于Codex生命周期运行。
- 冻结plain e100状态已连续完成到e105：15000→15750 updates，子进程exit code 0。
- 输入checkpoint SHA256：`aa73748e...4c983c`；输出checkpoint SHA256：
  `1de7ebf0...05672a`。
- supervisor在接纳e105后自动启动e105→e110，证明恢复不是一次性手动续跑。
- 该门只证明耐久执行、身份冻结与状态续接，不提供任何算法收益结论。
- confirmation20仍未打开。

## 事故归因

plain先后在e60附近和e100 checkpoint保存后无traceback硬终止。e100 checkpoint、
优化器、sampler和RNG均完整；隔离的420张LPIPS评估正常。重复硬终止发生在不同科学
阶段，支持“隐藏子进程生命周期失效”，不支持e100算法、LPIPS、显存或数据故障。

## 已通过恢复证据

- e100正式milestone从冻结checkpoint执行两次，完整逐图payload SHA256均为
  `25eb7687...159c6`。
- 评估前后模型与RNG状态哈希均为`fc4ba2d2...c26a7f`。
- 独立scratch中在e1完整边界后、e2训练中人为终止Python进程；e1原子checkpoint
  完整且无`.tmp`残留。
- 从幸存e1状态恢复到e2后，科学状态哈希为`8f4a334d...b98db5`，与既有连续/
  resume工程门参考逐位一致。
- confirmation20全程未打开。

## 新执行结构

- 科学训练固定在detached worktree `FINAL_UNSB_EXECUTOR_0da2a37`。
- 主分支的运维代码不进入冻结科学fingerprint。
- 每个训练子进程最多5 data epochs；每epoch保留原生heartbeat。
- 外部supervisor记录PID、父PID、exit code、stdout/stderr、输入/输出checkpoint hash、
  commit、protocol、manifest、updates和data epochs。
- 15分钟无heartbeat触发终止和恢复；同一chunk三次失败后停止。
- 缺失registered milestone评估先执行两次一致回填，不能静默跳过。

Windows当前用户计划任务注册和首个正式e100→e105 chunk仍是本门最后的运行时验收。
