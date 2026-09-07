# DEC-20260908：ST-CGR 与 Proposal 长训外层恢复守护

## 裁决

ST-CGR 与 Proposal 当前训练均健康，不允许重启或迁移；但它们的内层 supervisor 已分别
累计 2 次和 1 次工程故障。内层失败计数不会自行清零，ST-CGR 再发生一次故障就可能以
`BLOCKED_AFTER_REPEATED_ENGINEERING_FAILURE`退出。单纯依赖 heartbeat 只能发现停止，不能
保证长训自动续接，因此这是北极星任务中的真实静默停机风险。

为两条 lane 部署候选感知、身份锁定的外层 guard。它在原 supervisor 存活时只收养监控，
不会操作 trainer；只有 supervisor 和 trainer 同时消失，且代码、Git commit、协议、授权、
Python 解释器以及 latest full-state 全部重新验签后，才按冻结命令启动新的内层 supervisor。
出现重复进程或任一身份差异时 fail closed。

## 实施结果

- ST-CGR 原 PID `336345/863376` 未变化；外层 guard `942101`、health `942182`，均 PPID=1，
  状态为 `MONITORING_EXISTING_SUPERVISOR` / `HEALTHY`，零重启、零告警。
- Proposal 原 PID `9729/366768` 未变化；外层 guard `431648`、health `431712`，均 PPID=1，
  状态为 `MONITORING_EXISTING_SUPERVISOR` / `HEALTHY`，零重启、零告警。
- ST-CGR e81 与 Proposal e89 的当前 full-state 都在部署前完成了文件哈希预检。
- guard 明确区分普通 lane 与 candidate authorization；ST-CGR 使用候选 ID
  `G4-01-STRATIFIED-TIME-CONDITIONAL-GF`，不会被误识别为普通静态 lane。
- 两个运行时都绑定实际解释器 SHA256；没有完整运行时树 manifest，因此回执明确写为
  `runtime_identity_bound=false`，不夸大验证范围。两条 lane 既有 exact-resume 工程门保持有效。

这次操作没有读取性能指标、没有改变算法协议、没有打开 confirmation20，也没有触碰正在
执行的训练进程。完整机器可读回执见
`evidence/paper_aio/PAPER_AIO_STCGR_PROPOSAL_OUTER_RECOVERY_GUARDS_20260908T000000.json`。
