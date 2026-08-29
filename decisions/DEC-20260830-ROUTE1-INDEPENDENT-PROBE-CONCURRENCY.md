# DEC-20260830：以独立进程并行长期探针，不改变 batch 或训练更新

状态：`AUTHORIZED_IMPLEMENTATION_GATE`  
上游：`DEC-20260830-ROUTE1-REMOTE-OFFLOAD.md`、
`DEC-20260830-ROUTE1-REMOTE5090.md`

## 背景

4090/5090 单个 batch1 UNSB 进程只使用约 1.6--1.8GB 显存，不能充分利用服务器。
用户授权由研究主控决定是否提高 batch 或并行更多任务。

batch size 不是单纯的数值精度参数。改变它会同时改变无配对 B 的经验采样、
PatchNCE negatives、Adam 轨迹和 HJ/DT 的校正对象，因此当前 long anchors、candidate
e200 和 seed validation 继续固定 batch1。

## 决定

允许同一主机在隔离 output root 中并行运行相互不消费状态的探针。第一项只允许
HNEK：它与 HJ 都从同一个 shared e0 独立开始，原先“必须等 HJ e200”只是 runner 的
串行调度门，并不是算法依赖。

并行 wrapper 必须：

- 从不可变 `0da2a37` worktree 导入全部训练代码；
- 保持原 protocol fingerprint、manifest、batch1、seed2026 和 200 data epochs；
- 只在同机 plain e200 已完成且 checkpoint/metadata/hash 全部匹配时启动；
- 复制 exact shared e0 到隔离 root，不读取任何另一方法 checkpoint；
- source-bind wrapper，并明确记录唯一绕过项是 HJ-before-HNEK 的调度依赖；
- 独立保存 checkpoint、日志和 lock；不得与 canonical HNEK 同时写同一目录；
- 保持 paired controller 禁止和 confirmation20 封存。

5090 的补充调度允许在同机 matched plain 仍运行时提前启动 HNEK，但必须同时满足：

- canonical executor contract、运行状态、训练 worktree、manifest、train view、data root
  和 shared e0 全部精确匹配；
- 输出标记为 `QUARANTINED`，在本机 plain e200 checkpoint/hash 验证完成前不得比较、
  排名、审计或导入 canonical root；
- HNEK 返回时再次验证 plain e200；若仍未完成，结果继续封存并以非成功状态退出；
- 这一放宽只改变任务启动顺序，不改变训练 update，也不放宽 DT 的 proxy gate。

并行结果仍只能与本机 plain 比较。它完成后若要提供给 canonical auditor，必须先停止
canonical 对同一 lane 的写入，核验完整 checkpoint/hash，再以显式 import evidence
合并；不得静默覆盖。

## 后续并行边界

- proxy 校准前不得提前运行 DT；
- 因果审计可将不同 checkpoint/branch 分配给独立进程；
- 最多两个已冻结候选可在隔离 root 并行，但每个仍需自己的 gate/full state；
- 较大 batch 只可作为显式标注的非科学 throughput micro-run，不能进入路线一排行榜。
