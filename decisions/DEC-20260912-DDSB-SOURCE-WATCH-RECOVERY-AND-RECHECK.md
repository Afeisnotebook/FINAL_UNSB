# DEC-20260912：恢复 DDSB 权威源码守望，不放行猜测实现

## 故障与影响

DDSB 权威源码 watcher PID `20388` 于 2026-09-11 16:49 因 Windows 对固定临时状态文件
`DDSB_SOURCE_WATCH_STATE.json.tmp` 的一次共享访问拒绝而退出。原 health watcher PID `10780`
已正确报告 `ALERT_PID_DEAD`；这不是正常等待，也不能继续在项目状态中写成健康。

故障只发生在公开来源观察链：没有训练被启动或停止，没有 checkpoint 被读取，没有中间 paired
指标或 confirmation20 被访问，也没有改变 DDSB 的科学裁决。旧输出和 fatal receipt 原样保留。

## 工程恢复

提交 `40c4b37916844bb10b456926d8e44f0df8ce66c3` 将状态写入改为每个 writer 唯一的临时文件，
并只对短暂的 Windows sharing violation 做有界重试，最终仍使用原子的 `os.replace`。这不改变
来源列表、候选接受规则或训练授权。定向测试为 `8 passed`，compile 与 `git diff --check` 通过。

新的 detached control worktree 为 `E:/UNSB_Expl/FINAL_UNSB_DDSB_WATCH_40C4B37`。source watcher
PID `22104` 与 health watcher PID `16220` 已连续通过两个 health poll，状态分别为
`WAITING_FOR_AUTHORITATIVE_SOURCE` 与 `HEALTHY`，零告警。替代链稳定后，旧的报警 health PID
`10780` 已退出；任何科学训练进程均未触碰。

## 来源裁决

本次实时轮询重新访问 NeurIPS 2025 官方论文页和 Westlake 作者实验室论文页：两处仍没有与
DDSB 绑定的 repository；两条 GitHub repository 查询也没有相关作者实现。因而 DDSB 继续是
`REPRODUCTION_INCOMPLETE`，不是性能负结果。公开论文和补充材料尚未唯一确定 U-Net、MINE、
判别器、优化器、更新顺序、stop-gradient、损失归一化及当前 All-in-One 协议适配，禁止用猜测
实现补表或占用 GPU。

机器可读证据：
`evidence/paper_aio/DDSB_SOURCE_WATCH_RECOVERY_AND_RECHECK_20260912T060322.json`。
