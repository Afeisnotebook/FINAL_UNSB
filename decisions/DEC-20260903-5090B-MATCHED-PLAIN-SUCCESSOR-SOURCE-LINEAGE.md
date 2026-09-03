# 5090B matched plain 后继控制器源码谱系冻结

日期：2026-09-03

5090B 当前 CUT 与 CycleGAN 训练保持连续运行；本次只替换尚在等待 CUT e200 的
matched-plain successor 及其专属 health watcher。旧控制器已经锁定 scientific
training checkout，但等待期没有把自身 control checkout、控制源码、manifest 和
5090A peer runtime receipt 一并冻结。这不会立即改变训练，却会留下数日后用漂移
控制代码静默启动 171 万步任务的延迟风险。

v2 successor 在进入等待前生成不可变合同，并在每次轮询、每个工程门、容量门和真正
启动前重复核对：control commit/clean、control source SHA256、scientific commit/clean、
manifest SHA256、peer receipt SHA256、协议 fingerprint、fresh-e0 与禁止跨宿主 checkpoint
resume 的语义。任何漂移均 fail closed，不启动训练。

第一次远端部署暴露了共驻容量合同中的 `Path` 不能 JSON 序列化；进程在取得训练授权前
立即退出，CUT/CycleGAN 未受影响。实现随即把共驻路径规范化为绝对字符串，增加真实
共驻配置的 JSON 序列化测试，并再次通过完整 591 项测试。最终部署 commit 为
`a9c63cff5b5f23baf6ed8548b01abcc51e0f7be8`。

新 successor PID `115562`、health watcher PID `115614`，均由 PID 1 托管；健康状态为
零告警。旧 PID `61721/61725` 已退出。CUT 与 CycleGAN 的 supervisor/trainer PID
`3739/5771` 和 `11845/11846` 均未收到信号、未迁移、未恢复或重启。

本次没有读取任何性能值，没有启动 matched plain，没有打开 confirmation20，也没有
改变容量门决策。CUT e200 后仍须依次通过 preflight、resume、2000-update exact runtime
twin、重复评估和 authorization；随后才允许按 target-blind makespan 容量门决定与
CycleGAN 共驻还是精确暂停等待。
