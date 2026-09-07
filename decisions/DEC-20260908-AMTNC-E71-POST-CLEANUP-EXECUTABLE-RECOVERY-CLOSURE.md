# DEC-20260908：4090A 错误清理后的 AM-TNC e71 可执行恢复闭环

## 裁决

错误清理造成了真实损坏：原 `unsb_cov` 完整环境不存在，运行中的 AM-TNC trainer
PID 3446758 仍持有 deleted inode。这个标签不能在不中断训练的情况下消除，也不能靠原地
覆盖旧目录来“修好”。健康的 supervisor/trainer 不重启、不迁移，继续训练到 e200。

但这不再等同于“当前进程退出后无路可走”。主项目现在同时具备当前 full-state 的结构加载、
执行级单步恢复、隔离运行时全量验签、直接隔离启动的训练 guard、独立导出/评估恢复链和
异机备份。事故风险已被限制在一个尚未原子保存的 data epoch，而不是全部训练进度。

## 本次现场证明

- AM-TNC 已由原 PID 自然推进到 e71 / 607,263 updates；full-state SHA256 为
  `07c3d71d...c588ce`，scientific-state SHA256 为 `11e77db8...42a0f`。
- 隔离训练 runtime 的 111 项、4,198,514,872 bytes 重新逐文件验签：0 缺失、0 不匹配、
  0 可写；guard 的 `--verify-only` 对 e71 checkpoint 返回
  `PASS_RUNTIME_AND_CHECKPOINT_RECOVERY_PREFLIGHT`。
- 使用隔离 runtime 在 CPU 加载 e71 full-state，确认 G/F/D/E 所在 network 容器、四类
  optimizer/scheduler、AM-TNC 方法状态、Python/NumPy/CPU/CUDA RNG 和两个 sampler 均在。
- 新建完全隔离的恢复探针目录，从 e71 checkpoint 真实执行一项 update：
  `607263 -> 607264`，正常以 `ENGINEERING_PAUSE` 退出，stderr 为 0；主 run 目录和 GPU
  训练进程均未修改。
- e71 checkpoint、sidecar、heartbeat 和 lane authorization 已复制到
  `E:/UNSB_Expl/recovery_backups/4090A_AMTNC_20260908_E71`，逐文件哈希与远端只读源一致，
  本机文件已设为只读。此前的精确 runtime 归档继续保留在异机。

## 对旧路径和后继的准确解释

旧 prefix 不是完整环境；其中三个 326 字节 `python*` 文件只是 fail-closed relay，SHA256
为 `a479dc46...e731cdb`，会直接 `exec` 已验签隔离 runtime。它们不是重建环境，也不会被
当作恢复证据本身。训练 guard 的冻结重启命令直接使用隔离 Python，在启动前重新验签
Python、manifest、111 项依赖、代码、协议和最新 full-state；不依赖旧 prefix。

仍持有 deleted inode 的 AM-TNC terminal exporter、incremental exporter、算法评估、统一
评估和最终交付 child 均保留运行，但各自已经被从隔离评估 runtime 启动的 supervisor 收养；
未来重启命令的第一个 argv 已逐项核对为隔离 runtime。全部恢复计数为 0、health 为健康。
两个旧 progress watcher 只负责观测，不拥有科学状态或交付链；它们退出不会影响训练恢复、
source-bound export 或统一评估。

## 剩余边界

不能诚实声称隔离 runtime 在 GPU 上的“下一 update”与仍存活的 deleted-inode 进程逐位相同，
因为证明该反事实需要中断健康训练。可以严格声称的是：当前 e71 状态可加载、可执行、可
从冻结命令恢复，所有活动依赖已逐字节捕获并只读验签，且关键后继不再依赖被删环境。

机器可读回执：
`evidence/paper_aio/PAPER_AIO_AMTNC_E71_POST_CLEANUP_EXECUTABLE_RECOVERY_CLOSURE_20260908T070849.json`。
