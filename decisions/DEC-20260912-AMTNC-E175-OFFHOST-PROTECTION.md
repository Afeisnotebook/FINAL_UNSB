# DEC-20260912：为 AM-TNC e175 增加异机保护触发器

## 原因

4090A AM-TNC 已连续到 e168，但原源码根和完整环境已经被误删，活动进程仍映射 deleted inode。
现有增量异机链覆盖 e100/e150/e200；e175 会作为完整 milestone 保存在 4090A，却要等 e200
终端导出时才进入普通交付。因此 e175 到 e200 之间仍存在一段只依赖该宿主磁盘的窗口。

## 动作

在现有两小时 `final-unsb-goal` heartbeat 中加入一次性触发条件：e175 checkpoint 与 sidecar
完整生成后，不暂停训练，只读复制 e175、当时 heartbeat 和 lane authorization 到
`E:/UNSB_Expl/recovery_backups/4090A_AMTNC_E175`；逐文件核对源/目标 SHA256 后将本地副本设为
只读并提交 compact evidence。

不复制仍可能写入的 `full_state_latest.pt`，不从旧 `/home/yc/unsb_cov` 启动进程，不改动训练、
算法、数据或评估协议。本次只建立持久动作，实际 e175 文件尚未到达，因此当前状态是
`ARMED_WAITING_FOR_IMMUTABLE_E175_MILESTONE`。
