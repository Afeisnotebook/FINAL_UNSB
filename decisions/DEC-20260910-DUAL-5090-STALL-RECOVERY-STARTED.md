# DEC-20260910：Proposal 与 ST-CGR 长epoch停滞的哈希恢复

## 裁决

5090C Proposal 在 e135 后超过冻结的 14,400 秒无完整epoch进展，5090A ST-CGR 在 e125 后超过冻结的 7,200 秒无完整epoch进展；两者均连续观测到GPU利用率为零。诊断过程没有读取性能指标。两个lane都先通过源码commit、协议fingerprint、唯一进程和最新full-state文件哈希门，再只向已精确匹配的停滞trainer发送SIGTERM。

Proposal的原supervisor PID 9729按既有冻结命令从e135自动恢复，产生新trainer PID 766090；outer guard保持零重启。ST-CGR原supervisor因既有累计失败预算正常退出，outer guard在重验e125 full-state后启动新supervisor/trainer PID 357236/357237，guard重启计数为1。两条新trainer均已恢复GPU计算。ST-CGR原progress watcher绑定已退出的旧supervisor，因此已用相同冻结阈值启动replacement PID 358447，并在确认健康后退役旧watcher PID 436478。

## 边界与后继

- Proposal e135状态已先复制到本机只读备份；ST-CGR e125固定状态已在前一里程碑完成异机保护。
- 已完成epoch和全部full-state科学状态没有丢失，损失仅限各自未完成的停滞epoch。
- 当前只裁决为“恢复启动且GPU活动”；必须等Proposal e136与ST-CGR e126生成新full-state并核验哈希后，才关闭exact-resume连续性。
- 未修改batch、AMP、TF32、算法、数据顺序或runtime cohort；未改变matched control关系，confirmation20继续封存。

机器可读证据：`evidence/paper_aio/PAPER_AIO_DUAL_5090_STALL_HASH_VERIFIED_RECOVERY_STARTED_20260910T170500.json`。
