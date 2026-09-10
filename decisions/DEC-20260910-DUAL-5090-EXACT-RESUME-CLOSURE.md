# DEC-20260910：Proposal e136 与 ST-CGR e126 恢复闭环

## 裁决

Proposal和ST-CGR在上一份哈希恢复启动记录之后，已分别从e135与e125的完整状态自然产出
e136和e126。新checkpoint、sidecar与scientific-state哈希一致，原协议、seed、sampler/RNG
恢复边界和runtime cohort均未改变。因此两次工程恢复现可从“已启动”升级为
`EXACT_RESUME_CLOSURE`，两条lane继续运行至e200。

Proposal保持原supervisor PID 9729并由trainer PID 766090继续；ST-CGR由outer guard的第一次
受控恢复生成supervisor/trainer PID 357236/357237，replacement progress watcher PID 358447
健康。两条恢复均只丢弃并重放停滞时尚未完成的epoch，不损失任何已完成epoch。

## 同步只读核查

4090A AM-TNC已自然推进至e131，原deleted-inode进程与隔离恢复guard均健康且guard零重启；
本地DCLGAN已推进至e112。两者均未被修改。5090B本次没有可用认证入口，因此保留已提交的
matched-plain权威状态，不虚构刷新。

全程未读取中间性能值，未打开confirmation20，也未改变matched-control关系。机器可读证据为
`evidence/paper_aio/PAPER_AIO_DUAL_5090_EXACT_RESUME_CLOSURE_20260910T183300.json`。
