# DEC-20260910：Proposal e125 固定里程碑闭环

5090C Proposal 按冻结协议自然完成 e125 / 1,069,125 updates，并继续运行至 e127。
supervisor/trainer PID 9729/366768 保持不变，outer guard PID 431648 零重启；本次没有
停止、恢复或修改训练。

e125 checkpoint、sidecar 与指标文件均只做逐文件哈希，指标数值未读取。三项文件已复制到
`E:/UNSB_Expl/recovery_backups/5090C_PROPOSAL_20260910_E125` 并设为只读，源checkpoint
未改变。e125不是预注册的source-bound export节点；下一次增量source export仍为e150。

这一事件只证明固定里程碑和恢复资产完整，不产生Proposal相对plain的科学结论。合法matched
delta仍须等待已准入的5090B matched plain完成e200，confirmation20继续封存。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_PROPOSAL_E125_HASH_AND_OFFHOST_CLOSURE_20260910T023000.json`。
