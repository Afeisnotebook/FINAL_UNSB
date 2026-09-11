# DEC-20260912：ST-CGR e150 固定里程碑闭环

ST-CGR 在5090A按冻结协议完成 e150 / 1,282,950 updates，并继续向 e200 训练。e150
full-state、sidecar、scientific state 与 source-bound export receipt 均在来源侧重新计算
SHA256；本地既有 relay 自动发布 e150 导入，checkpoint、sidecar 和 receipt 的实际哈希与
来源完全一致。CPU只读加载确认 G/F/D/E、方法状态、optimizers、schedulers、两套sampler与
全部RNG状态均在完整checkpoint中。

e150固定评估artifact只做字节级异机备份与哈希核验，未解析指标值。训练、算法、采样、
runtime cohort和调度均未改变；该固定节点不是best-checkpoint选择。ST-CGR的matched delta
仍必须等待已准入的5090B matched plain完成e200和统一评估，不能从当前绝对轨迹推断。

证据：`evidence/paper_aio/PAPER_AIO_STCGR_E150_HASH_AND_OFFHOST_CLOSURE_20260912T042600.json`。
