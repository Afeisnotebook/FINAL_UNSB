# 5090A：ST-CGR终点到plain恢复的持久接力

日期：2026-09-03

## 背景

用户明确把5090A的时间优先级改为立即运行full-data ST-CGR，plain已在e9完整状态暂停。
该调度正确消除了ST-CGR前置等待，但也产生了一个新的长程断点：若ST-CGR e200后仍需
人工上线才能恢复plain，ST-CGR的matched裁决可能再次停滞。现有plain exporter的单次
等待期限也短于ST-CGR按首epoch外推的训练时间。

## 决策

新增并部署独立的`paper_aio_paused_plain_resume_successor.py`。它只读取工程状态：

1. 等待ST-CGR continuation写出`COMPLETE_CANDIDATE_E200`；中间paired值不参与决定；
2. 再次验证5090A原scientific checkout的commit、clean状态和协议指纹；
3. 逐字节复验plain e9 full-state SHA256、scientific-state SHA256、76,977 updates、
   seed2026、batch1、授权和confirmation封存；
4. 拒绝任何已存在的同output plain trainer；
5. 如旧GPU-free exporter已过期，按完全相同的冻结合同重新启动它；
6. 调用原scientific checkout的plain supervisor从e9 exact resume到e200。

该接力不跨宿主搬checkpoint、不修改训练配置、不依据ST-CGR成绩决定是否恢复plain，
也不把ST-CGR绝对轨迹写成matched收益。它只把用户已经决定的执行顺序持久化，避免
Codex会话结束后停机。
