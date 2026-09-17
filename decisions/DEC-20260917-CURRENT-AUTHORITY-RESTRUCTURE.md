# DEC-20260917：当前权威层重构与历史执行状态隔离

## 问题

full-data discovery 已完成，但多个根目录文件、启动提示和 append-only 账本仍包含训练期间的
`running`、PID、服务器队列和“active reconstruction”措辞。即使文件顶部有覆盖说明，
无旧上下文的接手者仍可能把历史阶段状态当成当前任务。

## 决定

1. 建立`configs/DOCUMENT_AUTHORITY_REGISTRY.json`作为机器可读的文档分类与冲突顺序；
2. 当前入口只保留最终交接、当前状态、claim boundary、最终 archive 和灾难恢复；
3. hash-bound protocol与路线一材料保持原文，但明确归类为 frozen scientific input；
4. append-only 状态文件增加 final overlay，旧嵌套 PID/heartbeat不再拥有调度权；
5. 旧 Master/Server bootstrap 与 active paper plan改为不可启动训练的墓碑/重定向；
6. 更新项目合同和验证器，使 clean clone 验证当前 archived phase，而不是旧 running phase。

## 不改变的内容

- 不改最终指标、算法 disposition、runtime relation或checkpoint身份；
- 不改 pre-result hash-bound methods/theory/checklist正文；
- 不删除 evidence、decisions、checkpoint或服务器资产；
- 不打开 confirmation20，不启动实验。

重构前历史快照固定为`17cd2edeb5263d7e3b46cf1f9ed24d9ab1854488`。
