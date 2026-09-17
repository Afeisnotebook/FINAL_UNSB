# 历史文档与当前权威的分界

本项目保留大量阶段计划，是为了审计研究如何走到最终结果，不代表这些计划仍在执行。

## 为什么不直接删除旧文件

部分路线一合同、算法卡和 pre-result 文档参与过协议指纹或哈希绑定；删除或重写会破坏
历史复现关系。其文件名中的`ACTIVE`、内容中的`running`或旧 PID 只能按当时语境解释。

## 历史文件分类

- `ACTIVE_LOCAL_ROUTE1_PLAN_CN.md`、`LOCAL_ROUTE1_RESEARCH_CONTRACT_CN.md`、
  `configs/LOCAL_ROUTE1_PROBES.json`、`HYPOTHESIS_LEDGER.json`：small25 路线一历史与协议输入。
- `PAPER_AIO_RESEARCH_CONTRACT_CN.md`、`configs/PAPER_AIO_UNPAIRED_V1.json`：full-data
  冻结协议；`ACTIVE`是当时的 schema token，不是当前任务状态。
- `configs/FULL_DATA_METHOD_PORTFOLIO.json`、`configs/PAPER_DELIVERY_COMPLETION_MATRIX.json`、
  `PROJECT_STATE.json`的大部分嵌套字段：append-only 执行账本。顶层 final overlay 和最终
  archive 才是当前结论。
- `evidence/`、`decisions/`：按时间保存的事实与决定，不应被“更新成当前措辞”。
- `ACTIVE_PAPER_AIO_PLAN_CN.md`：已经关闭的执行计划墓碑。

## 如何查看重构前原文

重构前最后完整快照为 commit：

`17cd2edeb5263d7e3b46cf1f9ed24d9ab1854488`

例如：

```bash
git show 17cd2ed:START_HERE_CN.md
git show 17cd2ed:CONTEXT_CAPSULE_CN.md
git show 17cd2ed:ACTIVE_PAPER_AIO_PLAN_CN.md
```

## 冲突处理

若旧文件与`FINAL_HANDOFF.json`或最终 portfolio 冲突，一律以
`configs/DOCUMENT_AUTHORITY_REGISTRY.json`规定的顺序解决。不得用较长、较详细或带 PID 的
旧记录覆盖较短的最终裁决。
