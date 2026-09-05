# 论文表格复现边界元数据闭环

## 问题

预结果冻结的基线配置已正确区分 CUT 的受控复现、CycleGAN 的官方损失/共享骨干实现，以及
DCLGAN 的作者源码受控曝光复现。但原最终表格接口只稳定导出 `lane_id`、数值和通用
`comparison_scope`，没有强制保留人类可读的 `paper_label` 与具体复现边界。由此生成的表格
仍可能在手工写作阶段把当前 CycleGAN 行误写成逐字官方实现。

## 裁决与实现

最终 portfolio 现在从已冻结的 `PAPER_BASELINE_PORTFOLIO.json` 物化规范化的逐行报告
元数据。`MAIN_E200.csv` 和 Markdown 摘要必须同时携带 paper label 与
reproduction/comparison scope；任一行缺失其中之一都会 fail closed。CycleGAN、CUT 和
DCLGAN 的受控复现标签因而与最终数值处于同一哈希闭环，不能在导表时静默丢失。

这项修改不改变冻结基线集合、训练协议、模型、checkpoint 或结果选择。它只修复结果到
论文表格之间的语义传递，未读取任何性能值，也未授权 confirmation20。

实现提交：`fb8daf47636ddfd183829d5ec1832df61dfb5153`。

证据：`evidence/paper_aio/PAPER_AIO_REPORTING_METADATA_CHAIN_20260906T040100.json`。
