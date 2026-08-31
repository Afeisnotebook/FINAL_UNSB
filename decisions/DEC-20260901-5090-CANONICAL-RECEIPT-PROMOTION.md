# 5090跨运行时回执的标准名称晋升

状态：`ACCEPTED_BEFORE_SOURCE_E200`

## 问题

5090跨运行时执行器在完整portfolio结束后生成
`<candidate>_5090.json`。HJCGR资源接力和related host裁决按统一接口等待
`<candidate>.json`。若不建立显式边界，Proposal-only与AM-TNC可以正常完成e200，
但HJCGR和终局裁决会永久等待错误文件名。

## 决策

增加一个持久、fail-closed的回执晋升后继。它只在完整
`CROSS_RUNTIME_PORTFOLIO_5090_RESULT.json`形成后执行，并要求：

- portfolio包含固定的Proposal-only与AM-TNC两个算法身份；
- 每份`_5090`回执通过原terminal receipt验证，路径位于同一run root且hash与portfolio一致；
- 标准回执不存在，或已经与源回执逐字节相同；
- 使用原子替换生成标准名称，并再次验证源/目标SHA256相同；
- 不传checkpoint，不改trajectory、指标、公式、排名或训练状态。

HJCGR等待标准Proposal-only回执仍只是GPU资源槽安排；目的宿主paired结果没有重新冻结、
修改或否决HJCGR公式的权力。

## 科学边界

这是命名/编排修复，不是新实验、重跑、结果复制或跨宿主delta合并。修复发生在两条5090
源轨迹完成e200之前，未丢失训练、checkpoint或科学证据。
