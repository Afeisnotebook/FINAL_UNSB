# DEC-20260912：终端病理裁决成为论文 claim freeze 的强制前置门

## 发现的问题

现有工程已经部署了完整的 target-blind terminal audit、统一评估和 posthoc
terminal-pathology successor，但此前的 claim-freeze 接口只强制绑定最终算法组合、理论包、
参考文献账本和论文声明。于是存在一个真实的交付缺口：即使长期因果裁决尚未完成，数值结果
齐备的 portfolio 在接口层面仍可能被提前冻结。

这不是训练失败，也没有改变任何算法结果，但它会让论文对“末段低方差/谱漂移是否先于性能
变化”这一预注册问题缺少不可绕过的最终答案。

## 裁决与实现

`freeze-draft` 和 `freeze-materialize` 现在都必须显式接收
`TERMINAL_PATHOLOGY_DECISION.json`。验证器同时接受严格完成的阳性和阴性结论，但会重新检查：

- 固定四个 probe、六个共同域形成的24个cell；
- e100/e150/e200 共12份 target-blind audit receipt；
- 对应的12份 posthoc unified metric receipt；
- metric binding、所有引用路径和SHA256；
- target-blind先于paired读取、不得控制训练、不得自动启动模块、不得选择最佳checkpoint、
  confirmation20仍封存。

review draft、人工/Codex committed review、materialized freeze receipt 与后续 distribution
阶段必须保持同一个 hash-bound terminal reference。任何缺失、篡改或不完整结果都会 fail
closed。

阳性裁决仍只授权撰写 derivation card；阴性裁决明确阻止添加终端修复模块。两者都不会修改
正在运行的训练，也不会把因果审计变成 paired controller。

## 验证与运行影响

冻结、分布指标、confirmation、稿件表格/分支、终端裁决和复现清单联合测试52项通过；全仓库
测试788项通过。compileall、JSON解析和`git diff --check`均通过。

本改动只约束未来的论文声明冻结，不要求重启当前 terminal-pathology successor，也没有触碰
任何健康GPU训练或队列。当前 claim freeze 仍未授权，必须等待全部固定e200结果和终端裁决。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_TERMINAL_PATHOLOGY_CLAIM_FREEZE_GATE_20260912T074454.json`。
