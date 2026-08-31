# DEC-20260831：完整多候选前沿，而非单算法科学剪枝

## 决策

路线一最终仍写出一个`CANDIDATE.json`，但它只表示下一阶段的默认行动优先级，不表示
其余算法被科学淘汰。完整e200链结束后必须额外写出`RESEARCH_FRONTIER.json`：保存
4090同宿主全部可排名轨迹、5090宿主分离的full/proposal-only/observable-only证据、
每条算法的前沿处置以及完整来源身份。

只要一个分支具有独立的长期机制证据，并且属于严格通过、因果上可修复的近边界或
证据型递补，它就继续留在研究前沿。单seed下几个十分之一dB以内的名次差只决定下一步
算力顺序，不能升级成父机制证伪。只有当前实现完成e200且既不满足正向证据、又没有
target-blind可修复缺陷时，才标记为`closed_current_implementation_on_current_protocol`；
这个词仍不等于`mechanism_falsified`。

## 当前算力使用

- 5090继续并行完成RF-AMMCRB与RF-MCRB的共同e0、batch1、seed2026、真实e200；随后为
  所有符合证据门的父算法完成各自proposal-only和observable-only流。
- 4090最多复跑两个来源绑定的修复算法，并分别保留Adam几何与欧氏几何的条件合成。
- 这不是超参网格：两条修复线对应不同约束几何，两个合成分支只有在各自严格父项和
  target-blind兼容门通过时才运行。
- seed2027/2028继续延期。额外算力先用于扩大独立算法/消融证据，而不是重复seed。

## 最终接口

- `CANDIDATE.json`：默认下一步行动主项；
- `ALTERNATES.json`：兼容既有合同的前两项递补；
- `RESEARCH_FRONTIER.json`：全部值得保留、修订或扩尺度验证的算法前沿；
- `RESULTS.json`：4090与5090完整宿主分离证据，不合并跨宿主delta。

最终交付守护器最初实现在`0f267ae2cf7cc99de9c7ae0decf888289e03da65`，只在4090完整
同宿主裁决和5090完整可携带机理证据同时到达后原子发布。中间paired指标仍不得参与
训练、路由、退出或checkpoint选择，confirmation20继续封存。

## 部署前语义修正

终点输入到达前的链路审计发现，研究前沿展示曾用错误字面量
`strict_sustained_local`识别共同严格分支，而权威分类常量是`strict_sustained`。这不会
改变训练、receipt或科学排名，但会把严格第二名错误显示成普通机制递补。`b717058`已改为
直接导入权威常量，同时补上终点输入hash复验和“唯一算法数/宿主轨迹数”的区分；旧等待器
在两个终点输入均未出现时替换。完整测试随`6fa361b`增至317项。修正证据见
`evidence/remote_route1_offload/COMPLETE_FINAL_STRICT_DISPOSITION_FIX_20260831.json`。

## 完整证据范围

后续预检确认，仅保存候选排名仍不够：终交付还必须把DT/HJ/HNEK的完整e200锚点、
474条反转与140条采样方差图谱、proxy校准和假设谱系带入主结果；不能让这些只存在于
旧归档。同时，4090每条候选以及5090每个full/proposal/observable角色都必须携带来源
receipt、推导卡、实现和逐域candidate/plain绝对/相对轨迹。`874a09b`已实现这组要求，
并在4090真实运行目录通过4条现有候选、3个历史探针和474/140图谱的静态预检。

因此最终的“多候选”是两层前沿：一层是可执行算法轨迹，另一层是产生这些算法的历史
探针、失败机理和数学谱系。5090导出、本地原字节relay和4090终交付器均已在终点生成前
替换为完整证据版本，证据见
`evidence/remote_route1_offload/COMPLETE_FRONTIER_FULL_EVIDENCE_CHAIN_20260831.json`。

为避免远端完成后再次依赖人工或Codex会话在线，`11be901`又增加了本地持久结果接收器。
它只在4090原子终点pointer出现后下载最终五个交付文件及两份宿主前沿，逐文件比较远端、
pointer和本地SHA256；目标已有不同内容时拒绝覆盖，不传checkpoint、不持久化密码。这样
终点结果会自动落回本地运行目录，随后再由主控审阅并提交compact裁决。部署见
`evidence/remote_route1_offload/COMPLETE_FINAL_RESULT_RELAY_ARMED_20260831.json`。

## 终点报告边界补全

后续Goal完成审计发现：`874a09b`已经把结论边界写入JSON，但Markdown报告尚未显式分开
科学结论、工程失败、proxy失真和未测试假设。`280d685`只补齐这四类报告文字和对应
fail-closed终点审计，不改变训练、checkpoint、候选公式或排名。由于两份终点前沿输入和
最终pointer在替换时都尚未产生，4090等待器已安全替换为
`tmux:FINAL_UNSB_COMPLETE_FINAL_280d685`；远端7项测试通过，stderr为0。证据见
`evidence/remote_route1_offload/COMPLETE_FINAL_REPORT_BOUNDARY_SUCCESSOR_REDEPLOYED_20260831.json`。
