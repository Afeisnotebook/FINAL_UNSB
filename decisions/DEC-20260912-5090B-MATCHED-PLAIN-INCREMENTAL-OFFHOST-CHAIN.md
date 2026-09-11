# DEC-20260912：5090B matched plain 增量异机保护链

## 裁决

5090B matched plain 是 Proposal 与 ST-CGR 合法 delta 的共同关键控制。原有 terminal exporter
只会在 e200 后发布 e100/e125/e150/e175/e200 五个固定checkpoint；因此在e200之前，已完成的
关键里程碑仍只保存在租用宿主上。这个串行风险不改变科学结论，却会放大宿主或租期故障的损失。

在不触碰 trainer、supervisor、checkpoint 或既有 terminal delivery 的前提下，新增一条只读、
source-bound、metric-blind 的 e100/e150/e200 增量链。它不产生实验、不选择checkpoint，只在
预注册里程碑存在后验签并复制到本地评估节点。

## 实际部署

- 5090B source exporter PID `107734`，恢复supervisor PID `107809`，health PID `107830`；
  三者已脱离SSH会话、零重启、零告警。
- 本地relay PID `11704`，恢复supervisor PID `25228`，health PID `20796`；当前健康等待e100。
- source与destination控制均固定在commit
  `21b224035d7ce296050d0d06ad4b80c0ad873bd9`，checkout干净；远端22项、当地30项相关测试通过。
- 训练身份固定为commit `e4a5eed9fe14e671e07329a970d93cd9828240ac`、protocol
  `e5704e445a51dd9c5c12369c94df01cf9532364a71c806b9914ef3963994b07b`、manifest
  `02c01df...e36744`。
- 现有terminal exporter PID `534982`与本地terminal relay PID `19496`保持不变；e125/e175仍由
  terminal链在e200后交付。

首次本地relay命令多出一个`-u`，恢复监督器按设计以
`BLOCKED_LIVE_PID_COMMAND_MISMATCH`拒绝收养。该空等待relay在确认精确PID与命令后被替换，
没有复制checkpoint；第二次命令逐项匹配冻结合同后才进入持久监督。这是fail-closed门生效，
不是科学任务或训练失败。

## 结果边界

5090B训练PID `76358`未变化，已自然推进到e76；不读取paired性能、不改调度、不打开
confirmation20。按最新单epoch实测，e100大约在9月12日21:45到达；届时增量链将自动完成
source receipt、checkpoint传输、SHA256复验和publish-last import，使关键matched control在
e200之前就具备异机恢复点。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_5090B_MATCHED_PLAIN_INCREMENTAL_OFFHOST_CHAIN_20260912T051022.json`。
