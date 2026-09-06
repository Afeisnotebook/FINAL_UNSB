# DCLGAN全量e20固定里程碑验收

## 决定

接受本地GTX1660 DCLGAN的e20完整checkpoint，继续原冻结协议训练至e200，不重启、
不重排队列，也不启动与terminal JVP的同卡共驻。下一固定里程碑为e40。

## 依据

- e20严格对应`171060`次更新；milestone checkpoint实算SHA256为
  `d0ce9339f44de9164d34d7d7220157b670e26af627e0b3fa35fa649f146a0f10`，与sidecar一致。
- milestone与`full_state_latest`的物理文件哈希因序列化产物不同而不同，但二者的
  scientific-state SHA256同为
  `76b9717da9a3a9e868dc417c1fe5b64644c66abcc5fb39465c397f8b819e45f8`。
- trainer PID 22020在5秒探针内增加14.703 CPU秒；GPU利用率87%，显存2768 MiB。
- 训练、e200 exporter和本地到4090A的source-bound push恢复链均存活、零健康告警；
  当前磁盘余量115.636 GiB，高于已闭环的21.162 GiB组合剩余写入上界。
- e20不要求生成早期指标产物；本次没有读取paired性能值。

完整回执见
`evidence/paper_aio/PAPER_AIO_DCLGAN_E20_HASH_CLOSURE_20260906T124522.json`。

## 科学边界

e20只是预注册轨迹里程碑，不用于早停、算法选择、调度或论文主表。主表仍固定使用
e200，confirmation20继续封存，DCLGAN与其他方法的统一评估仍等待source-bound e200
导入及合法评估cohort。
