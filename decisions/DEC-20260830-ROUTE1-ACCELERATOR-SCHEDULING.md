# DEC-20260830：路线一加速卡调度与 batch 语义

## 裁决

4090 和 5090 在当前 small25 路线一阶段采用“最多两条独立科学训练流/卡”的调度；正式 matched 训练、探针、反事实分支和候选裁决继续保持 `batch_size=1`。

这不是算力保守策略。实测表明双流提高了每小时完成的总 data epochs，但第三流当前既没有通过证据门的合法任务，也很可能只会继续摊薄单流吞吐。空闲流只能承接当前研究阶段已经允许的任务：长期锚点、冻结 checkpoint 上的 target-blind 反事实审计，或因果矩阵完成后生成并冻结的新候选。

## 实测依据

测量使用同一 host、同一 lane 在并发 HNEK 启动前后的 5-data-epoch chunk 记录，避免用瞬时 GPU utilization 推断吞吐。

| Host | 单流基准 | 双流期间同 lane | 总吞吐变化 |
|---|---:|---:|---:|
| RTX 4090 | HJ，114.78 data-epochs/hour | HJ + HNEK，121.58 aggregate data-epochs/hour | +5.9% |
| RTX 5090 | plain，80.76 data-epochs/hour | plain + HNEK，104.04 aggregate data-epochs/hour | +28.8% |

两张卡双流时总显存仅约 3.3–3.5 GB，但 GPU utilization 已约 90%。显存空闲不等于存在等比例的计算吞吐；UNSB batch1 主要受大量小 kernel、同步和模型更新路径影响。

## 为什么不增大正式 batch

`batch_size` 不是纯工程参数。增大它会同时改变：

- 每次 optimizer update 的梯度估计与方差；
- Adam moments 的时间尺度；
- A/B 无配对 sampler 的消费顺序；
- HJ、DT、HNEK 所观察和修正的对象；
- 30000 updates 与 200 data epochs 的既定对应关系；
- 与历史正窗口及 clean batch1 基座的 matched 可比性。

因此，大 batch 可以在未来另立“吞吐代理协议”做工程筛查，但不能替代本 Goal 的 batch1 e200 科学裁决，也不能与 batch1 plain 直接计算收益。

## 动态调度规则

1. 4090 的 plain/HJ/HNEK 已完成且 proxy 已 `CALIBRATED`。当前 DT 是解锁完整因果矩阵的关键路径，保持单流直到 e200；4090 实测双流总吞吐仅增加 5.9%，此时插入第二流会延后算法发现门。
2. 5090 当前保持 canonical plain 与独立 HNEK 两条 batch-1 流。plain 完成后 canonical runner 接续 HJ；独立 HNEK 完成并经同机 plain e200 验证后才导入 canonical run root。第三流仍禁止。
3. 本地 1660 保持一条 HJ 流和独立只读 verifier/auditor watcher，不与远端 checkpoint 混用。
4. DT e200 验证完成后，4090 立即把训练槽转为因果审计；可并行的审计单元必须读取独立 parent checkpoint、写入独立 row，并通过父状态 hash 隔离。是否开双流由实际审计吞吐决定，不以显存空闲推断。
5. 因果矩阵冻结前，不用空闲 GPU 预跑候选名字、公式模板或超参网格。400--800 updates 工程 micro gate 即使使用不同吞吐设置，也不得产生科学晋级结论。
6. 候选生成后，最多两个候选可在 4090/5090 上并行，但每个候选只与同 host、同 e0、同 seed、同 protocol、同 `batch_size=1` 的 plain 比较。
7. 不跨 host 延续 checkpoint，不把不同 host 的绝对 PSNR 直接相减；跨 host 结果仅作为重复性证据。
8. confirmation20 继续封存，paired 指标不进入调度或控制器。

## 防漂移结论

路线一的瓶颈不是显存，而是“哪些任务已经由证据授权”。算力并行用于缩短证据链，不用于越过 proxy、因果审计和数学推导门。该裁决不改变训练 commit `0da2a37086cca5bc4ad4488bb07c53096a7152ed`、protocol fingerprint `b0786b222790b84379802996448b8a68b86d69a6892ea0cdc04670cfcb1fb9b2` 或任何正在运行的 full state。
