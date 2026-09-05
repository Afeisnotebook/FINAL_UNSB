# 新提供的44804端点再次确认为既有5090B

日期：2026-09-05

用户将`connect.weste.seetacloud.com:44804`作为新增RTX 5090再次提供。实时只读物理身份门证明该端点仍是已登记的5090B，不是新增物理算力：

- GPU UUID仍为`GPU-578d4047-4c22-8c6a-d216-0f7938e99194`；
- hostname仍为`autodl-container-1qjylxm03k-35c3df8b`；
- machine-id仍为`5d04e104cf744bddb12c08eb61a501e9`；
- 端点内存在同一CUT trainer PID 5771、CycleGAN trainer PID 11846及相同run目录；
- CUT已连续推进至e176，CycleGAN已连续推进至e114；两条continuity guard均为0次重启；
- 两个health watcher均为`HEALTHY`、0告警，GPU快照利用率98%，不存在一张空闲新卡。

因此不克隆第二套环境、不登记5090D、不启动第三条lane，也不改变现有健康训练。这样不是缩小北极星，而是防止把同一物理GPU错误计算两次并破坏论文关键路径。5090B继续执行CUT和CycleGAN；CUT e200后，已有PID 148465的fresh-e0 matched-plain successor必须先通过既定容量门再启动。

现有两小时Goal heartbeat继续监督全部真实宿主。若需要新增物理容量，判定标准是得到一个与注册表不同的GPU UUID；端口或密码变化本身不构成新宿主。

本裁决没有读取paired性能，没有访问confirmation20，没有改变任何训练协议，也没有把容量重复写成算法失败或任务完成。
