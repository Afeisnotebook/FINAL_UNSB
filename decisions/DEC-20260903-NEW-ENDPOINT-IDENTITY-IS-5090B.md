# 新提供SSH端点的物理宿主同一性裁决

日期：2026-09-03

用户将`connect.weste.seetacloud.com:44804`作为新的RTX 5090提供。连接后的只读审计证明，
该端点不是新增物理算力，而是已登记5090B的另一个访问入口：

- hostname均为`autodl-container-1qjylxm03k-35c3df8b`；
- `/etc/machine-id`为`5d04e104cf744bddb12c08eb61a501e9`；
- 训练仓仍是`31f2fb8badaf8293a2ed2744963035575df7d7a6`，manifest SHA256仍是
  `02c01df580b882763fb0ff28dbdeac4b3729deb8bb772005f26f3e7bc2e36744`；
- 同一run目录、CUT PID 5771、CycleGAN PID 11846、两个exporter和matched-plain
  successor PID 61721同时存在；
- 当前CUT e54、CycleGAN e31，与5090B的既有连续轨迹一致；GPU上也正是这两个进程。

因此不得把该端点登记为5090D、不得重复物化数据/runtime，也不得把同一GPU上的第三个
进程误写成新增并行实验。本次不停止、迁移或重启任何训练，5090B继续CUT+CycleGAN，
CUT e200后的fresh-e0 exact-runtime matched plain后继保持不变。该卡230.97 GiB可用空间、
两个健康监控均为`HEALTHY`且0告警。

新的访问凭据不写入Git。若用户确实租到了另一张物理5090，需要提供能得到不同hostname/
machine-id且没有上述PID/run inode的新端点；在此之前北极星任务继续由当前四张远端GPU
路径推进，不会为了表面占用率在5090B上启动无证据第三lane。

本裁决不读取paired性能、不改变算法、batch或epoch定义，不打开confirmation20，也不把
端点重复误写成工程失败或机制证伪。
