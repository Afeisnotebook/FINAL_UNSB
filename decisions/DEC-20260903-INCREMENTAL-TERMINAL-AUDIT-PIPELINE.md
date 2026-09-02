# 本地终端审计的增量source-bound通道

日期：2026-09-03

## 问题

本地GTX 1660评估节点和终端谱审计继任器已经通过完整数据门，但既有
`paper_aio_export_successor.py`只在训练达到e200后一次性发布
e100/e125/e150/e175/e200全套checkpoint。于是预注册的e100/e150/e200
target-blind审计无法与剩余长训并行，形成没有科学必要的末端串行尾巴。

## 决策

增加一条独立的增量通道：

1. source exporter只观察冻结的e100/e150/e200 milestone文件；每个checkpoint必须经
   full-state、scientific-state、commit、protocol、manifest和confirmation边界验证后才
   发布source receipt；
2. 本地relay只接受上述三个固定epoch，锁定SSH host key和训练身份，逐文件SHA256验证后
   发布允许为partial的import receipt；
3. partial import必须同时绑定source export receipt、checkpoint、sidecar、lane receipt和
   relay-set receipt。lane/set之间的原子发布窗口只被解释为`not ready`，不会跳过验签；
4. terminal-audit successor优先接受完整既有import，完整import尚未到达时才读取通过上述
   验证的增量import；固定审计集合、GPU锁和父状态/RNG不变门不变；
5. 既有e200 exporter/relay和统一论文评估链全部保留。增量通道只提前target-blind诊断，
   不提前计算matched delta，也不改变主表必须使用e200的协议。

该通道不读取PSNR/SSIM/LPIPS，不控制训练、不选择checkpoint、不打开confirmation20。
它是墙钟时间优化，不是新的算法假设或科学裁决。
