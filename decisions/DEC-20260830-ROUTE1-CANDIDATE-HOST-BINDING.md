# DEC-20260830：候选只能与同环境产生的 plain e200 比较

## 裁决

每个候选 executor 合同必须同时绑定：

- 当前 host 的 Python、PyTorch、CUDA、cuDNN、GPU 和平台指纹；
- 该 run root 的 accepted plain e200 milestone verification 哈希；
- plain e200 checkpoint 文件哈希和 scientific-state 哈希；
- 候选算法、代码、manifest、共同 e0 和 CRN 身份。

合同初始化时，当前运行环境必须与产生 matched plain 的冻结环境记录完全一致；每次
executor 启动时再次检查。把4090 run root复制到5090后直接训练候选、跨主机延续
checkpoint或用另一主机plain计算delta都会在执行前失败。

这项门禁不改变候选数学、batch1或e200协议。它只保证4090和5090并行候选时，每条
结果仍是合法的same-host matched比较，而不是用算力并发换取不可解释的跨环境差值。

