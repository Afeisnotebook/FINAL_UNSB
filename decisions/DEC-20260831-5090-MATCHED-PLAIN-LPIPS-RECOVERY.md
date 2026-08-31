# DEC-20260831：5090 matched plain晚期LPIPS恢复

## 问题

5090同宿主plain的e100/e125/e150/e175指标包含完整PSNR、SSIM与420张CRN输出，但原评估
环境无法加载可选LPIPS模型，因此`lpips_available=false`。候选评估环境后来具备LPIPS，
而长期严格门要求e150/e175/e200三个LPIPS delta均可计算。若保持缺失值，任何5090候选
都会因为没有三项LPIPS权威值而机械失败；这属于工程证据缺失，不是算法退化。

## 裁决

允许从冻结plain milestone checkpoint只读恢复缺失的LPIPS字段，但必须同时满足：

- 使用训练commit `0da2a37`、原protocol fingerprint与同一discovery70 CRN；
- 每个milestone独立执行两次完整420图评估，完整payload hash一致；
- 原指标与恢复指标除LPIPS值及availability外逐字段完全一致；
- checkpoint文件/scientific-state hash与sidecar一致；
- 评估前后模型和RNG科学状态hash一致；
- 原指标在运行目录中保留，不进入Git；新指标原子替换；
- LPIPS只作事后护栏，不控制正在进行的训练、公式或调度。

e100/e125/e150/e175均已满足上述条件并通过独立milestone verifier。恢复后的plain LPIPS
依次为`0.5080493164 / 0.3747691379 / 0.3877784425 / 0.3530203853`。e200原指标已经完整，
未改动。

## 后续门禁

commit `c5c0cd6`起，新候选长训在启动前必须验证matched plain的e150/e175/e200均具有完整
LPIPS权威payload；缺失时fail closed，不能再把`None`写成算法护栏失败。正在运行的
PCNR/AM-MCRB轨迹未重启、未改写，其e200总结会从恢复后的plain原始指标重新计算delta。

compact证据见
`evidence/remote_route1_offload/REMOTE5090_PLAIN_LPIPS_RECOVERY_20260831.json`。
