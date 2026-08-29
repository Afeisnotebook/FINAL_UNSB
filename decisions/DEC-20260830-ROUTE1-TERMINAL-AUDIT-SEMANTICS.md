# DEC-20260830：e200 终点反事实的学习率边界

## 问题

路线一协议为 200 个 constant data epochs、`n_epochs_decay=0`。锚点 runner 在每个完整 epoch 后先执行 `scheduler.step()`，再保存 full state。因此 e200 checkpoint 中所有 optimizer 的 LR 都已经从 `1e-4` 变为 `0.0`。

原审计实现从 e200 状态继续执行 1/8/32/200 个 optimizer updates。由于 LR 为零，它会把所有参数位移记录成零，并可能错误地把“训练日程已结束”解释成方法具有 identity/self-null 性质。若继续跨过下一个 epoch 边界，原 LambdaLR 还会进入训练协议之外的负倍率区域。两者都不是有效的机制证据。

## 直接证据

从已验收的本地 plain full state 直接读取：

| Checkpoint | Step | 四个 optimizer LR | scheduler `last_epoch` |
|---|---:|---|---:|
| e175 | 26250 | `[1e-4, 1e-4, 1e-4, 1e-4]` | 175 |
| e200 | 30000 | `[0, 0, 0, 0]` | 200 |

checkpoint 和 scientific-state 身份记录在 `evidence/local_route1_progress/TERMINAL_AUDIT_LR_BOUNDARY.json`。

## 裁决

- e20/e100/e150/e175 使用注册训练 continuation，运行 1/8/32/200-step 分支；200-step paired discovery70 标签只在两个分支都冻结后离线加入。
- e175 新增为固定审计点，保证有一个晚期、仍位于原 200-epoch 训练域内的 200-update 未来标签。
- e200 只运行 1/8/32-step `terminal_base_lr_vector_field`：在每个 disposable branch 中保留完整网络、Adam moments、scheduler、sampler 和 RNG，仅把各 optimizer LR 恢复为其 scheduler 的冻结 `base_lrs`。
- e200 分支不跨 epoch/scheduler 边界，不运行 200-step 外推，不产生 future-PSNR 标签，也不参与 paired 阈值拟合。
- e200 的作用是回答“终点状态上的原生/方法局部算子是否仍有方向和幅值差异”，不是虚构第 201 个训练 epoch。

## 不变项

该修正不改变任何已经运行或正在运行的锚点、训练 commit、protocol fingerprint、checkpoint 或指标；只改变尚未开始的审计语义。paired target 仍不可进入 `StateObservation`、proposal 或 controller，confirmation20 仍封存。
