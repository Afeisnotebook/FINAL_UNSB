# 5090B matched plain 容量门通过并进入正式 e200 长训

时间：2026-09-06 08:14:46 +08:00

## 裁决

5090B 的两 epoch、metric-blind 共驻容量门已经通过，冻结决策为
`CONTINUE_PLAIN_NOW`。从容量探针留下的 e2 完整状态精确续训到 e200；不得等待
CycleGAN 结束后另起一条 plain，也不得重复启动第二个 matched plain。

## 依据

- 容量门回执 SHA256 为
  `7f1d96be51e3731098ecf2b6f1337b4ea61011ccfa06f97c2bb95bd145c83136`。
- 继续共驻的预计总剩余时间为 `552547.20 s`，等待 CycleGAN 释放后再训练为
  `580501.43 s`，净缩短 `27954.23 s`（约 `7.77 h`），超过预注册的一小时门槛。
- e2 状态闭环为：17106 updates，checkpoint SHA256
  `e73d3b788ed743b8bc34fa3aacf55d7a16c5519925ed4849f95b03f085a396d9`，
  scientific state SHA256
  `5ab8f3c9e4e65553d754e828762c0622f7098bb0f20007499d1a81de28f0a7ba`。
- 正式 successor、supervisor、trainer、exporter 分别为 PID
  `523993`、`534983`、`534984`、`534982`，均在只读复核时存活；CycleGAN 仍在原
  supervisor/trainer 下健康运行至已完成 e135，没有被重启或修改。

## 公式版本边界

在飞容量门继续使用已冻结 control commit `f8c38c1`。主分支后续提交的通用
companion-tail 修正 `a4f4b46` 不热修改该运行。此次真实分支是“CycleGAN 先完成”，
旧在飞公式已经计入 plain 的独占尾部，因此回执无需重算，也不失效。

## 科学边界

本裁决只使用吞吐、进程、完整状态和哈希，不读取性能值，不使用 paired 指标，不选择
最佳 checkpoint，不打开 confirmation20。runtime relation 已获准，但 matched delta 只有
在 plain e200、source-bound export 和统一评估全部完成后才可产生。

证据：
`evidence/paper_aio/PAPER_AIO_5090B_MATCHED_PLAIN_CAPACITY_PASS_AND_LONG_START_20260906T081446.json`
