# 本地路线一工程门报告

日期：2026-08-29  
验证代码：`a4883ebc5c060e5726acf4ef0a7d701cc20fa088`  
研究协议指纹：`b0786b222790b84379802996448b8a68b86d69a6892ea0cdc04670cfcb1fb9b2`

## 裁决

`PASS`。当前runner可以开始small25、batch1、真实200 data epochs的plain长期锚点。
这只是工程门通过，不是任何算法获得正收益，也不是proxy已经校准。

## 精确通过项

- manifest SHA256为`1a66cf...42b7b`，六域各100 train / 80 discovery /
  20 confirmation；训练只取冻结order前25张，confirmation未打开。
- DT权威核心的语义源码哈希与历史重构一致；仓库迁移只改变import/诊断依赖。
- HJ按物理e5激活并持续到e200；不再使用“总步数20%”起点。
- DT按物理e21映射active age 1；e21为`0.0002`、e25至e35为`0.001`、
  e44仍非零、e45自然归零并保持到e200。
- plain双跑一update的科学状态哈希相同：
  `f1f383976bfb51b0ff23389995ab4c67dc28a7008c9a93247e1fee5c27bca479`。
- inactive HJ、inactive DT、disabled HNEK的一update状态均与上述plain哈希相同。
- 连续2 epoch与1 epoch保存后恢复再训练1 epoch的完整科学状态哈希相同：
  `8f4a334dead6de398045005a6dec90df7ea41fe6e1a2772a58bd90b4e7b98db5`。
- discovery70共420图，lane-blind CRN评估连续执行两次逐图结果完全相同；评估
  前后模型、optimizer、scheduler、sampler和全部RNG状态哈希不变。
- observable schema会拒绝paired/PSNR/SSIM/LPIPS/discovery/confirmation字段；
  counterfactual branch在深拷贝上执行并证明父状态不变。
- 代码合同检查全通过；pytest为26/26；compileall与`git diff --check`通过。

## Git与外部运行物边界

Git只保存本报告、CPU/GPU门禁JSON和lineage JSON。共享e0、resume门checkpoint、
option dump及完整终端日志位于
`E:\UNSB_Expl\runs\FINAL_UNSB_LOCAL_ROUTE1_E200`，没有进入Git。

## 下一步边界

只允许启动plain e1-e200。plain完成后才允许HJ，HJ完成后才允许HNEK。只有HJ或
HNEK满足预注册的e150/e175/e200 proxy校准规则，DT和后续新算法长训才会解锁。
中间paired PSNR不能触发早停或改配置。
