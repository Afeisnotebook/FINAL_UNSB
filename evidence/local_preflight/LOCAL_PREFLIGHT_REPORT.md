# 本地工程验收结论

结论：本仓库已经可以交给四台4090做服务器门禁；尚未授权长训，也没有从本地
smoke宣称任何算法收益。

## 已通过

- 真实六域共9153个identity；train/discovery/confirmation为8553/480/120。所有
  input/target均做了内容SHA256，canonical manifest为
  `02c01df580b882763fb0ff28dbdeac4b3729deb8bb772005f26f3e7bc2e36744`。
- 18,306个A/B视图文件全是硬链接，没有复制或改写像素。
- 完整8553训练视图上，plain/HJ/HNEK/macro的e0网络hash均为
  `2318fc4556af6207e552945ea0980d9332d823382f79b37fdc8a6fc4246fd5f7`。
- plain连续跑e2与e1保存后重启到e2，在网络、优化器、scheduler、全部RNG和方法
  状态上逐项完全相同。
- HJ正式窗口按真实数据长度解析为`[13685,68424)`；12图smoke跨过窗口起点并记录
  到5个真实HJ-active optimizer step。
- 四种checkpoint均能走通统一CRN评估；plain重复评估JSON字节一致；未授权
  confirmation在打开图像前被拒绝。
- 19项pytest、compileall、契约检查和授权合并dry-run全部通过。

## 本地发现并修复的关键问题

HJ未激活时原先直接在CUDA抽PatchNCE latent，plain则在CPU抽完再搬到GPU。二者
分布相同但RNG流不同，导致HJ在声明的1.6 epoch之前已经与plain分叉。现在HJ严格
复用plain的CPU抽样语义；修复后e0及介入前的科学训练状态全部相同。

## 不能从本报告推出

smoke只有每域2张训练图、1张discovery图，所有PSNR均被丢弃。它不能说明HJ、HNEK
或macro谁有效，也不能替代4090 e200结果、跨seed或confirmation。下一步只有一件
事：四台4090在同一commit上重建manifest/runtime/e0并通过身份合并工具。
