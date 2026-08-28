# Claim boundaries

## 本轮允许回答

- 四条冻结 lane 在 seed=2026、泄漏安全8553张训练集、e200时谁最好；
- HJ的有限早期介入能否在长达192个native epoch后仍保留matched收益；
- HNEK历史e200信号能否迁移到clean全量六域；
- 将endpoint empirical measure改为六域宏边际是否改善宏评价；
- 每个结果的逐域、绝对、相对和计算成本。

## 本轮不能宣称

- 跨seed稳定；
- 论文最终确认；
- 完成UNSB论文原始400-epoch训练；
- 图像bootstrap代表训练seed不确定性；
- paired PSNR是训练时可访问的信号；
- time-dead bug修复是我们的新算法；
- macro-marginal若为正就必然是SB专属贡献。

## 结果分类

- `strong_single_seed_development`：e200宏PSNR为正、至少4/6域正、最差域
  >−1 dB，SSIM不退化，e150到e200没有>0.3 dB回撤。
- `positive_but_fragile`：e200宏PSNR为正但域覆盖或轨迹不足。
- `weak_fallback`：全负时e200排名第一的非plain方法。
- `engineering_invalid`：身份、resume、数据或访问门失败，禁止科学解释。

最终候选冻结后才允许一次性打开confirmation20。confirmation不用于回改方法。
