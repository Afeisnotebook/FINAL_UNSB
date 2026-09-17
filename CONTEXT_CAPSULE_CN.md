# FINAL_UNSB 最小上下文胶囊

这份胶囊只保存会影响下一步判断的当前事实，不保存旧PID或在线队列。

## 研究是怎样走到这里的

1. 早期在原始 UNSB 上观察到 DT、HJ、HNEK 等方法的阶段性正收益，但源码随机性和时间
   尺度混乱使消融难以解释。
2. 项目建立 deterministic clean UNSB，随后发现许多旧实验实际只覆盖很短 data epochs，
   不能据此判定长期机制。
3. small25路线一把DT/HJ/HNEK及后续机制当作因果探针，完成真实e200图谱，并产生
   Proposal-only、HJCGR、AM-TNC及ST-CGR等构造。
4. full-data阶段固定每侧8,553张、batch1、seed2026、e200，完成plain、Proposal、
   ST-CGR、AM-TNC和外部基线的统一评估。
5. 最终结果证明Proposal-only能保持相对plain的长期收益；ST-CGR/AM-TNC当前实现不能。

核心问题从来不是寻找更聪明的退出阈值，而是构造能与UNSB长期动力学相容、具有明确
自稳定/无偏/条件方差性质的算法。最终正证据支持的是Proposal的player-selective、
post-D/E条件iid双视图G/F估计；并不证明所有方差缩减都有效。

## 当前结论

- Proposal-only：通过full-data长期门，late-three `+1.723565 dB`，e200 `+0.839347 dB`。
- ST-CGR：当前实现失败，late-three `-0.692040 dB`，e200 `-2.744077 dB`。
- AM-TNC：当前实现失败，late-three `-2.107622 dB`，e200 `-4.072023 dB`。
- matched plain e200：PSNR 19.675085、SSIM 0.642470、LPIPS 0.236559。
- Proposal e200：PSNR 20.514432、SSIM 0.663835、LPIPS 0.221469。
- CUT/DCLGAN/CycleGAN e200绝对PSNR分别为23.728935/23.328284/21.655429。

这不是总体SOTA结论。论文的可靠主线是UNSB内部长期条件梯度方差控制及其边界。

## 当前科学边界

- 仅seed2026；confirmation20未打开；不选最佳checkpoint。
- ST-CGR/AM-TNC只关闭当前实现；HJCGR、DDSB不能写成机制失败。
- terminal singular drift未确认。
- DCLGAN为standalone fixed-protocol外部基线，没有matched delta。
- runtime事故、legacy one-sample stream phase offset及AM-TNC数值恢复必须披露。

## 现在做什么

下一步是论文claim review和confirmation policy freeze，不是自动重训。若之后明确授权新研究，
必须创建新的decision/contract，不能复活旧active plan。

文档权威顺序见`configs/DOCUMENT_AUTHORITY_REGISTRY.json`；完整结果见
`archive/paper_aio/final_v10/PAPER_ALGORITHM_PORTFOLIO_WITH_DCLGAN.json`。
