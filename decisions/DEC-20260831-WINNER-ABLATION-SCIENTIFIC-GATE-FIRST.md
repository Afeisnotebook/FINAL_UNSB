# DEC-20260831：赢家消融先过长期科学门，再比较排序字段

## 决策

proposal-only、observable-only与projected/full都完成共同e0、small25、seed2026、
batch1、e200之后，候选选择必须与全局路线一裁决一致：

1. 先区分是否通过完整持续收益门；
2. 任何严格通过者优先于`long_horizon_negative_current_implementation`；
3. 只在同一资格层内部依次比较late-three宏PSNR、e200、域覆盖、最差域、回撤、
   SSIM/LPIPS与成本；
4. 若所有实现都失败，才在失败层内部选择当前最优fallback。

旧赢家消融裁决直接使用数值排序键，导致PC-RSMG full凭late-three
`+0.620959 dB`压过proposal-only的`+0.541507 dB`，虽然full的e200为
`-0.001379 dB`且严格门失败，而proposal-only的e200为`+0.451092 dB`并通过完整
护栏。这与已注册的候选晋级规则和跨版本裁决器的“eligible-first”语义不一致。

修复后，proposal-only成为当前canonical seed2026开发候选；full与observable-only仍
完整保留为同一算法家族的机制证据。这个变化只读取三条已经完成并冻结的e200 receipt，
不选择checkpoint、不重训、不改变算法，也不使用paired指标控制训练。

## 边界

- 这是单seed、small25长期信号，不声称跨seed或全量一万张稳定。
- PCNR与AM-MCRB仍继续到e200；它们可能在后续同宿主裁决与4090复跑后改变主排名。
- 唯一主候选仍只是规范交付入口；两个备选和完整可信前沿必须保留。
- confirmation20、路线二、handoff、退出阈值和跨宿主delta合并继续禁止。

