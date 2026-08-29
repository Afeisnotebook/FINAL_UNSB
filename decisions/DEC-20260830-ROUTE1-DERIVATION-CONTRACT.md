# DEC-20260830：新算法推导合同不得退化为旧实现换名

## 裁决

长期因果矩阵仍是 Generation-1 的主要构造依据；历史 DT/HJ/HNEK、
PCOA/NPOOA、LBST、PTQ、DCUM、AEB、TA 和 KCK 只提供谱系、反例与等价性边界。
任何进入工程门的新候选都必须同时绑定当前 causal matrix、reversal atlas 和三份历史
上下文证据，并明确说明其 operator 与已测试实现的实质差别。

候选 derivation card 现在强制包含：

- 父因果证据、历史谱系和 prior-equivalence audit；
- 被改变的 UNSB 数学对象与完整公式；
- identity/self-null/无偏条件以及 paired target 不可访问证明；
- 是否改变 objective、estimator、coordinate 和 endpoint law；
- 预期适用状态与可明确判死的反例；
- 算力、显存和恢复状态成本；
- proposal-only、observable-only、projected/full 三项消融。

`prior_equivalence_audit.equivalent_rerun` 必须为 `false`。这不禁止从旧探针获得思想，
但禁止把一个已经等价测试过的 operator 改名后当作“算法发现”。所有字段进入算法
fingerprint，算法冻结后修改任一数学或消融定义都会使已有 gate 失效。

## 不改变的边界

- 不预写候选公式；候选仍只能在 causal matrix 完成后生成。
- 不使用 paired PSNR/SSIM/LPIPS 作为 observable 或 controller 输入。
- 不改变正在运行的 `0da2a37` 锚点、batch1、e200 协议或任何 full state。
- 不把旧短协议失败升级成父机制证伪。

