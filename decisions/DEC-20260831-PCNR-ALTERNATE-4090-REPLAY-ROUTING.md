# DEC-20260831：PCNR证据备选进入4090共同e0复赛

状态：`ACCEPTED_SOURCE_BOUND_ALTERNATE_REPLAY`

## 问题与裁决

5090完整e200前沿中，PCNR（`F1-01-PLAYER-CONDITIONAL-NATIVE-RESAMPLING`）
是唯一被保留的当前修复方向：late-three宏PSNR为`+0.034821 dB`，e200为
`-0.300890 dB`，晚期SSIM为正、LPIPS改善，分类为`evidence_backed_alternate`。
它不是严格通过项，但它与4090上的PC-RSMG/AM-TNC属于不同的条件采样机制证据。

旧的便携复赛导出器把推荐队列再次限制到RF-AMMCRB和RF-MCRB两个residual-feasible
身份。源5090裁决正确推荐PCNR复赛，但过滤后生成了空4090 portfolio。这是编排漏路由，
不是PCNR的科学淘汰，也不应把“唯一主候选”误解释为“只保留一个算法”。

因此新增PCNR专属、source-bound的4090持久后继：它只在便携完整5090前沿同时证明
PCNR为action priority、`evidence_preserved`且分类为`evidence_backed_alternate`时启动；
算法、源码commit、derivation card、implementation及完整e200 receipt必须逐项绑定。
4090从本机共同e0、seed2026、batch1重新训练到真实e200，再与4090 plain比较。

## 已发现的未执行注册差异

4090运行根中已有一个从未过gate、从未训练的旧PCNR注册。它的card与5090权威完全
一致，但implementation只在`generation1_gates.py`源码哈希上过时。新注册器只允许在以下
条件全部满足时修正该注册：

- 旧/新implementation除`source_files[*].sha256`外完全相同；
- 当前c874e37源码逐项匹配5090完整e200 receipt；
- 4090不存在PCNR gate、executor、candidate目录或terminal receipt；
- 修改前的implementation和ledger record先写入不可变事故归档；
- 重算后的algorithm fingerprint必须等于5090的
  `3423a51ff77f5605c7daac7d0b0317d7596e0e7cde256671fcb19f2c5b4ac186`。

这不是公式修订，也不使用5090 checkpoint。任何语义差异或已执行痕迹都会失败关闭。

## 多候选边界

- `CANDIDATE.json`继续只表达下一步动作优先级，不删除完整候选前沿。
- PC-RSMG proposal-only、PC-RSMG full、AM-TNC和PCNR在各自同宿主完整e200结果中保留；
  closed-current-operator只关闭具体算子，不扩张为父机制死亡。
- 额外算力按独立机制证据使用，不机械复跑RF-AMMCRB/RF-MCRB，也不做超参网格。
- PCNR复赛完成后，4090完整前沿必须重新物化；旧的空portfolio裁决不能直接触发最终
  pointer。

## 不变边界

- paired指标只在完整源轨迹冻结后用于资源分配，不进入公式、训练或控制；
- 不合并4090和5090的delta；
- 不迁移模型、优化器或checkpoint；
- 不改变batch、seed、data epochs或里程碑；
- 不使用退出窗口、handoff、paired controller或最佳checkpoint；
- confirmation20继续封存，seed2027/2028继续延期。

