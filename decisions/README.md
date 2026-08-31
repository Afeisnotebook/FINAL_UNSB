# Decision order

Read decisions numerically. A later accepted decision may supersede an earlier
one only when it names the superseded fields explicitly. Proposal files do not
authorize compute.

- `DEC-0001-FOUR-LANE-FREEZE.md`: scientific portfolio freeze.
- `RUN_AUTHORIZATION.example.json`: schema/example only; not authorization.
- `CONFIRMATION_UNLOCK.example.json`: schema/example only; confirmation remains sealed.
- `DEC-0002-LOCAL-PREFLIGHT-ACCEPTED.md`: local engineering gate result and HJ RNG correction.

## 当前路线一覆盖顺序（2026-08-31）

旧的四lane冻结已被后续用户授权明确暂停。接手者在上述历史文件后应按以下顺序读取：

1. `DEC-20260829-LOCAL-ROUTE1-REOPEN.md`：重新打开长期算法发现，旧短门不再充当e200结论。
2. `DEC-20260830-ROUTE1-REMOTE-OFFLOAD.md`与
   `DEC-20260830-ROUTE1-REMOTE5090.md`：授权4090/5090仅作host-matched路线一加速。
3. `DEC-20260830-ROUTE1-SINGLE-SEED-EMERGENCY.md`：当前以seed2026完整e200作开发选择，
   seed2027/2028延期但不得声称稳定。
4. `DEC-20260831-ROUTE1-FRONTIER-EXPANSION.md`：PCNR与AM-MCRB组成两条证据驱动前沿，
   不是超参网格。
5. `DEC-20260831-FRONTIER-PRESERVATION-AND-WINNER-ABLATION-BINDING.md`与
   `DEC-20260831-FRONTIER-FINAL-EVIDENCE-COMPLETENESS.md`：唯一主候选只是交付入口，
   两个备选、完整前沿和主候选自身三项e200消融都必须保留。
6. `DEC-20260831-WINNER-ABLATION-SCIENTIFIC-GATE-FIRST.md`：严格通过者优先于数值更高
   但长期门失败的fallback；当前PC-RSMG proposal-only因此优先于full。
7. `DEC-20260831-ROUTE1-CONDITIONAL-GENERATION3-SYNTHESIS.md`已因旧AM-MCRB固定绝对
   余量事故标为`SUPERSEDED_DO_NOT_RUN`；替代文件
   `DEC-20260831-RESIDUAL-FEASIBLE-CONDITIONAL-SYNTHESIS.md`要求两个同宿主严格父项、
   residual-feasible屏障与target-blind兼容门，才允许一个最多两组件的Generation-3合成。
8. `DEC-20260831-EVIDENCE-QUALIFIED-MULTI-CANDIDATE-ADVANCEMENT.md`：canonical主候选
   只是行动接口；严格通过和因果可修复近边界机制继续组成受限可信前沿。
9. `DEC-20260831-5090-MATCHED-PLAIN-LPIPS-RECOVERY.md`：晚期plain LPIPS缺失属于工程
   证据缺口，必须从冻结checkpoint双重确定性恢复并通过父状态隔离，不能机械记为算法失败。
10. `DEC-20260831-ANTITHETIC-GAUSSIAN-GRADIENT-AUDIT.md`：梯度级Gaussian反号虽保持
    期望无偏，但e20/e100/e200相对iid两视图均增加方差，因此当前involution不启动e200；
    该结论不外推到所有无偏variance estimator。
11. `DEC-20260831-RESIDUAL-FEASIBLE-EUCLIDEAN-CONDITIONAL-SYNTHESIS.md`：在对应父项
    同宿主严格通过的前提下，保留条件采样与Euclidean residual-feasible屏障的G3-03；
    它与G3-02的Adam度量是两个独立约束几何，不是强度网格，也不预先删除任一父算法。

同日文件若涉及不同对象并非互相覆盖；只有明确写出被替代字段的后决策才覆盖前决策。
