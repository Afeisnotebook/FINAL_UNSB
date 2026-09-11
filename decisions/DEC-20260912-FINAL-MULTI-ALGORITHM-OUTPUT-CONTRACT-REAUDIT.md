# DEC-20260912：最终多算法论文交付合同复核

## 裁决

当前4090A上实际部署的最终交付等待器不会把研究结果事后压缩为单冠军，也不会选择最佳
checkpoint。它使用的冻结脚本 SHA256 与当前已审查实现完全一致，supervisor/child PID
`3879674/3802122` 正常等待全部固定 e200 结果，restart count 为0。本次不需要替换或重启
等待器，也没有修改任何训练。

## 终局必须保留的内容

- 第一波结果必须包含 Input、plain、Proposal、CUT、CycleGAN 和 ST-CGR；缺一项即 fail closed。
- 算法组合固定包含 Proposal、ST-CGR、AM-TNC 三条完整长期结果。通过门的算法列入
  `accepted_algorithms`；未通过的算法仍保留完整轨迹，并仅标记为
  `failed_current_implementation_and_protocol`，不会从论文证据中消失或被写成机制证伪。
- HJCGR保持 `deferred`，DDSB保持 `reproduction_incomplete`；两者都不会因未产生数字而被伪装
  成负结果。
- DCLGAN通过独立、非阻塞的固定 e200 addendum 并入；不存在用其阻塞核心论文组合或把它
  错写成 matched delta 的路径。
- complexity固定覆盖plain、Proposal、CUT、CycleGAN、AM-TNC、ST-CGR，主表只使用e200，
  sustained只使用e150/e175/e200。

## 防止最后阶段目标漂移

提交后的 manuscript branching contract 对 Proposal/ST-CGR/AM-TNC 的八种 PASS/FAIL 组合全部
预注册，显式禁止推断唯一冠军，并保留全负结果分支。论文表格必须等待人工/Codex claim review
及已提交的 claim-freeze receipt，且要求六域轨迹、合法runtime relation和固定e200；因此终局
数字到达后不能临时删算法、挑checkpoint或跨不等价runtime合并delta。

五组针对性测试共34项全部通过。本次没有发现需要修改冻结实现的缺口，保持健康等待器原样
运行是风险最低且科学上正确的选择。

机器可读证据：
`evidence/paper_aio/PAPER_AIO_FINAL_MULTI_ALGORITHM_OUTPUT_CONTRACT_REAUDIT_20260912T071947.json`。
