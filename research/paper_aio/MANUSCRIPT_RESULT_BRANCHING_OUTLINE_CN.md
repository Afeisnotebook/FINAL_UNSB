# FINAL_UNSB 结果条件化论文骨架

状态：`PRE-RESULT STRUCTURE FROZEN / NO EMPIRICAL CLAIM`

这不是论文结果草稿，而是结果到达前冻结的写作路由。任何“提升、稳定、优于、有效”以及
最终保留几条算法，都只能从已提交的claim freeze和确定性表格导出中获得，不能由作者在看见
结果后临时改故事。

机器可读合同：`configs/PAPER_MANUSCRIPT_BRANCHING_CONTRACT.json`。

## 1. 中心研究问题

UNSB是一个顺序更新的无配对随机博弈：D和E先改变状态，随后G/F在该post-D/E状态上估计
更新。历史DT、HJ、HNEK及后续机制曾出现短期收益和长程反转。本论文不把问题重新写成
“寻找更好的退出窗口”，而问：在不读取paired target、不改变UNSB bridge/endpoint law的
条件下，能否通过条件随机测度或优化几何，得到在真实200 data epochs仍成立的长期结果？

这段问题陈述在正结果和负结果下都不变。

## 2. 建议章节结构

### 2.1 引言

1. 无配对All-in-One restoration的训练与评估难点；
2. UNSB顺序player更新中早期帮助、长期反转的实证问题；
3. 研究目标是长期operator设计，不是fixed-window handoff；
4. 贡献列表留空，直到claim freeze决定实际通过的方法集合。

引言不得把“条件方差存在”“two-view averaging”“timestep stratification”或“gradient
surgery”本身写成首创。

### 2.2 相关工作

- unpaired translation/restoration：CycleGAN、CUT、DCLGAN及DDSB；
- neural/diffusion Schrödinger bridge：UNSB、DBIM、Consistency Diffusion Bridge；
- stochastic estimator variance reduction：importance/non-uniform timestep sampling、
  antithetic/stratified sampling；
- gradient geometry：PCGrad、CAGrad与随机gradient manipulation的收敛限制；
- endpoint/terminal工作：NADB与SDDBM，明确当前方法没有改变terminal law。

投稿前必须重新执行`RELATED_WORK_NOVELTY_BOUNDARY_CN.md`中的一手来源检索门。

### 2.3 方法

先写共同状态边界，再写三条operator，避免把它们包装成一个预定冠军：

- **Proposal-only**：D/E保持原生单视图；在同一post-D/E状态上为G/F抽取两个条件iid完整
  视图并在pre-Adam合并。只主张条件均值保持与条件协方差减半。
- **ST-CGR**：在Proposal的两次G/F视图中，对T=5 bridge time做有序均匀无放回耦合；
  每个replica边际仍均匀，相对Proposal删除明确的between-time协方差项。
- **AM-TNC**：对同一player的exchangeable replicas，在冻结的pre-step Adam对角度量内
  分解disagreement并保留切向分量；交换反对称性只支持pre-Adam条件均值边界。

HJCGR只作为有证据的父机制/族成员讨论；未进行full-data不等于机制证伪。

### 2.4 实验协议

- 六域单模型All-in-One，8553张/侧，官方image-proportional unpaired sampler；
- seed2026、batch1、128px、200 data epochs，主结果固定e200；
- sustained结果固定e150/e175/e200；
- confirmation20在算法、基线与claim全部冻结后只打开一次；
- Proposal/ST-CGR只使用审核通过的5090B fresh-e0 matched plain；AM-TNC只使用4090A同宿主
  plain；不能合并非等价runtime delta；
- 外部方法保留其论文标签和复现范围，受控backbone复现不得冒充官方逐字复现；
- 报告参数、受控optimizer-step时间、峰值显存和相对plain成本，不以跨宿主epoch时间作为
  算法成本比。

## 3. 固定图表接口

1. **主表**：Input、CycleGAN、CUT、plain UNSB、Proposal、ST-CGR、AM-TNC的e200
   PSNR/SSIM/LPIPS；DCLGAN作为不阻塞核心结论的合法addendum。
2. **长期轨迹图**：e1--e200绝对轨迹，以及有合法control时的matched delta；突出固定
   e150/e175/e200，不突出峰值checkpoint。
3. **逐域图**：六域delta、正域数量和最差域，不能只报告pooled平均。
4. **随机性/分布表**：e200五个固定rollout bundle、KID主指标与标注小样本限制的FID。
5. **机制图**：UNSB顺序D→E→G/F状态边界，Proposal→ST-CGR lineage及AM-TNC独立分支。
6. **因果审计图**：只使用预注册target-blind信号；paired结果只能作为审计结束后的标签。
7. **复杂度表**：同一评估/计时运行时下的受控成本及相对plain比值。

所有数值必须由`MANUSCRIPT_TABLES_RECEIPT.json`绑定的导出文件填入，禁止手工复制。

## 4. 八种结果分支

“通过”指完整paper disposition门通过，不是单个e200 delta为正。

| Proposal | ST-CGR | AM-TNC | 论文主线 |
|---|---|---|---|
| 否 | 否 | 否 | 长程负结果与因果边界；不能改用最佳checkpoint |
| 否 | 否 | 是 | 独立Adam-metric stochastic geometry |
| 否 | 是 | 否 | 结构化time coupling是当前实现中必要的经验组件 |
| 否 | 是 | 是 | time-coupled estimator与独立geometry两条贡献 |
| 是 | 否 | 否 | player-selective conditional averaging |
| 是 | 否 | 是 | conditional estimator与独立geometry两条贡献 |
| 是 | 是 | 否 | Proposal→ST-CGR两级conditional estimator family |
| 是 | 是 | 是 | 两级estimator family加独立geometry，不强行合成单算法 |

如果Proposal失败而ST-CGR通过，只能说time coupling对当前full-data operator是必要经验组件，
不能反向声称Proposal的small25结果代表full-data成功。如果三条都失败，数学性质仍可报告，
但不能把数学新意替代长期性能结论。

## 5. 结果到达后的执行顺序

1. 验证全部source-bound e200 export及逐文件SHA256；
2. 在同一冻结评估运行时生成统一结果；
3. 审核runtime relation，只生成合法matched delta；
4. 生成AM-TNC、ST-CGR及Proposal的paper disposition；
5. 由动态portfolio物化完整算法集合，不预设唯一冠军；
6. 完成一手来源新颖性刷新；
7. 提交人工/Codex claim review，再生成并提交freeze receipt；
8. 运行确定性manuscript-table exporter；
9. 按上表选择唯一对应的写作分支；
10. 完整分布cohort冻结后，另行审核是否授权一次confirmation20。

## 6. 无论结果如何都必须披露

- 第一波主要是seed2026，不能宣称多seed稳定；
- 三条方法FLOP不同，optimizer-step数量相同不等于计算量相同；
- 条件均值结论停在pre-Adam，不能外推为Adam位移或完整训练kernel无偏；
- 当前方法没有证明解决terminal singularity；
- DDSB缺少可唯一复现的官方实现是复现边界，不是DDSB负结果；
- confirmation20不得反向修改算法、checkpoint、NFE或论文主张。

## 7. 摘要占位规则

摘要在claim freeze之前只能保留四个槽位，不得填结论：问题、方法集合、实验协议、经过审核的
结果。结果槽必须直接引用冻结`PAPER_CLAIMS.csv`中的允许主张；如果没有方法通过，摘要必须
转为长期负结果/诊断性研究，而不是改写窗口或筛选峰值。
