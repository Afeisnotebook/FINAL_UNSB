# FINAL_UNSB 全量论文实验与算法重构契约

状态：`ACTIVE_FULL_DATA_PAPER_RESEARCH`  
生效：2026-09-02 用户明确授权  
历史父阶段：small25 路线一已完成，证据保留但不再限制 full-data runner

## 1. 北极星

在六域 All-in-One 无配对任务上，同时完成两件不可互相替代的工作：

1. 用论文级、固定 e200 协议获得 plain UNSB、Proposal-only、CUT、CycleGAN及可可靠
   复现的当代对手的完整轨迹；
2. 继续依据 small25 长期因果证据构造和裁决新算法，目标是一个或多个在全量 e200 仍
   有持续收益、数学对象明确的算法。

这不是“只验证 Proposal”，也不要求最后只有一个冠军。`ACTION_PRIORITY`只安排下一份
算力；`ALGORITHM_SET`保存所有严格可行、脆弱正向、关闭当前实现和复现未完成的方法。

## 2. 冻结科学协议

- manifest：`manifests/FULL_DATA_MANIFEST.csv`，SHA256
  `02c01df580b882763fb0ff28dbdeac4b3729deb8bb772005f26f3e7bc2e36744`；
- train/discovery/confirmation：每侧 `8553/480/120`；
- 一次 data epoch = 8553 个 batch-1 optimizer updates；e200 = 1,710,600 updates；
- seed2026，128×128，no flip，num_workers=0；
- 主训练 measure 为 `official_image_proportional_unpaired`：A 每 epoch 一次 permutation，
  B 从全体边际独立均匀抽样，不读取 paired target 或域标签；
- UNSB 家族固定 lr `1e-4`、Adam `(.5,.999)`、200 constant epochs、GAN/SB/NCE
  权重均1、tau `.01`、T=5、NCE layers `0,4,8,12,16`；
- 外部 CycleGAN/CUT 保持各自官方损失和 `100 constant + 100 linear decay`；
- 不增大 batch、不梯度累积、不用中间指标改算法、不选择最佳 checkpoint。

当前 full-paper 科学 checkout 固定为 commit
`31f2fb8badaf8293a2ed2744963035575df7d7a6`，协议指纹为
`68f53a8e9d6fdafd750956d16fbd537aed6e727e081b1db6d0b62258e09b4e41`。
任何新编排代码只能等待、验门和精确恢复，不能改变正在跑的转移。

## 3. 宿主与比较边界

- 4090A：full plain；完成后同一 output root、同一 runtime、同一 e0 启动 Proposal。
- 5090B：CUT；完成后启动 CycleGAN。二者是独立外部基线，不与4090做 matched delta。
- 5090A：当前只运行 ST-CGR 的 small25/e200 证据门后轨迹；其结果仍为宿主分离证据。
- 本地 GTX1660：代码门、只读审计和不影响远端吞吐的研究工作。
- 计划建议的另外两张4090在用户提供实际连接前只是假设资源，禁止写成在线或已分配。

method-minus-plain 只能在相同 runtime cohort 中计算。checkpoint 不跨宿主续接，5090
结果不能减4090 plain。所有最终模型复制到一个固定4090评估容器做统一推理，但复制模型
不等于合并训练轨迹。

统一评估必须先由source host为e100/e125/e150/e175/e200 checkpoint生成同时绑定文件hash、
scientific-state hash、lane语义、训练commit/fingerprint与宿主标签的export receipt；复制后
只能只读加载。plain、Proposal、CUT、CycleGAN五个epoch全部在同一environment与当前
evaluator fingerprint复算并锁成`UNIFIED_EVALUATION_COHORT`后，第一波才可标记完成。

## 4. 评估和确认集

- 固定里程碑：e1/5/10/20/40/60/80/100/125/150/175/200；
- e1–e200固定discovery70与CRN单rollout，PSNR/SSIM；晚期增加LPIPS；
- UNSB家族e100/e150/e200固定报告NFE1–5，不据此选择最佳NFE；
- e200补全discovery80并运行5个固定rollout bundle；
- CRN bundle seed身份固定为首波paper指纹`68f53a8e...`，与之后新增候选导致的训练
  源码指纹变化解耦；候选和parent plain只有逐图stem/order/replicate/NFE/bundle hash
  全部一致时才允许计算delta；
- 主表只使用e200，sustained指标固定e150/e175/e200；
- KID/FID和补充分辨率推理只在算法冻结后计算；
- confirmation20直到算法、基线配置和论文主张全部冻结后才允许一次性打开。

## 5. 新算法防漂移门

- HJCGR、Proposal-only、AM-TNC是已完成的算法前沿，不是唯一候选名单；
- 新算法必须绑定父证据、UNSB数学对象、公式、不变量或无偏性质、target不可访问证明、
  判死反例和完整恢复状态；
- 当前终端谱审计没有跨两算法/三域确认低方差奇异漂移，因此禁止加入“终端修复模块”；
- time-stratum异方差证据产生了ST-CGR：两个post-D/E G/F视图的时间索引按均匀无放回
  有序对采样，单视图边际仍为原生均匀分布，期望不变；
- ST-CGR只有在固定状态target-blind门和small25 e200完整裁决通过后，才能形成全量
  candidate lock；中间paired指标不得决定是否继续；
- 全量candidate采用双阶段授权：先绑定source-bound small25 e200收据、完整trajectory、
  derivation card、implementation源码哈希与同宿主paper plain e200；再证明跨代码e0核心、
  原生2000步转移、zero-intervention、exact resume和重复评估全部精确一致。candidate
  lock本身固定写`full_data_authorized=false`，只有与当前commit/协议指纹绑定的第二份
  authorization才能启动171万步；
- Bridge Martingale Consistency只在系统性末段条件均值漂移证据成立后才可生成，当前
  不具备长训授权；
- 最多保留两个独立Generation-1方向，每个只允许一次由新因果证据支持的修订，禁止
  窗口、退出阈值、paired controller或无证据堆叠。

## 6. 外部基线真实性

- CUT已绑定官方上游语义并通过full-state门；
- CycleGAN使用仓库内官方损失适配器并在CUT后重新执行宿主门；
- DDSB是直接当代比较，但截至2026-09-02未找到作者公开源码。正文/补充不足以唯一恢复
  全部网络、更新顺序、stop-gradient和full-state语义，因此状态固定为
  `reproduction_incomplete`，不得用猜测实现冒充DDSB结果；
- DCLGAN/NEGCUT等第二优先级方法只有在第一波资源释放且源码/协议门通过后启动。

## 7. 持久性与完成条件

每条长训必须有完整状态、epoch heartbeat、同宿主精确resume和最多三次相同工程故障的
监督器。前序lane到`COMPLETE_E200`后，后继必须重新通过磁盘、数据、resume、重复评估
和方法专属身份门；失败时写明阻断，不得静默跳过。

本阶段完成需同时满足：

- 所有已授权第一波lane完成e200或给出非科学性的明确工程/复现阻断；
- ST-CGR small25完成并被诚实裁决，任何晋级算法均有自己的全量e200证据；
- 统一评估、逐域轨迹、绝对/相对分解、成本和随机性报告齐备；
- `PAPER_RESULTS.json`与`ALGORITHM_SET.json`区分科学失败、工程失败和复现未完成；
- 算法/基线/主张冻结后，confirmation20只打开一次；
- Git仅保存代码、合同、compact evidence与决定，不保存数据、checkpoint或完整日志。
