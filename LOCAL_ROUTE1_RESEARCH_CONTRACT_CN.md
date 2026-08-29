# 本地路线一长期算法发现契约

## 1. 唯一目标

从 deterministic clean UNSB 出发，利用 DT、HJ、HNEK 及后续机制的历史正窗口、
反转和失败证据，主动构造一个能在真实 200 data epochs 上保持 matched 收益的新
算法。最终方法不必保留任何旧算法的名字或形式。

当前已允许本地、4090和5090承担严格同宿主matched的路线一计算；这不把任务降级为
旧算法排名，也不允许跨宿主delta、全量数据、confirmation20、退出阈值、固定窗口、
paired-PSNR控制器或plain输出模仿。

## 2. 时间尺度纠错

- small25：六域共150张，batch1，2400 updates = 16 data epochs；
- full100：六域共600张，batch1，12000 updates = 20 data epochs；
- small25 的真实 e200 = 30000 updates；
- 所有新报告必须同时写 `updates` 与 `data_epochs`，科学裁决以data epochs为准。

旧 SEARCH-001/003/005 的短程负结果只关闭命名实现及已测试协议。除非已有等价
matched e200 证据，不得写成父机制的长期证伪。

## 3. 多算法而非 HJ-only

第一层长期锚点包含：

1. plain：共同基准；
2. continuous HJ：时间量尺正对照，历史轨迹 e100负、e125后正；
3. HNEK：历史e200正、clean e20内频繁变号的bridge-coordinate探针；
4. DT：历史有正窗口、deterministic clean短程为负的方向/协方差探针。

HJ先执行不代表只研究HJ。它的特殊作用是检验本地协议能否看见已知的延迟收益。
HJ校准结束不能作为项目完成条件。

第二层机制证据包含 PCOA/NPOOA、LBST、PTQ、DCUM/macro-marginal、AEB/BCAVP 及
HJ/DT/HNEK派生。它们不必全部原样跑到e200；其证据用于解释失败机理和生成新算法。
若其中某个原实现被重新晋级，必须先给出为什么短程失败仍值得长程运行的证据卡。

TA_MINIMAL直接恢复实现已有matched e200负结果，只作为长期负对照；这不否定所有
time/coordinate构造。

## 4. 执行顺序

### A. 语义谱系审计

逐项核对旧正结果与clean实现的数学对象、参数、forward/backward、batch composition、
sampler、RNG、optimizer顺序、数据视图和评估器。输出多方法lineage表，不能只审HJ。

### B. 长期锚点图谱

从共同small25 e0运行plain/HJ/HNEK/DT至e200。固定seed2026、batch1、confirmation20
封存；在e1/5/10/20/40/60/80/100/125/150/175/200评估。除工程故障外，不因中间
paired结果为负而停止。

记录target-blind内部量：原生/校正梯度关系、下一独立batch一致性、block/time/domain
符号、GAN/SB/NCE方向导数、endpoint dispersion、bridge状态、G/D/E平衡、Adam
moment角度和采样方差。paired指标只作事后标签。

### C. 算法生成

比较各探针在长程中的共同与专属转折，选择证据最清楚的失败机理，先生成一个最小
新算法。每个候选必须有derivation card：父证据、UNSB数学对象、更新推导、identity/
self-null或无偏条件、target不可访问证明、判死实验和成本。

候选可以融合多个探针揭示的同一个机理，但不得把多个未经独立支持的组件堆叠起来。

### D. 长期裁决

晋级候选从所在宿主的共同e0完成small25 e200 matched轨迹。排序看e150/175/200晚期均值、e200、
正域数、最差域、绝对轨迹和回撤；不挑最佳checkpoint。若失败，只允许按新观测到的
失败原因修订一次，不得改成窗口搜索。

## 5. 防漂移判据

下列表述出现时必须立即停下并回读本契约：

- “后续只做HJ”；
- “四个冻结算法直接上服务器比较”；
- “找到更聪明的退出点就是路线一”；
- “2400/12000 updates已经证明200-epoch长期失败”；
- “当前实现失败等于整个父机制被证明不可能”；
- “先验证预设算法，算法构造以后再说”。

本项目当前成功标准是完成一次证据驱动的长期算法发现与裁决，而不是让某个历史名字
获胜。
