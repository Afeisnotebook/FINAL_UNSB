# 三个月研究的最小上下文胶囊

这份文件专门给完全没有旧对话上下文的新 Codex。它只保留会改变最后一轮决策的
事实，不要求接手者重新阅读旧仓库。

## 0. 2026-08-29 当前优先级覆盖

本胶囊原来的结尾曾把项目收口为“四张4090、四条冻结lane”。用户随后明确暂停
4090，并要求继续本地路线一算法发现。复核又发现旧 local long gate 实际只有
16--20 data epochs，而历史 HJ 的收益在 e125 后才出现。因此：

- 当前目标不是HJ-only；HJ只是第一项延迟收益时间正对照；
- DT、HJ、HNEK是首层长期锚点，PCOA/LBST/PTQ/DCUM/AEB等是算法生成证据池；
- 旧短程失败关闭当前实现/协议，不自动判死父机制；
- 当前要建立多方法e200长期图谱并据此构造新算法；
- 4090、固定四lane和HJ有限handoff全部暂停。

下文第5--7节保留的是2026-08-28形成旧四lane计划时的推理，已降级为历史背景；
当前执行以 `LOCAL_ROUTE1_RESEARCH_CONTRACT_CN.md` 和最新decision为准。

## 1. 原始问题

我们从 UNSB 的 All-in-One 六域无配对图像恢复出发，曾观察到多个方法在训练早期
或中期出现正PSNR窗口，但继续训练后相对plain反转。DT、HJ、HNEK是三个主要历史
算法，后来又检查了time-active、path consistency、teacher、time sampling、domain
marginal、antithetic latent和耦合优化器等方向。

问题曾被错误简化成“找到更聪明的退出阈值”。用户真正授权的路线一是：从UNSB和
历史反转证据出发重新构造数学算法，争取长期收益；路线二才是：当算法必须有限期
介入时，检查native UNSB能否继承完整状态。

## 2. 干净基座为何必要

官方 `ResnetBlock_cond.forward()` 在循环中重复执行 `out = layer(x)`，显式time
embedding和首段卷积不会按论文意图顺序传播。更早的非确定性CUDA reflection-pad
反向又会让同seed训练漂移，足以覆盖小消融。当前canonical因此：

- 保留官方time-dead语义作为实际baseline；
- 用确定性reflection padding；
- 固定CuBLAS、cuDNN、TF32与全部RNG；
- 训练保持官方unpaired B sampling；
- 方法从同一seed/e0出发，完整保存G/F/D/E及优化器、scheduler、RNG和方法状态。

这不意味着官方time-dead是理论上正确的，只意味着它是当前性能更强且已审计的实际
baseline。

## 3. 已关闭方向

### TA_MINIMAL / KCK

UNSB论文明确描述共享time-conditional endpoint predictor，官方代码却功能性
time-dead。但严格的FINAL-1实验让plain和TA_MINIMAL从相同e0、相同随机bundle训练
到e200：TA为19.2485 dB，plain为20.3407 dB，delta `−1.09224 dB`，五域全负。
因此“直接恢复旧time branch会提高恢复质量”被否定。

KCK在同一TA_MINIMAL e5 full-state上做共同锚点分叉，到e10时目标path discrepancy
反而恶化3.4165%，4/4时间组合和5/5域均反向；不继续调lambda或延长。

### SEARCH-005路线一

路线一确实构造和测试了target-blind、自消隐或有不变量的算子，而非只搜索退出点。
其中PCOA在400/800/1200为正，到1600/2400反转；其范数保持修订NPOOA在400更好、
800更差。其他DT/HJ/HNEK派生算子也未通过2400步门禁。结论是“本轮算子没有持续
赢家”，不是“所有可能算法在数学上已被证明不可能”。

### DCUM等固定机制

DCUM强制B从A同域、不同stem选择。它短程出现过正信号，但与HNEK组合在full100的
4k降到−2.506 dB。LBST/PTQ/AEB当前实现也为负。它们不能换名进入最后四lane。

## 4. 仍存活的两条历史线

### HJ finite navigation

HJ的forward恒等，只在Layer-0 PatchNCE backward中删除与源结构方向冲突的梯度
分量。冻结协议是：

- `[0,1.6)` data epoch：plain；
- `[1.6,8.0)`：HJ；
- `8.0+`：只关闭HJ correction，完整保留G/F/D/E、Adam moments/age、scheduler、
  sampler与RNG，由native UNSB继续。

small25、seed2026中，total step 2400/2800/3200相对plain为
`+0.536/+2.133/+0.871 dB`，最后6/6域正。关键边界：总计只到约21.3个data
epoch，离本轮e200很远。

### HNEK

HNEK把官方time-dead generator输出解释成剩余时域归一化残差，并用真实物理horizon
修改endpoint/entropy坐标。冻结变体是`gamma=.25,residual,physical,all`。历史
900张视图、seed2026、e200为`+0.7884 dB`、4/5域正；但它从9个变体中筛选而来，
没有跨seed，clean小视图延长又频繁变号，所以只是高风险锚点。

## 5. 历史背景：为什么曾新增macro-marginal（当前未激活）

过去small25和full100都按“每域相同张数”构造，天然平衡。新的全量训练则是：

- FoggyCityscapes 4475/8553 = 52.3%；
- RainDS-syn 100/8553 = 1.17%；
- 主评价仍然是六域等权宏平均。

官方pooling还会让B target marginal按图片数加权。这个训练measure与评价measure的
错配从未在旧平衡视图被等价测试。

macro-marginal lane把经验端点measure定义为

`mu_macro = (1/6) sum_d mu_d`, `nu_macro = (1/6) sum_d nu_d`，

每步独立均匀抽A域和B域，再域内随机。A/B仍无配对，B不条件于A域，推理无域标签。
它是全量规模诊断，也是一个可能的训练方法；若为正，仍需额外论证其SB专属性。

## 6. 已暂停方案：为什么当时选择四条lane

我们只负担四张4090一周。plain不可缺；HJ是本地最强继承证据；HNEK是唯一历史
e200正锚点；macro-marginal是全量规模新增且未测试的问题。第四张卡若改成新的
fancy算子，会在没有本地归因和对照的情况下消耗整周。

四lane是一个面向“找到当前最值得继续候选”的风险组合，不是穷尽性科学证明。

## 7. 已暂停方案的结果流程

e200按宏PSNR、正域数、最差域、SSIM和e150→e200回撤排序。先在discovery80选择并
冻结唯一候选，再一次性打开confirmation20；不能用confirmation修改方法。若有正
候选，下一批算力才运行winner/plain seed2027，必要时2028。若全负，冻结第一名为
weak fallback并停止把单seed图像bootstrap写成算法稳定性。
