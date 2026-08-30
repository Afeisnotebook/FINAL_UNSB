# DEC-20260831：路线一近失配候选前沿扩展

## 决策

在紧急单seed开发协议下，不把PC-RSMG的唯一fallback交付视为算法搜索的自动终点。
保留4090上已经冻结的PC-RSMG三项e200消融，同时使用当前空闲5090推进两条有长期因果
依据、无paired控制、无窗口且不依赖超参网格的新数学分支：

1. `F1-01-PLAYER-CONDITIONAL-NATIVE-RESAMPLING`（PCNR）：保留原生单样本随机性，
   只解除D/E实现状态与随后G/F随机视图之间的条件耦合。
2. `F1-02-ADAM-METRIC-MOVING-COVARIANCE-BARRIER`（AM-MCRB）：保留MCRB的移动
   covariance-rate安全集，把欧氏最近点改写为Adam对角信任度量中的唯一最近点。

两者都必须先通过数学不变量、disabled identity、精确resume、e20/e100/e200跨状态父hash
隔离、target-blind和400-update有限性门，之后才可从共同e0运行seed2026、small25、batch1、
真实e200。中间paired指标只作固定里程碑描述，不能控制训练、调度、修改或退出。

## 为什么这不改变北极星

- 目标仍是发现能够在真实200 data epochs维持收益的新UNSB算法；不是保住旧名字。
- PC-RSMG、AM-TNC与MCRB都比历史50-epoch快速反转更接近长期可用边界，但尚无严格赢家；
  此时砍成唯一候选会丢失机制信息。
- 追加分支分别修复“条件耦合与方差缩减混在一起”和“安全约束与优化器几何不一致”两个
  已观测缺陷，不是对旧配置做网格搜索。
- 最终仍输出明确的主候选和排序；同时交付一个带完整e200证据的候选前沿，不预留旧算法
  名额，也不把单seed写成跨seed证明。

## 资源与停止边界

- 4090继续完成PC-RSMG proposal-only、observable-only和既有full的固定消融，不改代码身份。
- 5090最多并行两条batch1前沿轨迹；先用短门实测吞吐，若双流总完工时间劣于单流则顺序跑。
- seed2027/2028继续延期；释放算力用于数学分支而非重复初始化。
- confirmation20、全量数据、路线二、handoff、退出阈值和paired训练控制继续封存。
- 每条前沿只允许本文件冻结的一个公式，不调窗口、强度、teacher half-life、replica数或batch。
- 完整e200失败只关闭该具体算子；不得自动升级为父机制死亡。
