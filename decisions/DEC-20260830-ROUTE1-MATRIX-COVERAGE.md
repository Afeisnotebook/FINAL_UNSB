# DEC-20260830：补齐路线一因果矩阵的机制覆盖

状态：`PRE_EVIDENCE_ANALYSIS_SCHEMA_FROZEN`

## 问题

冻结审计器 `b7ebc4e` 已能产生真实的跨状态更新、虚拟分支、rollout/time 诊断和
sampling-variance atlas，但其矩阵排序只会提升 correction 符号、幅值、state feedback
和采样方差。已经采集的 rollout velocity 与 bridge-time conditioning 因而可能永远
无法成为算法构造来源，这与路线一预注册的机制空间不一致。

## 决定

在看到长期锚点和 atlas 结果之前，补充两个只读后处理路线：

- `rollout_distribution_speed`：使用 proposal/native rollout-speed ratio，以及相邻已审计
  状态之间的 native rollout-velocity growth；
- `coordinate_horizon_imbalance`：使用 latent/time 重采样下，各 bridge-time correction
  norm 的变异系数。

它们只有在对应 target-blind、数学零阈值信号通过既定跨方法筛选时，才具有有偏控制
算法的生成资格；否则只保留为解释证据。无偏重参数化仍按原规则独立授权。

## 证据不可变性

正在运行的训练 worktree、审计 worktree、checkpoint、分支和 atlas 行均不修改。
新增 `reanalyze` 只从冻结 atlas/variance atlas/queue 重建矩阵，并记录分析 commit、
分析源码、三个输入文件 hash 及 `branch_rows_modified_by_analysis=false`。候选之后绑定
重建矩阵和原 atlas 的 hash；不得把分析代码升级伪装成原始审计 commit。

该决定不新增预写候选、不打开 paired controller、不改变 confirmation20、数据、seed、
milestone、200 data epochs 或任何正在运行的轨迹。
