# DCLGAN以非阻塞addendum进入最终论文组合

日期：2026-09-03

交付链审计发现：DCLGAN已经拥有官方源码绑定、完整状态适配、5090B长训队列、source
export、统一固定评估和健康监控，但当前核心`PAPER_ALGORITHM_PORTFOLIO.json`只收录
Input、CUT和CycleGAN三条外部结果。若不补接口，DCLGAN完成后会形成一份独立结果，却
不会自动进入最终论文组合，也没有统一复杂度回执。

本次新增非阻塞的DCLGAN portfolio addendum。它不成为核心论文交付的前驱：核心plain、
Proposal、ST-CGR、AM-TNC、CUT和CycleGAN可以先完成统一结果与组合；DCLGAN自己的长串行
链完成后，addendum才读取固定e100/e125/e150/e175/e200结果，在共享评估GPU上只读加载
e200 checkpoint，测量参数量、训练步延迟、推理延迟和峰值显存，最终产生带DCLGAN的
增强组合。

该接口冻结DCLGAN既有evaluation contract和官方上游/adapter身份，重验所有结果、metric、
receipt和checkpoint哈希。等待期不解析性能值；完成后的指标仅用于论文汇总，不控制训练、
调度或最佳checkpoint。DCLGAN仍是standalone external comparator，不与UNSB plain制造
matched delta。

当前只提交实现和完成矩阵，暂不部署新的等待进程。原因不是工程未就绪，而是5090A plain
取消后，未来合法核心portfolio的真实output路径只有在5090B runtime receipt经过Git人工
准入、dynamic evaluator/final-delivery successor冻结时才能确定。届时从干净commit部署
addendum；现在绑定旧的已被取代output反而会制造一条必然无法完成的僵尸链。

实现后的真实1000-update checkpoint烟测还提前发现一项跨checkout身份污染：冻结adapter
动态载入新评估控制进程时，Python模块缓存会让它把control commit误当adapter commit，
合法checkpoint因而会在评估开始前被拒绝。修复只把adapter的Git身份函数绑定回已经过
源码hash验证的adapter checkout，不改变网络、损失、数据、checkpoint或指标。现有旧
DCLGAN评估等待器须在未开始评估、未读取性能值时由包含该修复的新冻结checkout安全替换。

本次没有停止、重启或修改任何远端训练，没有读取中间paired性能，没有打开
confirmation20。高分辨率推理和KID/FID仍属于算法/基线冻结后的补充项，不阻塞e200核心
组合。
