# 动态matched-control交付的不可变产物闭环

日期：2026-09-03

在5090A plain取消后的交付DAG复核中发现：既有future final-delivery接口会等待
first-wave、AM-TNC和ST-CGR successor的完成状态，但对完成状态所声明的cohort、
`PAPER_RESULTS.json`、`ALGORITHM_SET.json`和algorithm disposition没有再次核对路径与
SHA256。正常生成路径仍会给出正确文件，但陈旧输出根、后续覆盖或接线错误可能使“完成
状态”和最终实际读取文件分离，直到所有e200训练结束后才fail或被误接收。

commit `1dbb5f56ddf83b4dca164940280339df3d80e30c`把尚未部署的动态替换链升级为：

- unified-evaluation successor state v2在完成时绑定cohort、paper results和algorithm set的
  绝对路径与SHA256；
- final-delivery contract v3只按固定完成状态、路径和哈希放行，不解析metric payload决定
  调度；
- AM-TNC/ST-CGR完成状态必须绑定预期output root中的唯一disposition及SHA256；
- complexity结束、生成最终portfolio前再次验证全部依赖，防止长profiling期间产物变化；
- Proposal/ST-CGR跨宿主compact relation再次要求2000-step、64位e0/step core哈希、固定
  宿主和`performance_values_read=false`，不能只凭PASS字符串通过。

当前已部署但无法完成的5090A-plain legacy waiter继续原样保留；当前训练、exporter、
runtime-relation successor和registry-review successor均不受影响。等两份关系候选经Codex
显式Git准入后，新的5090B-matched-plain unified evaluator及final delivery必须从包含本提交
和新registry的干净checkout部署，不得复用v1/v2旧完成状态冒充v3闭环。

此修复只强化未来固定结果交付，不读取中间paired性能、不控制训练、不打开confirmation20。
