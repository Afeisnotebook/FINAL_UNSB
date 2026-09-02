# Runtime relation候选的确定性Git审核入口

日期：2026-09-03

## 问题

Proposal与ST-CGR的两个持久successor会在5090B runtime receipt出现后分别生成review-only
关系候选，但此前从候选到Git registry之间没有独立验证入口。直接手工拼接JSON容易漏掉
关系类型、宿主对、proof chain或重复项，并可能让训练完成后的统一评估再次停住。

## 决策

新增`operations.paper_aio_relation_registry_review`。它同时接收冻结候选路径、预期SHA256、
lane、method host、plain host与当前registry，并重新验证：

- Proposal只能使用标准exact runtime relation；
- ST-CGR只能使用cross-host/cross-code candidate relation；
- 2000-step、manifest、e0/step core与所有原始回执哈希齐全；
- candidate proof chain精确为candidate-to-parent加parent-to-plain两段证明；
- 无PSNR/SSIM/LPIPS/FID/KID/ranking/delta字段；
- 同一method/plain host pair不存在歧义或冲突。

通过后只在Git外生成`PROPOSED_RUNTIME_RELATION_REGISTRY.json`和审核receipt，不修改tracked
registry，不授权比较，不启动评估。Codex仍必须审查精确差异并使用`apply_patch`形成独立Git
commit。这样既保留论文级人工裁决边界，又把容易出错的机械验证自动化。

## 边界

- 当前真实候选尚未产生，因此本次只提交接口和测试，不预先写入任何5090B关系。
- 关系审核完全metric-blind，不读取训练性能或confirmation20。
- 缺候选、哈希漂移、类型不匹配、宿主不匹配、重复或冲突一律fail closed。
- 该接口不改变任何在飞训练、后继队列或科学协议。

