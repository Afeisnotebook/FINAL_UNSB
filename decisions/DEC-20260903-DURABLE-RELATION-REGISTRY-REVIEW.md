# 5090B关系候选到Git审核提案的持久接力

日期：2026-09-03

5090B matched plain在CUT e200后会先执行2000-update exact runtime gate。现有两条持久
successor随后分别生成Proposal标准关系候选和ST-CGR跨代码两段证明候选；但是候选到
`PROPOSED_RUNTIME_RELATION_REGISTRY.json`之间此前只有手动命令。如果候选在无人在线时
出现，评估关键路径仍会停住。

新增`operations.paper_aio_relation_registry_review_successor`。它冻结控制commit、基础
registry和全部相关源码哈希，等待两份successor state同时达到各自精确终态。等待期间只读
状态、PID和无性能边界字段；任一上游报告BLOCKED、宿主/lane不符、exact equivalence不成立、
候选路径非绝对路径或哈希格式异常，均fail closed。

两份状态就绪后，successor从状态中取得不可变候选路径和SHA256，再调用既有确定性review
接口重新验证2000步identity、manifest、e0/step core、两类proof chain、原始receipt、宿主对、
候选文件及successor state自身哈希。输出只有Git外的proposed registry和review receipt。

该自动化明确不修改tracked registry、不授权matched delta、不启动统一评估，也不读取任何
PSNR/SSIM/LPIPS/FID/KID/ranking/delta或confirmation20。Codex仍需在候选产生后检查精确diff，
用`apply_patch`单独提交registry准入，然后才能部署替换评估链。这样消除无人接线造成的工程
停滞，同时保留跨宿主科学裁决所需的显式人工边界。
