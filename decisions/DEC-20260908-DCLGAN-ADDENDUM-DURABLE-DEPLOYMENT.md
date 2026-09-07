# DEC-20260908：DCLGAN 非阻塞论文 addendum 已持久部署

## 发现与裁决

此前已经实现并测试 DCLGAN portfolio addendum，但当时合法的动态核心交付目录尚未冻结，
因此有意没有部署。当前核心目录已经固定为
`/home/yc/runs/FINAL_UNSB_PAPER_FINAL_DELIVERY_DYNAMIC_V1`，而项目状态仍停留在
`CODE_READY_NONBLOCKING_WAITING_FOR_DYNAMIC_CORE_OUTPUT_PATH`。若继续不部署，DCLGAN e200
会得到独立评估结果，却不会自动加入最终论文组合，这是一个真实的交付缺口。

本次给通用 control supervisor 增加了严格的 `dclgan_addendum` 角色：它只接受固定模块、
干净 Git commit、绝对状态路径和无训练/confirmation 参数的命令；允许在 DCLGAN 完整固定评估
完成后读取性能并测量 e200 复杂度，但继续拒绝 paired 控制和 confirmation20。

## 已部署链路

- 控制 commit：`589125f552d2f3e9f35bdaf036a48a2b16e49bf2`。
- 4090A 独立干净 checkout：`/home/yc/FINAL_UNSB_DCLGAN_ADDENDUM_589125f`。
- supervisor PID 638006，child PID 638015，health PID 638474；三者均脱离 SSH 存活。
- child 当前只等待核心 `PAPER_ALGORITHM_PORTFOLIO.json` 和固定
  `DCLGAN_PAPER_RESULT.json`，等待期不申请 GPU lock、不读取性能值。
- 两个依赖完成后，它会在共享评估锁下只读测量 DCLGAN e200 复杂度，并生成
  `PAPER_EXTERNAL_BASELINE_ADDENDUM.json` 与
  `PAPER_ALGORITHM_PORTFOLIO_WITH_DCLGAN.json`。这条链不阻塞更早的核心论文交付。
- GitHub 直连 clone 因4090A网络问题失败且没有留下目录；随后用本机完整 Git bundle 传输，
  本地与远端 SHA256 相同，远端只读，从中得到 clean detached checkout。没有覆盖旧环境或
  触碰 AM-TNC；PID 3446757/3446758 在部署后仍连续运行。

## 边界

这不是新增实验，也不让 DCLGAN 成为核心结果的前驱。它只关闭“外部基线已经跑完却没有进入
最终论文组合”的尾部风险。等待期不解析中间结果；完成后的指标只用于论文汇总，不能控制
训练、调度、早停或最佳 checkpoint，confirmation20 继续封存。

机器可读回执：
`evidence/paper_aio/PAPER_AIO_DCLGAN_ADDENDUM_DURABLE_DEPLOYMENT_20260908T025845.json`。
